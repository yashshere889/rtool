"""Starting a run of a validity case: the research pipeline from the case's start, with nobody at the gates."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ..agents.coder.agent import GATE as GPU_RUNS_GATE
from ..agents.study.tasks import GATE as DATA_UPLOAD_GATE
from ..agents.tasks.fetch_resources import GATE as MANUAL_RESOURCES_GATE
from ..core.artifacts import ArtifactStore, Producer
from ..core.config import AppConfig, apply_overrides
from ..core.engine import REVISION_LIMIT_GATE
from ..core.fs import atomic_write_json, atomic_write_text
from ..core.pipeline import find_pipeline, pipeline_from
from ..core.run import RunHandle, is_valid_name
from ..packs import load_pack, pack_refusal
from ..schemas.common import SelectedItem
from ..schemas.literature import Corpus, KnowledgeMap
from ..tools.resources.fetch import content_hash
from .cases import EXPERIMENTS, MAIN_VARIANT, Case, CaseError

MARKER = "validity.json"  # in the run folder: the case as it was, and which variant and repeat this run is
OVERLAY = "validity-overlay.yaml"  # in the run folder; every episode applies it last
AGENT = "rtool validity"
# Gates opened inside a stage or by the engine; the pipeline's own gates come from its spec.
IN_STAGE_GATES = (MANUAL_RESOURCES_GATE, GPU_RUNS_GATE, DATA_UPLOAD_GATE, REVISION_LIMIT_GATE)
# Stage 0 cases are CPU-only, so experiments get no GPU-hours; the LLM's own episodes are not charged here.
DEFAULT_BUDGETS: dict[str, Any] = {"max_gpu_hours": 0.0, "max_llm_calls": 3000, "max_jobs": 40}
# The seeded corpus has no knowledge base. This path must not exist: open_corpus_kb takes any existing
# path for the knowledge base, and an empty kb_path would resolve to the run folder itself.
NO_KNOWLEDGE_BASE = "seeded/no-knowledge-base"
# Case data, generated at submit time: one folder per content hash, under the runs root (the leading
# dot keeps it out of run listings), so runs on compute nodes can read it.
DATA_FOLDER = ".validity-data"
GENERATE_TIMEOUT_S = 600


def case_overlay(gate_ids: Iterable[str], budgets: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Config for a run nobody watches: every gate decides itself, so the budgets are the only limit on spend."""
    limits = {**DEFAULT_BUDGETS, **(budgets or {})}
    if limits.get("max_gpu_hours") is None:
        raise CaseError("a validity run needs a GPU-hour budget: its GPU-runs checkpoint approves itself")
    return {
        "gates": {"default": "auto", "modes": dict.fromkeys(sorted(set(gate_ids)), "auto")},
        "budgets": {"run": limits},
    }


def seeded_artifacts(case: Case, variant: str = MAIN_VARIANT) -> dict[str, BaseModel]:
    """What the stages before the case's start would have made, as the case (and this variant) fixes it."""
    brief = case.research_brief(variant)
    seeded: dict[str, BaseModel] = {
        "research_brief": brief,
        "corpus": Corpus(
            topic=brief.topic, queries=[], papers=[], kb_path=NO_KNOWLEDGE_BASE, kb_sha256="", bibtex_path=""
        ),
        "knowledge_map": KnowledgeMap(topic=brief.topic, overview=""),
    }
    if case.hypothesis is not None:
        item = case.hypothesis.hypothesis(case.id).model_dump(mode="json")
        seeded["selected_hypothesis"] = SelectedItem(index=0, item=item)
    return seeded


def generate_data(case: Case, variant: str, data_root: Path) -> dict[str, dict[str, str]]:
    """Run each of the case's generators with this variant's arguments; the run's ``local_data`` entries.

    A generator is ``python -I <script> <output folder> <arguments as JSON>``, standard library only. Its
    output is kept under its content hash: the same data is shared, and changed data never overwrites a
    folder an earlier run read.
    """
    folder = case._folder
    if case.data and folder is None:
        raise CaseError(f"case {case.id!r} provides data: load it with load_case, which finds its generators")
    found = next((v for v in case.variants if v.name == variant), None)
    entries: dict[str, dict[str, str]] = {}
    for spec in case.data:
        assert folder is not None
        args = {**spec.args, **(found.data_args if found else {})}
        data_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data_root, prefix=".partial-") as scratch:
            out = Path(scratch) / "data"
            out.mkdir()
            try:
                result = subprocess.run(
                    [sys.executable, "-I", str(folder / spec.generator), str(out), json.dumps(args, sort_keys=True)],
                    cwd=scratch,
                    capture_output=True,
                    text=True,
                    timeout=GENERATE_TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise CaseError(f"case {case.id!r}: {spec.generator} ran over {GENERATE_TIMEOUT_S} s") from None
            if result.returncode != 0:
                last = (result.stderr.strip().splitlines() or ["no error output"])[-1]
                raise CaseError(f"case {case.id!r}: {spec.generator} failed (exit {result.returncode}): {last}")
            if not any(out.iterdir()):
                raise CaseError(f"case {case.id!r}: {spec.generator} wrote nothing into its output folder")
            target = data_root / f"{spec.name}-{content_hash(out)[:12]}"
            if not target.exists():
                out.rename(target)
        entries[spec.name] = {"path": str(target), "description": spec.description}
    return entries


def case_run_id(case: Case, variant: str, repeat: int, commit: str) -> str:
    return f"{case.id}-{variant}-r{repeat}-{commit[:7]}"


def create_case_run(
    config: AppConfig,
    case: Case,
    *,
    commit: str,
    variant: str = MAIN_VARIANT,
    repeat: int = 1,
    budgets: Mapping[str, Any] | None = None,
    project: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> RunHandle:
    """A run of ``case`` that needs nobody: seeded with what the case fixes, every gate auto, hard budgets.

    Nothing runs until an episode is started. The run folder keeps the case as it was (``validity.json``),
    so the run is scored against that, whatever happens to ``case.yaml`` later.
    """
    case.expectation(variant)  # an unknown variant fails here
    brief = case.research_brief(variant)
    texts = [brief.topic, *brief.questions, *brief.constraints, *brief.seed_queries, *brief.datasets]
    refused = pack_refusal(load_pack(case.pack), *texts, *([case.hypothesis.statement] if case.hypothesis else []))
    if refused:
        raise CaseError(f"rtool refuses this case: {refused}")
    if repeat < 1:
        raise CaseError(f"repeat must be 1 or more, not {repeat}")
    spec = pipeline_from(find_pipeline(case.pipeline_name), case.start)
    if case.kind in EXPERIMENTS and "selected_hypothesis" not in spec.initial_artifacts:
        raise CaseError(
            f"case {case.id!r} starts at {case.start!r}, where the pipeline still chooses the hypothesis; "
            f"a {case.kind} case must fix it, so it starts at 'plan' or later"
        )
    seeded = seeded_artifacts(case, variant)
    missing = [name for name in spec.initial_artifacts if name not in seeded]
    if missing:
        raise CaseError(f"a case starting at {case.start!r} would have to provide {missing}, which cases cannot")
    run_id = case_run_id(case, variant, repeat, commit)
    project = project or f"validity-{case.pack}"
    if not (is_valid_name(run_id) and is_valid_name(project)):
        raise CaseError(f"invalid run id {run_id!r} or project {project!r}")

    overlay = case_overlay([*config.gates.modes, *spec.gates, *IN_STAGE_GATES], budgets)
    runs_root = config.paths.resolved_runs_root()
    data = generate_data(case, variant, runs_root / DATA_FOLDER) if case.data else {}
    if data:
        overlay["experiments"] = {"local_data": {**config.experiments.model_dump(mode="json")["local_data"], **data}}
    overlay_path = runs_root.resolve() / project / run_id / OVERLAY
    settings = apply_overrides(config, overlay, env={})
    # episodes re-read their config files over the run's snapshot: the overlay must come last, or a file
    # such as barkla.yaml could turn a gate back to "ask" and leave the run waiting for nobody
    settings.slurm.episode.config_files = [*config.slurm.episode.config_files, str(overlay_path)]
    run = RunHandle.create(runs_root, project, spec, settings, run_id=run_id, pack=case.pack)
    atomic_write_text(run.root / OVERLAY, yaml.safe_dump(overlay, sort_keys=False))
    store = ArtifactStore(run.artifacts_dir)
    producer = Producer(stage="user", agent=AGENT)
    for name in spec.initial_artifacts:
        store.put(name, seeded[name], producer=producer)
    atomic_write_json(
        run.root / MARKER,
        {
            "case": case.model_dump(mode="json"),
            "variant": variant,
            "repeat": repeat,
            "commit": commit,
            "data": {name: entry["path"] for name, entry in data.items()},
            "provenance": dict(provenance or {}),  # which code made the run (validity.provenance)
        },
    )
    return run
