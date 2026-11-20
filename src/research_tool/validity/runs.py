"""Starting a run of a validity case: the research pipeline from the case's start, with nobody at the gates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import yaml
from pydantic import BaseModel

from ..agents.coder.agent import GATE as GPU_RUNS_GATE
from ..agents.tasks.fetch_resources import GATE as MANUAL_RESOURCES_GATE
from ..core.artifacts import ArtifactStore, Producer
from ..core.config import AppConfig, apply_overrides
from ..core.engine import REVISION_LIMIT_GATE
from ..core.fs import atomic_write_json, atomic_write_text
from ..core.pipeline import find_pipeline, pipeline_from
from ..core.run import RunHandle, is_valid_name
from ..schemas.common import SelectedItem
from ..schemas.literature import Corpus, KnowledgeMap
from .cases import MAIN_VARIANT, Case, CaseError

MARKER = "validity.json"  # in the run folder: the case as it was, and which variant and repeat this run is
OVERLAY = "validity-overlay.yaml"  # in the run folder; every episode applies it last
AGENT = "rtool validity"
# Gates opened inside a stage or by the engine; the pipeline's own gates come from its spec.
IN_STAGE_GATES = (MANUAL_RESOURCES_GATE, GPU_RUNS_GATE, REVISION_LIMIT_GATE)
# Stage 0 cases are CPU-only, so experiments get no GPU-hours; the LLM's own episodes are not charged here.
DEFAULT_BUDGETS: dict[str, Any] = {"max_gpu_hours": 0.0, "max_llm_calls": 3000, "max_jobs": 40}
# The seeded corpus has no knowledge base. This path must not exist: open_corpus_kb takes any existing
# path for the knowledge base, and an empty kb_path would resolve to the run folder itself.
NO_KNOWLEDGE_BASE = "seeded/no-knowledge-base"


def case_overlay(gate_ids: Iterable[str], budgets: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Config for a run nobody watches: every gate decides itself, so the budgets are the only limit on spend."""
    limits = {**DEFAULT_BUDGETS, **(budgets or {})}
    if limits.get("max_gpu_hours") is None:
        raise CaseError("a validity run needs a GPU-hour budget: its GPU-runs checkpoint approves itself")
    return {
        "gates": {"default": "auto", "modes": dict.fromkeys(sorted(set(gate_ids)), "auto")},
        "budgets": {"run": limits},
    }


def seeded_artifacts(case: Case) -> dict[str, BaseModel]:
    """What the stages before the case's start would have made, as the case fixes it."""
    brief = case.research_brief()
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
) -> RunHandle:
    """A run of ``case`` that needs nobody: seeded with what the case fixes, every gate auto, hard budgets.

    Nothing runs until an episode is started. The run folder keeps the case as it was (``validity.json``),
    so the run is scored against that, whatever happens to ``case.yaml`` later.
    """
    if case.data:
        raise CaseError(f"case {case.id!r} provides data, which needs the `local` resource source (Stage 0 PR 2)")
    case.expectation(variant)  # an unknown variant fails here
    if repeat < 1:
        raise CaseError(f"repeat must be 1 or more, not {repeat}")
    spec = pipeline_from(find_pipeline("research"), case.start)
    if case.kind != "open" and "selected_hypothesis" not in spec.initial_artifacts:
        raise CaseError(
            f"case {case.id!r} starts at {case.start!r}, where the pipeline still chooses the hypothesis; "
            f"a {case.kind} case must fix it, so it starts at 'plan' or later"
        )
    seeded = seeded_artifacts(case)
    missing = [name for name in spec.initial_artifacts if name not in seeded]
    if missing:
        raise CaseError(f"a case starting at {case.start!r} would have to provide {missing}, which cases cannot")
    run_id = case_run_id(case, variant, repeat, commit)
    project = project or f"validity-{case.pack}"
    if not (is_valid_name(run_id) and is_valid_name(project)):
        raise CaseError(f"invalid run id {run_id!r} or project {project!r}")

    overlay = case_overlay([*config.gates.modes, *spec.gates, *IN_STAGE_GATES], budgets)
    runs_root = config.paths.resolved_runs_root()
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
        {"case": case.model_dump(mode="json"), "variant": variant, "repeat": repeat, "commit": commit},
    )
    return run
