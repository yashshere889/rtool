"""Coder: turns the approved plan into running experiment code, runs it on Slurm and analyses the results.

    write ─▶ smoke (CPU, inside the episode) ─▶ [gpu_runs gate] ─▶ submit (Slurm array) ─▶ collect ─▶ finish
      ▲        │ failure: playbook fix first, then the LLM (bounded)             │ failures: fix, rerun (bounded)
      └────────┘                                                                ◀┘

The LLM writes one file, ``experiment.py``, starting from the domain pack's
skeleton. rtool owns everything else: the harness (``run.py``, ``launch.py``),
``config.json``, the environment, the jobs, and every number in the results.
The stage keeps its progress in ``ctx.state``, so it survives suspensions
(waiting for jobs or a human) and crashes.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import jinja2
from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...core.context import AgentContext
from ...core.errors import ConfigError, StageFailed
from ...core.fs import atomic_write_json, atomic_write_text, canonical_json, read_json, sha256_text
from ...core.registry import register_agent
from ...hpc.jobspec import format_time_limit, parse_time_limit
from ...hpc.sbatch import validate_job
from ...llm.prompts import PromptLibrary
from ...packs import DEFAULT_PACK, Pack, load_pack
from ...schemas.code import CodeBundle, CodeFile, ExperimentResults, FixRecord, RunRecord, SmokeResult
from ...schemas.common import ResearchBrief
from ...schemas.plan import ExperimentPlan, FetchedResource, ResourceManifest
from ...tools.resources.fetch import tree_size
from . import sanity
from .env import ExperimentEnv
from .playbook import Diagnosis, declared_requirements, diagnose, halve_batch_size, merge_requirements
from .results import (
    aggregate,
    analysis_notes,
    collect,
    correction_used,
    output_name,
    planned_runs,
    read_output,
    render_figures,
    render_tables,
    results_payload,
    validate,
    variation_notes,
)
from .stats import flat

SYSTEM = (
    "You are a careful research software engineer. Write correct, simple, runnable Python that follows "
    "the stated contract exactly. Never invent data paths or results: every metric must be measured by "
    "really carrying out the experiment."
)
GATE = "gpu_runs"
HARNESS = ("run.py", "launch.py")
EDITABLE = "code/experiment.py"  # workspace-relative: the one file researchers may edit
GENERATED = "generated/experiment.py"  # the latest version the tool wrote (outside the code: for the diff)
_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)
RUN_ARGS = ("condition", "seed", "resources", "smoke", "workdir")
BROAD = ("Exception", "BaseException")


class CoderConfig(BaseModel):
    max_fix_rounds: int = Field(default=4, ge=0, le=12)  # smoke-test repairs before asking a human
    max_rerun_rounds: int = Field(default=2, ge=0, le=5)  # repairs after failed full runs
    smoke_all_conditions: bool = True  # False: smoke-test only the baseline
    resource_files: int = Field(default=15, ge=0)  # files listed per resource in the prompt


def extract_code(text: str) -> str | None:
    """The Python file in a reply: the longest fenced block (or the bare reply if it is code)."""
    blocks = _BLOCK.findall(text)
    code = max(blocks, key=len) if blocks else text.strip()
    return code.strip() + "\n" if re.search(r"^def run\(", code, re.M) else None


def run_function(tree: ast.Module) -> ast.FunctionDef | None:
    return next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)


def uses_condition(run: ast.FunctionDef) -> bool:
    """Does run()'s body read ``condition`` (or ``kwargs["condition"]``, or hand on ``locals()``)?"""
    nodes = [node for stmt in run.body for node in ast.walk(stmt)]
    for node in nodes:
        if isinstance(node, ast.Name) and node.id == "condition" and isinstance(node.ctx, ast.Load):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("locals", "vars"):
            return True
    return run.args.kwarg is not None and any(isinstance(n, ast.Constant) and n.value == "condition" for n in nodes)


def run_uses_condition(source: str) -> bool | None:
    """:func:`uses_condition` for a whole file; None if it does not parse or has no run()."""
    try:
        run = run_function(ast.parse(source))
    except SyntaxError:
        return None
    return None if run is None else uses_condition(run)


def _number(node: ast.expr) -> bool:
    """A literal int or float (possibly signed), not a bool."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def _fixed_metrics(node: ast.expr | None) -> bool:
    """A dict of fixed numbers: ``{"acc": 0.0}``, ``{m: 0.0 for m in METRICS}`` or ``dict(acc=0.0)``."""
    if isinstance(node, ast.Dict):
        return bool(node.values) and None not in node.keys and all(_number(v) for v in node.values)
    if isinstance(node, ast.DictComp):
        return _number(node.value)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        return bool(node.keywords) and not node.args and all(k.arg and _number(k.value) for k in node.keywords)
    return False


def _broad(handler: ast.ExceptHandler) -> bool:
    """A bare ``except:`` or one that catches Exception or BaseException."""
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any((getattr(t, "id", None) or getattr(t, "attr", None)) in BROAD for t in types)


def fallback_metrics(tree: ast.Module) -> list[int]:
    """Lines of broad ``except`` handlers that return fixed metric values instead of failing."""
    lines = []
    for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and _broad(n)):
        nodes = [node for stmt in handler.body for node in ast.walk(stmt)]
        fixed = {
            t.id
            for n in nodes
            if isinstance(n, ast.Assign) and _fixed_metrics(n.value)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        returned = [n.value for n in nodes if isinstance(n, ast.Return)]
        if any(_fixed_metrics(v) or (isinstance(v, ast.Name) and v.id in fixed) for v in returned):
            lines.append(handler.lineno)
    return lines


def condition_literals(source: str, conditions: list[str]) -> set[str]:
    """Condition names written as string literals, outside the skeleton's descriptive ``CONDITIONS`` table."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {c for c in conditions if re.search(rf"""['"]{re.escape(c)}['"]""", source)}
    table = {
        id(node)
        for stmt in tree.body
        if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(t, ast.Name) and t.id == "CONDITIONS"
            for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
        )
        for node in ast.walk(stmt)
    }
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in conditions and id(n) not in table
    }


def condition_code_removed(before: str, after: str, conditions: list[str]) -> list[str]:
    """Signs that a rewrite dropped condition-specific code; empty if there are none."""
    lost = sorted(condition_literals(before, conditions) - condition_literals(after, conditions))
    signs = [f"the condition name {name!r} no longer appears in the code" for name in lost]
    if run_uses_condition(before) and run_uses_condition(after) is False:
        signs.append("run() no longer uses its `condition` argument")
    return signs


def static_problems(source: str, path: Path) -> list[str]:
    """Syntax, the run() signature, undefined names and fake-result patterns, checked without running anything.

    Ignoring ``condition`` and a broad ``except`` that returns fixed metrics are
    problems (the repair loop fixes them), not gate warnings: neither has a
    legitimate use in an experiment, since run.py already records failed runs.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]
    run = run_function(tree)
    if run is None:
        return ["experiment.py must define a top-level function run(...)"]
    names = {a.arg for a in [*run.args.args, *run.args.kwonlyargs]}
    missing = [a for a in RUN_ARGS if a not in names]
    if missing and run.args.kwarg is None:
        return [f"run() must accept {', '.join(RUN_ARGS)} (missing: {', '.join(missing)})"]
    problems = []
    if not uses_condition(run):
        problems.append(
            "run() never uses its `condition` argument, so every condition would give the same results; "
            "make run() do what each condition describes"
        )
    problems += [
        f"line {line}: a broad `except` returns fixed metric values, which turns a crash into fake results; "
        "let the exception propagate (rtool records failed runs and repairs them)"
        for line in fallback_metrics(tree)
    ]
    return problems + undefined_names(path)


def undefined_names(path: Path) -> list[str]:
    """ruff's undefined-name findings for the file, if ruff is available."""
    ruff = shutil.which("ruff") or str(Path(sys.executable).with_name("ruff"))
    if not Path(ruff).exists():
        return []
    try:
        out = subprocess.run(
            [
                ruff,
                "check",
                "--no-cache",
                "--isolated",
                "--select",
                "F821,F822,F823",
                "--target-version",
                "py310",
                "--output-format",
                "concise",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.split(": ", 1)[-1] for line in out.stdout.splitlines() if re.search(r":\d+:\d+: F8", line)]


def file_listing(path: Path, limit: int) -> str:
    """'a.parquet (12 MB), b/c.json (0.1 MB), ...' — what the code will find at a resource path."""
    if limit == 0 or not path.exists():
        return ""
    if path.is_file():
        return f"{path.name} ({path.stat().st_size / 1e6:.1f} MB)"
    found: list[str] = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            file = Path(root) / name
            found.append(f"{file.relative_to(path)} ({file.stat().st_size / 1e6:.1f} MB)")
            if len(found) >= limit:
                return ", ".join(found) + ", ..."
    return ", ".join(found)


# what the experiment code depends on: a change elsewhere (wording, expected results, risks, statistics, which
# rtool computes itself) keeps the existing experiment.py instead of having it written again
CODE_FIELDS = {
    "hypothesis": True,
    "design": True,
    "independent_variables": True,
    "dependent_variables": True,
    "controls": True,
    "conditions": True,
    "ablations": True,
    "metrics": True,
    "seeds": True,
    "procedure": True,
    "resources": {"__all__": {"kind", "name", "source", "identifier", "subset", "split"}},
    "compute": {"gpus_per_run", "gpu_memory_gb", "gpu_type"},
}


def plan_hash(plan: ExperimentPlan) -> str:
    return sha256_text(canonical_json(plan.model_dump(mode="json", include=CODE_FIELDS)))


def same_metrics(a: dict[str, float], b: dict[str, float], names: Sequence[str] | None = None) -> bool:
    """The planned metrics (all of them if ``names`` is None) are equal up to rounding in both runs.

    Extra values an experiment reports, such as timings, always differ a little and would hide identical results.
    """
    shared = [k for k in a if k in b and (names is None or k in names)]
    return bool(shared) and all(flat([a[k], b[k]]) for k in shared)


def smoke_warnings(results: list[SmokeResult], baseline: str, names: Sequence[str] | None = None) -> list[str]:
    """Signs in passing smoke runs that run() ignores the seed or the condition.

    Shown at the gpu_runs gate, never fatal: some experiments are deterministic
    by design, and a tiny smoke problem can give equal numbers by chance.
    ``names``: the plan's metrics, the only values compared.
    """
    if not results:
        return []
    warnings = []
    base = [r for r in results if r.condition == baseline]
    if len(base) >= 2:
        seeds = (base[0].seed or 0, base[1].seed or 0)
        warnings.append(sanity.seed_warning(baseline, seeds, base[0].metrics, base[1].metrics, names))
    first = [r for r in results if r.seed == results[0].seed]  # one run per condition, same seed
    if len(first) >= 2 and all(same_metrics(r.metrics, first[0].metrics, names) for r in first[1:]):
        warnings.append(
            f"all {len(first)} conditions gave identical smoke metrics: run() may not implement the conditions"
        )
    return [w for w in warnings if w]


@register_agent("coder")
class CoderAgent(Agent):
    description = "Writes the experiment code, smoke-tests and debugs it, runs it on Slurm and analyses the results."
    consumes = ("research_brief", "experiment_plan", "resource_manifest")
    produces = ("code_bundle", "experiment_results")
    config_model = CoderConfig
    editable_files = (EDITABLE,)  # rtool rewrites every other file on each attempt

    def run(self, ctx: AgentContext) -> None:
        CoderSession(ctx, self.config).drive()


class CoderSession:
    """One attempt of the coder stage; every step reads and writes ``ctx.state``."""

    def __init__(self, ctx: AgentContext, cfg: CoderConfig) -> None:
        self.ctx, self.cfg, self.state = ctx, cfg, ctx.state
        self.plan: ExperimentPlan = ctx.inputs.get("experiment_plan")
        self.manifest: ResourceManifest = ctx.inputs.get("resource_manifest")
        brief: ResearchBrief = ctx.inputs.get("research_brief")
        try:
            self.pack: Pack = load_pack(brief.domain_pack)
        except ConfigError:
            ctx.log(f"unknown domain pack {brief.domain_pack!r}; using {DEFAULT_PACK!r}")
            self.pack = load_pack(DEFAULT_PACK)
        self.prompts = PromptLibrary.for_agent(__package__, "coder", pack=self.pack.name)
        self.provenance: dict[str, str] = self.state.setdefault("prompts", {})
        ws = ctx.workspace
        self.code, self.logs = ws.mkdir("code"), ws.mkdir("logs")
        self.runs_dir, self.scratch, self.results_dir = ws.mkdir("runs"), ws.mkdir("scratch"), ws.mkdir("results")
        self.resources = self.resource_paths()
        root = Path(self.manifest.root)
        binds = [str(root)] if self.manifest.root and root.is_absolute() else []
        factory = ctx.tools.get("experiment_env")
        self.env: ExperimentEnv = (
            factory(ctx.app_config.experiments, self.code)
            if factory
            else ExperimentEnv(ctx.app_config.experiments, self.code, binds=binds)
        )
        self.metric_names = [m.name for m in self.plan.metrics]
        self.conditions = [c.name for c in self.plan.run_conditions()]
        self.plan_hash = plan_hash(self.plan)

    # ------------------------------------------------------------------ driver
    def drive(self) -> None:
        steps = {
            "write": self.write,
            "smoke": self.smoke,
            "approve": self.approve,
            "submit": self.submit,
            "collect": self.collect,
        }
        if "phase" not in self.state:
            self.goto("write")
        while (phase := self.state["phase"]) != "finish":
            steps[phase]()
        self.finish()

    def goto(self, phase: str) -> None:
        if self.state.get("phase") != phase:
            self.ctx.emit("phase_changed", phase=phase)  # the web UI follows the stage through these
        self.state["phase"] = phase
        # waiting for the GPU-runs decision, submitting, collecting and analysing need no model: they
        # can resume on a CPU node instead of queueing for a GPU (and hand over if a fix needs the LLM)
        self.ctx.resume_without_llm(phase in ("approve", "submit", "collect", "finish"))
        self.ctx.checkpoint()

    def require_llm(self) -> None:
        """On a CPU node this hands the stage to a GPU episode, so call it before changing any state."""
        _ = self.ctx.llm

    # -------------------------------------------------------------- prompts
    def render(self, name: str, **variables: Any) -> str:
        prompt = self.prompts.render(name, **variables)
        self.provenance[name] = prompt.template_hash
        return prompt.text

    def packages(self) -> list[str]:
        site = self.ctx.app_config.experiments.packages
        return list(dict.fromkeys([*site, *self.pack.packages])) or ["(the Python standard library)"]

    def feedback(self) -> list[str]:
        items = [
            f.text + (f" — required: {'; '.join(f.required_changes)}" if f.required_changes else "")
            for f in self.ctx.feedback
        ]
        return items + list(self.state.get("gate_feedback", []))

    def ask_for_code(self, template: str, **variables: Any) -> str | None:
        prompt = self.render(template, **variables)
        for overrides in ({}, {"reasoning": "off"}):  # second try: no long chain of thought before the code
            reply = self.ctx.llm.chat(prompt, profile="code", system=SYSTEM, **overrides)
            code = extract_code(reply.text)
            if code is not None:
                return code
        return None

    # ---------------------------------------------------------------- files
    def resource_paths(self) -> dict[str, str]:
        return {r.name: r.path for r in self.manifest.resources if r.path and r.status in ("ready", "manual")}

    @staticmethod
    def provided(resource: FetchedResource) -> bool:
        """Downloaded, or (manual) somebody put files in its folder; an empty manual folder is not data."""
        if not resource.path or resource.status not in ("ready", "manual"):
            return False
        return resource.status == "ready" or (Path(resource.path).is_dir() and tree_size(Path(resource.path))[1] > 0)

    def available(self) -> dict[str, str]:
        """What the code may read: name -> local path."""
        return {r.name: r.path for r in self.manifest.resources if self.provided(r) and r.path}

    def missing(self) -> list[FetchedResource]:
        """Manual resources nobody provided: the code has to build them (or the researcher add them)."""
        return [r for r in self.manifest.resources if r.status == "manual" and not self.provided(r)]

    def missing_text(self) -> list[dict[str, str]]:
        purposes = {r.name: r.purpose for r in self.plan.resources}
        return [{"name": r.name, "kind": r.kind, "purpose": purposes.get(r.name, "")} for r in self.missing()]

    def prepare(self) -> None:
        """Harness, config.json and the code README's facts: rtool's files, rewritten every attempt."""
        harness = resources.files(__package__).joinpath("harness")
        for name in HARNESS:
            atomic_write_text(self.code / name, harness.joinpath(name).read_text(encoding="utf-8"))
        config = {
            "hypothesis": self.plan.hypothesis,
            "conditions": {
                c.name: {"description": c.description, "baseline": c.is_baseline} for c in self.plan.run_conditions()
            },
            "metrics": {
                m.name: {"definition": m.definition, "higher_is_better": m.higher_is_better} for m in self.plan.metrics
            },
            "seeds": self.plan.seeds,
            "resources": self.resources,
        }
        atomic_write_json(self.code / "config.json", config)

    def skeleton(self) -> str:
        source = self.pack.template or load_pack(DEFAULT_PACK).template
        return jinja2.Template(source, keep_trailing_newline=True).render(
            hypothesis=self.plan.hypothesis.replace('"""', "'''"),
            conditions=repr({c.name: c.description for c in self.plan.run_conditions()}),
            metrics=repr(self.metric_names),
        )

    def marker(self) -> dict[str, Any]:
        path = self.code / ".rtool.json"
        return read_json(path) if path.exists() else {}

    def save_source(self, source: str, origin: str) -> None:
        atomic_write_text(self.code / "experiment.py", source)
        atomic_write_text(self.ctx.workspace.path(GENERATED), source)  # what a hand edit is compared with
        atomic_write_json(
            self.code / ".rtool.json", {"plan": self.plan_hash, "sha": sha256_text(source), "origin": origin}
        )
        self.ctx.checkpoint()

    def source(self) -> str:
        return (self.code / "experiment.py").read_text(encoding="utf-8")

    def requirements(self) -> list[str]:
        return merge_requirements(
            declared_requirements(self.source()),
            list(self.state.get("extra_requirements", [])),
            provided=self.ctx.app_config.experiments.packages,
        )

    # ---------------------------------------------------------------- write
    def write(self) -> None:
        self.prepare()
        path, marker, feedback = self.code / "experiment.py", self.marker(), self.feedback()
        same_plan = path.exists() and marker.get("plan") == self.plan_hash
        resources_text = [
            {**r.model_dump(), "files": file_listing(Path(r.path), self.cfg.resource_files) if r.path else ""}
            for r in self.manifest.resources
            if r.path and self.provided(r)
        ]
        if same_plan and not feedback:
            edited = sha256_text(self.source()) != marker.get("sha")
            self.ctx.log("keeping the existing experiment.py" + (" (edited by hand)" if edited else ""))
            if edited:
                self.record(phase="check", signature="human_edit", action="kept the researcher's edits")
        else:
            if same_plan:
                code = self.ask_for_code(
                    "revise.j2",
                    feedback=feedback,
                    plan=self.plan,
                    conditions=self.conditions,
                    metrics=self.metric_names,
                    resources=self.available(),
                    missing=self.missing_text(),
                    packages=self.packages(),
                    source=self.source(),
                )
            else:
                code = self.ask_for_code(
                    "implement.j2",
                    plan=self.plan,
                    conditions=self.plan.run_conditions(),
                    resources=resources_text,
                    deferred=[r for r in self.manifest.resources if r.status == "deferred"],
                    missing=self.missing_text(),
                    packages=self.packages(),
                    gpu=self.gpu_text(),
                    pack=self.pack,
                    feedback=feedback,
                    skeleton=self.skeleton(),
                )
            if code is None:
                raise StageFailed("the model did not return experiment.py in a ```python block", needs_human=True)
            self.save_source(code, "llm")
        self.state.pop("gate_feedback", None)
        self.check()
        self.goto("smoke")

    def check(self) -> None:
        """Static problems go through the repair loop after writing and before every smoke test (fixes add some)."""
        while problems := static_problems(self.source(), self.code / "experiment.py"):
            self.repair(Diagnosis("static_check", problems[0]), log="\n".join(problems), condition="", phase="check")

    def gpu_text(self) -> str:
        c = self.plan.compute
        return f"{c.gpus_per_run}x {c.gpu_type or 'GPU'} on {c.partition}" if c.gpus_per_run else "no GPU (CPU only)"

    # ---------------------------------------------------------------- smoke
    def smoke(self) -> None:
        if self.state.pop("needs_human", False):  # a person fixed things and ran `rtool retry --resume`
            self.state["fix_rounds"] = 0
            if sha256_text(self.source()) != self.marker().get("sha"):
                self.record(phase="smoke", signature="human_edit", action="re-tested the researcher's edits")
        while True:
            self.check()
            installed = self.env.install(self.requirements(), log_path=self.logs / "install.log")
            if installed is not None and not installed.ok:
                self.repair(
                    Diagnosis("install_failed", f"pip could not install {self.requirements()}"),
                    log=installed.tail(40),
                    condition="",
                    phase="smoke",
                )
                continue
            failure = self.run_smoke()
            if failure is None:
                self.state["tested_sha"] = sha256_text(self.source())  # what the GPU-runs approval refers to
                self.goto("approve")
                return
            condition, diagnosis, log = failure
            self.repair(diagnosis, log=log, condition=condition, phase="smoke")

    def run_smoke(self) -> tuple[str, Diagnosis, str] | None:
        """Every condition (or only the baseline) with the first seed, then the baseline with a second seed."""
        baseline = next((c.name for c in self.plan.conditions if c.is_baseline), self.conditions[0])
        conditions = self.conditions if self.cfg.smoke_all_conditions else [baseline]
        seeds = self.plan.seeds or [0]
        seed = seeds[0]
        second = next((s for s in seeds if s != seed), seed + 1)  # does the seed change anything?
        smoke_dir = self.ctx.workspace.mkdir("smoke")
        results: list[SmokeResult] = []
        failure = None
        for condition, s in [*((c, seed) for c in conditions), (baseline, second)]:
            name = condition if s == seed else f"{condition}__{s}"
            out = smoke_dir / f"{name}.json"
            out.unlink(missing_ok=True)
            run = self.env.run(
                [
                    "run.py",
                    "--condition",
                    condition,
                    "--seed",
                    str(s),
                    "--output",
                    str(out),
                    "--smoke",
                    "--workdir",
                    str(self.scratch / "smoke" / name),
                ],
                log_path=self.logs / f"smoke-{name}.log",
                timeout_s=self.ctx.app_config.experiments.smoke_timeout_s,
            )
            data, problem = read_output(out, self.metric_names)
            if run.ok and data is not None:
                results.append(
                    SmokeResult(condition=condition, seed=s, ok=True, seconds=run.seconds, metrics=data["metrics"])
                )
                if s == seed:  # the UI lists one smoke test per condition; the second seed only checks the seed
                    self.ctx.emit("smoke_result", **results[-1].model_dump())
                continue
            log = run.tail(200)
            found = diagnose(log, exit_code=run.exit_code, timed_out=run.timed_out)
            if found.signature == "time_limit":
                limit = self.ctx.app_config.experiments.smoke_timeout_s
                found = Diagnosis(
                    "smoke_timeout",
                    f"the smoke run did not finish within {limit:g} s; smoke mode must use a tiny problem",
                )
            elif run.ok:
                found = Diagnosis("contract", problem)
            results.append(SmokeResult(condition=condition, seed=s, ok=False, seconds=run.seconds, error=found.summary))
            self.ctx.emit("smoke_result", **results[-1].model_dump())
            failure = (condition, found, "\n".join(log.splitlines()[-80:]))
            break
        self.state["smoke"] = [r.model_dump() for r in results]
        warnings = [] if failure else smoke_warnings(results, baseline, self.metric_names)
        self.state["smoke_warnings"] = warnings
        if warnings:
            self.ctx.emit("smoke_warning", warnings=warnings)
        return failure

    def warnings(self) -> list[str]:
        """Signs that the code may not measure anything real (see sanity.py), for people to check."""
        found = [sanity.missing_resources_warning([r.name for r in self.missing()])]
        found += sanity.code_warnings(self.source(), has_resources=bool(self.available()))
        found += self.state.get("smoke_warnings", [])
        return [w for w in found if w]

    # --------------------------------------------------------------- repair
    def record(self, *, phase: str, signature: str, action: str, detail: str = "") -> None:
        fixes = self.state.setdefault("fixes", [])
        fixes.append(
            FixRecord(
                round=len(fixes) + 1,
                phase=phase,
                signature=signature,
                action=action,  # type: ignore[arg-type]
                detail=detail[:300],
            ).model_dump()
        )
        self.ctx.emit("code_fix", phase=phase, signature=signature, action=action, detail=detail[:300])

    def repair(self, diagnosis: Diagnosis, *, log: str, condition: str, phase: str) -> None:
        """One repair round: the playbook if it knows the failure, else the LLM. Bounded."""
        rounds = self.state.get("fix_rounds", 0) + 1
        if rounds > self.cfg.max_fix_rounds:
            self.state["needs_human"] = True
            self.ctx.checkpoint()
            where = self.ctx.workspace.path("code/experiment.py")
            raise StageFailed(
                f"the experiment code still fails after {rounds - 1} repair rounds ({diagnosis.summary}). "
                f"Fix {where} by hand (logs: {self.logs}), then run `rtool retry <run> --resume`.",
                needs_human=True,
            )
        if not self.playbook_applies(diagnosis):
            self.require_llm()  # before anything changes: a hand-over must not count a round twice
        self.state["fix_rounds"] = rounds
        action = self.playbook_fix(diagnosis)
        if action is None:
            history = [f["detail"] for f in self.state.get("fixes", []) if f.get("detail")][-4:]
            code = self.ask_for_code(
                "fix.j2",
                summary=diagnosis.summary,
                condition=condition,
                phase=phase,
                log_tail=log[-6000:],
                source=self.source(),
                metrics=self.metric_names,
                conditions=self.conditions,
                resources=self.available(),
                missing=self.missing_text(),
                packages=self.packages(),
                smoke=self.pack.smoke,
                history=history,
            )
            if code is None:
                action = "the model returned no code; trying again"
            else:
                removed = condition_code_removed(self.source(), code, self.conditions)
                if removed:
                    self.reject_rewrite(code, removed, diagnosis=diagnosis, phase=phase)
                self.save_source(code, "llm")
                action = "rewrote experiment.py (LLM)"
        self.record(phase=phase, signature=diagnosis.signature, action=action, detail=diagnosis.summary)
        self.ctx.checkpoint()

    def reject_rewrite(self, code: str, removed: list[str], *, diagnosis: Diagnosis, phase: str) -> None:
        """Stop for a person: the model "fixed" the code by dropping what makes the conditions differ.

        experiment.py stays as it was; the rejected rewrite is kept in the logs for comparison.
        """
        rejected = self.logs / f"experiment.rejected-{self.state.get('fix_rounds', 0)}.py"
        atomic_write_text(rejected, code)
        self.record(
            phase=phase,
            signature="condition_code_removed",
            action="rejected the model's rewrite",
            detail="; ".join(removed),
        )
        self.state["needs_human"] = True
        if phase == "full":  # once fixed: a smoke test, a fresh approval and every run again
            self.state.get("_gates", {}).pop(GATE, None)
            self.state.pop("pending", None)
            self.goto("smoke")
        self.ctx.checkpoint()
        where = self.ctx.workspace.path("code/experiment.py")
        raise StageFailed(
            f"the model's fix for '{diagnosis.summary}' removed condition-specific code ({'; '.join(removed)}), "
            f"so the conditions might no longer differ. {where} is unchanged; the rejected rewrite is {rejected}. "
            "Fix experiment.py by hand, then run `rtool retry <run> --resume`.",
            needs_human=True,
        )

    def playbook_applies(self, diagnosis: Diagnosis) -> bool:
        """Can the playbook handle this without the LLM? (No side effects.)"""
        if diagnosis.signature == "time_limit":
            return True
        if diagnosis.signature == "missing_module" and diagnosis.requirement:
            extra = self.state.get("extra_requirements", [])
            return diagnosis.requirement not in extra and diagnosis.requirement not in self.requirements()
        return diagnosis.memory and halve_batch_size(self.source()) is not None

    def playbook_fix(self, diagnosis: Diagnosis) -> str | None:
        extra: list[str] = self.state.setdefault("extra_requirements", [])
        if diagnosis.signature == "missing_module" and diagnosis.requirement:
            if diagnosis.requirement in extra or diagnosis.requirement in self.requirements():
                return None  # installed already and still missing: let the LLM look at it
            extra.append(diagnosis.requirement)
            return f"added requirement {diagnosis.requirement}"
        if diagnosis.memory:
            smaller = halve_batch_size(self.source())
            if smaller is not None:
                self.save_source(smaller, "llm")
                size = re.search(r"^BATCH_SIZE\s*=\s*(\d+)", smaller, re.M)
                return f"halved BATCH_SIZE to {size.group(1) if size else '?'}"
        return None

    # -------------------------------------------------------------- approve
    def approve(self) -> None:
        tested = self.state.get("tested_sha")
        if tested and sha256_text(self.source()) != tested:
            # experiment.py was edited (by hand, at this gate) after its smoke test passed: test the edit
            # before anyone can approve GPU time for it; any earlier decision referred to the old code
            self.retest("the researcher's edits")
            return
        runs = planned_runs(self.plan)
        hours = self.gpu_hours(len(runs), self.plan.compute.time_limit)
        smoke = [SmokeResult.model_validate(r) for r in self.state.get("smoke", [])]
        c = self.plan.compute
        warnings = self.warnings()
        check = (
            " Check experiment.py before approving: "
            + " ".join(f"({n}) {w}" + ("" if w.endswith(("?", ".")) else ".") for n, w in enumerate(warnings, 1))
            if warnings
            else ""
        )
        summary = {
            "message": (
                f"Smoke tests passed on CPU for {len({r.condition for r in smoke})} condition(s). Approve to run "
                f"{len(runs)} jobs on {c.partition} ({self.gpu_text()}, up to {c.time_limit} each; at most "
                f"{hours:g} GPU-hours). Reject with feedback to have the code revised first." + check
            ),
            "warnings": warnings,
            "smoke": ", ".join(
                f"{r.condition}{'' if r.seed == smoke[0].seed else f' seed {r.seed}'} ok in {r.seconds:g}s"
                for r in smoke
            ),
            "runs": len(runs),
            "partition": c.partition,
            "gpu_hours": hours,
            "time_limit": c.time_limit,
            "tested_sha": tested,  # the experiment.py these smoke tests ran (a later edit is tested again first)
            "review_files": [str(self.code / "experiment.py"), str(self.code / "config.json")],
        }
        decision = self.ctx.gate(GATE, summary)
        if decision.approved:
            self.state.setdefault("pending", [list(r) for r in runs])  # a rerun keeps its own list
            self.goto("submit")
            return
        if decision.data.get("retest"):  # "test it again", not "rewrite it": no model involved
            self.retest("the code, as asked")
            return
        self.state.get("_gates", {}).pop(GATE, None)  # ask again after the revision
        self.state["gate_feedback"] = [decision.feedback or "The researcher rejected the GPU runs; improve the code."]
        self.state["fix_rounds"] = 0
        self.goto("write")

    def retest(self, what: str) -> None:
        """Static checks and smoke tests again, then a fresh GPU-runs gate."""
        self.state.get("_gates", {}).pop(GATE, None)
        self.state["fix_rounds"] = 0
        if sha256_text(self.source()) != self.marker().get("sha"):
            self.record(phase="smoke", signature="human_edit", action=f"re-testing {what}")
        self.check()
        self.goto("smoke")

    def gpu_hours(self, runs: int, time_limit: str) -> float:
        return round(runs * self.plan.compute.gpus_per_run * parse_time_limit(time_limit) / 3600, 2)

    # --------------------------------------------------------------- submit
    def submit(self) -> None:
        ctx, s = self.ctx, self.state
        pending = [tuple(r) for r in s.pop("pending", None) or planned_runs(self.plan)]
        jobs: list[dict[str, Any]] = s.setdefault("jobs", [])
        number = len(jobs) + 1
        batch = f"batches/full-{number}.json"
        entries = [
            {
                "condition": condition,
                "seed": seed,
                "output": str(self.runs_dir / output_name(condition, seed)),
                "workdir": str(self.scratch / f"{condition}__{seed}"),
            }
            for condition, seed in pending
        ]
        atomic_write_json(self.code / batch, {"runs": entries})
        for entry in entries:
            Path(entry["output"]).unlink(missing_ok=True)  # never mistake an old result for a new one
        time_limit = s.get("time_limit", self.plan.compute.time_limit)
        compute = self.plan.compute.model_copy(update={"time_limit": time_limit})
        spec = self.env.array_job(
            name=f"rtool-exp-{ctx.stage_id}-{number}",
            workdir=ctx.workspace.root,
            batch_file=batch,
            tasks=len(entries),
            compute=compute,
        )
        slurm_cfg = ctx.app_config.slurm
        allowed = (
            [*slurm_cfg.experiment_partitions, slurm_cfg.episode.cpu_partition]
            if slurm_cfg.experiment_partitions
            else None
        )
        problems = validate_job(spec, slurm_cfg, allowed_partitions=allowed)
        if problems:
            raise StageFailed("cannot submit the experiment: " + "; ".join(problems), needs_human=True)
        hours = self.gpu_hours(len(entries), time_limit)
        over = ctx.budget.would_exceed(gpu_hours=hours, jobs=1)
        if over:
            raise StageFailed(f"{over}; raise budgets.run.max_gpu_hours or shrink the plan", needs_human=True)
        job_id = ctx.slurm.submit(spec)
        ctx.budget.charge(gpu_hours=hours, jobs=1)
        jobs.append(
            {"id": job_id, "name": spec.name, "batch": batch, "runs": [list(r) for r in pending], "gpu_hours": hours}
        )
        s.setdefault("job_of", {}).update({output_name(c, seed): job_id for c, seed in pending})
        ctx.emit("experiment_submitted", job=job_id, runs=len(entries), partition=spec.partition, gpu_hours=hours)
        self.goto("collect")
        ctx.suspend_until([job_id], reason=f"{len(entries)} experiment runs on {spec.partition}")

    # -------------------------------------------------------------- collect
    def task_log(self, condition: str, seed: int) -> str:
        """End of the Slurm log of the array task that ran (condition, seed)."""
        for job in reversed(self.state.get("jobs", [])):
            runs = [tuple(r) for r in job["runs"]]
            if (condition, seed) in runs:
                log = (
                    self.ctx.workspace.path(".rtool/logs")
                    / f"{job['name']}-{job['id']}_{runs.index((condition, seed))}.out"
                )
                try:
                    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
                except OSError:
                    return ""
        return ""

    def collect(self) -> None:
        s, job = self.state, self.state["jobs"][-1]
        result = self.ctx.job_result(job["id"])
        if result is None:
            self.ctx.suspend_until([job["id"]], reason="waiting for experiment runs")

        def failure_of(condition: str, seed: int) -> str:
            log = self.task_log(condition, seed)
            return diagnose(log).summary if log else ""

        records = collect(self.plan, self.runs_dir, job_of=s.get("job_of", {}), failure_of=failure_of)
        s["records"] = [r.model_dump() for r in records]
        failed = [r for r in records if r.status != "ok"]
        if not failed or s.get("rerun_rounds", 0) >= self.cfg.max_rerun_rounds:
            self.goto("finish")
            return
        first = failed[0]
        log = self.task_log(first.condition, first.seed)
        state = str(result.get("state", ""))
        if state == "CANCELLED":  # someone cancelled the job: do not undo that; report what there is
            self.record(phase="full", signature="cancelled", action="job cancelled; kept the finished runs")
            self.goto("finish")
            return
        if not log.strip() or state in ("PREEMPTED", "NODE_FAIL"):  # never ran, or killed: not the code's fault
            s["rerun_rounds"] = s.get("rerun_rounds", 0) + 1
            self.record(
                phase="full", signature="not_run", action=f"resubmitted {len(failed)} runs ({state or 'no log'})"
            )
            s["pending"] = [[r.condition, r.seed] for r in failed]
            self.goto("submit")
            return
        found = diagnose(log, job_state=state)
        if not self.playbook_applies(found):
            self.require_llm()  # collecting ran on a CPU node; this repair needs the model
        s["rerun_rounds"] = s.get("rerun_rounds", 0) + 1
        self.ctx.emit("experiment_runs_failed", failed=len(failed), signature=found.signature, summary=found.summary)
        rerun = self.fix_full_runs(found, log)
        if rerun == "none":
            self.goto("finish")
        elif rerun == "all":  # the code changed: every run again, after a smoke test and a fresh approval
            s.get("_gates", {}).pop(GATE, None)
            s.pop("pending", None)
            s["fix_rounds"] = 0
            self.goto("smoke")
        else:
            s["pending"] = [[r.condition, r.seed] for r in failed]
            # new packages are installed (and smoke-tested) first; the approval still stands
            self.goto("smoke" if rerun == "install" else "submit")

    def fix_full_runs(self, diagnosis: Diagnosis, log: str) -> str:
        """What to repeat: 'failed' (more time), 'install' (new packages), 'all' (code changed) or 'none'."""
        if diagnosis.signature == "time_limit":
            partition = self.ctx.app_config.slurm.partitions.get(self.plan.compute.partition)
            current = parse_time_limit(self.state.get("time_limit", self.plan.compute.time_limit))
            ceiling = parse_time_limit(partition.max_time) if partition else current * 2
            if current >= ceiling:
                self.record(phase="full", signature="time_limit", action="already at the partition's limit")
                return "none"
            self.state["time_limit"] = format_time_limit(min(ceiling, current * 1.5))
            self.record(phase="full", signature="time_limit", action=f"time limit raised to {self.state['time_limit']}")
            return "failed"
        before = sha256_text(self.source())
        requirements_before = self.requirements()
        self.state["fix_rounds"] = 0
        self.repair(diagnosis, log=log, condition="", phase="full")
        if sha256_text(self.source()) != before:
            return "all"
        return "install" if self.requirements() != requirements_before else "none"

    # --------------------------------------------------------------- finish
    def finish(self) -> None:
        ctx = self.ctx
        records = [RunRecord.model_validate(r) for r in self.state.get("records", [])]
        ok = [r for r in records if r.status == "ok"]
        if not ok:
            raise StageFailed(
                f"no experiment run produced results; see {self.logs} and {ctx.workspace.path('.rtool/logs')}",
                needs_human=True,
            )
        summaries, comparisons = aggregate(self.plan, records)
        notes = (
            [f"{len(records) - len(ok)} of {len(records)} planned runs have no results"]
            if len(ok) < len(records)
            else []
        )
        notes += analysis_notes(self.plan, comparisons)
        notes += variation_notes(self.plan, records, comparisons)
        warnings = self.warnings() + sanity.results_warnings(self.plan, records)
        notes += [f"Warning: {w}" for w in warnings]
        payload = results_payload(records, notes)
        problems = validate(payload, self.plan.results_schema)
        if problems:
            notes.append("results.json does not fully match the plan's schema: " + "; ".join(problems[:5]))
            payload = results_payload(records, notes)
        results_json = self.results_dir / "results.json"
        atomic_write_text(results_json, json.dumps(payload, indent=2) + "\n")
        figures = render_figures(self.plan, summaries, self.results_dir / "figures")
        tables = self.results_dir / "results.md"
        atomic_write_text(
            tables,
            render_tables(
                self.plan,
                records,
                summaries,
                comparisons,
                {m: f"figures/{p.name}" for m, p in figures.items()},
                warnings=warnings,
            ),
        )
        bundle = self.bundle()
        ctx.outputs.put("code_bundle", bundle, meta={"review": bundle.readme_path, "prompts": self.provenance})
        jobs = self.state.get("jobs", [])
        ctx.outputs.put(
            "experiment_results",
            ExperimentResults(
                runs=records,
                summaries=summaries,
                comparisons=comparisons,
                complete=len(ok) == len(records),
                results_path=ctx.relative(results_json),
                tables_path=ctx.relative(tables),
                figures=[ctx.relative(p) for p in figures.values()],
                jobs=[j["id"] for j in jobs],
                gpu_hours_requested=round(sum(j.get("gpu_hours", 0) for j in jobs), 2),
                code_fingerprint=bundle.fingerprint,
                notes=notes,
                warnings=warnings,
                correction=correction_used(self.plan, comparisons),
            ),
            meta={"review": ctx.relative(tables)},
        )

    def bundle(self) -> CodeBundle:
        marker = self.marker()
        files: list[CodeFile] = []
        for path in sorted(self.code.rglob("*")):
            rel = path.relative_to(self.code)
            skip = {".venv", "batches", "__pycache__"} & set(rel.parts) or rel.name == ".rtool.json"
            if skip or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            sha = sha256_text(text)
            origin = "harness" if rel.name != "experiment.py" else ("llm" if sha == marker.get("sha") else "human")
            files.append(CodeFile(path=str(rel), sha256=sha, lines=text.count("\n"), origin=origin))
        fingerprint = sha256_text("\n".join(f"{f.path}:{f.sha256}" for f in files))
        readme = self.code / "README.md"
        atomic_write_text(readme, self.readme(files))
        return CodeBundle(
            root=self.ctx.relative(self.code),
            files=files,
            pack=self.pack.name,
            container=self.ctx.app_config.experiments.container,
            requirements=self.requirements(),
            smoke=[SmokeResult.model_validate(r) for r in self.state.get("smoke", [])],
            warnings=list(self.state.get("smoke_warnings", [])),
            fixes=[FixRecord.model_validate(f) for f in self.state.get("fixes", [])],
            readme_path=self.ctx.relative(readme),
            fingerprint=fingerprint,
        )

    def readme(self, files: list[CodeFile]) -> str:
        python = " ".join(self.env.python(["run.py"], gpu=True, for_job=True))
        lines = [
            "# Experiment code",
            "",
            f"Hypothesis: {self.plan.hypothesis}",
            "",
            f"Domain pack: `{self.pack.name}` · environment: "
            + (f"`{self.ctx.app_config.experiments.container}`" if self.env.containerized else "local Python"),
            "",
            "## Files",
            "",
            *[f"- `{f.path}` ({f.lines} lines, {f.origin})" for f in files],
            "",
            "`experiment.py` is the experiment (edit freely, then `rtool retry <run> --resume`); `run.py` and "
            "`launch.py` are rtool's harness; `config.json` holds the plan's conditions, metrics and data paths.",
            "",
            "## Run one condition by hand",
            "",
            f"    cd {self.code}",
            f"    {python} --condition {self.conditions[0]} --seed 0 --output out.json --smoke",
            "",
            "## Smoke tests",
            "",
            *[
                f"- `{r['condition']}`"
                + (f" (seed {r['seed']})" if r.get("seed") is not None else "")
                + ": "
                + (f"ok in {r['seconds']:g} s" if r["ok"] else f"failed: {r.get('error', '')}")
                for r in self.state.get("smoke", [])
            ],
            *[f"- Warning: {w}" for w in self.state.get("smoke_warnings", [])],
        ]
        fixes = self.state.get("fixes", [])
        if fixes:
            lines += ["", "## Repairs", ""]
            lines += [f"{f['round']}. {f['phase']} · {f['signature']}: {f['action']}" for f in fixes]
        return "\n".join(lines) + "\n"
