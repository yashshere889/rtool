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
from ...schemas.plan import ExperimentPlan, ResourceManifest
from .env import ExperimentEnv
from .playbook import Diagnosis, declared_requirements, diagnose, halve_batch_size, merge_requirements
from .results import (
    aggregate,
    collect,
    output_name,
    planned_runs,
    read_output,
    render_figures,
    render_tables,
    results_payload,
    validate,
)

SYSTEM = (
    "You are a careful research software engineer. Write correct, simple, runnable Python that follows "
    "the stated contract exactly. Never invent data paths or results."
)
GATE = "gpu_runs"
HARNESS = ("run.py", "launch.py")
EDITABLE = "code/experiment.py"  # workspace-relative: the one file researchers may edit
GENERATED = "generated/experiment.py"  # the latest version the tool wrote (outside the code: for the diff)
_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)
RUN_ARGS = ("condition", "seed", "resources", "smoke", "workdir")


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


def static_problems(source: str, path: Path) -> list[str]:
    """Syntax, the run() signature and undefined names, checked without running anything."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]
    run = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if run is None:
        return ["experiment.py must define a top-level function run(...)"]
    names = {a.arg for a in [*run.args.args, *run.args.kwonlyargs]}
    missing = [a for a in RUN_ARGS if a not in names]
    if missing and run.args.kwarg is None:
        return [f"run() must accept {', '.join(RUN_ARGS)} (missing: {', '.join(missing)})"]
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
            if r.path and r.status in ("ready", "manual")
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
                    resources=self.resources,
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
        self.static_checks()
        self.goto("smoke")

    def static_checks(self) -> None:
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
        baseline = next((c.name for c in self.plan.conditions if c.is_baseline), self.conditions[0])
        conditions = self.conditions if self.cfg.smoke_all_conditions else [baseline]
        seed = self.plan.seeds[0] if self.plan.seeds else 0
        smoke_dir = self.ctx.workspace.mkdir("smoke")
        results: list[SmokeResult] = []
        failure = None
        for condition in conditions:
            out = smoke_dir / f"{condition}.json"
            out.unlink(missing_ok=True)
            run = self.env.run(
                [
                    "run.py",
                    "--condition",
                    condition,
                    "--seed",
                    str(seed),
                    "--output",
                    str(out),
                    "--smoke",
                    "--workdir",
                    str(self.scratch / "smoke" / condition),
                ],
                log_path=self.logs / f"smoke-{condition}.log",
                timeout_s=self.ctx.app_config.experiments.smoke_timeout_s,
            )
            data, problem = read_output(out, self.metric_names)
            if run.ok and data is not None:
                results.append(SmokeResult(condition=condition, ok=True, seconds=run.seconds, metrics=data["metrics"]))
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
            results.append(SmokeResult(condition=condition, ok=False, seconds=run.seconds, error=found.summary))
            self.ctx.emit("smoke_result", **results[-1].model_dump())
            failure = (condition, found, "\n".join(log.splitlines()[-80:]))
            break
        self.state["smoke"] = [r.model_dump() for r in results]
        return failure

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
                resources=self.resources,
                packages=self.packages(),
                smoke=self.pack.smoke,
                history=history,
            )
            if code is None:
                action = "the model returned no code; trying again"
            else:
                self.save_source(code, "llm")
                action = "rewrote experiment.py (LLM)"
        self.record(phase=phase, signature=diagnosis.signature, action=action, detail=diagnosis.summary)
        self.ctx.checkpoint()

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
        summary = {
            "message": (
                f"Smoke tests passed on CPU for {len(smoke)} condition(s). Approve to run {len(runs)} jobs on "
                f"{c.partition} ({self.gpu_text()}, up to {c.time_limit} each; at most {hours:g} GPU-hours). "
                "Reject with feedback to have the code revised first."
            ),
            "smoke": ", ".join(f"{r.condition} ok in {r.seconds:g}s" for r in smoke),
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
        self.static_checks()
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
        if comparisons:
            notes.append(f"comparisons use the {comparisons[0].test} against the baseline")
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
                self.plan, records, summaries, comparisons, {m: f"figures/{p.name}" for m, p in figures.items()}
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
                f"- `{r['condition']}`: "
                + (f"ok in {r['seconds']:g} s" if r["ok"] else f"failed: {r.get('error', '')}")
                for r in self.state.get("smoke", [])
            ],
        ]
        fixes = self.state.get("fixes", [])
        if fixes:
            lines += ["", "## Repairs", ""]
            lines += [f"{f['round']}. {f['phase']} · {f['signature']}: {f['action']}" for f in fixes]
        return "\n".join(lines) + "\n"
