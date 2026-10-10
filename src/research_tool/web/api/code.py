"""The Code & runs page: the experiment code, its smoke tests and repairs, the runs on the cluster, the results.

The Coder's progress is read from its public traces only: the files in its
workspace, its events (``phase_changed``, ``smoke_result``, ``code_fix``,
``experiment_submitted``, ``jobs_finished``), the open checkpoint's summary and
its artifacts. Edits of ``experiment.py`` are sent as ``write_file`` actions.
"""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ...agents.coder.agent import EDITABLE, GENERATED, static_problems
from ...agents.coder.results import output_name, planned_runs, read_output
from ...core.artifacts import ArtifactStore
from ...core.engine import load_status
from ...core.errors import ArtifactError, PathEscape
from ...core.events import EventLog
from ...core.fs import read_json, sha256_text
from ...core.run import RunHandle
from ...core.state import GateStatus, StageState, StageStatus
from ...core.workspace import Workspace
from ...schemas.code import ExperimentResults
from ...schemas.plan import ExperimentPlan
from ..deps import get_run

router = APIRouter(tags=["code"])

BUNDLE, RESULTS, PLAN = "code_bundle", "experiment_results", "experiment_plan"
HIDDEN = {".venv", "__pycache__", "batches"}
MAX_TEXT = 1_000_000
LOG_TAIL = 64_000
ROLES = {"run.py": "harness", "launch.py": "harness", "config.json": "config", "README.md": "readme"}


class CodeFileView(BaseModel):
    path: str  # relative to the code folder
    size: int
    lines: int
    role: Literal["experiment", "harness", "config", "readme", "other"]
    editable: bool


class SmokeView(BaseModel):
    condition: str
    ok: bool
    seconds: float
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str = ""


class FixView(BaseModel):
    ts: str
    phase: str
    signature: str
    action: str
    detail: str


class JobView(BaseModel):
    id: str
    runs: int
    partition: str
    gpu_hours: float
    submitted_at: str
    state: str | None  # once the job has finished (COMPLETED, FAILED, CANCELLED, ...)


class RunCell(BaseModel):
    condition: str
    seed: int
    status: Literal["waiting", "running", "ok", "failed", "missing"]
    metrics: dict[str, float] = Field(default_factory=dict)
    seconds: float | None = None
    error: str = ""
    log: str | None = None  # workspace-relative path of its job log, for /code/log


class LogView(BaseModel):
    name: str  # workspace-relative, for /code/log
    label: str
    size: int


class CodeView(BaseModel):
    stage: str
    status: StageStatus
    phase: str  # write | smoke | approve | submit | collect | finish ("" before the stage starts)
    code_root: str  # run-relative
    files: list[CodeFileView]
    editable: bool
    why_not_editable: str
    experiment_sha: str | None
    generated_sha: str | None  # the version the tool wrote last (None: never written, e.g. an old run)
    tested_sha: str | None  # what the open GPU checkpoint's smoke tests ran
    plan_version: int | None
    conditions: list[str]
    baseline: str | None
    seeds: list[int]
    metrics: list[str]
    smoke: list[SmokeView]  # the latest round of smoke tests
    fixes: list[FixView]
    jobs: list[JobView]
    runs: list[RunCell]
    logs: list[LogView]
    bundle_version: int | None
    results_version: int | None


class FileText(BaseModel):
    path: str
    content: str
    sha: str
    editable: bool


class CheckRequest(BaseModel):
    content: str = Field(max_length=MAX_TEXT)


class CodeCheck(BaseModel):
    problems: list[str]


class LogText(BaseModel):
    name: str
    text: str
    truncated: bool


class MetricInfo(BaseModel):
    name: str
    definition: str
    higher_is_better: bool


class ResultsView(BaseModel):
    version: int
    results: ExperimentResults
    metrics: list[MetricInfo]
    baseline: str | None
    conditions: list[str]


# ------------------------------------------------------------------ helpers
class _Code:
    def __init__(self, run: RunHandle) -> None:
        _, states, pipeline = load_status(run)
        sid = next((s for s in pipeline.order if BUNDLE in pipeline.stage(s).produces), None)
        if sid is None:
            raise HTTPException(404, "this run has no coding stage")
        self.run, self.sid, self.st = run, sid, states[sid]
        self.editable_files = pipeline.stage(sid).agent_cls.editable_files
        self.ws = Workspace(run.work_dir(sid))
        self.code = self.ws.path("code")
        self.store = ArtifactStore(run.artifacts_dir)

    def plan(self) -> ExperimentPlan | None:
        try:
            return ExperimentPlan.model_validate(self.store.envelope(PLAN, self.st.input_versions.get(PLAN)).payload)
        except ArtifactError:
            return None

    def attempt_events(self) -> list[dict[str, Any]]:
        """This stage's events since its last fresh start (a resumed attempt continues the same story)."""
        events, _ = EventLog(self.run.events_path, self.run.run_id).read_from(0)
        mine = [e for e in events if e.get("stage") == self.sid]
        starts = [
            i
            for i, e in enumerate(mine)
            if e.get("kind") == "stage_started" and not (e.get("data") or {}).get("resumed")
        ]
        return mine[starts[-1] :] if starts else mine

    def editable_now(self) -> str | None:
        """Why experiment.py cannot be edited now (None if it can)."""
        st: StageState = self.st
        at_gate = st.status is StageStatus.AWAITING_GATE and st.gate is not None and st.gate.in_stage
        if at_gate or st.status is StageStatus.FAILED:
            return None
        if st.status is StageStatus.SUSPENDED:
            return "the experiment is running on the cluster"
        if st.status is StageStatus.DONE:
            return "the stage has finished; reopen it or edit the plan to run it again"
        return "the coding agent is working on it"

    def text(self, rel: str) -> str | None:
        try:
            path = self.ws.path(rel)
        except PathEscape:
            return None
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None

    def task_logs(self) -> dict[tuple[str, int], str]:
        """(condition, seed) -> its Slurm log, from the job names and batch files (the latest job wins)."""
        logs = self.ws.path(".rtool/logs")
        pattern = re.compile(rf"^rtool-exp-{re.escape(self.sid)}-(\d+)-[^_]+_(\d+)\.out$")
        found: dict[tuple[str, int], tuple[int, str]] = {}
        for path in sorted(logs.glob("*.out")) if logs.is_dir() else []:
            match = pattern.match(path.name)
            if not match:
                continue
            number, index = int(match.group(1)), int(match.group(2))
            try:
                entry = read_json(self.code / "batches" / f"full-{number}.json")["runs"][index]
            except (OSError, ValueError, KeyError, IndexError):
                continue
            key = (str(entry["condition"]), int(entry["seed"]))
            if key not in found or found[key][0] <= number:
                found[key] = (number, f".rtool/logs/{path.name}")
        return {key: rel for key, (_, rel) in found.items()}


def _files(code: _Code) -> list[CodeFileView]:
    out = []
    root = code.code
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        rel = path.relative_to(root)
        if not path.is_file() or HIDDEN & set(rel.parts) or rel.name.startswith("."):
            continue
        name = str(rel)
        role = "experiment" if name == "experiment.py" else ROLES.get(name, "other")
        size = path.stat().st_size
        lines = path.read_text(encoding="utf-8", errors="replace").count("\n") if size <= MAX_TEXT else 0
        editable = f"code/{name}" in code.editable_files
        out.append(CodeFileView(path=name, size=size, lines=lines, role=role, editable=editable))  # type: ignore[arg-type]
    return out


def _runs(code: _Code, plan: ExperimentPlan | None, jobs: list[JobView]) -> list[RunCell]:
    if plan is None:
        return []
    logs = code.task_logs()
    st = code.st
    if st.status is StageStatus.DONE and st.output_versions.get(RESULTS):
        results = ExperimentResults.model_validate(code.store.envelope(RESULTS, st.output_versions[RESULTS]).payload)
        return [
            RunCell(
                condition=r.condition,
                seed=r.seed,
                status=r.status,
                metrics=r.metrics,
                seconds=r.seconds,
                error=r.error,
                log=logs.get((r.condition, r.seed)),
            )
            for r in results.runs
        ]
    running = st.status is StageStatus.SUSPENDED and bool(jobs)
    names = [m.name for m in plan.metrics]
    cells = []
    for condition, seed in planned_runs(plan):
        out = code.ws.path("runs") / output_name(condition, seed)
        data, problem = read_output(out, names)
        log = logs.get((condition, seed))
        if data is not None:
            cells.append(RunCell(condition=condition, seed=seed, status="ok", log=log, **data))
        elif out.exists():
            cells.append(RunCell(condition=condition, seed=seed, status="failed", error=problem, log=log))
        else:
            cells.append(RunCell(condition=condition, seed=seed, status="running" if running else "waiting", log=log))
    return cells


def _log_label(stem: str) -> str:
    if stem.startswith("smoke-"):
        return f"Smoke test: {stem.removeprefix('smoke-')}"
    return "Installing packages" if stem == "install" else stem


# ------------------------------------------------------------------ routes
@router.get("/runs/{project}/{run_id}/code", response_model=CodeView)
def code_view(project: str, run_id: str, request: Request) -> CodeView:
    code = _Code(get_run(request, project, run_id))
    st = code.st
    events = code.attempt_events()
    phases = [e["data"]["phase"] for e in events if e.get("kind") == "phase_changed" and e.get("data")]
    phase = "finish" if st.status is StageStatus.DONE else (phases[-1] if phases else "")
    smoke_from = max((i for i, e in enumerate(events) if (e.get("data") or {}).get("phase") == "smoke"), default=-1)
    smoke = [SmokeView.model_validate(e["data"]) for e in events[smoke_from + 1 :] if e.get("kind") == "smoke_result"]
    finished: dict[str, str] = {}
    for e in events:
        if e.get("kind") == "jobs_finished":
            finished.update({str(k): str(v) for k, v in ((e.get("data") or {}).get("jobs") or {}).items()})
    jobs = [
        JobView(
            id=str(d.get("job")),
            runs=int(d.get("runs") or 0),
            partition=str(d.get("partition") or ""),
            gpu_hours=float(d.get("gpu_hours") or 0),
            submitted_at=e.get("ts", ""),
            state=finished.get(str(d.get("job"))),
        )
        for e in events
        if e.get("kind") == "experiment_submitted" and (d := e.get("data") or {})
    ]
    fixes = [
        FixView(
            ts=e.get("ts", ""),
            phase=str(d.get("phase", "")),
            signature=str(d.get("signature", "")),
            action=str(d.get("action", "")),
            detail=str(d.get("detail", "")),
        )
        for e in events
        if e.get("kind") == "code_fix"
        for d in [e.get("data") or {}]
    ]
    plan = code.plan()
    experiment, generated = code.text(EDITABLE), code.text(GENERATED)
    gate = st.gate if st.gate is not None and st.gate.status is GateStatus.OPEN and st.gate.in_stage else None
    logs_dir = code.ws.path("logs")
    logs = [
        LogView(name=f"logs/{p.name}", label=_log_label(p.stem), size=p.stat().st_size)
        for p in (sorted(logs_dir.glob("*.log")) if logs_dir.is_dir() else [])
    ]
    refusal = code.editable_now()
    return CodeView(
        stage=code.sid,
        status=st.status,
        phase=phase,
        code_root=f"{code.run.work_dir(code.sid).relative_to(code.run.root)}/code",
        files=_files(code),
        editable=refusal is None and experiment is not None,
        why_not_editable=refusal or ("" if experiment is not None else "the code has not been written yet"),
        experiment_sha=sha256_text(experiment) if experiment is not None else None,
        generated_sha=sha256_text(generated) if generated is not None else None,
        tested_sha=(gate.summary.get("tested_sha") if gate else None),
        plan_version=st.input_versions.get(PLAN),
        conditions=[c.name for c in plan.run_conditions()] if plan else [],
        baseline=next((c.name for c in plan.conditions if c.is_baseline), None) if plan else None,
        seeds=plan.seeds if plan else [],
        metrics=[m.name for m in plan.metrics] if plan else [],
        smoke=smoke,
        fixes=fixes,
        jobs=jobs,
        runs=_runs(code, plan, jobs),
        logs=logs,
        bundle_version=st.output_versions.get(BUNDLE),
        results_version=st.output_versions.get(RESULTS),
    )


@router.get("/runs/{project}/{run_id}/code/file", response_model=FileText)
def code_file(project: str, run_id: str, path: str, request: Request) -> FileText:
    """One file of the experiment code (``path`` relative to the code folder)."""
    code = _Code(get_run(request, project, run_id))
    if HIDDEN & set(Path(path).parts):
        raise HTTPException(404, "no such file")
    try:
        target = code.ws.path(f"code/{path}")
    except PathEscape:
        raise HTTPException(404, "no such file") from None
    if not target.is_file() or target.stat().st_size > MAX_TEXT:
        raise HTTPException(404, "no such file (or too large to show)")
    content = target.read_text(encoding="utf-8", errors="replace")
    editable = f"code/{path}" in code.editable_files
    return FileText(path=path, content=content, sha=sha256_text(content), editable=editable)


@router.get("/runs/{project}/{run_id}/code/generated", response_model=FileText)
def generated(project: str, run_id: str, request: Request) -> FileText:
    """experiment.py as the tool last wrote it, to compare with the current file."""
    code = _Code(get_run(request, project, run_id))
    content = code.text(GENERATED)
    if content is None:
        raise HTTPException(404, "this run has no copy of the generated code")
    return FileText(path="experiment.py", content=content, sha=sha256_text(content), editable=False)


@router.post("/runs/{project}/{run_id}/code/check", response_model=CodeCheck)
def check(project: str, run_id: str, body: CheckRequest, request: Request) -> CodeCheck:
    """The checks the coding agent runs before a smoke test: syntax, the run() contract, undefined names."""
    get_run(request, project, run_id)
    with tempfile.TemporaryDirectory(prefix="rtool-check-") as folder:
        path = Path(folder) / "experiment.py"
        path.write_text(body.content, encoding="utf-8")
        return CodeCheck(problems=static_problems(body.content, path))


@router.get("/runs/{project}/{run_id}/code/log", response_model=LogText)
def log(project: str, run_id: str, name: str, request: Request) -> LogText:
    """The end of a smoke-test, install or Slurm job log (``name`` from the code view)."""
    code = _Code(get_run(request, project, run_id))
    if not (name.startswith("logs/") or name.startswith(".rtool/logs/")):
        raise HTTPException(404, "no such log")
    try:
        path = code.ws.path(name)
    except PathEscape:
        raise HTTPException(404, "no such log") from None
    if not path.is_file():
        raise HTTPException(404, "no such log")
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - LOG_TAIL))
        text = handle.read().decode("utf-8", errors="replace")
    return LogText(name=name, text=text, truncated=size > LOG_TAIL)


@router.get("/runs/{project}/{run_id}/code/archive")
def archive(project: str, run_id: str, request: Request) -> Response:
    """The experiment code as a zip file (without the environment and rtool's bookkeeping)."""
    code = _Code(get_run(request, project, run_id))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(code.code.rglob("*")) if code.code.is_dir() else []:
            rel = path.relative_to(code.code)
            if path.is_file() and not HIDDEN & set(rel.parts) and not rel.name.startswith("."):
                zf.write(path, f"{run_id}-code/{rel}")
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"content-disposition": f'attachment; filename="{run_id}-code.zip"'},
    )


@router.get("/runs/{project}/{run_id}/results", response_model=ResultsView)
def results(project: str, run_id: str, request: Request, version: int | None = None) -> ResultsView:
    """The analysed results: per-condition summaries, comparisons with the baseline, figures, notes."""
    run = get_run(request, project, run_id)
    store = ArtifactStore(run.artifacts_dir)
    try:
        envelope = store.envelope(RESULTS, version)
    except ArtifactError:
        raise HTTPException(404, "no results yet") from None
    try:
        plan = ExperimentPlan.model_validate(store.envelope(PLAN, envelope.inputs.get(PLAN)).payload)
    except ArtifactError:
        plan = None
    return ResultsView(
        version=envelope.version,
        results=ExperimentResults.model_validate(envelope.payload),
        metrics=[
            MetricInfo(name=m.name, definition=m.definition, higher_is_better=m.higher_is_better)
            for m in (plan.metrics if plan else [])
        ],
        baseline=next((c.name for c in plan.conditions if c.is_baseline), None) if plan else None,
        conditions=[c.name for c in plan.run_conditions()] if plan else [],
    )
