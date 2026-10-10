"""Runs as the CLI and the web UI see them: typed summaries instead of printed tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.actions import Action, ActionInbox
from ..core.artifacts import ArtifactStore, Producer
from ..core.config import AppConfig, deep_merge, default_config_files, load_config, parse_set_overrides
from ..core.editing import edit_refusal, run_editor
from ..core.engine import REVISION_LIMIT_GATE, load_status
from ..core.errors import ActionError, RToolError
from ..core.events import EventLog
from ..core.pipeline import ResolvedPipeline, find_pipeline, resolve_pipeline
from ..core.run import RunHandle, is_valid_name
from ..core.services import build_slurm
from ..core.state import GateStatus, RunState, StageState, StageStatus
from ..hpc.episode import EpisodeTracker, submit_episode
from ..packs import load_pack
from ..schemas.common import ResearchBrief

# What a human can answer at each kind of gate (select gates also take an edit with the choice;
# a reopened select gate also takes approve, which keeps the current choice).
GATE_ACTIONS = {
    "approve": ["approve", "reject"],
    "select": ["select", "reject"],
    "revision_limit": ["approve", "reject"],
}


class OpenGate(BaseModel):
    stage: str
    gate: str
    kind: str  # approve | select | revision_limit
    in_stage: bool = False  # asked by the agent mid-stage (e.g. the Coder's gpu_runs)
    nonce: str = ""  # send it back as payload.expect.gate_nonce so a stale page cannot decide
    opened_at: str = ""
    actions: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class StageSummary(BaseModel):
    id: str
    agent: str
    status: StageStatus
    attempt: int = 0
    checkpoint: str | None = None  # the gate this stage ends with, if any (a finished stage can reopen it)
    needs: list[str] = Field(default_factory=list)  # the stages it builds on
    gate: OpenGate | None = None
    stale_reason: str | None = None
    last_error: str | None = None
    waiting_on: list[str] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    output_versions: dict[str, int] = Field(default_factory=dict)
    usage: dict[str, float] = Field(default_factory=dict)
    pending_feedback: int = 0


class RunSummary(BaseModel):
    project: str
    run_id: str
    topic: str = ""
    pipeline: str = ""
    pack: str = ""
    created_at: str = ""
    updated_at: str | None = None
    last_outcome: str = "new"
    stages: list[StageSummary] = Field(default_factory=list)
    open_gates: list[OpenGate] = Field(default_factory=list)
    error: str | None = None  # the run directory could not be read


class RunStatus(RunSummary):
    path: str = ""
    tool_version: str = ""
    run_state: RunState | None = None
    queued_actions: list[Action] = Field(default_factory=list)


class KickResult(BaseModel):
    submitted: bool
    job_id: str | None = None
    detail: str = ""


class StopResult(BaseModel):
    cancelled: list[str] = Field(default_factory=list)


class EditCheck(BaseModel):
    """What an edit would store: the value as the owning agent normalised it, and what changed."""

    stage: str
    artifact: str
    base_version: int
    value: dict[str, Any]
    changed: list[str]  # top-level fields the edit changes (as submitted, before recomputing)


class Affected(BaseModel):
    """A later stage, and what happens to it if a stage's outputs change."""

    stage: str
    status: StageStatus
    effect: str  # redo | cancel_jobs | not_started
    detail: str = ""


# ------------------------------------------------------------------ config
def load_app_config(
    config_files: Iterable[Path] = (),
    sets: Iterable[str] = (),
    *,
    runs_root: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Defaults < $RTOOL_CONFIG / ~/.config/rtool/config.yaml < ``config_files`` < ``overrides`` < ``sets``."""
    merged: dict[str, Any] = deep_merge(dict(overrides or {}), parse_set_overrides(list(sets)))
    if runs_root is not None:
        merged.setdefault("paths", {})["runs_root"] = str(runs_root)
    return load_config(*default_config_files(), *config_files, overrides=merged)


# ------------------------------------------------------------ create / find
def create_run(
    config: AppConfig,
    project: str,
    topic: str,
    *,
    pipeline: str = "research",
    questions: Iterable[str] = (),
    constraints: Iterable[str] = (),
    seed_queries: Iterable[str] = (),
    pack: str = "generic",
    run_id: str | None = None,
    created_by: str = "rtool new",
) -> RunHandle:
    """Create a run directory with its research brief; nothing runs until an episode is started."""
    spec = find_pipeline(pipeline)
    resolve_pipeline(spec)  # fail now, not in the first episode
    load_pack(pack)  # a typo in the pack should not surface days later in the coder
    run = RunHandle.create(config.paths.resolved_runs_root(), project, spec, config, run_id=run_id, pack=pack)
    brief = ResearchBrief(
        topic=topic,
        questions=list(questions),
        constraints=list(constraints),
        seed_queries=list(seed_queries),
        domain_pack=pack,
    )
    ArtifactStore(run.artifacts_dir).put("research_brief", brief, producer=Producer(stage="user", agent=created_by))
    return run


def open_run(runs_root: Path, project: str, run_id: str) -> RunHandle:
    """The run ``<runs_root>/<project>/<run_id>``; both parts must be plain names (safe for user input)."""
    if not (is_valid_name(project) and is_valid_name(run_id)):
        raise RToolError(f"invalid run reference {project!r}/{run_id!r}")
    return RunHandle(Path(runs_root).expanduser() / project / run_id)


def list_runs(runs_root: Path) -> list[RunSummary]:
    """Every run under ``runs_root``, newest first. A broken run is listed with its error, never hidden."""
    root = Path(runs_root).expanduser()
    found: list[RunSummary] = []
    if not root.is_dir():
        return found
    for project in sorted(p for p in root.iterdir() if p.is_dir() and is_valid_name(p.name)):
        for run_dir in sorted(p for p in project.iterdir() if (p / "run.json").is_file() and is_valid_name(p.name)):
            try:
                found.append(run_summary(RunHandle(run_dir)))
            except Exception as exc:  # noqa: BLE001 - one unreadable run must not hide the others
                found.append(RunSummary(project=project.name, run_id=run_dir.name, error=str(exc)))
    found.sort(key=lambda s: (s.created_at, s.project, s.run_id), reverse=True)
    return found


# ----------------------------------------------------------------- inspect
def _topic(run: RunHandle) -> str:
    try:
        brief = ArtifactStore(run.artifacts_dir).get("research_brief")
    except RToolError:
        return ""
    return brief.topic if isinstance(brief, ResearchBrief) else ""


def _open_gate(sid: str, st: StageState, pipeline: ResolvedPipeline) -> OpenGate | None:
    gate = st.gate
    if st.status is not StageStatus.AWAITING_GATE or gate is None or gate.status is not GateStatus.OPEN:
        return None
    if gate.gate_id == REVISION_LIMIT_GATE:
        kind = "revision_limit"
    elif gate.in_stage:
        kind = "approve"
    else:
        spec = pipeline.stage(sid).gate
        kind = spec.kind.value if spec is not None else "approve"
    actions = list(GATE_ACTIONS[kind])
    if kind == "select" and gate.summary.get("reopened"):
        actions.insert(1, "approve")
    return OpenGate(
        stage=sid,
        gate=gate.gate_id,
        kind=kind,
        in_stage=gate.in_stage,
        nonce=gate.nonce,
        opened_at=gate.opened_at,
        actions=actions,
        summary=gate.summary,
    )


def _stage_summary(sid: str, st: StageState, pipeline: ResolvedPipeline) -> StageSummary:
    spec = pipeline.stage(sid).spec
    return StageSummary(
        id=sid,
        agent=spec.agent,
        status=st.status,
        attempt=st.attempt,
        checkpoint=spec.gate,
        needs=list(spec.needs),
        gate=_open_gate(sid, st, pipeline),
        stale_reason=st.stale_reason,
        last_error=st.last_error,
        waiting_on=list(st.waiting_on),
        started_at=st.started_at,
        finished_at=st.finished_at,
        output_versions=dict(st.output_versions),
        usage=dict(st.usage),
        pending_feedback=len(st.pending_feedback()),
    )


def run_summary(run: RunHandle) -> RunSummary:
    return RunSummary.model_validate(run_status(run).model_dump(include=set(RunSummary.model_fields)))


def run_status(run: RunHandle) -> RunStatus:
    """Stages, open gates, usage and queued actions — what ``rtool status`` shows, as data."""
    run_state, stages, pipeline = load_status(run)
    summaries = [_stage_summary(sid, stages[sid], pipeline) for sid in pipeline.order]
    return RunStatus(
        project=run.spec.project,
        run_id=run.run_id,
        topic=_topic(run),
        pipeline=run.spec.pipeline.name,
        pack=run.spec.pack,
        created_at=run.spec.created_at,
        updated_at=run_state.updated_at if run_state else None,
        last_outcome=run_state.last_outcome if run_state else "new",
        stages=summaries,
        open_gates=[s.gate for s in summaries if s.gate is not None],
        path=str(run.root),
        tool_version=run.spec.tool_version,
        run_state=run_state,
        queued_actions=[action for _, action in ActionInbox(run.actions_dir).pending()],
    )


def check_edit(run: RunHandle, artifact: str, base_version: Any, value: Any) -> EditCheck:
    """Check and normalise a hand edit without storing it (the engine repeats this when it applies the edit).

    Raises ActionError (an RToolError) when the edit would be refused.
    """
    _, stages, pipeline = load_status(run)
    owner, previous, result = run_editor(
        pipeline, ArtifactStore(run.artifacts_dir), run.config, artifact, value, base_version, run.root
    )
    refusal = edit_refusal(owner, stages[owner])
    if refusal:
        raise ActionError(refusal)
    # what the researcher changed (derived fields such as the issues list do not count)
    changed = [key for key, old in previous.payload.items() if key in value and value[key] != old]
    new = result.value.model_dump(mode="json")
    return EditCheck(stage=owner, artifact=artifact, base_version=previous.version, value=new, changed=changed)


def downstream_impact(run: RunHandle, stage: str) -> list[Affected]:
    """What a change to ``stage``'s outputs would do to the stages built on it (directly or not), in order."""
    _, stages, pipeline = load_status(run)
    if stage not in stages:
        raise RToolError(f"no stage {stage!r} in this run")
    out: list[Affected] = []
    for sid in pipeline.order:
        if stage not in pipeline.ancestors(sid):
            continue
        st = stages[sid]
        if st.status is StageStatus.SUSPENDED:
            jobs = ", ".join(st.waiting_on)
            out.append(Affected(stage=sid, status=st.status, effect="cancel_jobs", detail=f"cluster jobs {jobs}"))
        elif st.status in (StageStatus.DONE, StageStatus.AWAITING_GATE, StageStatus.RUNNING) or st.attempt > 0:
            out.append(Affected(stage=sid, status=st.status, effect="redo"))
        else:
            out.append(Affected(stage=sid, status=st.status, effect="not_started"))
    return out


# ------------------------------------------------------------------ act
def kick(run: RunHandle, reason: str, *, needs_llm: bool = False) -> KickResult:
    """Make sure an episode job will pick up queued work (real Slurm only).

    Decisions only need a CPU episode: it hands over to a GPU episode by itself
    if a stage then needs the model.
    """
    if run.config.slurm.backend != "real":
        return KickResult(submitted=False, detail="no real Slurm here: the next `rtool run` applies it")
    job_id = submit_episode(run, build_slurm(run), needs_llm=needs_llm, reason=reason)
    if job_id is None:
        return KickResult(submitted=False, detail="an episode job is already queued or running; it will pick this up")
    kind = "GPU" if needs_llm else "CPU"
    return KickResult(submitted=True, job_id=job_id, detail=f"submitted {kind} episode job {job_id}")


def queue_action(run: RunHandle, action: Action, *, wake: bool = True) -> KickResult | None:
    """Queue a decision for the next episode and (on real Slurm) make sure one comes."""
    ActionInbox(run.actions_dir).submit(action)
    return kick(run, f"action:{action.kind.value}") if wake else None


def stop_run(run: RunHandle, *, cancel_experiments: bool = False) -> StopResult:
    """Cancel the run's episode job (it stops cleanly and can be resumed) and, if asked, its experiment jobs.

    Cancelled experiment runs count as finished: the Coder reports what completed.
    """
    slurm = build_slurm(run)
    cancelled: list[str] = []
    current = EpisodeTracker(run, slurm).current()
    if current is not None:
        slurm.cancel(current["job_id"])
        cancelled.append(str(current["job_id"]))
    if cancel_experiments:
        _, stages, _ = load_status(run)
        for st in stages.values():
            if st.status is StageStatus.SUSPENDED:
                for job_id in st.waiting_on:
                    slurm.cancel(job_id)
                    cancelled.append(job_id)
    EventLog(run.events_path, run.run_id).emit("stop_requested", jobs=cancelled, experiments=cancel_experiments)
    return StopResult(cancelled=cancelled)
