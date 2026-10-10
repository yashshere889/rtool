"""The pipeline engine: advances a run until it has to wait.

One call to :meth:`Engine.advance` is an *episode*. It takes the run lease,
applies queued human actions, wakes stages whose Slurm jobs have finished,
runs every stage that is ready, and returns as soon as the run needs a human
(a gate), a Slurm job, or more time than the episode has. Progress is
persisted after every transition, so the process can be killed at any point;
the next episode recovers (an interrupted stage resumes from its last
checkpoint, an interrupted commit is completed).

Stage lifecycle::

    PENDING ─run─▶ RUNNING ─commit─▶ DONE
                     │  ├─ ctx.suspend_until(jobs) ─▶ SUSPENDED ─jobs end─▶ PENDING(resume)
                     │  ├─ ctx.gate(id)            ─▶ AWAITING_GATE ─decision─▶ PENDING(resume)
                     │  └─ error                   ─▶ PENDING(retry) | FAILED
                     └─ outputs committed, stage gate declared ─▶ AWAITING_GATE ─approve─▶ DONE
                                                                               └─reject─▶ STALE
    DONE ─input changed / change requested─▶ STALE ─run─▶ ...
"""

from __future__ import annotations

import copy
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

from ..schemas.common import SelectedItem
from .actions import Action, ActionInbox, ActionKind
from .artifacts import ArtifactStore, Producer, StagedOutputs, artifact_model
from .budgets import BudgetTracker, Usage
from .config import GateMode
from .context import RESUME_WITHOUT_LLM, AgentContext, InputView
from .editing import describe, edit_refusal, run_editor
from .errors import (
    ActionError,
    ArtifactError,
    BudgetExceeded,
    GateRequested,
    LLMRequired,
    RToolError,
    StageFailed,
    SuspendRequested,
)
from .events import EventLog
from .fs import FileLease, now_iso, sha256_text
from .pipeline import GateKind, GateSpec, ResolvedPipeline, ResolvedStage, get_by_path, resolve_pipeline
from .registry import AgentRegistry
from .registry import registry as default_registry
from .run import RunHandle
from .services import Services
from .state import (
    ChangeRequest,
    FeedbackItem,
    FeedbackKind,
    GateResolution,
    GateState,
    GateStatus,
    RunState,
    StageState,
    StageStatus,
    StateStore,
    new_gate_nonce,
)
from .workspace import Workspace

REVISION_LIMIT_GATE = "revision_limit"
MAX_EDITED_FILE_BYTES = 1_000_000


class Outcome(StrEnum):
    DONE = "done"  # every stage finished
    WAITING_GATE = "waiting_gate"  # a human must approve / reject / select
    WAITING_JOB = "waiting_job"  # Slurm jobs are pending or running
    NEEDS_HUMAN = "needs_human"  # a stage failed (rtool retry) or the run is blocked
    NEEDS_LLM = "needs_llm"  # a ready stage needs the LLM, which this episode can't provide (no GPU)
    CONTINUE = "continue"  # time budget used up; more work is ready
    BUSY = "busy"  # another episode holds the run


@dataclass
class AdvanceResult:
    outcome: Outcome
    ran: list[str] = field(default_factory=list)
    open_gates: list[dict[str, Any]] = field(default_factory=list)
    waiting_jobs: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)  # ready stages this episode was not allowed to run
    message: str = ""


def _option_title(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("title", "name", "statement", "summary", "text"):
            if isinstance(item.get(key), str):
                return item[key][:160]
    return str(item)[:160]


def _option_model(gate: GateSpec) -> type[BaseModel] | None:
    """The schema of one option at a selection gate, when the options artifact declares ``list[Model]`` there."""
    assert gate.options_artifact and gate.options_path
    try:
        annotation: Any = artifact_model(gate.options_artifact)
    except ArtifactError:
        return None
    for part in gate.options_path.split("."):
        fields = annotation.model_fields if isinstance(annotation, type) and issubclass(annotation, BaseModel) else {}
        if part not in fields:
            return None
        annotation = fields[part].annotation
    args = get_args(annotation)
    if get_origin(annotation) is list and args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
        return args[0]
    return None


def edit_option(gate: GateSpec, option: Any, edit: Any) -> dict[str, Any]:
    """The chosen option with the human's changes, checked against the option's schema (ActionError if invalid)."""
    item = dict(option) if isinstance(option, dict) else {"value": option}
    if not edit:
        return item
    if not isinstance(edit, dict):
        raise ActionError("an edit must map field names to new values")
    model = _option_model(gate)
    if model is None:
        return item | edit
    unknown = sorted(set(edit) - set(model.model_fields))
    if unknown:
        raise ActionError(f"the choice has no field {', '.join(map(repr, unknown))}")
    try:
        return model.model_validate(item | edit).model_dump(mode="json")
    except ValidationError as exc:
        raise ActionError(f"the edited choice is not valid: {describe(exc, 3)}") from None


class Engine:
    def __init__(
        self,
        run: RunHandle,
        services: Services | None = None,
        *,
        registry: AgentRegistry = default_registry,
        episode_id: str | None = None,
        stage_filter: Callable[[ResolvedStage], bool] | None = None,
    ) -> None:
        """``stage_filter`` limits which ready stages this episode may run (e.g. no LLM stages on a CPU node)."""
        self.run = run
        self.config = run.config
        self.services = services or Services()
        self.clock = self.services.clock
        self.episode_id = episode_id or f"ep-{uuid.uuid4().hex[:8]}"
        self.stage_filter = stage_filter
        self._deferred: set[str] = set()
        self.pipeline: ResolvedPipeline = resolve_pipeline(run.spec.pipeline, registry)
        self._ancestors = {sid: self.pipeline.ancestors(sid) for sid in self.pipeline.order}
        self.store = ArtifactStore(run.artifacts_dir)
        self.states = StateStore(run.state_dir)
        self.inbox = ActionInbox(run.actions_dir)
        self.events = EventLog(run.events_path, run.run_id, self.episode_id)
        self.run_state: RunState | None = None
        self.stages: dict[str, StageState] = {}

    # ------------------------------------------------------------------ episode
    def advance(self, *, max_seconds: float | None = None) -> AdvanceResult:
        lease = FileLease(
            self.run.lease_path,
            ttl=self.config.episode.lease_ttl_seconds,
            owner=self.episode_id,
            clock=self.clock,
        )
        if not lease.acquire():
            holder = lease.holder()
            owner = holder.owner if holder else "another episode"
            return AdvanceResult(Outcome.BUSY, message=f"run is held by {owner}")
        deadline = None if max_seconds is None else self.clock() + max_seconds
        ran: list[str] = []
        time_up = False
        try:
            self._load()
            assert self.run_state is not None
            self.run_state.active_episode = self.episode_id
            self.run_state.episodes += 1
            self._save_run()
            self.events.emit("episode_started", number=self.run_state.episodes)
            self._recover()
            self._propagate_staleness(None)  # artifacts may have been changed outside the engine
            while True:
                lease.renew()
                self._apply_actions()
                self._poll_jobs()
                stage_id = self._next_runnable()
                if stage_id is None:
                    break
                if deadline is not None and self.clock() >= deadline:
                    time_up = True
                    break
                self._run_stage(stage_id)
                ran.append(stage_id)
            result = self._summarize(ran, time_up)
            self.run_state.last_outcome = result.outcome.value
            self.events.emit("episode_finished", outcome=result.outcome.value, ran=ran)
            return result
        finally:
            if self.run_state is not None:
                self.run_state.active_episode = None
                self._save_run()
            lease.release()

    # ------------------------------------------------------------- persistence
    def _load(self) -> None:
        self.run_state = self.states.load_run() or RunState(run_id=self.run.run_id, nonce=uuid.uuid4().hex[:12])
        self.stages = self.states.load_stages(self.pipeline.order)

    def _save_run(self) -> None:
        assert self.run_state is not None
        self.states.save_run(self.run_state)

    def _save(self, st: StageState) -> None:
        self.states.save_stage(st)

    def _recover(self) -> None:
        """Repair stages left RUNNING by a killed episode."""
        for sid in self.pipeline.order:
            st = self.stages[sid]
            if st.status is not StageStatus.RUNNING:
                continue
            if st.commit_pending:
                self.events.emit("commit_recovered", stage=sid, attempt=st.attempt)
                self._finish_commit(sid)
            else:
                st.status = StageStatus.PENDING
                st.resume = True
                self._save(st)
                self.events.emit("stage_interrupted", stage=sid, attempt=st.attempt)

    # --------------------------------------------------------------- scheduling
    def _next_runnable(self) -> str | None:
        for sid in self.pipeline.order:
            st = self.stages[sid]
            if st.status not in (StageStatus.PENDING, StageStatus.STALE) or sid in self._deferred:
                continue
            # Every upstream stage (transitively) must be settled, not just direct deps:
            # a revision of a grand-parent must finish before this stage re-runs.
            if all(self.stages[a].status is StageStatus.DONE for a in self._ancestors[sid]):
                cpu_resume = st.resume and bool(st.agent_state.get(RESUME_WITHOUT_LLM))
                if self.stage_filter is not None and not cpu_resume and not self.stage_filter(self.pipeline.stage(sid)):
                    self._deferred.add(sid)
                    continue
                return sid
        return None

    def _resolve_inputs(self, rs: ResolvedStage) -> dict[str, int]:
        versions: dict[str, int] = {}
        for name in rs.consumes:
            head = self.store.head(name)
            if head is None:
                raise ArtifactError(f"required input {name!r} does not exist")
            versions[name] = head
        for name in rs.optional_inputs:
            head = self.store.head(name)
            if head is not None:
                versions[name] = head
        return versions

    # ---------------------------------------------------------- running stages
    def _run_stage(self, sid: str) -> None:
        assert self.run_state is not None
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        staged = StagedOutputs(self.store, sid, allowed=set(rs.agent_produces))
        resumed = st.resume
        if not resumed:
            st.attempt += 1
            st.token = f"{self.run_state.nonce}:{sid}:{st.attempt}"
            st.agent_state = {}
            st.job_results = {}
            st.gate = None
            st.started_at = now_iso()
            staged.discard()
            try:
                st.input_versions = self._resolve_inputs(rs)
            except ArtifactError as exc:
                self._fail(sid, str(exc), retryable=False)
                return
        st.status = StageStatus.RUNNING
        st.resume = False
        st.waiting_on = []
        st.stale_reason = None
        self._save(st)
        self.events.emit("stage_started", stage=sid, agent=rs.spec.agent, attempt=st.attempt, resumed=resumed)

        budget = BudgetTracker(
            self.config.budgets.for_stage(sid),
            self.config.budgets.run,
            stage_usage=Usage.model_validate(st.usage),
            run_usage=Usage.model_validate(self.run_state.usage),
            label=f"stage {sid}",
        )
        feedback = st.pending_feedback()
        ctx = self._context(rs, st, staged, budget, resumed, feedback)
        started = self.clock()
        try:
            rs.agent_cls(rs.config).run(ctx)
        except SuspendRequested as signal:
            self._record_usage(st, budget, started)
            st.agent_state = ctx.state
            st.status = StageStatus.SUSPENDED
            st.waiting_on = list(signal.job_ids)
            self._save(st)
            self.events.emit("stage_suspended", stage=sid, jobs=signal.job_ids, reason=signal.reason)
        except LLMRequired:
            # Started on a CPU node (it said it could) but needs the model after all: hand over.
            self._record_usage(st, budget, started)
            st.agent_state = ctx.state
            st.status = StageStatus.PENDING
            st.resume = True
            self._deferred.add(sid)
            self._save(st)
            self.events.emit("stage_deferred", stage=sid, reason="needs the LLM")
        except GateRequested as signal:
            self._record_usage(st, budget, started)
            st.agent_state = ctx.state
            st.status = StageStatus.AWAITING_GATE
            st.gate = GateState(gate_id=signal.gate_id, in_stage=True, summary=signal.summary, nonce=new_gate_nonce())
            self._save(st)
            self.events.emit("gate_opened", stage=sid, gate=signal.gate_id, in_stage=True)
        except BudgetExceeded as exc:
            self._record_usage(st, budget, started)
            self._fail(sid, str(exc), agent_state=ctx.state, retryable=False)
        except StageFailed as exc:
            self._record_usage(st, budget, started)
            self._fail(sid, str(exc), agent_state=ctx.state, retryable=not exc.needs_human)
        except Exception as exc:  # noqa: BLE001 - an agent bug must fail the stage, not the episode
            self._record_usage(st, budget, started)
            self._fail(sid, f"{type(exc).__name__}: {exc}", agent_state=ctx.state, tb=traceback.format_exc())
        else:
            self._record_usage(st, budget, started)
            self._complete(sid, ctx, staged, feedback)

    def _context(
        self,
        rs: ResolvedStage,
        st: StageState,
        staged: StagedOutputs,
        budget: BudgetTracker,
        resumed: bool,
        feedback: list[FeedbackItem],
    ) -> AgentContext:
        sid = rs.id
        llm = None
        if rs.agent_cls.needs_llm and self.services.llm is not None:
            llm = self.services.llm.bind(
                budget=budget,
                on_call=lambda info: self.events.emit("llm_call", stage=sid, **info),
                stage=sid,
            )

        def checkpoint(state: dict[str, Any]) -> None:
            st.agent_state = copy.deepcopy(state)
            self._save(st)

        return AgentContext(
            stage_id=sid,
            agent_name=rs.spec.agent,
            attempt=st.attempt,
            config=rs.config,
            app_config=self.config,
            inputs=InputView(self.store, st.input_versions),
            outputs=staged,
            state=copy.deepcopy(st.agent_state),
            resumed=resumed,
            feedback=feedback,
            events=self.events,
            budget=budget,
            workspace=Workspace(self.run.work_dir(sid)),
            llm=llm,
            slurm=self.services.slurm,
            job_results=dict(st.job_results),
            checkpoint=checkpoint,
            run_root=self.run.root,
            tools=self.services.tools,
            # on a CPU-only episode an LLM stage may only run because it said it could resume without the
            # model; if it reaches for the model anyway, it is handed to a GPU episode (LLMRequired)
            defer_without_llm=rs.agent_cls.needs_llm and llm is None and self.stage_filter is not None,
        )

    def _record_usage(self, st: StageState, budget: BudgetTracker, started: float) -> None:
        assert self.run_state is not None
        wall = max(0.0, self.clock() - started)
        st.usage = budget.stage_usage.plus(wall_seconds=wall).model_dump()
        self.run_state.usage = budget.run_usage.plus(wall_seconds=wall).model_dump()
        self._save_run()

    def _producer(self, rs: ResolvedStage, st: StageState) -> Producer:
        return Producer(
            stage=rs.id,
            agent=rs.spec.agent,
            agent_version=rs.agent_cls.version,
            attempt=st.attempt,
            token=st.token,
        )

    def _complete(self, sid: str, ctx: AgentContext, staged: StagedOutputs, feedback: list[FeedbackItem]) -> None:
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        missing = [name for name in rs.agent_produces if not staged.has(name)]
        if missing:
            self._fail(sid, f"agent finished without producing {missing}", agent_state=ctx.state)
            return
        st.agent_state = ctx.state
        st.outbox = ctx.change_requests
        for item in feedback:
            item.consumed_by_attempt = st.attempt
        st.commit_pending = True
        self._save(st)
        self._finish_commit(sid)

    def _finish_commit(self, sid: str) -> None:
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        versions = StagedOutputs(self.store, sid).commit(self._producer(rs, st), dict(st.input_versions))
        for name in rs.agent_produces:
            if name not in versions:  # committed before a crash; its staging file is already gone
                head = self.store.head(name)
                if head is None:
                    self._fail(sid, f"output {name!r} was lost while committing")
                    return
                versions[name] = head
        st.output_versions.update(versions)
        for name in rs.all_inputs:  # refining one's own input is not an input change
            if name in versions:
                st.input_versions[name] = versions[name]
        st.commit_pending = False
        st.last_error = None
        st.failures = 0
        self._save(st)
        for name, version in sorted(versions.items()):
            self.events.emit("artifact_committed", stage=sid, artifact=name, version=version)
        self._propagate_staleness(sid)
        waiting = self._apply_change_requests(sid, st.outbox) if st.outbox else False
        st.outbox = []
        self._save(st)
        if not waiting:
            self._open_gate_or_finish(sid)

    def _fail(
        self,
        sid: str,
        message: str,
        *,
        agent_state: dict[str, Any] | None = None,
        tb: str | None = None,
        retryable: bool = True,
    ) -> None:
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        if agent_state is not None:
            st.agent_state = agent_state
        st.failures += 1
        st.last_error = message
        st.waiting_on = []
        st.commit_pending = False
        if retryable and st.failures <= rs.spec.retries:
            st.status = StageStatus.PENDING
            st.resume = False
            self._save(st)
            self.events.emit("stage_retrying", stage=sid, attempt=st.attempt, error=message, failures=st.failures)
        else:
            st.status = StageStatus.FAILED
            self._save(st)
            self.events.emit("stage_failed", stage=sid, attempt=st.attempt, error=message, traceback=tb)

    def _finish(self, sid: str) -> None:
        st = self.stages[sid]
        st.status = StageStatus.DONE
        st.finished_at = now_iso()
        self._save(st)
        self.events.emit("stage_finished", stage=sid, attempt=st.attempt, outputs=st.output_versions)

    # ------------------------------------------------------------- staleness
    def _mark_stale(self, sid: str, reason: str) -> None:
        st = self.stages[sid]
        if st.status is StageStatus.SUSPENDED and st.waiting_on and self.services.slurm is not None:
            for job_id in st.waiting_on:
                try:
                    self.services.slurm.cancel(job_id)
                except Exception as exc:  # noqa: BLE001 - best effort
                    self.events.emit("job_cancel_failed", stage=sid, job=job_id, error=str(exc))
            self.events.emit("jobs_cancelled", stage=sid, jobs=st.waiting_on)
        if st.gate is not None and st.gate.status is GateStatus.OPEN:
            st.gate.status = GateStatus.CANCELLED
            st.gate.resolved_at = now_iso()
        st.status = StageStatus.STALE
        st.stale_reason = reason
        st.resume = False
        st.waiting_on = []
        self._save(st)
        self.events.emit("stage_stale", stage=sid, reason=reason)

    def _propagate_staleness(self, changed_sid: str | None) -> None:
        heads = self.store.heads()
        for sid in self.pipeline.order:
            if sid == changed_sid:
                continue
            st = self.stages[sid]
            live = st.status in (StageStatus.DONE, StageStatus.AWAITING_GATE, StageStatus.SUSPENDED) or (
                st.status is StageStatus.PENDING and st.resume
            )
            if not live:
                continue
            for name in self.pipeline.stage(sid).all_inputs:
                if name in st.input_versions and heads.get(name) != st.input_versions[name]:
                    self._mark_stale(sid, f"input {name} changed (v{st.input_versions[name]} -> v{heads.get(name)})")
                    break

    # -------------------------------------------------------- change requests
    def _apply_change_requests(self, sid: str, requests: list[ChangeRequest]) -> bool:
        """Apply revisions requested by ``sid``. Returns True if ``sid`` must now wait."""
        limit = self.config.revisions.max_rounds
        accepted: list[ChangeRequest] = []
        over_limit: list[ChangeRequest] = []
        for request in requests:
            if request.target_stage not in self._ancestors[sid]:
                self.events.emit(
                    "change_request_ignored",
                    stage=sid,
                    target=request.target_stage,
                    reason="target is not upstream of the requesting stage",
                )
            elif self.stages[request.target_stage].revision_count >= limit:
                over_limit.append(request)
            else:
                accepted.append(request)
        for request in accepted:
            self._apply_revision(request)
        st = self.stages[sid]
        pending_targets = sorted({r.target_stage for r in accepted})
        if over_limit and self.config.gates.mode(REVISION_LIMIT_GATE) is GateMode.AUTO:
            # Unattended runs accept the current outputs instead of stopping.
            self.events.emit("gate_auto_resolved", stage=sid, gate=REVISION_LIMIT_GATE, ignored=len(over_limit))
            over_limit = []
        if over_limit:
            st.status = StageStatus.AWAITING_GATE
            st.gate = GateState(
                gate_id=REVISION_LIMIT_GATE,
                nonce=new_gate_nonce(),
                summary={
                    "message": (
                        f"Revision limit ({limit}) reached. Approve to accept the current outputs; "
                        "reject to force one more revision round."
                    ),
                    "requests": [r.model_dump(mode="json") for r in over_limit],
                    "pending_revisions": pending_targets,
                },
            )
            self._save(st)
            self.events.emit("gate_opened", stage=sid, gate=REVISION_LIMIT_GATE, in_stage=False)
            return True
        if accepted:
            st.status = StageStatus.STALE
            st.stale_reason = "waiting for revisions of " + ", ".join(pending_targets)
            st.resume = False
            self._save(st)
            self.events.emit("stage_waiting_for_revisions", stage=sid, targets=pending_targets)
            return True
        return False

    def _apply_revision(self, request: ChangeRequest, *, forced: bool = False) -> None:
        target = self.stages[request.target_stage]
        target.revision_count += 1
        target.feedback.append(
            FeedbackItem(
                kind=FeedbackKind.CHANGE_REQUEST,
                text=request.reason,
                from_stage=request.from_stage,
                required_changes=request.required_changes,
                severity=request.severity,
            )
        )
        self._mark_stale(request.target_stage, f"change requested by {request.from_stage}: {request.reason}")
        self.events.emit(
            "revision_requested",
            stage=request.target_stage,
            by=request.from_stage,
            round=target.revision_count,
            forced=forced,
        )

    # ------------------------------------------------------------------ gates
    def _open_gate_or_finish(self, sid: str) -> None:
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        if rs.gate is None or rs.spec.gate is None:
            self._finish(sid)
            return
        gate_id = rs.spec.gate
        try:
            summary = self._gate_summary(rs, st)
        except ActionError as exc:
            self._fail(sid, f"cannot open gate {gate_id!r}: {exc}", retryable=False)
            return
        st.gate = GateState(gate_id=gate_id, summary=summary, nonce=new_gate_nonce())
        if self.config.gates.mode(gate_id) is GateMode.AUTO:
            self._save(st)
            self.events.emit("gate_auto_resolved", stage=sid, gate=gate_id)
            self._resolve_stage_gate(sid, approved=True, index=0, auto=True)
            return
        st.status = StageStatus.AWAITING_GATE
        self._save(st)
        self.events.emit("gate_opened", stage=sid, gate=gate_id, in_stage=False, prompt=rs.gate.prompt)

    def _gate_options(self, rs: ResolvedStage, st: StageState) -> list[Any]:
        gate = rs.gate
        assert gate is not None and gate.options_artifact and gate.options_path
        version = st.output_versions.get(gate.options_artifact)
        payload = self.store.envelope(gate.options_artifact, version).payload
        try:
            options = get_by_path(payload, gate.options_path)
        except KeyError:
            raise ActionError(f"{gate.options_artifact} has no field {gate.options_path!r}") from None
        if not isinstance(options, list) or not options:
            raise ActionError(f"{gate.options_artifact}.{gate.options_path} must be a non-empty list")
        return options

    def _gate_summary(self, rs: ResolvedStage, st: StageState) -> dict[str, Any]:
        assert rs.gate is not None
        summary: dict[str, Any] = {
            "prompt": rs.gate.prompt,
            "artifacts": {name: st.output_versions.get(name) for name in rs.agent_produces},
        }
        # Agents can point reviewers at human-readable files via artifact meta {"review": "<run-relative path>"}.
        review = []
        for name in rs.agent_produces:
            version = st.output_versions.get(name)
            if version is not None:
                path = self.store.envelope(name, version).meta.get("review")
                if path:
                    review.append(str(self.run.root / path))
        if review:
            summary["review_files"] = review
        if rs.gate.kind is GateKind.SELECT:
            summary["options"] = [
                {"index": i, "title": _option_title(item)} for i, item in enumerate(self._gate_options(rs, st))
            ]
        return summary

    def _resolve_stage_gate(
        self,
        sid: str,
        approved: bool,
        *,
        feedback: str = "",
        index: int | None = None,
        edit: dict[str, Any] | None = None,
        auto: bool = False,
        by: str | None = None,
    ) -> dict[str, Any]:
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        gate = rs.gate
        assert gate is not None and st.gate is not None
        if not approved:
            st.gate.status = GateStatus.REJECTED
            st.gate.resolved_at = now_iso()
            st.gate.resolution = {"approved": False, "feedback": feedback, "by": by}
            st.feedback.append(
                FeedbackItem(
                    kind=FeedbackKind.HUMAN_REJECTION,
                    text=feedback or "rejected without comment",
                    gate_id=st.gate.gate_id,
                )
            )
            self.events.emit("gate_resolved", stage=sid, gate=st.gate.gate_id, approved=False)
            self._mark_stale(sid, f"gate {st.gate.gate_id} rejected")
            return {"approved": False}

        resolution: dict[str, Any] = {"approved": True, "auto": auto, "by": by, "note": feedback}
        reopened = bool(st.gate.summary.get("reopened")) and gate.selection_artifact in st.output_versions
        if gate.kind is GateKind.SELECT and index is None and reopened:
            resolution.update(kept=True)  # a reopened choice approved as it stands
        elif gate.kind is GateKind.SELECT:
            assert gate.options_artifact and gate.selection_artifact
            if index is None:
                raise ActionError(f"gate {st.gate.gate_id!r} needs a choice: rtool select <run> <index>")
            options = self._gate_options(rs, st)
            if not 0 <= index < len(options):
                raise ActionError(f"index {index} is out of range (0..{len(options) - 1})")
            item = edit_option(gate, options[index], edit)
            version = self.store.put(
                gate.selection_artifact,
                SelectedItem(index=index, item=item, edited=bool(edit), note=feedback),
                producer=Producer(stage=sid, agent=f"gate:{st.gate.gate_id}", attempt=st.attempt),
                inputs={gate.options_artifact: st.output_versions[gate.options_artifact]},
            )
            st.output_versions[gate.selection_artifact] = version
            resolution.update(index=index, edited=bool(edit))
            self.events.emit("artifact_committed", stage=sid, artifact=gate.selection_artifact, version=version)
        st.gate.status = GateStatus.APPROVED
        st.gate.resolved_at = now_iso()
        st.gate.resolution = resolution
        self.events.emit("gate_resolved", stage=sid, gate=st.gate.gate_id, approved=True, auto=auto)
        self._finish(sid)
        self._propagate_staleness(sid)
        return resolution

    def _resolve_revision_limit(self, sid: str, approved: bool, feedback: str, by: str) -> dict[str, Any]:
        st = self.stages[sid]
        assert st.gate is not None
        summary = st.gate.summary
        st.gate.status = GateStatus.APPROVED if approved else GateStatus.REJECTED
        st.gate.resolved_at = now_iso()
        st.gate.resolution = {"approved": approved, "feedback": feedback, "by": by}
        self.events.emit("gate_resolved", stage=sid, gate=REVISION_LIMIT_GATE, approved=approved)
        pending = list(summary.get("pending_revisions", []))
        if approved:
            if pending:
                st.status = StageStatus.STALE
                st.stale_reason = "waiting for revisions of " + ", ".join(pending)
                self._save(st)
            else:
                self._save(st)
                self._open_gate_or_finish(sid)
        else:
            for raw in summary.get("requests", []):
                request = ChangeRequest.model_validate(raw)
                if feedback:
                    request = request.model_copy(update={"reason": f"{request.reason} (human: {feedback})"})
                self._apply_revision(request, forced=True)
            st.status = StageStatus.STALE
            st.stale_reason = "forced revision round"
            self._save(st)
        return {"stage_id": sid, "gate": REVISION_LIMIT_GATE, "approved": approved}

    # ---------------------------------------------------------------- actions
    def _apply_actions(self) -> None:
        assert self.run_state is not None
        for path, action in self.inbox.pending():
            if action.id in self.run_state.processed_actions:
                self.inbox.archive(path, action, {"status": "duplicate"})
                continue
            try:
                outcome = {"status": "applied", **self._apply_action(action)}
                self.events.emit("action_applied", action=action.kind.value, by=action.by, outcome=outcome)
            except ActionError as exc:
                outcome = {"status": "rejected", "error": str(exc)}
                self.events.emit("action_rejected", action=action.kind.value, by=action.by, error=str(exc))
            self.run_state.processed_actions.append(action.id)
            self._save_run()
            self.inbox.archive(path, action, outcome)

    def _find_open_gate(self, stage: str | None, gate: str | None) -> str:
        candidates = [
            sid
            for sid in self.pipeline.order
            if self.stages[sid].status is StageStatus.AWAITING_GATE
            and self.stages[sid].gate is not None
            and self.stages[sid].gate.status is GateStatus.OPEN  # type: ignore[union-attr]
        ]
        if stage is not None:
            candidates = [sid for sid in candidates if sid == stage]
        if gate is not None:
            candidates = [sid for sid in candidates if self.stages[sid].gate.gate_id == gate]  # type: ignore[union-attr]
        if not candidates:
            where = (f" for stage {stage!r}" if stage else "") + (f" named {gate!r}" if gate else "")
            raise ActionError(f"no open gate{where}")
        if len(candidates) > 1:
            names = [f"{sid}:{self.stages[sid].gate.gate_id}" for sid in candidates]  # type: ignore[union-attr]
            raise ActionError(f"several gates are open ({names}); choose one with --gate or --stage")
        return candidates[0]

    def _check_expected_gate(self, action: Action, gate: GateState) -> None:
        """Refuse a decision made while looking at an earlier version of this gate.

        ``payload["expect"]`` (optional) is what the decider saw: ``gate_nonce`` (new
        each time a gate opens) and/or ``artifacts`` ({name: version} under review).
        A UI sends both, so a click from a stale page never approves something unseen.
        """
        expect = action.payload.get("expect")
        if not expect:
            return
        if not isinstance(expect, dict):
            raise ActionError("expect must be a mapping")
        nonce = expect.get("gate_nonce")
        if nonce and gate.nonce and nonce != gate.nonce:
            raise ActionError(
                f"gate {gate.gate_id!r} was opened again since this decision was made; look at it again and decide"
            )
        shown = gate.summary.get("artifacts") or {}
        changed = [
            f"{name} (v{seen} -> v{shown.get(name)})"
            for name, seen in (expect.get("artifacts") or {}).items()
            if shown.get(name) != seen
        ]
        if changed:
            raise ActionError(f"changed since this decision was made: {', '.join(changed)}; look again and decide")

    def _single(self, status: StageStatus, label: str) -> str:
        matches = [sid for sid in self.pipeline.order if self.stages[sid].status is status]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ActionError(f"no {label} stage")
        raise ActionError(f"several {label} stages ({matches}); choose one with --stage")

    def _apply_action(self, action: Action) -> dict[str, Any]:
        kind = action.kind
        if kind in (ActionKind.APPROVE, ActionKind.REJECT, ActionKind.SELECT):
            sid = self._find_open_gate(action.stage, action.gate)
            st = self.stages[sid]
            gate_state = st.gate
            assert gate_state is not None
            self._check_expected_gate(action, gate_state)
            approved = kind is not ActionKind.REJECT
            feedback = str(action.payload.get("feedback") or action.payload.get("note") or "")
            if gate_state.gate_id == REVISION_LIMIT_GATE:
                return self._resolve_revision_limit(sid, approved, feedback, action.by)
            if gate_state.in_stage:
                if kind is ActionKind.SELECT:
                    raise ActionError("in-stage gates take approve or reject, not select")
                return self._resolve_in_stage_gate(sid, approved, feedback, dict(action.payload), action.by)
            index = action.payload.get("index")
            if kind is ActionKind.SELECT and index is None:
                raise ActionError("select needs an index")
            resolution = self._resolve_stage_gate(
                sid,
                approved,
                feedback=feedback,
                index=None if index is None else int(index),
                edit=action.payload.get("edit"),
                by=action.by,
            )
            return {"stage_id": sid, "gate": gate_state.gate_id, **resolution}

        if kind is ActionKind.RETRY:
            sid = action.stage or self._single(StageStatus.FAILED, "failed")
            if sid not in self.stages:
                raise ActionError(f"unknown stage {sid!r}")
            st = self.stages[sid]
            if st.status is not StageStatus.FAILED:
                raise ActionError(f"stage {sid!r} is {st.status.value}, not failed")
            resume = bool(action.payload.get("resume", False))
            st.status = StageStatus.PENDING
            st.resume = resume
            st.failures = 0
            st.last_error = None
            note = str(action.payload.get("note") or "").strip()
            if note:
                st.feedback.append(FeedbackItem(kind=FeedbackKind.NOTE, text=note))
            self._save(st)
            return {"stage_id": sid, "resume": resume}

        if kind is ActionKind.REOPEN:
            return self._reopen(action.stage)

        if kind is ActionKind.EDIT:
            return self._edit(action)

        if kind is ActionKind.WRITE_FILE:
            return self._write_file(action)

        if kind is ActionKind.NOTE:
            sid = action.stage
            if sid is None or sid not in self.stages:
                raise ActionError("a note needs a valid --stage")
            text = str(action.payload.get("text") or "").strip()
            if not text:
                raise ActionError("note text is empty")
            st = self.stages[sid]
            st.feedback.append(FeedbackItem(kind=FeedbackKind.NOTE, text=text))
            self._save(st)
            return {"stage_id": sid}

        raise ActionError(f"unsupported action {kind.value!r}")

    def _resolve_in_stage_gate(
        self, sid: str, approved: bool, feedback: str, data: dict[str, Any], by: str
    ) -> dict[str, Any]:
        """Record the decision for the agent (it reads it from ctx.gate when it resumes) and let it resume."""
        st = self.stages[sid]
        gate_state = st.gate
        assert gate_state is not None
        decisions = st.agent_state.setdefault("_gates", {})
        decisions[gate_state.gate_id] = GateResolution(approved=approved, feedback=feedback, data=data).model_dump()
        gate_state.status = GateStatus.APPROVED if approved else GateStatus.REJECTED
        gate_state.resolved_at = now_iso()
        gate_state.resolution = {"approved": approved, "feedback": feedback, "by": by}
        st.status = StageStatus.PENDING
        st.resume = True
        self._save(st)
        self.events.emit("gate_resolved", stage=sid, gate=gate_state.gate_id, approved=approved, in_stage=True)
        return {"stage_id": sid, "gate": gate_state.gate_id, "approved": approved}

    def _write_file(self, action: Action) -> dict[str, Any]:
        """Rewrite a file the stage's agent lets researchers edit (e.g. the Coder's experiment.py).

        Allowed while the stage waits at its own checkpoint or after it failed. With ``retest`` the stage
        then continues: the checkpoint is answered with "test it again", a failed stage resumes.
        """
        sid = action.stage
        if sid is None or sid not in self.stages:
            raise ActionError("write_file needs a valid --stage")
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        payload = action.payload
        path = str(payload.get("path") or "")
        if path not in rs.agent_cls.editable_files:
            allowed = ", ".join(rs.agent_cls.editable_files) or "no file"
            raise ActionError(f"in stage {sid!r} you can edit {allowed}, not {path!r}")
        at_gate = (
            st.status is StageStatus.AWAITING_GATE
            and st.gate is not None
            and st.gate.in_stage
            and st.gate.status is GateStatus.OPEN
        )
        if not (at_gate or st.status is StageStatus.FAILED):
            raise ActionError(
                f"{path} can be edited while stage {sid!r} waits for your decision or after it failed "
                f"(it is {st.status.value})"
            )
        content = payload.get("content")
        if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_EDITED_FILE_BYTES:
            raise ActionError(f"{path} must be text of at most {MAX_EDITED_FILE_BYTES // 1_000_000} MB")
        workspace = Workspace(self.run.work_dir(sid))
        target = workspace.path(path)
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        base = payload.get("base_sha")
        if base is not None and sha256_text(current) != base:
            raise ActionError(f"{path} changed since you opened it; reload it and edit again")
        changed = content != current
        if changed:
            workspace.write_text(path, content)
            self.events.emit("file_written", stage=sid, path=path, by=action.by, sha=sha256_text(content))
        outcome: dict[str, Any] = {"stage_id": sid, "path": path, "changed": changed}
        if payload.get("retest"):
            if at_gate:
                outcome.update(self._resolve_in_stage_gate(sid, False, "", {"retest": True}, action.by))
            else:  # failed: resume where it stopped (the agent re-tests what it finds)
                st.status = StageStatus.PENDING
                st.resume = True
                st.failures = 0
                st.last_error = None
                self._save(st)
                outcome["resumed"] = True
        return outcome

    def _reopen(self, sid: str | None) -> dict[str, Any]:
        """Open a finished stage's checkpoint again: choose differently, keep it as it is, or reject to re-run.

        Later stages are left alone until the decision: only a changed output makes them stale.
        """
        if sid is None or sid not in self.stages:
            raise ActionError("reopen needs a valid --stage")
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        if rs.gate is None or rs.spec.gate is None:
            raise ActionError(f"stage {sid!r} has no checkpoint to reopen")
        if st.status is not StageStatus.DONE:
            raise ActionError(f"stage {sid!r} is {st.status.value}; only a finished stage can be reopened")
        summary = self._gate_summary(rs, st)
        summary["reopened"] = True
        selection = rs.gate.selection_artifact
        if rs.gate.kind is GateKind.SELECT and selection in st.output_versions:
            chosen = self.store.envelope(selection, st.output_versions[selection]).payload
            summary["current"] = {"index": chosen.get("index"), "version": st.output_versions[selection]}
        st.gate = GateState(gate_id=rs.spec.gate, summary=summary, nonce=new_gate_nonce())
        st.status = StageStatus.AWAITING_GATE
        self._save(st)
        self.events.emit(
            "gate_opened", stage=sid, gate=rs.spec.gate, in_stage=False, prompt=rs.gate.prompt, reopened=True
        )
        return {"stage_id": sid, "gate": rs.spec.gate}

    def _edit(self, action: Action) -> dict[str, Any]:
        """Store a hand-edited output as a new version (normalised by the owning agent's editor).

        The owning stage keeps its state (an open gate stays open, now showing the edited version); later
        stages that used the old version become stale.
        """
        payload = action.payload
        name = str(payload.get("artifact") or "")
        value, base = payload.get("value"), payload.get("base_version")
        owner, previous, result = run_editor(self.pipeline, self.store, self.config, name, value, base, self.run.root)
        st = self.stages[owner]
        refusal = edit_refusal(owner, st)
        if refusal:
            raise ActionError(refusal)
        note = str(payload.get("note") or "").strip()
        meta = {**previous.meta, **result.meta, "edited_from": previous.version, "edited_by": action.by}
        if note:
            meta["edit_note"] = note
        version = self.store.put(
            name,
            result.value,
            producer=Producer(stage=owner, agent=f"human:{action.by}", attempt=st.attempt),  # no token: a new version
            inputs=dict(previous.inputs),
            meta=meta,
        )
        if version == previous.version:
            return {"stage_id": owner, "artifact": name, "version": version, "unchanged": True}
        workspace = Workspace(self.run.work_dir(owner))
        for path, text in result.files.items():
            workspace.write_text(path, text)
        rs = self.pipeline.stage(owner)
        st.output_versions[name] = version
        if name in rs.all_inputs:
            st.input_versions[name] = version
        if st.gate is not None and st.gate.status is GateStatus.OPEN and name in st.gate.summary.get("artifacts", {}):
            st.gate.summary["artifacts"][name] = version  # the checkpoint now asks about the edited version
        self._save(st)
        self.events.emit("artifact_committed", stage=owner, artifact=name, version=version)
        before = {sid: self.stages[sid].status for sid in self.pipeline.order}
        self._propagate_staleness(owner)
        stale = [
            s
            for s in self.pipeline.order
            if self.stages[s].status is StageStatus.STALE and before[s] is not StageStatus.STALE
        ]
        self.events.emit("artifact_edited", stage=owner, artifact=name, version=version, by=action.by, stale=stale)
        return {"stage_id": owner, "artifact": name, "version": version, "stale": stale}

    # ------------------------------------------------------------------- jobs
    def _poll_jobs(self) -> None:
        for sid in self.pipeline.order:
            st = self.stages[sid]
            if st.status is not StageStatus.SUSPENDED or not st.waiting_on:
                continue
            if self.services.slurm is None:
                raise RToolError(f"stage {sid!r} waits on jobs {st.waiting_on} but no Slurm client is configured")
            statuses = self.services.slurm.status_many(st.waiting_on)
            if all(s.state.is_terminal for s in statuses.values()):
                st.job_results.update({job: s.model_dump(mode="json") for job, s in statuses.items()})
                st.status = StageStatus.PENDING
                st.resume = True
                st.waiting_on = []
                self._save(st)
                self.events.emit("jobs_finished", stage=sid, jobs={job: s.state.value for job, s in statuses.items()})

    # ---------------------------------------------------------------- summary
    def _summarize(self, ran: list[str], time_up: bool) -> AdvanceResult:
        open_gates: list[dict[str, Any]] = []
        waiting: list[str] = []
        failed: list[dict[str, Any]] = []
        for sid in self.pipeline.order:
            st = self.stages[sid]
            if st.status is StageStatus.AWAITING_GATE and st.gate and st.gate.status is GateStatus.OPEN:
                open_gates.append(
                    {"stage": sid, "gate": st.gate.gate_id, "in_stage": st.gate.in_stage, "summary": st.gate.summary}
                )
            elif st.status is StageStatus.SUSPENDED:
                waiting.extend(st.waiting_on)
            elif st.status is StageStatus.FAILED:
                failed.append({"stage": sid, "error": st.last_error})
        deferred = [
            sid
            for sid in self.pipeline.order
            if sid in self._deferred and self.stages[sid].status in (StageStatus.PENDING, StageStatus.STALE)
        ]
        if time_up:
            outcome, message = Outcome.CONTINUE, "episode time budget used; more stages are ready"
        elif deferred:
            outcome, message = Outcome.NEEDS_LLM, f"ready stages need the LLM (GPU episode): {', '.join(deferred)}"
        elif all(self.stages[sid].status is StageStatus.DONE for sid in self.pipeline.order):
            outcome, message = Outcome.DONE, "all stages are done"
        elif open_gates:
            names = ", ".join(f"{g['gate']} ({g['stage']})" for g in open_gates)
            outcome, message = Outcome.WAITING_GATE, f"waiting for a decision at: {names}"
        elif waiting:
            outcome, message = Outcome.WAITING_JOB, f"waiting for Slurm jobs: {', '.join(waiting)}"
        elif failed:
            names = ", ".join(f["stage"] for f in failed)
            outcome, message = Outcome.NEEDS_HUMAN, f"failed stages: {names} (fix, then rtool retry)"
        else:
            outcome, message = Outcome.NEEDS_HUMAN, "no stage can run; check rtool status"
        return AdvanceResult(outcome, ran, open_gates, waiting, failed, deferred, message)


def load_status(
    run: RunHandle, registry: AgentRegistry = default_registry
) -> tuple[RunState | None, dict[str, StageState], ResolvedPipeline]:
    """Read-only view of a run's state (no lease needed)."""
    pipeline = resolve_pipeline(run.spec.pipeline, registry)
    states = StateStore(run.state_dir)
    return states.load_run(), states.load_stages(pipeline.order), pipeline
