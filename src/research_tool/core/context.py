"""AgentContext: everything an agent may touch while it runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from pydantic import BaseModel

from .artifacts import ArtifactEnvelope, ArtifactStore, StagedOutputs
from .budgets import BudgetTracker
from .config import AppConfig, GateMode
from .errors import ArtifactError, GateRequested, LLMRequired, RToolError, SuspendRequested
from .events import EventLog
from .state import ChangeRequest, FeedbackItem, GateResolution
from .workspace import Workspace

if TYPE_CHECKING:
    from ..hpc.slurm import SlurmClient
    from ..llm.client import LLMClient

M = TypeVar("M", bound=BaseModel)
RESUME_WITHOUT_LLM = "_resume_without_llm"  # key in a stage's saved state (see AgentContext.resume_without_llm)


class InputView:
    """Read-only access to the exact artifact versions this stage attempt consumes."""

    def __init__(self, store: ArtifactStore, versions: dict[str, int]) -> None:
        self._store = store
        self._versions = dict(versions)

    def names(self) -> list[str]:
        return sorted(self._versions)

    def has(self, name: str) -> bool:
        return name in self._versions

    def version(self, name: str) -> int:
        self._require(name)
        return self._versions[name]

    def get(self, name: str, model: type[M] | None = None) -> Any:
        self._require(name)
        return self._store.get(name, self._versions[name], model)

    def envelope(self, name: str) -> ArtifactEnvelope:
        self._require(name)
        return self._store.envelope(name, self._versions[name])

    def _require(self, name: str) -> None:
        if name not in self._versions:
            raise ArtifactError(f"{name!r} is not an input of this stage (inputs: {self.names()})")


class AgentContext:
    def __init__(
        self,
        *,
        stage_id: str,
        agent_name: str,
        attempt: int,
        config: Any,
        app_config: AppConfig,
        inputs: InputView,
        outputs: StagedOutputs,
        state: dict[str, Any],
        resumed: bool,
        feedback: list[FeedbackItem],
        events: EventLog,
        budget: BudgetTracker,
        workspace: Workspace,
        llm: LLMClient | None,
        slurm: SlurmClient | None,
        job_results: dict[str, dict[str, Any]],
        checkpoint: Callable[[dict[str, Any]], None],
        run_root: Path | None = None,
        tools: dict[str, Any] | None = None,
        defer_without_llm: bool = False,
    ) -> None:
        self.run_root = Path(run_root) if run_root is not None else workspace.root.parent.parent
        self._defer_without_llm = defer_without_llm
        self.tools: dict[str, Any] = dict(tools or {})  # injected tool factories (tests, sites)
        self.stage_id = stage_id
        self.agent_name = agent_name
        self.attempt = attempt
        self.config = config
        self.app_config = app_config
        self.inputs = inputs
        self.outputs = outputs
        self.state = state
        self.resumed = resumed
        self.feedback = feedback
        self.events = events
        self.budget = budget
        self.workspace = workspace
        self._llm = llm
        self._slurm = slurm
        self._job_results = job_results
        self._checkpoint = checkpoint
        self._change_requests: list[ChangeRequest] = []

    # -- services ----------------------------------------------------------
    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            if self._defer_without_llm:
                raise LLMRequired(f"stage {self.stage_id!r} needs the LLM")
            raise RToolError(
                f"stage {self.stage_id!r} has no LLM (no backend configured, or the agent sets needs_llm=False)"
            )
        return self._llm

    def resume_without_llm(self, allowed: bool = True) -> None:
        """Say whether the next resumption (after a gate or a job) can start on a CPU node.

        Use it for steps such as submitting or collecting jobs, so the run does not wait in the
        GPU queue just to do bookkeeping. If such a step does need the LLM after all, touching
        ``ctx.llm`` there moves the stage to a GPU episode, state intact.
        """
        self.state[RESUME_WITHOUT_LLM] = allowed

    @property
    def slurm(self) -> SlurmClient:
        if self._slurm is None:
            raise RToolError(f"stage {self.stage_id!r} has no Slurm client configured")
        return self._slurm

    # -- files shared with later stages --------------------------------------
    def relative(self, path: Path) -> str:
        """Run-relative form of ``path`` for storing in artifacts (stays valid if the run moves)."""
        return str(Path(path).resolve().relative_to(self.run_root.resolve()))

    def absolute(self, relative: str) -> Path:
        """Resolve a run-relative path from an artifact, refusing anything outside the run."""
        root = self.run_root.resolve()
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise RToolError(f"{relative!r} points outside the run directory")
        return target

    # -- logging & persistence --------------------------------------------
    def emit(self, kind: str, **data: Any) -> None:
        self.events.emit(kind, stage=self.stage_id, **data)

    def log(self, message: str, **data: Any) -> None:
        self.emit("agent_log", message=message, **data)

    def checkpoint(self) -> None:
        """Persist ``ctx.state`` now, so a crash resumes from here."""
        self._checkpoint(self.state)

    # -- waiting on the outside world ---------------------------------------
    def suspend_until(self, job_ids: Iterable[str], reason: str = "") -> NoReturn:
        """Pause this stage until the jobs finish; ``run()`` is called again afterwards.

        On the second call ``ctx.resumed`` is True and ``ctx.job_result(id)``
        returns each job's final status.
        """
        ids = [str(j) for j in job_ids]
        if not ids:
            raise RToolError("suspend_until needs at least one job id")
        raise SuspendRequested(ids, reason)

    def job_result(self, job_id: str) -> dict[str, Any] | None:
        return self._job_results.get(str(job_id))

    def gate(self, gate_id: str, summary: dict[str, Any] | None = None) -> GateResolution:
        """Ask a human to decide before continuing (e.g. before spending GPU hours).

        The first call pauses the stage; after the decision ``run()`` is called
        again and this returns the resolution (``approved``, ``feedback``).
        Gates configured as ``auto`` are approved immediately.
        """
        decisions = self.state.setdefault("_gates", {})
        if gate_id in decisions:
            return GateResolution.model_validate(decisions[gate_id])
        if self.app_config.gates.mode(gate_id) is GateMode.AUTO:
            resolution = GateResolution(approved=True, auto=True)
            decisions[gate_id] = resolution.model_dump()
            self.emit("gate_auto_resolved", gate=gate_id, summary=summary or {})
            return resolution
        raise GateRequested(gate_id, summary)

    # -- talking to upstream stages ----------------------------------------
    def request_change(
        self,
        *,
        reason: str,
        artifact: str | None = None,
        stage: str | None = None,
        required_changes: Iterable[str] = (),
        severity: str = "major",
    ) -> None:
        """Ask the producer of an input artifact (preferred) or an upstream stage to redo its work.

        Takes effect when this stage finishes; the engine re-runs the target
        with this request as feedback, then re-runs this stage.
        """
        if (artifact is None) == (stage is None):
            raise RToolError("request_change needs exactly one of artifact= or stage=")
        target = stage
        if artifact is not None:
            target = self.inputs.envelope(artifact).producer.stage
        request = ChangeRequest(
            from_stage=self.stage_id,
            target_stage=target,  # type: ignore[arg-type]
            target_artifact=artifact,
            reason=reason,
            required_changes=list(required_changes),
            severity=severity,  # type: ignore[arg-type]
        )
        self._change_requests.append(request)
        self.emit("change_request_raised", target=target, artifact=artifact, reason=reason)

    @property
    def change_requests(self) -> list[ChangeRequest]:
        return list(self._change_requests)
