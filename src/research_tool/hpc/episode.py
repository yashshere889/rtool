"""Episodes as Slurm jobs: submitting, chaining and running them.

An episode job runs ``rtool episode <run>``. Nothing stays resident between
episodes: when the engine stops, the episode decides what comes next.

    outcome          next step
    ---------------  -------------------------------------------------------------
    CONTINUE         resubmit now (the time budget ran out)
    NEEDS_LLM        submit a GPU episode now (this one ran on a CPU node)
    WAITING_JOB      submit an episode that starts when those jobs end (afterany)
    WAITING_GATE     nothing; `rtool approve` / `reject` / `select` submit the next one
    DONE/NEEDS_HUMAN nothing

CPU episodes are cheap and start quickly, so decisions are applied there; they
hand over to a GPU episode only when a stage that needs the LLM is ready. GPU
episodes start vLLM lazily, the first time such a stage actually runs.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..core.config import AppConfig, GPUChoice
from ..core.context import RESUME_WITHOUT_LLM
from ..core.engine import AdvanceResult, Engine, Outcome, load_status
from ..core.errors import RToolError
from ..core.events import EventLog
from ..core.fs import atomic_write_json, now_iso, read_json
from ..core.pipeline import ResolvedStage
from ..core.run import RunHandle
from ..core.services import LLMEndpoint, Services, build_llm
from ..core.state import StageState, StageStatus
from ..llm.client import LLMClient
from ..llm.server import ServerHandle, VLLMServer
from .jobspec import GPURequest, JobSpec
from .slurm import SlurmClient

EPISODE_FILE = "episode_job.json"
SAFETY_MARGIN_S = 15 * 60  # stop starting new stages this long before the job's time limit
# Jobs inherit the submitting shell's environment: never let other Python setups leak in.
ISOLATE_PYTHON = "unset PYTHONPATH PYTHONHOME && export PYTHONNOUSERSITE=1 && "


def gpu_choices(config: AppConfig) -> list[GPUChoice]:
    ep = config.slurm.episode
    return list(ep.gpu_choices) or [GPUChoice(partition=ep.partition, gpu_type=ep.gpu_type)]


def episode_job_spec(
    run: RunHandle, *, needs_llm: bool, dependency: list[str] | None = None, gpu: GPUChoice | None = None
) -> JobSpec:
    ep = run.config.slurm.episode
    gpu = gpu or gpu_choices(run.config)[0]
    activate = f"source {shlex.quote(os.path.expandvars(ep.activate))} && " if ep.activate else ""
    config_args = "".join(f" --config {shlex.quote(os.path.expanduser(path))}" for path in ep.config_files)
    return JobSpec(
        name=f"rtool-ep-{run.run_id}"[:100],
        partition=gpu.partition if needs_llm else ep.cpu_partition,
        command=f"{ISOLATE_PYTHON}{activate}exec rtool episode {shlex.quote(str(run.root))}{config_args}",
        workdir=run.root,
        time_limit=ep.time_limit,
        gpus=GPURequest(type=gpu.gpu_type, count=ep.gpus) if needs_llm else None,
        ntasks=None if needs_llm else 1,
        dependency=list(dependency or []),
        dependency_type="afterany",
        modules=list(ep.modules),
        requeue=True,  # episodes resume where they stopped, so preemption only costs time
        signal=f"B:USR1@{ep.signal_seconds}" if ep.signal_seconds else None,
        metadata={"kind": "episode", "run": run.run_id, "needs_llm": needs_llm},
    )


def choose_gpu(run: RunHandle, slurm: SlurmClient) -> tuple[GPUChoice, float | None]:
    """The GPU an LLM episode should ask for now, and when Slurm expects it to start.

    Each choice is priced with Slurm's own estimate (``sbatch --test-only``). The
    soonest wins, except that an earlier choice in the list is kept when it starts
    within ``prefer_within_minutes`` of the soonest (e.g. a non-preemptible partition).
    """
    choices = gpu_choices(run.config)
    if len(choices) == 1:
        return choices[0], None
    starts: list[float | None] = []
    for choice in choices:
        try:
            starts.append(slurm.estimate_start(episode_job_spec(run, needs_llm=True, gpu=choice)))
        except RToolError:
            starts.append(None)  # e.g. a partition missing from the site profile
    known = [s for s in starts if s is not None]
    if not known:
        return choices[0], None
    slack = run.config.slurm.episode.prefer_within_minutes * 60
    soonest = min(known)
    return next((c, s) for c, s in zip(choices, starts, strict=True) if s is not None and s <= soonest + slack)


def resumes_without_llm(st: StageState) -> bool:
    """The stage said its next resumption can start on a CPU node (ctx.resume_without_llm)."""
    waiting = st.resume or st.status in (StageStatus.SUSPENDED, StageStatus.AWAITING_GATE)
    return waiting and bool(st.agent_state.get(RESUME_WITHOUT_LLM))


def llm_needed(run: RunHandle, statuses: tuple[StageStatus, ...] | None = None) -> bool:
    """Would the stages that can run next need the LLM?

    Only the frontier counts: unfinished stages whose upstream stages are all done (or
    at a gate that is about to be decided). A stage at an in-stage gate, or waiting on
    jobs, counts unless it said it can resume without the LLM. ``statuses`` narrows the
    stages considered (e.g. only SUSPENDED ones, when jobs are about to finish).
    """
    _, stages, pipeline = load_status(run)
    settled = (StageStatus.DONE, StageStatus.AWAITING_GATE)
    for sid in pipeline.order:
        st = stages[sid]
        if st.status in (StageStatus.DONE, StageStatus.FAILED) or (statuses is not None and st.status not in statuses):
            continue
        if st.status is StageStatus.AWAITING_GATE and not (st.gate is not None and st.gate.in_stage):
            continue  # its work is done; approving it lets the stages after it run
        if not all(stages[a].status in settled for a in pipeline.ancestors(sid)):
            continue
        if pipeline.stage(sid).agent_cls.needs_llm and not resumes_without_llm(st):
            return True
    return False


class EpisodeTracker:
    """Remembers the run's queued or running episode job, so nobody submits a duplicate."""

    def __init__(self, run: RunHandle, slurm: SlurmClient) -> None:
        self.run = run
        self.slurm = slurm
        self.path = run.state_dir / EPISODE_FILE

    def current(self, *, ignore: str | None = None) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        record = read_json(self.path)
        if record.get("job_id") == ignore:
            return None
        status = self.slurm.status(record["job_id"])
        return None if status.state.is_terminal else record | {"state": status.state.value}

    def record(self, job_id: str, *, needs_llm: bool, reason: str, dependency: list[str]) -> None:
        atomic_write_json(
            self.path,
            {"job_id": job_id, "needs_llm": needs_llm, "reason": reason, "dependency": dependency, "at": now_iso()},
        )


def submit_episode(
    run: RunHandle,
    slurm: SlurmClient,
    *,
    needs_llm: bool,
    reason: str,
    dependency: list[str] | None = None,
    self_job_id: str | None = None,
) -> str | None:
    """Submit an episode unless one is already queued/running. Returns the new job id or None."""
    tracker = EpisodeTracker(run, slurm)
    existing = tracker.current(ignore=self_job_id)
    if existing is not None:
        return None
    dependency = list(dependency or [])
    if needs_llm and dependency and len(gpu_choices(run.config)) > 1:
        # Which GPU is free can only be judged when the wait is over: start on a CPU node,
        # which hands over to a GPU episode (chosen then) if a stage needs the LLM.
        needs_llm = False
    gpu, expected = choose_gpu(run, slurm) if needs_llm else (None, None)
    spec = episode_job_spec(run, needs_llm=needs_llm, dependency=dependency, gpu=gpu)
    job_id = slurm.submit(spec)
    tracker.record(job_id, needs_llm=needs_llm, reason=reason, dependency=dependency)
    EventLog(run.events_path, run.run_id).emit(
        "episode_submitted",
        job=job_id,
        needs_llm=needs_llm,
        reason=reason,
        dependency=dependency,
        partition=spec.partition,
        gpu=spec.gpus.gres() if spec.gpus else None,
        expected_start=datetime.fromtimestamp(expected).isoformat(timespec="minutes") if expected else None,
    )
    return job_id


def schedule_next(run: RunHandle, slurm: SlurmClient, result: AdvanceResult, *, self_job_id: str | None) -> str | None:
    if result.outcome is Outcome.NEEDS_LLM:
        return submit_episode(run, slurm, needs_llm=True, reason="needs_llm", self_job_id=self_job_id)
    if result.outcome is Outcome.CONTINUE:
        return submit_episode(run, slurm, needs_llm=llm_needed(run), reason="continue", self_job_id=self_job_id)
    if result.outcome is Outcome.WAITING_JOB:
        # The stage that resumes is SUSPENDED now; go straight to a GPU episode if it needs the LLM.
        needs = llm_needed(run, (StageStatus.SUSPENDED,))
        return submit_episode(
            run, slurm, needs_llm=needs, reason="waiting_job", dependency=result.waiting_jobs, self_job_id=self_job_id
        )
    return None


def gpu_available(env: dict[str, str] | None = None) -> bool:
    env = dict(os.environ) if env is None else env
    devices = env.get("CUDA_VISIBLE_DEVICES", "")
    return bool(devices and devices not in {"-1", "NoDevFiles"}) or bool(env.get("SLURM_JOB_GPUS"))


class LazyLLM:
    """Stands in for an LLMClient; starts the vLLM server the first time a stage needs it."""

    def __init__(self, config: AppConfig, log_dir: Any, factory: Callable[[], LLMClient] | None = None) -> None:
        self.config = config
        self.log_dir = log_dir
        self._factory = factory
        self._client: LLMClient | None = None
        self._handle: ServerHandle | None = None

    @property
    def started(self) -> bool:
        return self._client is not None

    def _ensure(self) -> LLMClient:
        if self._client is None:
            if self._factory is not None:
                self._client = self._factory()
            elif self.config.llm.backend != "vllm":
                self._client = build_llm(self.config)  # mock or an external server: nothing to start
            else:
                self._handle = VLLMServer(self.config.llm, self.log_dir).start()
                endpoint = LLMEndpoint(url=self._handle.url, api_key=self._handle.api_key)
                self._client = build_llm(self.config, endpoint)
        return self._client

    def bind(self, **kwargs: Any) -> LLMClient:
        return self._ensure().bind(**kwargs)

    def close(self) -> None:
        if self._handle is not None:
            VLLMServer.stop(self._handle)
            self._handle = None


def run_episode(
    run: RunHandle,
    *,
    slurm: SlurmClient,
    llm_factory: Callable[[], LLMClient] | None = None,
    has_gpu: bool | None = None,
    max_seconds: float | None = None,
    episode_id: str | None = None,
) -> AdvanceResult:
    """One episode inside a job: run what this node can run, then return the outcome."""
    gpu = gpu_available() if has_gpu is None else has_gpu
    lazy = LazyLLM(run.config, run.jobs_dir / "llm", factory=llm_factory) if gpu else None

    def cpu_only(stage: ResolvedStage) -> bool:
        return not stage.agent_cls.needs_llm

    try:
        engine = Engine(
            run,
            Services(llm=lazy, slurm=slurm),  # type: ignore[arg-type]
            episode_id=episode_id,
            stage_filter=None if gpu else cpu_only,
        )
        return engine.advance(max_seconds=max_seconds)
    finally:
        if lazy is not None:
            lazy.close()
