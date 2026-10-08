"""Helpers for driving a run in tests (and for trying a new agent quickly)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.actions import Action, ActionInbox, ActionKind
from ..core.artifacts import ArtifactStore, Producer
from ..core.config import AppConfig, load_config
from ..core.engine import AdvanceResult, Engine, Outcome
from ..core.pipeline import PipelineSpec, find_pipeline, parse_pipeline
from ..core.run import RunHandle
from ..core.services import Services
from ..hpc.fake_slurm import FakeSlurm
from ..llm.client import LLMClient
from ..llm.mock_backend import MockLLM
from ..schemas.common import ResearchBrief


def make_run(
    runs_root: Path,
    pipeline: PipelineSpec | dict[str, Any] | str,
    *,
    topic: str = "toy topic",
    overrides: dict[str, Any] | None = None,
    project: str = "test",
) -> RunHandle:
    if isinstance(pipeline, str):
        spec = find_pipeline(pipeline)
    elif isinstance(pipeline, dict):
        spec = parse_pipeline(pipeline)
    else:
        spec = pipeline
    config: AppConfig = load_config(overrides=overrides or {}, env={})
    run = RunHandle.create(runs_root, project, spec, config)
    ArtifactStore(run.artifacts_dir).put(
        "research_brief",
        ResearchBrief(topic=topic),
        producer=Producer(stage="user", agent="harness"),
    )
    return run


def make_services(run: RunHandle, llm: MockLLM | None = None, slurm: FakeSlurm | None = None) -> Services:
    return Services(
        llm=LLMClient(llm or MockLLM(), run.config.llm),
        slurm=slurm or FakeSlurm(run.jobs_dir / "fake_slurm.json"),
    )


def queue(run: RunHandle, kind: ActionKind, **fields: Any) -> None:
    payload = fields.pop("payload", {})
    ActionInbox(run.actions_dir).submit(Action(kind=kind, payload=payload, **fields))


def drive(run: RunHandle, services: Services, *, max_episodes: int = 20) -> AdvanceResult:
    """Advance until the run needs a human, finishing fake Slurm jobs as they appear."""
    result = AdvanceResult(Outcome.NEEDS_HUMAN)
    for _ in range(max_episodes):
        result = Engine(run, services).advance()
        if result.outcome is Outcome.WAITING_JOB and isinstance(services.slurm, FakeSlurm):
            services.slurm.finish_all()
            continue
        if result.outcome is Outcome.CONTINUE:
            continue
        return result
    return result
