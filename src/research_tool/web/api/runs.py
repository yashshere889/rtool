"""Runs: list, start a new one, look at one, pause and resume."""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.actions import Action
from ...core.errors import RToolError
from ...core.run import RunHandle, is_valid_name
from ...core.state import RunState
from ...hpc.episode import llm_needed
from ...service import KickResult, RunSummary, create_run, list_runs, load_app_config, run_status, stop_run
from ..deps import get_run, public_summary, runner_of, settings_of
from ..runner import WorkerState

router = APIRouter(tags=["runs"])

_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "on",
    "in",
    "for",
    "to",
    "with",
    "during",
    "does",
    "do",
    "is",
    "are",
    "how",
}

ReviewMode = Literal["every_step", "gpu_and_final", "automatic"]
REVIEW_MODES: dict[str, dict[str, Any]] = {
    "every_step": {},  # stop at every checkpoint (the default)
    "gpu_and_final": {
        "gates": {"modes": {"literature_review": "auto", "hypothesis_selection": "auto", "experiment_plan": "auto"}}
    },
    "automatic": {"gates": {"default": "auto"}},
}


class RunDetail(RunSummary):
    """A run as the browser sees it: no server paths, plus what is advancing it right now."""

    tool_version: str = ""
    run_state: RunState | None = None
    queued_actions: list[Action] = Field(default_factory=list)
    worker: WorkerState = Field(default_factory=WorkerState)


class NewRun(BaseModel):
    topic: str = Field(min_length=3, max_length=2000, description="The research question or topic")
    project: str | None = Field(default=None, max_length=80, description="Group name; made from the topic if empty")
    questions: list[str] = Field(default_factory=list, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=20)
    seed_queries: list[str] = Field(default_factory=list, max_length=20)
    pack: str = "generic"
    review: ReviewMode = "every_step"
    max_gpu_hours: float | None = Field(default=None, gt=0)


class Created(BaseModel):
    run: RunDetail
    kick: KickResult


class StopRequest(BaseModel):
    experiments: bool = False  # also cancel running experiment jobs


def project_name(topic: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", topic.lower()) if w not in _STOPWORDS]
    name = "-".join(words[:4])[:60].strip("-")
    return name if name and is_valid_name(name) else "research"


def detail(request: Request, run: RunHandle) -> RunDetail:
    status = run_status(run)
    for stage in status.stages:
        if stage.gate is not None:
            stage.gate.summary = public_summary(run, stage.gate.summary)
    status.open_gates = [s.gate for s in status.stages if s.gate is not None]
    fields = {name: getattr(status, name) for name in RunDetail.model_fields if name != "worker"}
    return RunDetail(**fields, worker=runner_of(request).state(run))


@router.get("/runs", response_model=list[RunSummary])
def all_runs(request: Request) -> list[RunSummary]:
    return list_runs(settings_of(request).runs_root)


@router.post("/runs", response_model=Created, status_code=201)
def new_run(body: NewRun, request: Request) -> Created:
    settings = settings_of(request)
    project = body.project or project_name(body.topic)
    if not is_valid_name(project):
        raise HTTPException(422, "project names may use letters, digits, '.', '_' and '-' only")
    overrides: dict[str, Any] = dict(REVIEW_MODES[body.review])
    if body.max_gpu_hours is not None:
        overrides["budgets"] = {"run": {"max_gpu_hours": body.max_gpu_hours}}
    questions = [q.strip() for q in body.questions if q.strip()]
    try:
        if settings.runs_locally:
            from ...testing.demo_run import create_demo_run

            run = create_demo_run(
                settings.runs_root, project=project, topic=body.topic.strip(), questions=questions, overrides=overrides
            )
        else:
            config = load_app_config(settings.config_files, runs_root=settings.runs_root, overrides=overrides)
            run = create_run(
                config,
                project,
                body.topic.strip(),
                questions=questions,
                constraints=[c.strip() for c in body.constraints if c.strip()],
                seed_queries=[s.strip() for s in body.seed_queries if s.strip()],
                pack=body.pack,
                created_by="rtool ui",
            )
    except RToolError as exc:
        raise HTTPException(422, str(exc)) from None
    kick = runner_of(request).wake(run, "new run", needs_llm=llm_needed(run))
    return Created(run=detail(request, run), kick=kick)


@router.get("/runs/{project}/{run_id}", response_model=RunDetail)
def one_run(project: str, run_id: str, request: Request) -> RunDetail:
    return detail(request, get_run(request, project, run_id))


@router.post("/runs/{project}/{run_id}/resume", response_model=KickResult)
def resume(project: str, run_id: str, request: Request) -> KickResult:
    """Make sure something is advancing the run (after a pause, or if a decision was not picked up)."""
    run = get_run(request, project, run_id)
    return runner_of(request).wake(run, "resume", needs_llm=llm_needed(run))


@router.post("/runs/{project}/{run_id}/stop")
def stop(project: str, run_id: str, body: StopRequest, request: Request) -> dict[str, list[str]]:
    run = get_run(request, project, run_id)
    if settings_of(request).runs_locally:
        raise HTTPException(409, "local runs cannot be paused; they stop at the next checkpoint by themselves")
    return {"cancelled": stop_run(run, cancel_experiments=body.experiments).cancelled}
