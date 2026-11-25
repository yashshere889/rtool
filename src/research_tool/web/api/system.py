"""What this deployment can do: its fields (packs), pipelines, agents, and whether this machine is set up for it.

The same facts `rtool packs`, `pipelines`, `agents` and `doctor` print, for the Settings page. Read-only: nothing
here starts a run or changes configuration.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ...core.errors import RToolError
from ...core.pipeline import builtin_pipelines, find_pipeline, resolve_pipeline
from ...core.registry import registry
from ...hpc.doctor import run_doctor
from ...packs import load_pack, pack_sources
from ...service import load_app_config
from ..deps import settings_of
from .session import offered_packs

router = APIRouter(tags=["system"])


class FieldInfo(BaseModel):
    name: str
    title: str
    description: str
    pipeline: str
    origin: str  # "built-in", or the installed package it comes from
    state: str  # offered | not_offered | shadowed | broken
    detail: str  # why it is shadowed or broken
    max_gpu_hours: float | None


class StageInfo(BaseModel):
    id: str
    agent: str
    after: list[str]
    gate: str
    produces: list[str]


class PipelineInfo(BaseModel):
    name: str
    description: str
    stages: list[StageInfo]


class AgentInfo(BaseModel):
    name: str
    description: str
    consumes: list[str]
    optional: list[str]
    produces: list[str]
    uses_llm: bool


class CheckInfo(BaseModel):
    name: str
    status: str  # ok | warn | fail | skip
    detail: str


@router.get("/fields", response_model=list[FieldInfo])
def fields(request: Request) -> list[FieldInfo]:
    """Every pack, as `rtool packs` lists them, and whether this server offers it to new runs."""
    settings = settings_of(request)
    try:
        packs = load_app_config(settings.config_files, runs_root=settings.runs_root).packs
    except RToolError:
        return []
    offered = {p.name for p in offered_packs(packs.allowed, demo=settings.runs_locally)}
    out = []
    for source in pack_sources():
        base = {"name": source.name, "origin": source.origin, "max_gpu_hours": packs.max_gpu_hours.get(source.name)}
        if source.shadowed:
            detail = "another pack has this name, so this one is never used"
            out.append(FieldInfo(**base, title="", description="", pipeline="", state="shadowed", detail=detail))
            continue
        try:
            pack = load_pack(source.name)
        except RToolError as exc:
            out.append(FieldInfo(**base, title="", description="", pipeline="", state="broken", detail=str(exc)))
            continue
        out.append(
            FieldInfo(
                **base,
                title=pack.title,
                description=pack.description,
                pipeline=pack.pipeline or "research",
                state="offered" if source.name in offered else "not_offered",
                detail="",
            )
        )
    return out


@router.get("/pipelines", response_model=list[PipelineInfo])
def pipelines() -> list[PipelineInfo]:
    out = []
    for name in builtin_pipelines():
        resolved = resolve_pipeline(find_pipeline(name))
        stages = [
            StageInfo(
                id=sid,
                agent=(rs := resolved.stage(sid)).spec.agent,
                after=list(rs.deps),
                gate=rs.spec.gate or "",
                produces=list(rs.produces),
            )
            for sid in resolved.order
        ]
        out.append(PipelineInfo(name=name, description=resolved.spec.description, stages=stages))
    return out


@router.get("/agents", response_model=list[AgentInfo])
def agents() -> list[AgentInfo]:
    return [
        AgentInfo(
            name=name,
            description=cls.description,
            consumes=list(cls.consumes),
            optional=list(cls.optional_inputs),
            produces=list(cls.produces),
            uses_llm=cls.needs_llm,
        )
        for name, cls in registry.items()
    ]


@router.get("/doctor", response_model=list[CheckInfo])
def doctor(request: Request, network: bool = False) -> list[CheckInfo]:
    """Whether this machine can run the pipeline with this config (`network`: also reach the paper APIs)."""
    settings = settings_of(request)
    config = load_app_config(settings.config_files, runs_root=settings.runs_root)
    return [CheckInfo(name=c.name, status=c.status, detail=c.detail) for c in run_doctor(config, network=network)]
