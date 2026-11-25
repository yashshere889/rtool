"""Signing in (token → cookie), health and server settings."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ... import __version__
from ...core.errors import RToolError
from ...core.secrets import secrets_status
from ...packs import DEFAULT_PACK, available_packs, load_pack
from ...service import load_app_config
from ..deps import settings_of
from ..security import cookie_name, session_value

router = APIRouter(tags=["session"])


class Health(BaseModel):
    ok: bool = True
    version: str
    mode: str


class SessionRequest(BaseModel):
    token: str


class PackChoice(BaseModel):
    name: str
    title: str
    pipeline: str  # the pipeline its runs use
    description: str = ""
    max_gpu_hours: float | None = None  # this deployment's cap on a run's GPU budget in this field


class ServerSettings(BaseModel):
    version: str
    mode: str
    runs_locally: bool  # true: a local demo/development server (simulated papers, model and cluster)
    runs_root: str
    secrets: dict[str, bool]  # which API keys the server sees (never their values)
    packs: list[PackChoice] = []  # the domain packs this deployment offers new runs


def offered_packs(allowed: list[str], caps: dict[str, float] | None = None, *, demo: bool = False) -> list[PackChoice]:
    """The packs new runs may use here; a pack that does not load is left out (``rtool packs`` shows why).

    ``demo``: a demo server can only simulate the packs it has a scenario for.
    """
    demo_packs: tuple[str, ...] = ()
    if demo:  # the demo module and its fakes are only loaded for a demo server
        from ...testing.demo_run import DEMO_PACKS as demo_packs

    choices = []
    for name in sorted(available_packs(), key=lambda n: n != DEFAULT_PACK):  # the default first
        if (allowed and name not in allowed) or (demo and name not in demo_packs):
            continue
        try:
            pack = load_pack(name)
        except RToolError:
            continue
        choices.append(
            PackChoice(
                name=name,
                title=pack.title,
                pipeline=pack.pipeline or "research",
                description=pack.description,
                max_gpu_hours=(caps or {}).get(name),
            )
        )
    return choices


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    return Health(version=__version__, mode=settings_of(request).mode)


@router.post("/session", status_code=204)
def sign_in(body: SessionRequest, request: Request, response: Response) -> None:
    settings = settings_of(request)
    if not hmac.compare_digest(body.token, settings.token):
        raise HTTPException(401, "wrong access token")
    response.set_cookie(
        cookie_name(request.headers.get("host", "")),
        session_value(settings.token),
        httponly=True,
        samesite="strict",
        path="/",
        secure=False,
    )


@router.get("/settings", response_model=ServerSettings)
def server_settings(request: Request) -> ServerSettings:
    settings = settings_of(request)
    try:
        config = load_app_config(settings.config_files, runs_root=settings.runs_root).packs
        packs = offered_packs(config.allowed, config.max_gpu_hours, demo=settings.runs_locally)
    except RToolError:
        packs = []  # a broken site config is reported when a run is created; settings still load
    return ServerSettings(
        version=__version__,
        mode=settings.mode,
        runs_locally=settings.runs_locally,
        runs_root=str(settings.runs_root),
        secrets=secrets_status(),
        packs=packs,
    )
