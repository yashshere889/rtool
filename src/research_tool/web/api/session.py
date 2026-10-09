"""Signing in (token → cookie), health and server settings."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ... import __version__
from ...core.secrets import secrets_status
from ..deps import settings_of
from ..security import SESSION_COOKIE, session_value

router = APIRouter(tags=["session"])


class Health(BaseModel):
    ok: bool = True
    version: str
    mode: str


class SessionRequest(BaseModel):
    token: str


class ServerSettings(BaseModel):
    version: str
    mode: str
    runs_locally: bool  # true: a local demo/development server (simulated papers, model and cluster)
    runs_root: str
    secrets: dict[str, bool]  # which API keys the server sees (never their values)


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    return Health(version=__version__, mode=settings_of(request).mode)


@router.post("/session", status_code=204)
def sign_in(body: SessionRequest, request: Request, response: Response) -> None:
    settings = settings_of(request)
    if not hmac.compare_digest(body.token, settings.token):
        raise HTTPException(401, "wrong access token")
    response.set_cookie(
        SESSION_COOKIE, session_value(settings.token), httponly=True, samesite="strict", path="/", secure=False
    )


@router.get("/settings", response_model=ServerSettings)
def server_settings(request: Request) -> ServerSettings:
    settings = settings_of(request)
    return ServerSettings(
        version=__version__,
        mode=settings.mode,
        runs_locally=settings.runs_locally,
        runs_root=str(settings.runs_root),
        secrets=secrets_status(),
    )
