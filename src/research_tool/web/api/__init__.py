"""JSON API routes, all under ``/api``."""

from __future__ import annotations

from fastapi import APIRouter

from . import actions, artifacts, events, hypotheses, literature, runs, session

router = APIRouter()
for module in (session, runs, actions, artifacts, events, literature, hypotheses):
    router.include_router(module.router)
