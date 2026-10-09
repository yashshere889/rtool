"""JSON API routes, all under ``/api``."""

from __future__ import annotations

from fastapi import APIRouter

from . import actions, artifacts, events, runs, session

router = APIRouter()
for module in (session, runs, actions, artifacts, events):
    router.include_router(module.router)
