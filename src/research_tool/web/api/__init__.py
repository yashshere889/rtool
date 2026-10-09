"""JSON API routes, all under ``/api``."""

from __future__ import annotations

from fastapi import APIRouter

from . import actions, artifacts, events, literature, runs, session

router = APIRouter()
for module in (session, runs, actions, artifacts, events, literature):
    router.include_router(module.router)
