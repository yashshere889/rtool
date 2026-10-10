"""JSON API routes, all under ``/api``."""

from __future__ import annotations

from fastapi import APIRouter

from . import actions, artifacts, code, drafts, events, hypotheses, literature, plan, runs, session

router = APIRouter()
for module in (session, runs, actions, artifacts, events, literature, hypotheses, plan, drafts, code):
    router.include_router(module.router)
