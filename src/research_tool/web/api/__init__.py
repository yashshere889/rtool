"""JSON API routes, all under ``/api``."""

from __future__ import annotations

from fastapi import APIRouter

from . import actions, artifacts, code, drafts, events, hypotheses, literature, plan, report, runs, session

router = APIRouter()
for module in (session, runs, actions, artifacts, events, literature, hypotheses, plan, drafts, code, report):
    router.include_router(module.router)
