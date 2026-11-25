"""JSON API routes, all under ``/api``."""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    actions,
    artifacts,
    code,
    copies,
    drafts,
    events,
    hypotheses,
    literature,
    plan,
    report,
    runs,
    session,
    study,
    synthesis,
    system,
    validity,
)

router = APIRouter()
for module in (
    session,
    runs,
    actions,
    artifacts,
    events,
    literature,
    hypotheses,
    plan,
    drafts,
    code,
    report,
    study,
    synthesis,
    system,
    validity,
    copies,
):
    router.include_router(module.router)
