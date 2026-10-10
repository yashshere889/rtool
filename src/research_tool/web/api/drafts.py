"""Drafts of hand edits, saved as the researcher types (one per run and output), so nothing is lost on reload.

They live in ``<run>/ui/drafts/``, outside everything the engine reads. Sending an
edit removes its draft; taking the edit back while it is queued restores it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ...core.fs import atomic_write_json, now_iso, read_json
from ...core.run import RunHandle
from ..deps import get_run

router = APIRouter(tags=["drafts"])

KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")  # the artifact name
MAX_BYTES = 4 * 1024 * 1024


class Draft(BaseModel):
    base_version: int  # the version the edit started from
    value: dict[str, Any]
    note: str = ""
    saved_at: str = ""


def draft_path(run: RunHandle, key: str) -> Path:
    if not KEY.match(key):
        raise HTTPException(422, "invalid draft name")
    return run.root / "ui" / "drafts" / f"{key}.json"


def read_draft(run: RunHandle, key: str) -> Draft | None:
    path = draft_path(run, key)
    try:
        return Draft.model_validate(read_json(path))
    except (OSError, ValueError):
        return None


def write_draft(run: RunHandle, key: str, draft: Draft) -> Draft:
    saved = draft.model_copy(update={"saved_at": now_iso()})
    atomic_write_json(draft_path(run, key), saved.model_dump(mode="json"))
    return saved


def remove_draft(run: RunHandle, key: str) -> None:
    draft_path(run, key).unlink(missing_ok=True)


@router.get("/runs/{project}/{run_id}/drafts/{key}", response_model=Draft)
def get_draft(project: str, run_id: str, key: str, request: Request) -> Draft:
    draft = read_draft(get_run(request, project, run_id), key)
    if draft is None:
        raise HTTPException(404, "no draft")
    return draft


@router.put("/runs/{project}/{run_id}/drafts/{key}", response_model=Draft)
def put_draft(project: str, run_id: str, key: str, draft: Draft, request: Request) -> Draft:
    if int(request.headers.get("content-length") or 0) > MAX_BYTES:
        raise HTTPException(413, "this draft is too large")
    return write_draft(get_run(request, project, run_id), key, draft)


@router.delete("/runs/{project}/{run_id}/drafts/{key}", status_code=204)
def delete_draft(project: str, run_id: str, key: str, request: Request) -> Response:
    remove_draft(get_run(request, project, run_id), key)
    return Response(status_code=204)
