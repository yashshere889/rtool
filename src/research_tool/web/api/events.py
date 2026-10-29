"""The run's event log: pages for history, and a live stream (Server-Sent Events) for the activity feed."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ...core.events import EventLog
from ..deps import get_run

router = APIRouter(tags=["events"])

NOISE = {"llm_call"}  # one per model call: thousands per run, useful only for accounting
POLL_SECONDS = 1.0
PING_SECONDS = 15.0


class EventPage(BaseModel):
    events: list[dict[str, Any]]
    cursor: int  # pass back as ?after= to continue


def visible(events: list[dict[str, Any]], include_noise: bool) -> list[dict[str, Any]]:
    return events if include_noise else [e for e in events if e.get("kind") not in NOISE]


@router.get("/runs/{project}/{run_id}/events", response_model=EventPage)
def events(
    project: str,
    run_id: str,
    request: Request,
    after: int = 0,
    limit: int = 500,
    tail: int | None = None,
    include_noise: bool = False,
) -> EventPage:
    """Events after the cursor ``after`` (at most ``limit``), or with ``tail`` the last ``tail`` events."""
    run = get_run(request, project, run_id)
    log = EventLog(run.events_path, run.run_id)
    if tail is not None:
        found, cursor = log.read_from(0)
        return EventPage(events=visible(found, include_noise)[-max(1, min(tail, 5000)) :], cursor=cursor)
    found, cursor = log.read_from(after, limit=max(1, min(limit, 5000)))
    return EventPage(events=visible(found, include_noise), cursor=cursor)


@router.get("/runs/{project}/{run_id}/events/stream")
async def stream(
    project: str, run_id: str, request: Request, after: int | None = None, timeout: float = 0.0
) -> StreamingResponse:
    """New events as they are written: one SSE message per batch, ``{"events": [...], "cursor": n}``.

    Resumes from ``Last-Event-ID`` (or ``?after=``); without either it starts at the end of the log.
    ``timeout`` (seconds) ends the stream, for tests.
    """
    run = get_run(request, project, run_id)
    log = EventLog(run.events_path, run.run_id)
    resume = request.headers.get("last-event-id")
    start = int(resume) if resume and resume.isdigit() else after

    async def generate() -> AsyncIterator[str]:
        cursor = start if start is not None else (log.path.stat().st_size if log.path.exists() else 0)
        began = last = time.monotonic()
        yield f"retry: 3000\nid: {cursor}\nevent: ready\ndata: {json.dumps({'cursor': cursor})}\n\n"
        while not await request.is_disconnected():
            found, new_cursor = await run_in_threadpool(log.read_from, cursor)
            if new_cursor != cursor:
                cursor = new_cursor
                shown = visible(found, include_noise=False)
                if shown:
                    payload = json.dumps({"events": shown, "cursor": cursor})
                    yield f"id: {cursor}\nevent: events\ndata: {payload}\n\n"
                    last = time.monotonic()
            now = time.monotonic()
            if now - last >= PING_SECONDS:
                yield ": ping\n\n"
                last = now
            if timeout and now - began >= timeout:
                return
            await asyncio.sleep(POLL_SECONDS if not timeout else min(POLL_SECONDS, timeout / 4))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )
