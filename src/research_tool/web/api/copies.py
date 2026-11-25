"""Copies and archives of a run: redo one stage in a copy, and pack a run's results into one file.

Neither changes the run itself. A replay copies the run and makes one stage run again there (the original stays as
it is). An archive reads the run and writes one .tar.gz under ``paths.archive_root``; it runs in a background thread
because a whole run can be large, and only one archive per run runs at a time.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.archive import SKIP_DIRS, SKIP_NAMES, SKIP_SUFFIXES, archive_run
from ...core.errors import RToolError
from ...core.run import RunHandle
from ...hpc.episode import llm_needed
from ...service import KickResult, replay_run
from ..deps import get_run, runner_of

router = APIRouter(tags=["copies"])

DEFAULT_MAX_FILE_MB = 200.0


class ReplayRequest(BaseModel):
    stage: str
    tag: str = Field(default="again", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,30}$")


class Replayed(BaseModel):
    project: str
    run_id: str
    kick: KickResult


@router.post("/runs/{project}/{run_id}/replay", response_model=Replayed, status_code=201)
def replay(project: str, run_id: str, body: ReplayRequest, request: Request) -> Replayed:
    """Copy the run and redo ``stage`` in the copy (same inputs); the copy stops at that stage's checkpoint."""
    source = get_run(request, project, run_id)
    try:
        copy = replay_run(source, body.stage, body.tag)
    except RToolError as exc:
        raise HTTPException(409, str(exc)) from None
    kick = runner_of(request).wake(copy, "replay", needs_llm=llm_needed(copy))
    return Replayed(project=project, run_id=copy.run_id, kick=kick)


# ------------------------------------------------------------------ archive
class ArchiveRequest(BaseModel):
    max_file_mb: float = Field(default=DEFAULT_MAX_FILE_MB, gt=0, le=DEFAULT_MAX_FILE_MB)


class Skipped(BaseModel):
    path: str
    bytes: int


class ArchiveView(BaseModel):
    can_archive: bool
    why_not: str
    run_bytes: int  # what would be packed, before compression, at the default file cap
    default_max_file_mb: float
    state: Literal["none", "running", "done", "failed"]
    path: str  # where the archive is (on the server), once done
    files: int
    bytes: int
    skipped: list[Skipped]  # files left out for being larger than the cap
    error: str


_lock = threading.Lock()
_jobs: dict[Path, ArchiveView] = {}


def _archive_root(run: RunHandle) -> Path | None:
    root = run.config.paths.archive_root
    return Path(os.path.expandvars(str(root))).expanduser() if root else None


def _packable_bytes(run: RunHandle) -> int:
    """About what ``archive_run`` would take: the same skips, without the size cap."""
    total = 0
    for root, dirs, files in os.walk(run.root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = Path(root) / name
            if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES) or path.is_symlink():
                continue
            total += path.stat().st_size
    return total


def _view(run: RunHandle) -> ArchiveView:
    target = _archive_root(run)
    with _lock:
        job = _jobs.get(run.root)
    base = ArchiveView(
        can_archive=target is not None,
        why_not="" if target is not None else "no archive folder is set: set paths.archive_root in the configuration",
        run_bytes=_packable_bytes(run),
        default_max_file_mb=DEFAULT_MAX_FILE_MB,
        state="none",
        path="",
        files=0,
        bytes=0,
        skipped=[],
        error="",
    )
    return base if job is None else base.model_copy(update={k: getattr(job, k) for k in _JOB_FIELDS})


_JOB_FIELDS = ("state", "path", "files", "bytes", "skipped", "error")


@router.get("/runs/{project}/{run_id}/archive", response_model=ArchiveView)
def archive_state(project: str, run_id: str, request: Request) -> ArchiveView:
    return _view(get_run(request, project, run_id))


@router.post("/runs/{project}/{run_id}/archive", response_model=ArchiveView, status_code=202)
def archive_start(project: str, run_id: str, body: ArchiveRequest, request: Request) -> ArchiveView:
    """Pack the run into ``paths.archive_root`` in the background; poll GET for the outcome."""
    run = get_run(request, project, run_id)
    target = _archive_root(run)
    if target is None:
        raise HTTPException(409, "no archive folder is set: set paths.archive_root in the configuration")
    with _lock:
        current = _jobs.get(run.root)
        if current is not None and current.state == "running":
            raise HTTPException(409, "this run is already being archived")
        _jobs[run.root] = ArchiveView(
            can_archive=True,
            why_not="",
            run_bytes=0,
            default_max_file_mb=DEFAULT_MAX_FILE_MB,
            state="running",
            path="",
            files=0,
            bytes=0,
            skipped=[],
            error="",
        )

    def work() -> None:
        try:
            result = archive_run(
                run.root, target, name=f"{run.spec.project}-{run.run_id}", max_file_bytes=int(body.max_file_mb * 1e6)
            )
            update = {
                "state": "done",
                "path": str(result.path),
                "files": result.files,
                "bytes": result.bytes,
                "skipped": [Skipped(path=str(s["path"]), bytes=int(s["bytes"])) for s in result.skipped],  # type: ignore[call-overload]
            }
        except Exception as exc:  # noqa: BLE001 - say why on the page instead of losing it in a thread
            update = {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        with _lock:
            _jobs[run.root] = _jobs[run.root].model_copy(update=update)

    threading.Thread(target=work, name=f"archive-{run.run_id}", daemon=True).start()
    return _view(run)


def wait_for_archive(root: Path, timeout: float = 30) -> None:
    """Tests: wait until the archive of the run at ``root`` has finished."""
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with _lock:
            job = _jobs.get(root)
        if job is None or job.state != "running":
            return
        time.sleep(0.05)
