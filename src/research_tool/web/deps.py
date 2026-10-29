"""Shared pieces of the API: settings, the runner, opening runs from URL parts, safe paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from ..core.errors import PathEscape, RToolError
from ..core.run import RunHandle
from ..core.workspace import Workspace
from ..service import open_run
from .runner import Runner
from .settings import UISettings


def settings_of(request: Request) -> UISettings:
    return request.app.state.settings


def runner_of(request: Request) -> Runner:
    return request.app.state.runner


def get_run(request: Request, project: str, run_id: str) -> RunHandle:
    """The run named in the URL; 404 for anything that is not a plain project/run name or does not exist."""
    try:
        return open_run(settings_of(request).runs_root, project, run_id)
    except RToolError:
        raise HTTPException(404, f"no run {project}/{run_id}") from None


def run_relative(run: RunHandle, path: str | Path) -> str | None:
    """``path`` relative to the run directory, or None if it lies outside (never hand out other paths).

    Gate summaries store absolute paths; if the runs folder was moved since, the part after
    ``<project>/<run_id>/`` still names the file inside this run.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = run.root / candidate
    try:
        return str(candidate.resolve().relative_to(run.root))
    except (ValueError, OSError):
        pass
    marker = f"/{run.spec.project}/{run.run_id}/"
    text = str(path)
    if marker in text:
        rel = text.split(marker, 1)[1]
        try:
            inside = (run.root / rel).resolve()
            if run.root in inside.parents and inside.is_file():
                return str(inside.relative_to(run.root))
        except OSError:
            return None
    return None


def jailed(run: RunHandle, rel: str) -> Path:
    """A path inside the run directory (symlinks resolved); 404 otherwise."""
    try:
        path = Workspace(run.root).path(rel)
    except PathEscape:
        raise HTTPException(404, "no such file") from None
    if not path.is_file():
        raise HTTPException(404, "no such file")
    return path


def public_summary(run: RunHandle, summary: dict[str, Any]) -> dict[str, Any]:
    """A gate summary fit for the browser: review files as run-relative paths only."""
    out = dict(summary)
    if "review_files" in out:
        paths = (run_relative(run, p) for p in out["review_files"])
        out["review_files"] = list(dict.fromkeys(p for p in paths if p))  # several outputs may share one file
    return out
