"""Run directories: layout, creation and lookup.

<runs_root>/<project>/<run_id>/
    run.json        immutable spec: pipeline snapshot + resolved config
    state/          run.json + stages/<stage>.json
    actions/        inbox of queued human actions (+ processed/)
    artifacts/      versioned artifacts (+ .staging/)
    work/<stage>/   per-stage workspaces
    jobs/           job scripts, logs, sentinels
    events.jsonl    append-only audit log
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .. import __version__
from .config import AppConfig
from .errors import RToolError
from .fs import atomic_write_json, now_iso, read_json
from .pipeline import PipelineSpec

_PROJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RunSpec(BaseModel):
    run_id: str
    project: str
    created_at: str
    tool_version: str
    pack: str = "generic"
    pipeline: PipelineSpec
    config: dict[str, Any]


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


class RunHandle:
    def __init__(self, root: Path) -> None:
        root = Path(root).expanduser()
        if not (root / "run.json").is_file():
            raise RToolError(f"{root} is not a run directory (no run.json)")
        self.root = root.resolve()
        self.spec = RunSpec.model_validate(read_json(self.root / "run.json"))
        self.config = AppConfig.model_validate(self.spec.config)

    @property
    def run_id(self) -> str:
        return self.spec.run_id

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def actions_dir(self) -> Path:
        return self.root / "actions"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def lease_path(self) -> Path:
        return self.root / "state" / "episode.lease"

    def work_dir(self, stage_id: str) -> Path:
        return self.root / "work" / stage_id

    @classmethod
    def create(
        cls,
        runs_root: Path,
        project: str,
        pipeline: PipelineSpec,
        config: AppConfig,
        *,
        run_id: str | None = None,
        pack: str = "generic",
    ) -> RunHandle:
        if not _PROJECT.match(project):
            raise RToolError(f"invalid project name {project!r} (letters, digits, . _ - only)")
        run_id = run_id or new_run_id()
        if not _PROJECT.match(run_id) or run_id in {".", ".."}:
            raise RToolError(f"invalid run id {run_id!r} (letters, digits, . _ - only)")
        root = Path(runs_root).expanduser() / project / run_id
        if root.exists():
            raise RToolError(f"run directory already exists: {root}")
        for sub in ("state/stages", "actions", "artifacts", "work", "jobs"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        spec = RunSpec(
            run_id=run_id,
            project=project,
            created_at=now_iso(),
            tool_version=__version__,
            pack=pack,
            pipeline=pipeline,
            config=config.model_dump(mode="json"),
        )
        atomic_write_json(root / "run.json", spec.model_dump(mode="json"))
        return cls(root)


def resolve_run_path(ref: str, runs_root: Path) -> Path:
    """Accept a run directory path, ``project/run_id``, a bare unique ``run_id``, or a project (its latest run)."""
    direct = Path(ref).expanduser()
    if (direct / "run.json").is_file():
        return direct
    root = Path(runs_root).expanduser()
    if (root / ref / "run.json").is_file():
        return root / ref
    if "/" not in ref and (root / ref).is_dir():
        runs = sorted((p.stat().st_mtime, p.parent) for p in (root / ref).glob("*/run.json"))
        if runs:
            return runs[-1][1]  # the project's most recent run
    matches = sorted(p.parent for p in root.glob(f"*/{ref}/run.json")) if root.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RToolError(f"run id {ref!r} is ambiguous: {[str(m) for m in matches]}")
    raise RToolError(f"no run found for {ref!r} (looked in {root})")
