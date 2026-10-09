"""Action inbox: how the CLI (or a future UI) talks to the engine.

The CLI never edits engine state directly. It drops an action file into
``actions/``; the next episode applies it while holding the run lease and moves
it to ``actions/processed/`` together with the outcome. File names sort
chronologically, so actions apply in the order they were queued.
"""

from __future__ import annotations

import getpass
import json
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .fs import atomic_write_json, now_iso, read_json


class ActionKind(StrEnum):
    APPROVE = "approve"  # approve an open gate
    REJECT = "reject"  # reject an open gate (with feedback) -> the stage re-runs
    SELECT = "select"  # choose an option at a selection gate
    RETRY = "retry"  # re-run a failed stage (fresh, or resume from checkpoint)
    NOTE = "note"  # attach guidance to a stage's next attempt
    REOPEN = "reopen"  # open a finished stage's checkpoint again (choose differently, or reject to re-run it)


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no passwd entry in some containers
        return "unknown"


class Action(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    kind: ActionKind
    stage: str | None = None
    gate: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
    by: str = Field(default_factory=_user)


class ActionInbox:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.processed_dir = self.root / "processed"
        self.invalid_dir = self.root / "invalid"

    def submit(self, action: Action) -> Path:
        path = self.root / f"{time.time_ns():020d}-{action.id[:12]}.json"
        atomic_write_json(path, action.model_dump(mode="json"))
        return path

    def pending(self) -> list[tuple[Path, Action]]:
        if not self.root.is_dir():
            return []
        out: list[tuple[Path, Action]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                out.append((path, Action.model_validate(read_json(path))))
            except (json.JSONDecodeError, ValidationError, OSError):
                self.invalid_dir.mkdir(parents=True, exist_ok=True)
                path.replace(self.invalid_dir / path.name)
        return out

    def archive(self, path: Path, action: Action, outcome: dict[str, Any]) -> None:
        record = action.model_dump(mode="json") | {"outcome": outcome, "applied_at": now_iso()}
        atomic_write_json(self.processed_dir / path.name, record)
        path.unlink(missing_ok=True)

    def processed(self) -> list[dict[str, Any]]:
        if not self.processed_dir.is_dir():
            return []
        return [read_json(p) for p in sorted(self.processed_dir.glob("*.json"))]

    def lookup(self, action_id: str) -> tuple[str, dict[str, Any] | None]:
        """Where action ``action_id`` is: ("queued" | "processed" | "invalid" | "unknown", its record)."""
        pattern = f"*-{action_id[:12]}.json"
        places = (("queued", self.root), ("processed", self.processed_dir), ("invalid", self.invalid_dir))
        for state, directory in places:
            for path in sorted(directory.glob(pattern)) if directory.is_dir() else []:
                try:
                    record = read_json(path)
                except (json.JSONDecodeError, OSError):
                    record = {}
                if record.get("id") == action_id or state == "invalid":
                    return state, record
        return "unknown", None

    def withdraw(self, action_id: str) -> bool:
        """Remove a queued action before an episode applies it. False if it is no longer queued."""
        for path in self.root.glob(f"*-{action_id[:12]}.json") if self.root.is_dir() else []:
            try:
                if read_json(path).get("id") != action_id:
                    continue
            except (json.JSONDecodeError, OSError):
                continue
            try:
                path.unlink()  # an episode may have taken it this very moment: then it is not queued any more
            except FileNotFoundError:
                return False
            return True
        return False
