"""Append-only, machine-readable event log (one JSON object per line).

The log is the audit trail of a run and the feed a future UI or REST layer can
tail. Only the episode holding the run lease writes to it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .fs import append_jsonl, now_iso


class EventLog:
    def __init__(self, path: Path, run_id: str, episode_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.episode_id = episode_id

    def emit(self, kind: str, stage: str | None = None, **data: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"ts": now_iso(), "run": self.run_id, "episode": self.episode_id, "kind": kind}
        if stage is not None:
            record["stage"] = stage
        if data:
            record["data"] = data
        append_jsonl(self.path, record)
        return record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn final line after a crash

    def read_from(self, offset: int = 0, *, limit: int | None = None) -> tuple[list[dict[str, Any]], int]:
        """Events after byte ``offset`` and the offset to continue from (a cursor for tailing).

        Only complete lines are returned, so a line still being appended is read
        next time. A cursor past the end (the log was replaced) starts over.
        """
        if not self.path.exists():
            return [], 0
        with open(self.path, "rb") as f:
            size = f.seek(0, 2)
            if offset > size:
                offset = 0
            f.seek(offset)
            data = f.read(size - offset)
        events: list[dict[str, Any]] = []
        position = offset
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break  # torn or still being written
            position += len(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if limit is not None and len(events) >= limit:
                break
        return events, position

    def read(
        self,
        *,
        kinds: Iterable[str] | None = None,
        stage: str | None = None,
        last: int | None = None,
    ) -> list[dict[str, Any]]:
        wanted = set(kinds) if kinds else None
        out = [
            e
            for e in self
            if (wanted is None or e.get("kind") in wanted) and (stage is None or e.get("stage") == stage)
        ]
        return out[-last:] if last else out
