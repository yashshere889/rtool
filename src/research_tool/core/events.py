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
