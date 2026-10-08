"""Filesystem helpers that are safe on shared (NFS/Lustre) storage.

Every state change is either a whole-file atomic replace (write temp + rename)
or a single O_APPEND write of one line, so a killed episode never leaves a
half-written state file behind.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import LeaseHeld


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical_json(obj: Any) -> str:
    """Stable JSON used for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line using a single O_APPEND write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


@dataclass
class LeaseInfo:
    owner: str
    host: str
    pid: int
    acquired_at: float
    expires_at: float


class FileLease:
    """A lease file that gives one process exclusive ownership of a run.

    Creation uses ``O_CREAT | O_EXCL``, which is atomic on local disks, NFSv3+
    and Lustre. The holder must ``renew()`` before ``ttl`` seconds pass; an
    expired lease may be taken over, which is how the lock of a crashed
    episode is recovered.
    """

    def __init__(
        self,
        path: Path,
        ttl: float = 300.0,
        owner: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.ttl = ttl
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.clock = clock
        self._held = False
        self._acquired_at = 0.0

    @property
    def held(self) -> bool:
        return self._held

    def holder(self) -> LeaseInfo | None:
        try:
            return LeaseInfo(**read_json(self.path))
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _info(self) -> LeaseInfo:
        now = self.clock()
        return LeaseInfo(
            owner=self.owner,
            host=socket.gethostname(),
            pid=os.getpid(),
            acquired_at=self._acquired_at or now,
            expires_at=now + self.ttl,
        )

    def _expired(self) -> bool:
        info = self.holder()
        if info is None:
            # Unreadable: a contender may be between create and write. Only stale if old.
            try:
                age = time.time() - self.path.stat().st_mtime
            except FileNotFoundError:
                return True
            return age > self.ttl
        return info.expires_at < self.clock()

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                info = self.holder()
                # Our own name on it: an earlier attempt of the same Slurm job, killed and requeued.
                ours = info is not None and info.owner == self.owner
                if not ours and not self._expired():
                    return False
                # Move the stale lease aside; only one contender's rename can succeed.
                aside = self.path.with_name(f"{self.path.name}.stale-{uuid.uuid4().hex[:8]}")
                try:
                    os.rename(self.path, aside)
                except FileNotFoundError:
                    continue
                aside.unlink(missing_ok=True)
                continue
            self._acquired_at = self.clock()
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self._info()), f)
                f.flush()
                os.fsync(f.fileno())
            self._held = True
            return True
        return False

    def renew(self) -> None:
        if not self._held:
            raise LeaseHeld(f"lease {self.path} is not held by {self.owner}")
        info = self.holder()
        if info is None or info.owner != self.owner:
            self._held = False
            taker = info.owner if info else "unknown"
            raise LeaseHeld(f"lease {self.path} was taken over by {taker}")
        atomic_write_json(self.path, asdict(self._info()))

    def release(self) -> None:
        if not self._held:
            return
        info = self.holder()
        if info is not None and info.owner == self.owner:
            self.path.unlink(missing_ok=True)
        self._held = False

    def __enter__(self) -> FileLease:
        if not self.acquire():
            info = self.holder()
            raise LeaseHeld(f"run is locked by {info.owner if info else 'another process'}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
