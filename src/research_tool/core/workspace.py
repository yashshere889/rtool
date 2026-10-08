"""Path-jailed working directory for a stage.

Agents (and later the Coder's file tools) get a ``Workspace`` instead of raw
paths; every relative path is resolved inside the jail, symlinks included.
"""

from __future__ import annotations

from pathlib import Path

from .errors import PathEscape
from .fs import atomic_write_text


class Workspace:
    def __init__(self, root: Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    def path(self, rel: str | Path = ".") -> Path:
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PathEscape(f"{str(rel)!r} resolves outside the workspace {self.root}")
        return candidate

    def exists(self, rel: str | Path) -> bool:
        return self.path(rel).exists()

    def mkdir(self, rel: str | Path) -> Path:
        target = self.path(rel)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_text(self, rel: str | Path, text: str) -> Path:
        target = self.path(rel)
        atomic_write_text(target, text)
        return target

    def read_text(self, rel: str | Path) -> str:
        return self.path(rel).read_text(encoding="utf-8")

    def listdir(self, rel: str | Path = ".") -> list[str]:
        return sorted(p.name for p in self.path(rel).iterdir())
