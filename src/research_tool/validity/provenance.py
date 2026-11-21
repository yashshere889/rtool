"""Which code a validity run ran: a hash of the installed package, and the git commit when there is one.

Barkla gets the code by rsync, without git metadata, so the hash is what identifies it there. The
same code gives the same hash anywhere: ``python -m research_tool.validity.provenance`` prints it, so
a card from Barkla can be matched to a commit on the Mac.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from pydantic import BaseModel

PACKAGE = Path(__file__).resolve().parents[1]  # src/research_tool


class CodeVersion(BaseModel):
    code: str  # sha256 over the package's files
    commit: str = ""  # git HEAD, when the package is in a git checkout
    dirty: bool = False  # the package has changes git has not committed

    @property
    def label(self) -> str:
        return self.code[:12]


def code_hash(root: Path = PACKAGE) -> str:
    """sha256 over every file of the package (path and bytes), compiled files left out."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        if path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_version(root: Path = PACKAGE) -> CodeVersion:
    commit = _git(root, "rev-parse", "HEAD") or ""
    status = _git(root, "status", "--porcelain", "--", ".") if commit else None
    return CodeVersion(code=code_hash(root), commit=commit, dirty=bool(status))


if __name__ == "__main__":
    version = code_version()
    print(
        f"code {version.code}\ncommit {version.commit or '(no git)'}{' (uncommitted changes)' if version.dirty else ''}"
    )
