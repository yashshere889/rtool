"""The built web app committed in src/research_tool/web/static must match the frontend sources.

The cluster has no Node, so the build is committed. If this fails, run scripts/build_frontend.sh.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
STATIC = ROOT / "src" / "research_tool" / "web" / "static"
SOURCES = ["index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json", "src", "public"]


def sources_hash() -> str:
    """Same as frontend/scripts/stamp.mjs."""
    paths: list[str] = []
    for name in SOURCES:
        target = FRONTEND / name
        if target.is_file():
            paths.append(name)
        elif target.is_dir():
            paths += [p.relative_to(FRONTEND).as_posix() for p in target.rglob("*") if p.is_file()]
    paths = sorted(p for p in paths if ".test." not in p and not p.startswith("src/test/"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode() + b"\0")
        digest.update((FRONTEND / path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@pytest.mark.skipif(not FRONTEND.is_dir(), reason="no frontend sources here (e.g. the copy on the cluster)")
def test_the_committed_build_matches_the_sources() -> None:
    stamp = json.loads((STATIC / "build-stamp.json").read_text())
    assert stamp["sources_sha256"] == sources_hash(), "the web app is out of date: run scripts/build_frontend.sh"


def test_the_build_is_complete() -> None:
    index = (STATIC / "index.html").read_text()
    assets = [line.split('"')[1] for line in index.split() if line.startswith(("src=", "href=")) and "/assets/" in line]
    assert assets and all((STATIC / a.lstrip("/")).is_file() for a in assets)
