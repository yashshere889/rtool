"""Pack a run's durable results into one .tar.gz (e.g. into Barkla's backed-up `data` area).

Runs live in scratch, which is not backed up. An archive keeps what matters —
artifacts, reports, code, results, logs, state and events — and leaves out what
can be rebuilt (virtualenvs, scratch space, caches). Files above a size limit are
skipped and listed in the archive's MANIFEST.json.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__
from .fs import now_iso

SKIP_DIRS = {".venv", "scratch", "__pycache__", ".pytest_cache", ".ruff_cache", "pip-cache", "container-home"}
SKIP_SUFFIXES = (".partial", ".tmp", ".lease", ".pyc")
SKIP_NAMES = {"http_cache.sqlite", "http_cache.sqlite-journal"}


@dataclass
class ArchiveResult:
    path: Path
    files: int = 0
    bytes: int = 0
    skipped: list[dict[str, object]] = field(default_factory=list)


def archive_run(run_root: Path, dest_dir: Path, *, name: str, max_file_bytes: int) -> ArchiveResult:
    run_root, dest_dir = Path(run_root), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"{name}.tar.gz"
    partial = dest_dir / f".{name}.tar.gz.partial"
    result = ArchiveResult(path=final)
    with tarfile.open(partial, "w:gz") as tar:
        for root, dirs, files in os.walk(run_root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for file_name in sorted(files):
                path = Path(root) / file_name
                rel = path.relative_to(run_root)
                if file_name in SKIP_NAMES or file_name.endswith(SKIP_SUFFIXES) or path.is_symlink():
                    continue
                size = path.stat().st_size
                if size > max_file_bytes:
                    result.skipped.append({"path": str(rel), "bytes": size})
                    continue
                tar.add(path, arcname=f"{name}/{rel}", recursive=False)
                result.files += 1
                result.bytes += size
        manifest = json.dumps(
            {
                "run": str(run_root),
                "created_at": now_iso(),
                "rtool_version": __version__,
                "files": result.files,
                "bytes": result.bytes,
                "skipped_large_files": result.skipped,
                "left_out": sorted(SKIP_DIRS),
            },
            indent=2,
        ).encode()
        info = tarfile.TarInfo(f"{name}/MANIFEST.json")
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
    os.replace(partial, final)
    return result
