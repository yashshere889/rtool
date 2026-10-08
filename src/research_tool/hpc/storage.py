"""Quota checks so downloads never push the user over Barkla's storage limits.

Barkla quotas (docs, 2025.1): scratch 2 TB / 300k files, fastscratch 500 GB /
500k files, home 75 GB / 100k files. ``lfs quota`` reports Lustre usage in
kilobytes; values over the limit carry a trailing ``*``.
"""

from __future__ import annotations

import getpass
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import RToolError

Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


class QuotaError(RToolError):
    """The planned write would exceed (or come too close to) a storage quota."""


@dataclass(frozen=True)
class Quota:
    mount: str
    used_bytes: int
    limit_bytes: int | None  # None = no limit
    used_files: int
    limit_files: int | None

    def headroom(self, extra_bytes: int = 0, extra_files: int = 0) -> tuple[float | None, float | None]:
        """Fraction of each limit that would be used after adding the extras."""
        bytes_frac = (self.used_bytes + extra_bytes) / self.limit_bytes if self.limit_bytes else None
        files_frac = (self.used_files + extra_files) / self.limit_files if self.limit_files else None
        return bytes_frac, files_frac


def _number(token: str) -> int:
    return int(token.rstrip("*"))


def parse_lfs_quota(output: str, mount: str) -> Quota:
    """Parse ``lfs quota -u USER MOUNT`` (kbytes columns; long mount names wrap onto their own line)."""
    numbers: list[str] = []
    seen_mount = False
    for line in output.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == mount:
            seen_mount = True
            tokens = tokens[1:]
        elif not seen_mount:
            continue
        numbers.extend(tokens)
        if len(numbers) >= 8:
            break
    if len(numbers) < 7:
        raise QuotaError(f"could not parse lfs quota output for {mount}: {output[:300]!r}")
    # kbytes quota limit grace files quota limit grace
    used_kb, _soft_kb, limit_kb = numbers[0], numbers[1], numbers[2]
    used_files, _soft_files, limit_files = numbers[4], numbers[5], numbers[6]
    limit_bytes = _number(limit_kb) * 1024
    limit_file_count = _number(limit_files)
    return Quota(
        mount=mount,
        used_bytes=_number(used_kb) * 1024,
        limit_bytes=limit_bytes or None,
        used_files=_number(used_files),
        limit_files=limit_file_count or None,
    )


def lustre_quota(mount: str, *, user: str | None = None, runner: Runner | None = None) -> Quota:
    run = runner or (lambda args: subprocess.run(args, capture_output=True, text=True, timeout=60, check=False))
    try:
        result = run(["lfs", "quota", "-u", user or getpass.getuser(), mount])
    except FileNotFoundError:
        raise QuotaError("lfs not found: quota checks only work on the cluster") from None
    if result.returncode != 0:
        raise QuotaError(f"lfs quota failed for {mount}: {result.stderr.strip()[:300]}")
    return parse_lfs_quota(result.stdout, mount)


def mount_for(path: Path, mounts: list[str]) -> str | None:
    resolved = str(Path(path).expanduser().resolve())
    matches = [m for m in mounts if resolved == m or resolved.startswith(m.rstrip("/") + "/")]
    return max(matches, key=len) if matches else None


def ensure_room(
    path: Path,
    *,
    extra_bytes: int = 0,
    extra_files: int = 0,
    max_fraction: float = 0.9,
    mounts: tuple[str, ...] = ("/mnt/scratch", "/mnt/fastscratch"),
    quota_fn: Callable[[str], Quota] = lustre_quota,
) -> Quota | None:
    """Raise QuotaError if writing ``extra_*`` under ``path`` would pass ``max_fraction`` of a quota.

    Returns the quota checked, or None when ``path`` is not on a known quota-managed mount.
    """
    mount = mount_for(path, list(mounts))
    if mount is None:
        return None
    quota = quota_fn(mount)
    bytes_frac, files_frac = quota.headroom(extra_bytes, extra_files)
    if bytes_frac is not None and bytes_frac > max_fraction:
        raise QuotaError(
            f"{mount}: writing {extra_bytes / 1e9:.1f} GB would use {bytes_frac:.0%} of your space quota "
            f"(limit {max_fraction:.0%}); free space or choose another location"
        )
    if files_frac is not None and files_frac > max_fraction:
        raise QuotaError(
            f"{mount}: {extra_files} more files would use {files_frac:.0%} of your file-count quota "
            f"(limit {max_fraction:.0%}); pack small files into archives"
        )
    return quota
