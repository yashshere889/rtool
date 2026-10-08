"""Download planned resources into a shared, quota-checked location on the cluster.

* Hugging Face datasets/models: ``huggingface_hub.snapshot_download`` (installed
  on Barkla via the ``[barkla]`` extra), into a cache shared across runs so a
  model is downloaded once.
* URLs: streamed with a size cap.
* torchvision / pip: deferred — the Coder's environment installs them.
* manual: the human provides them (the manifest says where to put them).
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...schemas.plan import FetchedResource, ResourceSpec
from ..http import HttpError, PoliteHttp

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
Snapshot = Callable[..., str]


def default_snapshot() -> Snapshot | None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError:
        return None
    return snapshot_download


def tree_size(path: Path) -> tuple[int, int]:
    total, count = 0, 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            count += 1
            with contextlib.suppress(OSError):  # a file vanished mid-walk
                total += (Path(root) / name).stat().st_size
    return total, count


class Fetcher:
    def __init__(
        self,
        root: Path,
        *,
        http: PoliteHttp | None = None,
        snapshot: Snapshot | None = None,
        hf_token: str | None = None,
        max_url_bytes: int = 50 * 1024**3,
    ) -> None:
        self.root = Path(root)
        self.http = http
        self.snapshot = snapshot if snapshot is not None else default_snapshot()
        self.hf_token = hf_token
        self.max_url_bytes = max_url_bytes

    def fetch(self, resource: ResourceSpec) -> FetchedResource:
        base: dict[str, Any] = {
            "name": resource.name,
            "kind": resource.kind,
            "source": resource.source,
            "identifier": resource.identifier,
        }
        if resource.source in ("torchvision", "pip"):
            return FetchedResource(**base, status="deferred", note="installed with the experiment environment")
        if resource.source == "manual":
            target = self.root / "manual" / _SAFE.sub("_", resource.name)
            target.mkdir(parents=True, exist_ok=True)
            return FetchedResource(
                **base, status="manual", path=str(target), note=f"place the files for '{resource.name}' in {target}"
            )
        try:
            if resource.source == "huggingface":
                return self._huggingface(resource, base)
            return self._url(resource, base)
        except (HttpError, OSError, RuntimeError, ValueError) as exc:
            return FetchedResource(**base, status="failed", note=str(exc)[:300])

    def _huggingface(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        if self.snapshot is None:
            return FetchedResource(
                **base, status="failed", note="huggingface_hub is not installed here (it is part of the [barkla] extra)"
            )
        repo_type = "model" if resource.kind == "model" else "dataset"
        target = self.root / "hf" / f"{repo_type}s" / _SAFE.sub("--", resource.identifier)
        path = self.snapshot(
            repo_id=resource.identifier, repo_type=repo_type, local_dir=str(target), token=self.hf_token
        )
        size, files = tree_size(Path(path))
        return FetchedResource(**base, status="ready", path=str(path), bytes=size, files=files)

    def _url(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        if self.http is None:
            return FetchedResource(**base, status="failed", note="no HTTP client configured")
        name = _SAFE.sub("_", resource.identifier.rstrip("/").rsplit("/", 1)[-1]) or "download"
        target = self.root / "url" / _SAFE.sub("_", resource.name) / name
        result = self.http.download(resource.identifier, target, max_bytes=self.max_url_bytes, accept=())
        return FetchedResource(
            **base, status="ready", path=str(result.path), bytes=result.bytes, files=1, note=result.sha256
        )
