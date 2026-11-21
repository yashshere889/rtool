"""Download planned resources into a shared, quota-checked location on the cluster.

* Hugging Face datasets/models: ``huggingface_hub.snapshot_download`` (installed
  on Barkla via the ``[barkla]`` extra), into a cache shared across runs so a
  model is downloaded once. A model already in the LLM server's cache
  (:func:`local_models`) is used from there instead.
* URLs: streamed with a size cap.
* torchvision datasets: downloaded once by torchvision itself (``torchvision``, a callable that runs it in
  the experiment environment), so smoke tests and every run read the same local folder.
* pip (and torchvision when no such callable is given): deferred — the Coder's environment handles them.
* local: a dataset the researcher provides on disk (``experiments.local_data``), copied into the
  resources folder under its content hash, so runs read a copy nobody changes under them.
* manual: the human provides them (the manifest says where to put them).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...core.fs import atomic_write_text
from ...schemas.plan import FetchedResource, ResourceSpec
from ..http import HttpError, PoliteHttp
from .models import biomodels_files, download_url, sbml_problem
from .omics import cellxgene_latest, geo_files, h5ad_asset
from .structures import RCSB_FILES, accession, alphafold_models, pdb_id, uniprot_fasta

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
Snapshot = Callable[..., str]
Torchvision = Callable[[str, Path], str]  # (identifier, target folder) -> the torchvision class it resolved to


def default_snapshot() -> Snapshot | None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError:
        return None
    return snapshot_download


def local_models(hf_home: str | None) -> dict[str, str]:
    """Hub models already in a Hugging Face cache (on Barkla: the LLM server's weights): id -> snapshot folder."""
    hub = Path(os.path.expandvars(os.path.expanduser(hf_home))) / "hub" if hf_home else None
    found: dict[str, str] = {}
    if hub is None or not hub.is_dir():
        return found
    for repo in sorted(hub.glob("models--*")):
        snapshots = repo / "snapshots"
        ref = repo / "refs" / "main"
        pinned = snapshots / ref.read_text().strip() if ref.is_file() else None
        newest = max(snapshots.glob("*"), key=lambda p: p.stat().st_mtime, default=None) if snapshots.is_dir() else None
        path = pinned if pinned is not None and pinned.is_dir() else newest
        if path is not None:
            found[repo.name.removeprefix("models--").replace("--", "/", 1)] = str(path)
    return found


def content_hash(path: Path) -> str:
    """sha256 over every file's relative path and bytes, in order: the same data gives the same hash."""
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in files:
        digest.update((file.relative_to(path).as_posix() if path.is_dir() else file.name).encode() + b"\0")
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


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
        torchvision: Torchvision | None = None,
        local: dict[str, str] | None = None,
        provided: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.local = local or {}  # Hub model id -> a folder that already holds it
        self.provided = dict(provided or {})  # local dataset name -> LocalDataset (its .path)
        self.http = http
        self.snapshot = snapshot if snapshot is not None else default_snapshot()
        self.hf_token = hf_token
        self.max_url_bytes = max_url_bytes
        self.torchvision = torchvision

    def fetch(self, resource: ResourceSpec) -> FetchedResource:
        base: dict[str, Any] = {
            "name": resource.name,
            "kind": resource.kind,
            "source": resource.source,
            "identifier": resource.identifier,
        }
        if resource.source == "pip" or (resource.source == "torchvision" and self.torchvision is None):
            return FetchedResource(**base, status="deferred", note="installed with the experiment environment")
        if resource.source == "local":
            return self._provided(resource, base)
        if resource.source == "manual":
            target = self.root / "manual" / _SAFE.sub("_", resource.name)
            target.mkdir(parents=True, exist_ok=True)
            return FetchedResource(
                **base, status="manual", path=str(target), note=f"place the files for '{resource.name}' in {target}"
            )
        try:
            if resource.source == "geo":
                return self._geo(resource, base)
            if resource.source == "cellxgene":
                return self._cellxgene(resource, base)
            if resource.source == "biomodels":
                return self._biomodels(resource, base)
            if resource.source in ("uniprot", "alphafold", "pdb"):
                return self._structure(resource, base)
            if resource.source == "huggingface":
                return self._huggingface(resource, base)
            if resource.source == "torchvision":
                return self._torchvision(resource, base)
            return self._url(resource, base)
        except (HttpError, OSError, RuntimeError, ValueError) as exc:
            return FetchedResource(**base, status="failed", note=str(exc)[:300])

    def _geo(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        if self.http is None:
            return FetchedResource(**base, status="failed", note="no HTTP client configured")
        urls = geo_files(self.http, resource.identifier)
        if not urls:
            return FetchedResource(**base, status="failed", note="the GEO series has no supplementary or matrix files")
        target = self.root / "geo" / resource.identifier.strip().upper()
        for url in urls:  # suppl/ and matrix/ keep their names; a file already there is not fetched again
            sub, name = url.rstrip("/").rsplit("/", 2)[-2:]
            dest = target / sub / name
            if not dest.exists():
                self.http.download(url, dest, max_bytes=self.max_url_bytes, accept=())
        size, files = tree_size(target)
        return FetchedResource(
            **base, status="ready", path=str(target), bytes=size, files=files, note=f"{files} GEO files"
        )

    def _cellxgene(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        if self.http is None:
            return FetchedResource(**base, status="failed", note="no HTTP client configured")
        version = cellxgene_latest(self.http, resource.identifier)
        asset = h5ad_asset(version)
        if asset is None:
            return FetchedResource(**base, status="failed", note="the CELLxGENE dataset has no H5AD file")
        dataset = resource.identifier.strip().lower()
        dest = self.root / "cellxgene" / dataset / f"{dataset}.h5ad"
        if not dest.exists():
            self.http.download(asset["url"], dest, max_bytes=self.max_url_bytes, accept=())
        size, files = tree_size(dest.parent)
        note = f"{version.get('title') or 'CELLxGENE dataset'}, version {version.get('dataset_version_id', '?')}"
        return FetchedResource(**base, status="ready", path=str(dest.parent), bytes=size, files=files, note=note)

    def _biomodels(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        if self.http is None:
            return FetchedResource(**base, status="failed", note="no HTTP client configured")
        model = resource.identifier.strip()
        files = biomodels_files(self.http, model)
        if not files:
            return FetchedResource(**base, status="failed", note=f"BioModels lists no main file for {model}")
        target = self.root / "biomodels" / model
        problems = []
        for entry in files:
            dest = target / Path(entry["name"]).name
            if not dest.exists():
                self.http.download(download_url(model, entry["name"]), dest, max_bytes=self.max_url_bytes, accept=())
            if dest.suffix.lower() == ".xml":
                problems += [p for p in (sbml_problem(dest),) if p]
        if problems:  # the experiment would read a broken model: better to stop here and say why
            return FetchedResource(**base, status="failed", path=str(target), note="; ".join(problems)[:300])
        size, count = tree_size(target)
        return FetchedResource(
            **base, status="ready", path=str(target), bytes=size, files=count, note=f"BioModels {model}"
        )

    def _structure(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        """A UniProt sequence (FASTA), an AlphaFold DB model or a PDB entry (mmCIF), kept once per id."""
        if self.http is None:
            return FetchedResource(**base, status="failed", note="no HTTP client configured")
        folder = self.root / resource.source
        if resource.source == "uniprot":
            dest = folder / f"{accession(resource.identifier)}.fasta"
            if not dest.exists():
                atomic_write_text(dest, uniprot_fasta(self.http, resource.identifier))
            note = dest.read_text(encoding="utf-8").splitlines()[0][1:120]
        elif resource.source == "alphafold":
            model = alphafold_models(self.http, resource.identifier)[0]
            url = model.get("cifUrl") or model["pdbUrl"]
            dest = folder / url.rsplit("/", 1)[-1]
            if not dest.exists():
                self.http.download(url, dest, max_bytes=self.max_url_bytes, accept=())
            note = f"AlphaFold model {model.get('entryId', '?')}, mean pLDDT {model.get('globalMetricValue', '?')}"
        else:
            dest = folder / f"{pdb_id(resource.identifier)}.cif"
            if not dest.exists():
                self.http.download(f"{RCSB_FILES}/download/{dest.name}", dest, max_bytes=self.max_url_bytes, accept=())
            note = "PDB entry (mmCIF)"
        return FetchedResource(**base, status="ready", path=str(dest), bytes=dest.stat().st_size, files=1, note=note)

    def _provided(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        found = self.provided.get(resource.identifier)
        source = Path(os.path.expandvars(os.path.expanduser(found.path))) if found is not None else None
        if source is None or not source.exists():
            return FetchedResource(**base, status="failed", note=f"no local dataset {resource.identifier!r} on disk")
        digest = content_hash(source)
        target = self.root / "local" / f"{_SAFE.sub('_', resource.identifier)}-{digest[:12]}"
        if not target.exists():  # the same content is copied once; changed content gets a new folder
            partial = target.with_name(target.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            if source.is_dir():
                shutil.copytree(source, partial)
            else:
                partial.mkdir(parents=True)
                shutil.copy2(source, partial / source.name)
            partial.rename(target)
        size, files = tree_size(target)
        return FetchedResource(
            **base, status="ready", path=str(target), bytes=size, files=files, note=f"provided locally, sha256 {digest}"
        )

    def _huggingface(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        if self.snapshot is None:
            return FetchedResource(
                **base, status="failed", note="huggingface_hub is not installed here (it is part of the [barkla] extra)"
            )
        repo_type = "model" if resource.kind == "model" else "dataset"
        if repo_type == "model" and resource.identifier in self.local:
            path = self.local[resource.identifier]
            size, files = tree_size(Path(path))
            return FetchedResource(
                **base, status="ready", path=path, bytes=size, files=files, note="already on the cluster"
            )
        target = self.root / "hf" / f"{repo_type}s" / _SAFE.sub("--", resource.identifier)
        path = self.snapshot(
            repo_id=resource.identifier, repo_type=repo_type, local_dir=str(target), token=self.hf_token
        )
        size, files = tree_size(Path(path))
        return FetchedResource(**base, status="ready", path=str(path), bytes=size, files=files)

    def _torchvision(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        assert self.torchvision is not None
        # shortcut: two runs fetching the same dataset at the same moment write the same files (as two Hugging
        # Face snapshots would); lock the folder if fetches ever run in parallel. A later fetch is safe: torchvision
        # checks what is there and downloads nothing twice.
        target = self.root / "torchvision" / _SAFE.sub("_", resource.identifier)
        name = self.torchvision(resource.identifier, target)
        size, files = tree_size(target)
        return FetchedResource(
            **base, status="ready", path=str(target), bytes=size, files=files, note=f"torchvision.datasets.{name}"
        )

    def _url(self, resource: ResourceSpec, base: dict[str, Any]) -> FetchedResource:
        if self.http is None:
            return FetchedResource(**base, status="failed", note="no HTTP client configured")
        name = _SAFE.sub("_", resource.identifier.rstrip("/").rsplit("/", 1)[-1]) or "download"
        target = self.root / "url" / _SAFE.sub("_", resource.name) / name
        result = self.http.download(resource.identifier, target, max_bytes=self.max_url_bytes, accept=())
        return FetchedResource(
            **base, status="ready", path=str(result.path), bytes=result.bytes, files=1, note=result.sha256
        )
