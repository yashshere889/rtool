"""Does a planned resource exist, what licence does it have, how big is it?

The Hugging Face Hub API (``/api/datasets/<id>`` and ``/api/models/<id>``)
answers all three without downloading anything. URLs get a HEAD request, pip
packages a look-up on PyPI. A "manual" resource is never verified: nothing
public was named, so someone has to provide it (or the experiment code build it).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ...core.config import LocalDataset
from ...packs.refusals import controlled_access
from ...schemas.plan import ResourceChoice
from ..http import HttpError, PoliteHttp
from .fetch import tree_size
from .models import biomodels_files
from .omics import cellxgene_latest, geo_files, h5ad_asset
from .structures import alphafold_models, pdb_header, uniprot_fasta

HF_API = "https://huggingface.co/api"
PYPI_API = "https://pypi.org/pypi"
MANUAL_NOTE = "not public: nothing will be downloaded; someone must provide it, or the experiment code must build it"
HTML_PAGE_NOTE = "a web page (HTML), not a data file: name the direct download URL of the file itself"
_PIP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # "pettingzoo[mpe]>=1.24" -> "pettingzoo"


@dataclass(frozen=True)
class ResourceCheck:
    exists: bool
    license: str = ""
    size_gb: float | None = None
    note: str = ""


class ResourceChecker(Protocol):
    def check(self, resource: ResourceChoice) -> ResourceCheck: ...


def _hf_size_gb(info: dict[str, Any]) -> float | None:
    used = info.get("usedStorage")
    if isinstance(used, int | float) and used > 0:
        return round(used / 1e9, 3)
    sizes = [s.get("size") for s in info.get("siblings") or [] if isinstance(s.get("size"), int)]
    return round(sum(sizes) / 1e9, 3) if sizes else None


def _hf_license(info: dict[str, Any]) -> str:
    card = info.get("cardData") or {}
    licence = card.get("license") or ""
    if not licence:
        licence = next((t.split(":", 1)[1] for t in info.get("tags") or [] if t.startswith("license:")), "")
    return licence if isinstance(licence, str) else ",".join(licence)


def local_check(resource: ResourceChoice, local: Mapping[str, LocalDataset]) -> ResourceCheck:
    """A local dataset exists if the researcher provided it under this name and its files are there."""
    found = local.get(resource.identifier)
    if found is None:
        known = ", ".join(sorted(local)) or "none"
        return ResourceCheck(False, note=f"no local dataset {resource.identifier!r}; the researcher provided: {known}")
    path = Path(os.path.expandvars(os.path.expanduser(found.path)))
    if not path.exists():
        return ResourceCheck(False, note=f"the local dataset {resource.identifier!r} is missing: {path} does not exist")
    size, files = tree_size(path) if path.is_dir() else (path.stat().st_size, 1)
    return ResourceCheck(True, size_gb=round(size / 1e9, 3), note=f"provided locally ({files} files)")


class HubChecker:
    """Checks resources against the Hugging Face Hub API and plain URLs, and local datasets on disk."""

    def __init__(
        self, http: PoliteHttp, token: str | None = None, local: Mapping[str, LocalDataset] | None = None
    ) -> None:
        self.http = http
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.local = dict(local or {})

    def check(self, resource: ResourceChoice) -> ResourceCheck:
        repository = controlled_access(resource.identifier, resource=True)
        if repository:  # before any request: the planner must choose open data instead
            return ResourceCheck(False, note=f"controlled-access data ({repository}): rtool does not use it")
        if resource.source == "local":
            return local_check(resource, self.local)
        try:
            if resource.source == "geo":
                files = geo_files(self.http, resource.identifier)
                return ResourceCheck(bool(files), note=f"GEO series with {len(files)} supplementary and matrix files")
            if resource.source == "uniprot":
                header = uniprot_fasta(self.http, resource.identifier).splitlines()[0]
                return ResourceCheck(True, note=header[1:120])
            if resource.source == "alphafold":
                model = alphafold_models(self.http, resource.identifier)[0]
                plddt = model.get("globalMetricValue", "?")
                return ResourceCheck(True, note=f"AlphaFold model {model.get('entryId', '?')}, mean pLDDT {plddt}")
            if resource.source == "pdb":
                pdb_header(self.http, resource.identifier)
                return ResourceCheck(True, note="PDB entry (mmCIF)")
            if resource.source == "biomodels":
                files = biomodels_files(self.http, resource.identifier)
                size = sum(f.get("fileSize") or 0 for f in files)
                return ResourceCheck(
                    bool(files), size_gb=round(size / 1e9, 6), note=f"main files: {', '.join(f['name'] for f in files)}"
                )
            if resource.source == "cellxgene":
                version = cellxgene_latest(self.http, resource.identifier)
                asset = h5ad_asset(version)
                size = asset.get("filesize") if asset else None
                return ResourceCheck(
                    asset is not None,
                    size_gb=round(size / 1e9, 3) if isinstance(size, int | float) else None,
                    note=f"{version.get('title') or 'untitled'}: {version.get('cell_count') or '?'} cells"
                    + ("" if asset else "; no H5AD file"),
                )
            if resource.source == "huggingface":
                kind = "models" if resource.kind == "model" else "datasets"
                info = self.http.get_json(
                    f"{HF_API}/{kind}/{resource.identifier}", params={"blobs": "true"}, headers=self.headers
                )
                gated = info.get("gated")
                note = f"gated ({gated}): needs HF_TOKEN and accepted terms" if gated else ""
                return ResourceCheck(True, _hf_license(info), _hf_size_gb(info), note)
            if resource.source == "url":
                text = self.http.get_text(resource.identifier, cache=False)
                # A quote or "history" page answers fine but is not the data (2026-11-17: a Yahoo Finance page
                # was fetched as the S&P 500 dataset, and the Coder spent every repair round parsing HTML).
                if resource.kind == "dataset" and text.lstrip()[:512].lower().startswith(("<!doctype html", "<html")):
                    return ResourceCheck(False, note=HTML_PAGE_NOTE)
                return ResourceCheck(True, note="reachable")
            if resource.source == "manual":
                return ResourceCheck(False, note=MANUAL_NOTE)
            if resource.source == "pip":
                name = _PIP_NAME.match(resource.identifier.strip())
                if name is None:
                    return ResourceCheck(False, note=f"not a pip package name: {resource.identifier!r}")
                info = self.http.get_json(f"{PYPI_API}/{name.group(0)}/json").get("info") or {}
                licence = info.get("license") or ""
                return ResourceCheck(
                    True,
                    licence if isinstance(licence, str) and len(licence) <= 40 else "",
                    note="on PyPI; installed with the experiment code",
                )
            return ResourceCheck(
                True, note="a torchvision dataset: downloaded with torchvision when the plan is approved"
            )
        except HttpError as exc:
            note = str(exc)[:200]
            # without a token the Hub answers 401 for an id that does not exist (it could be private)
            if resource.source == "huggingface" and exc.status in (401, 404):
                found = self.similar(resource)
                note += f"; real ids that may fit: {', '.join(found)}" if found else ""
            return ResourceCheck(False, note=note)

    def similar(self, resource: ResourceChoice) -> list[str]:
        """Up to 3 Hub ids close to a missing one, so a repair round picks a real id instead of guessing again."""
        kind = "models" if resource.kind == "model" else "datasets"
        found: list[str] = []
        for term in dict.fromkeys([resource.name, resource.identifier.rsplit("/", 1)[-1]]):
            try:
                hits = self.http.get_json(
                    f"{HF_API}/{kind}",
                    params={"search": term, "limit": 3, "sort": "downloads", "direction": -1},
                    headers=self.headers,
                )
            except HttpError:
                continue
            found += [h["id"] for h in hits if isinstance(h, dict) and h.get("id") and h["id"] not in found]
            if found:
                break
        return found[:3]
