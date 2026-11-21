"""GEO series and CELLxGENE Discover datasets: whether one exists, and which files to download.

GEO (its README, ftp.ncbi.nlm.nih.gov/geo/README.txt): series live under geo/series/GSE<digits but the last
three>nnn/GSE<digits>/, with supplementary files in suppl/ and series matrices in matrix/. The FTP site is
also served over HTTPS, so its folders are HTML listings.

CELLxGENE (its OpenAPI spec, api.cellxgene.cziscience.com/curation/openapi.json): GET
/curation/v1/datasets/{dataset_id}/versions lists a public dataset's versions, each with "assets"
(filetype H5AD or RDS, url, filesize), "title", "cell_count" and "published_at". No login for public data.
"""

from __future__ import annotations

import re
from typing import Any

from ..http import HttpError, PoliteHttp

GEO_SERIES = "https://ftp.ncbi.nlm.nih.gov/geo/series"
CELLXGENE_API = "https://api.cellxgene.cziscience.com/curation/v1"
_GSE = re.compile(r"^GSE(\d+)$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_HREF = re.compile(r'href="([^"]+)"')


def geo_series_url(accession: str) -> str | None:
    """The series folder of a GEO accession such as GSE15701 (bucket GSE15nnn), or None if it is not one."""
    match = _GSE.match(accession.strip().upper())
    if match is None:
        return None
    digits = match.group(1)
    return f"{GEO_SERIES}/GSE{digits[:-3]}nnn/GSE{digits}/"


def listing(html: str) -> list[str]:
    """The file names in an HTTPS folder listing (sub-folders, parent links and sort links left out)."""
    names = []
    for href in _HREF.findall(html):
        if href.endswith("/") or href.startswith(("?", "/", "http:", "https:", "ftp:", "#")) or href in names:
            continue
        names.append(href)
    return names


def geo_files(http: PoliteHttp, accession: str) -> list[str]:
    """Download URLs of a series' supplementary files and series matrices; HttpError if the series is unknown."""
    folder = geo_series_url(accession)
    if folder is None:
        raise HttpError(f"not a GEO series accession: {accession!r} (expected GSE followed by digits)", status=404)
    http.get_text(folder, cache=False)  # 404 for an unknown series
    urls: list[str] = []
    for sub in ("suppl/", "matrix/"):
        try:
            urls += [f"{folder}{sub}{name}" for name in listing(http.get_text(folder + sub, cache=False))]
        except HttpError as exc:
            if exc.status != 404:  # a series without supplementary files has no suppl/ folder
                raise
    return urls


def cellxgene_latest(http: PoliteHttp, dataset_id: str) -> dict[str, Any]:
    """The newest published version of a CELLxGENE dataset; HttpError if there is none."""
    if not _UUID.match(dataset_id.strip().lower()):
        raise HttpError(f"not a CELLxGENE dataset id: {dataset_id!r} (expected a UUID)", status=404)
    versions = http.get_json(f"{CELLXGENE_API}/datasets/{dataset_id.strip().lower()}/versions")
    if not isinstance(versions, list) or not versions:
        raise HttpError(f"CELLxGENE has no published version of dataset {dataset_id}", status=404)
    return max(versions, key=lambda v: str(v.get("published_at") or ""))


def h5ad_asset(version: dict[str, Any]) -> dict[str, Any] | None:
    return next((a for a in version.get("assets") or [] if a.get("filetype") == "H5AD" and a.get("url")), None)
