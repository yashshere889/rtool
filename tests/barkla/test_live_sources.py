"""Live probes of the outside services rtool's clients were written for (run on Barkla: `pytest -m barkla`).

The Mac makes no live calls to research services, so these are deselected there (pyproject's addopts).
Each probe makes a few small requests and checks that the fields the client reads are really there.
"""

from __future__ import annotations

import pytest

from research_tool.tools.http import PoliteHttp
from research_tool.tools.resources.omics import CELLXGENE_API, cellxgene_latest, geo_files, h5ad_asset
from research_tool.tools.scholarly.sources import EuropePMCSource, PubMedSource

pytestmark = pytest.mark.barkla


@pytest.fixture(scope="module")
def http() -> PoliteHttp:
    return PoliteHttp(limits={"eutils.ncbi.nlm.nih.gov": 0.34, "ftp.ncbi.nlm.nih.gov": 0.34})


def test_pubmed(http: PoliteHttp) -> None:
    hits = PubMedSource(http).search("interferon signature systemic lupus erythematosus", 3)
    assert hits and all(h.title and h.source_id.isdigit() for h in hits)
    assert any(h.abstract for h in hits) and any(h.doi for h in hits) and any(h.year for h in hits)


def test_europe_pmc(http: PoliteHttp) -> None:
    hits = EuropePMCSource(http).search("malaria vaccine", 5, year_from=2020)
    assert hits and all(h.title and h.year and h.year >= 2020 for h in hits)
    assert any(h.oa_urls for h in hits) and any(h.abstract for h in hits)


def test_geo(http: PoliteHttp) -> None:
    files = geo_files(http, "GSE2034")  # Wang et al. 2005, breast cancer: a long-standing public series
    assert any("/matrix/" in f for f in files), files


def test_cellxgene(http: PoliteHttp) -> None:
    collections = http.get_json(f"{CELLXGENE_API}/collections")
    collection = http.get_json(f"{CELLXGENE_API}/collections/{collections[0]['collection_id']}")
    version = cellxgene_latest(http, collection["datasets"][0]["dataset_id"])
    assert h5ad_asset(version) is not None and version.get("title")
