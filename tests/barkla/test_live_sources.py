"""Live probes of the outside services rtool's clients were written for (run on Barkla: `pytest -m barkla`).

The Mac makes no live calls to research services, so these are deselected there (pyproject's addopts).
Each probe makes a few small requests and checks that the fields the client reads are really there.
"""

from __future__ import annotations

from pathlib import Path

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


def test_biomodels(http: PoliteHttp, tmp_path_factory: pytest.TempPathFactory) -> None:
    from research_tool.schemas.plan import ResourceSpec  # noqa: PLC0415
    from research_tool.tools.resources.fetch import Fetcher  # noqa: PLC0415

    model = ResourceSpec(kind="model", name="m", source="biomodels", identifier="BIOMD0000000012", purpose="")
    fetched = Fetcher(tmp_path_factory.mktemp("biomodels"), http=http).fetch(model)  # Elowitz's repressilator
    assert fetched.status == "ready", fetched.note  # an SBML file with a model in it


@pytest.mark.parametrize(
    ("source", "identifier", "name"),
    [("uniprot", "P69905", "P69905.fasta"), ("alphafold", "P69905", "AF-P69905-F1"), ("pdb", "4HHB", "4hhb.cif")],
)
def test_structures(http: PoliteHttp, tmp_path: Path, source: str, identifier: str, name: str) -> None:
    from research_tool.schemas.plan import ResourceChoice, ResourceSpec  # noqa: PLC0415
    from research_tool.tools.resources.checks import HubChecker  # noqa: PLC0415
    from research_tool.tools.resources.fetch import Fetcher  # noqa: PLC0415

    choice = ResourceChoice(kind="other", name="s", source=source, identifier=identifier, purpose="")  # type: ignore[arg-type]
    checked = HubChecker(http).check(choice)
    assert checked.exists, checked.note
    fetched = Fetcher(tmp_path, http=http).fetch(ResourceSpec(**choice.model_dump()))
    assert fetched.status == "ready" and name in (fetched.path or ""), fetched.note
    assert Path(fetched.path or "").stat().st_size > 100  # a real record (hemoglobin alpha: 233 bytes)
