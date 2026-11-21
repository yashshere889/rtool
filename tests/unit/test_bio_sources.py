"""PubMed, Europe PMC, GEO and CELLxGENE clients, on responses in each service's documented format.

The formats come from the services' documentation (docs/build_log.md lists what was checked);
tests/barkla/test_live_sources.py calls the real services, on Barkla.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from research_tool.agents.literature.agent import LiteratureConfig
from research_tool.packs import load_pack
from research_tool.schemas.plan import ResourceChoice, ResourceSpec
from research_tool.tools.http import PoliteHttp
from research_tool.tools.resources.checks import HubChecker
from research_tool.tools.resources.fetch import Fetcher
from research_tool.tools.resources.omics import cellxgene_latest, geo_series_url, listing
from research_tool.tools.scholarly.sources import EuropePMCSource, PubMedSource


def http(handler: Callable[[httpx.Request], httpx.Response]) -> PoliteHttp:
    return PoliteHttp(
        client=httpx.Client(transport=httpx.MockTransport(handler)), default_interval_s=0, sleep=lambda _s: None
    )


PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">31712345</PMID>
      <Article>
        <Journal>
          <Title>Nature Immunology</Title>
          <JournalIssue><PubDate><MedlineDate>2019 Nov-Dec</MedlineDate></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Interferon <i>signatures</i> in lupus.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Type I interferon matters.</AbstractText>
          <AbstractText Label="RESULTS">IFI27 rose in patients.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>Ada</ForeName></Author>
          <Author><CollectiveName>Lupus Consortium</CollectiveName></Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList>
      <ArticleId IdType="pubmed">31712345</ArticleId>
      <ArticleId IdType="doi">10.1038/S41590-019-0001-X</ArticleId>
      <ArticleId IdType="pmc">PMC6900000</ArticleId>
    </ArticleIdList></PubmedData>
  </PubmedArticle>
  <PubmedArticle><MedlineCitation><PMID>1</PMID><Article><ArticleTitle></ArticleTitle></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>
"""


def test_pubmed_searches_ids_then_fetches_the_records() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": {"count": "2", "idlist": ["31712345", "1"]}})
        return httpx.Response(200, text=PUBMED_XML)

    (hit,) = PubMedSource(http(handler)).search("interferon lupus", 25, year_from=2015)
    term = seen[0].url.params["term"]
    assert term == '(interferon lupus) AND ("2015"[PDAT] : "3000"[PDAT])' and seen[0].url.params["retmode"] == "json"
    assert seen[0].url.params["tool"] == "rtool" and "email" not in seen[0].url.params  # no address unless set
    assert seen[1].url.params["id"] == "31712345,1" and seen[1].url.params["retmode"] == "xml"
    assert (hit.source, hit.source_id, hit.title) == ("pubmed", "31712345", "Interferon signatures in lupus.")
    assert hit.abstract == "BACKGROUND: Type I interferon matters. RESULTS: IFI27 rose in patients."
    assert hit.authors == ["Ada Smith", "Lupus Consortium"] and hit.year == 2019 and hit.venue == "Nature Immunology"
    assert hit.doi == "10.1038/s41590-019-0001-x" and hit.url == "https://pubmed.ncbi.nlm.nih.gov/31712345/"
    assert (
        PubMedSource(http(lambda r: httpx.Response(200, json={"esearchresult": {"idlist": []}}))).search("x", 5) == []
    )


EUROPE_PMC = {
    "hitCount": 2,
    "resultList": {
        "result": [
            {
                "id": "31712345",
                "source": "MED",
                "pmid": "31712345",
                "pmcid": "PMC6900000",
                "doi": "10.1038/s41590-019-0001-x",
                "title": "Interferon <i>signatures</i> in lupus.",
                "authorString": "Smith A, Jones B.",
                "authorList": {"author": [{"fullName": "Smith A"}, {"fullName": "Jones B"}]},
                "journalInfo": {"journal": {"title": "Nature immunology"}},
                "pubYear": "2019",
                "abstractText": "IFI27 <sup>high</sup> cells rose.",
                "isOpenAccess": "Y",
                "citedByCount": 12,
                "fullTextUrlList": {
                    "fullTextUrl": [
                        {"availabilityCode": "S", "documentStyle": "pdf", "url": "https://paywall.example/a.pdf"},
                        {
                            "availabilityCode": "OA",
                            "documentStyle": "html",
                            "url": "https://europepmc.org/articles/PMC6900000",
                        },
                        {"availabilityCode": "OA", "documentStyle": "pdf", "url": "https://europepmc.org/a.pdf"},
                        {"availabilityCode": "F", "documentStyle": "doi", "url": "https://doi.org/x"},
                    ]
                },
            },
            {"id": "PPR1", "source": "PPR", "title": "A preprint", "authorString": "Lee C.", "pubYear": "2024"},
        ]
    },
}


def test_europe_pmc_keeps_open_copies_and_preprints() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=EUROPE_PMC)

    paper, preprint = EuropePMCSource(http(handler)).search("interferon", 50, year_from=2018)
    params = seen[0].url.params
    assert params["query"] == "(interferon) AND PUB_YEAR:[2018 TO 3000]" and params["resultType"] == "core"
    assert params["format"] == "json" and params["pageSize"] == "50"
    assert paper.title == "Interferon signatures in lupus." and paper.abstract == "IFI27 high cells rose."
    assert paper.authors == ["Smith A", "Jones B"] and paper.year == 2019 and paper.citation_count == 12
    # only free copies (OA, F) a downloader can read, PDFs first
    assert paper.oa_urls == ["https://europepmc.org/a.pdf", "https://europepmc.org/articles/PMC6900000"]
    assert (preprint.source_id, preprint.authors, preprint.url) == (
        "PPR:PPR1",
        ["Lee C"],
        "https://europepmc.org/article/PPR/PPR1",
    )


def test_a_pack_can_choose_its_literature_sources() -> None:
    sources = load_pack("omics").literature_sources
    assert LiteratureConfig.model_validate({"sources": sources}).sources[:2] == ["pubmed", "europe_pmc"]


GEO_FOLDER = (
    '<html><a href="../">Parent</a><a href="?C=N;O=D">Name</a><a href="matrix/">matrix/</a><a href="suppl/">suppl/</a>'
)
GEO_SUPPL = '<a href="../">..</a><a href="GSE15701_RAW.tar">GSE15701_RAW.tar</a><a href="filelist.txt">filelist.txt</a>'
GEO_MATRIX = '<a href="GSE15701_series_matrix.txt.gz">GSE15701_series_matrix.txt.gz</a>'


def geo(request: httpx.Request) -> httpx.Response:
    pages = {
        "/geo/series/GSE15nnn/GSE15701/": GEO_FOLDER,
        "/geo/series/GSE15nnn/GSE15701/suppl/": GEO_SUPPL,
        "/geo/series/GSE15nnn/GSE15701/matrix/": GEO_MATRIX,
    }
    if request.url.path in pages:
        return httpx.Response(200, text=pages[request.url.path])
    if request.url.path.startswith("/geo/series/GSE15nnn/GSE15701/"):
        return httpx.Response(200, content=b"data of " + request.url.path.rsplit("/", 1)[-1].encode())
    return httpx.Response(404)


def test_geo_series_folders_and_listings() -> None:
    assert geo_series_url("GSE15701") == "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE15nnn/GSE15701/"
    assert geo_series_url("gse123") == "https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE123/"
    assert geo_series_url("GSM1") is None and geo_series_url("GSE") is None
    assert listing(GEO_FOLDER) == [] and listing(GEO_SUPPL) == ["GSE15701_RAW.tar", "filelist.txt"]


def choice(source: str, identifier: str) -> ResourceChoice:
    return ResourceChoice(kind="dataset", name="d", source=source, identifier=identifier, purpose="")  # type: ignore[arg-type]


def test_geo_series_are_checked_and_fetched(tmp_path: Path) -> None:
    checker = HubChecker(http(geo))
    found = checker.check(choice("geo", "GSE15701"))
    assert found.exists and found.note == "GEO series with 3 supplementary and matrix files"
    missing = checker.check(choice("geo", "GSE99999"))
    assert not missing.exists and "404" in missing.note
    assert "not a GEO series accession" in checker.check(choice("geo", "GSM5")).note
    fetched = Fetcher(tmp_path, http=http(geo)).fetch(ResourceSpec(**choice("geo", "GSE15701").model_dump()))
    assert fetched.status == "ready" and fetched.files == 3 and fetched.note == "3 GEO files"
    assert (tmp_path / "geo" / "GSE15701" / "suppl" / "GSE15701_RAW.tar").read_bytes() == b"data of GSE15701_RAW.tar"
    assert (tmp_path / "geo" / "GSE15701" / "matrix" / "GSE15701_series_matrix.txt.gz").exists()


DATASET = "0aab20b3-c30c-4606-bd2e-d20dae739c45"


def cellxgene(request: httpx.Request) -> httpx.Response:
    if request.url.path == f"/curation/v1/datasets/{DATASET}/versions":
        old = {"published_at": "2023-01-01T00:00:00", "dataset_version_id": "v1", "title": "Old", "assets": []}
        new = {
            "published_at": "2024-05-01T00:00:00",
            "dataset_version_id": "v2",
            "title": "Lupus PBMC",
            "cell_count": 1200,
            "assets": [
                {"filetype": "RDS", "url": "https://datasets.example/v2.rds", "filesize": 9},
                {"filetype": "H5AD", "url": "https://datasets.example/v2.h5ad", "filesize": 2_500_000_000},
            ],
        }
        return httpx.Response(200, json=[old, new])
    if request.url.host == "datasets.example":
        return httpx.Response(200, content=b"h5ad bytes")
    return httpx.Response(404)


def test_cellxgene_datasets_use_their_newest_version(tmp_path: Path) -> None:
    assert cellxgene_latest(http(cellxgene), DATASET)["dataset_version_id"] == "v2"
    found = HubChecker(http(cellxgene)).check(choice("cellxgene", DATASET))
    assert found.exists and found.size_gb == 2.5 and found.note == "Lupus PBMC: 1200 cells"
    assert "expected a UUID" in HubChecker(http(cellxgene)).check(choice("cellxgene", "lupus")).note
    fetched = Fetcher(tmp_path, http=http(cellxgene)).fetch(ResourceSpec(**choice("cellxgene", DATASET).model_dump()))
    assert fetched.status == "ready" and fetched.note == "Lupus PBMC, version v2"
    assert (tmp_path / "cellxgene" / DATASET / f"{DATASET}.h5ad").read_bytes() == b"h5ad bytes"
