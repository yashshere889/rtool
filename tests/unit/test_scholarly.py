"""Scholarly API clients against hand-written responses in each service's documented format (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from research_tool.tools.http import HttpError, PoliteHttp, RateLimiter, redact
from research_tool.tools.scholarly.base import (
    PaperHit,
    arxiv_from_doi,
    bib_key,
    normalize_arxiv,
    normalize_doi,
    paper_id,
    title_key,
)
from research_tool.tools.scholarly.merge import PaperIndex
from research_tool.tools.scholarly.sources import (
    ArxivSource,
    OpenAlexSource,
    SemanticScholarSource,
    UnpaywallResolver,
    openalex_abstract,
)

OPENALEX_WORK = {
    "id": "https://openalex.org/W111",
    "doi": "https://doi.org/10.1000/XYZ.123",
    "display_name": "Sleep Spindles Predict Memory Consolidation in Adults",
    "publication_year": 2021,
    "authorships": [{"author": {"display_name": "Ana Lopez"}}, {"author": {"display_name": "Ben Ode"}}],
    "primary_location": {"source": {"display_name": "Journal of Sleep"}, "pdf_url": None},
    "best_oa_location": {"pdf_url": "https://repo.example.org/paper.pdf"},
    "open_access": {"is_oa": True, "oa_url": "https://repo.example.org/landing"},
    "locations": [{"landing_page_url": "https://arxiv.org/abs/2101.00001v2"}],
    "cited_by_count": 42,
    "abstract_inverted_index": {"Spindles": [0], "predict": [1], "recall.": [2]},
    "referenced_works": ["https://openalex.org/W222", "https://openalex.org/W333"],
}
S2_PAPER = {
    "paperId": "abc123",
    "title": "Sleep spindles predict memory consolidation in adults",
    "abstract": "A longer abstract from Semantic Scholar about spindles and recall in sixty-four adults.",
    "year": 2020,
    "authors": [{"name": "Ana Lopez"}],
    "venue": "",
    "externalIds": {"DOI": "10.1000/xyz.123", "ArXiv": "2101.00001"},
    "openAccessPdf": {"url": "https://arxiv.org/pdf/2101.00001", "status": "GREEN"},
    "citationCount": 50,
}
ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v2</id>
    <published>2021-01-02T00:00:00Z</published>
    <title>Sleep Spindles Predict
      Memory Consolidation in Adults</title>
    <summary>  We find that spindles predict recall.  </summary>
    <author><name>Ana Lopez</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2101.00001v2" rel="related" type="application/pdf"/>
    <arxiv:doi>10.1000/xyz.123</arxiv:doi>
  </entry>
</feed>"""


def http_with(handler: Any) -> PoliteHttp:
    return PoliteHttp(
        client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _s: None, default_interval_s=0.0
    )


def test_identifier_normalisation() -> None:
    assert normalize_doi("https://doi.org/10.1000/XYZ.123") == "10.1000/xyz.123"
    assert normalize_doi("doi:10.5/A") == "10.5/a"
    assert normalize_arxiv("http://arxiv.org/abs/2101.00001v2") == "2101.00001"
    assert normalize_arxiv("arXiv:2101.00001v1") == "2101.00001"
    assert title_key("Sleep—Spindles: Predict!") == "sleep spindles predict"
    assert title_key("Café–Effects of Naïve Bayes") == "cafe effects of naive bayes"
    assert paper_id(doi=None, arxiv_id="2101.00001", openalex_id="W1", s2_id=None, title="t") == "arxiv:2101.00001"
    taken: set[str] = set()
    assert bib_key(["Ana Lopez"], 2021, "The Sleep Spindles", taken) == "lopez2021sleep"
    assert bib_key(["Ana Lopez"], 2021, "Sleep again", taken) == "lopez2021sleepa"


def test_openalex_parsing_and_api_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": [OPENALEX_WORK]})

    hit = OpenAlexSource(http_with(handler), api_key="KEY").search("sleep spindles", 5, year_from=2015)[0]
    assert seen[0].url.params["api_key"] == "KEY"
    assert seen[0].url.params["filter"] == "from_publication_date:2015-01-01"
    assert hit.openalex_id == "W111" and hit.doi == "10.1000/xyz.123" and hit.arxiv_id == "2101.00001"
    assert hit.abstract == "Spindles predict recall." and hit.authors == ["Ana Lopez", "Ben Ode"]
    assert hit.reference_ids == ["W222", "W333"] and hit.citation_count == 42
    assert "https://repo.example.org/paper.pdf" in hit.oa_urls
    assert openalex_abstract({"b": [1], "a": [0]}) == "a b"


def test_semantic_scholar_search_references_and_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "S2KEY"
        if request.url.path.endswith("/references"):
            return httpx.Response(200, json={"data": [{"citedPaper": S2_PAPER}, {"citedPaper": {"title": None}}]})
        return httpx.Response(200, json={"data": [S2_PAPER]})

    s2 = SemanticScholarSource(http_with(handler), api_key="S2KEY")
    hit = s2.search("spindles", 10)[0]
    assert (hit.s2_id, hit.doi, hit.arxiv_id) == ("abc123", "10.1000/xyz.123", "2101.00001")
    assert s2.reference_for(hit) == "abc123"
    assert [h.title for h in s2.references("abc123", 10)] == [S2_PAPER["title"]]


def test_arxiv_query_and_feed_parsing() -> None:
    assert ArxivSource.build_query("sleep spindles and the memory") == "all:sleep AND all:spindles AND all:memory"
    hits = ArxivSource(http_with(lambda r: httpx.Response(200, text=ARXIV_FEED))).search("sleep", 5)
    assert hits[0].arxiv_id == "2101.00001" and hits[0].title == "Sleep Spindles Predict Memory Consolidation in Adults"
    assert hits[0].year == 2021 and hits[0].doi == "10.1000/xyz.123"
    assert hits[0].oa_urls == ["http://arxiv.org/pdf/2101.00001v2"]


def test_unpaywall() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "missing" in request.url.path:
            return httpx.Response(404, json={"error": True})
        return httpx.Response(200, json={"best_oa_location": {"url_for_pdf": "https://x/a.pdf", "url": "https://x/a"}})

    resolver = UnpaywallResolver(http_with(handler), email="me@uni.ac.uk")
    assert resolver.oa_urls("10.1/abc") == ["https://x/a.pdf", "https://x/a"]
    assert resolver.oa_urls("10.1/missing") == []
    assert UnpaywallResolver(http_with(handler), email=None).oa_urls("10.1/abc") == []


def test_merging_across_sources() -> None:
    openalex = OpenAlexSource(http_with(lambda r: httpx.Response(200, json={"results": [OPENALEX_WORK]})))
    s2 = SemanticScholarSource(http_with(lambda r: httpx.Response(200, json={"data": [S2_PAPER]})))
    arxiv = ArxivSource(http_with(lambda r: httpx.Response(200, text=ARXIV_FEED)))
    index = PaperIndex()
    record, created = index.add(arxiv.search("q", 1)[0], found_by="q1")
    assert created and record.id == "doi:10.1000/xyz.123"  # a DOI is the preferred identifier
    for hit in (openalex.search("q", 1)[0], s2.search("q", 1)[0]):
        merged, created = index.add(hit, found_by="q2")
        assert not created and merged is record
    assert len(index) == 1
    assert record.id == "doi:10.1000/xyz.123"  # ids never change once assigned
    assert (record.doi, record.openalex_id, record.s2_id) == ("10.1000/xyz.123", "W111", "abc123")
    assert record.year == 2020 and record.citation_count == 50  # earliest year, highest count
    assert record.abstract.startswith("A longer abstract") and record.found_by == ["q1", "q2"]
    assert set(record.sources) == {"arxiv", "openalex", "semantic_scholar"}
    late, created = index.add(
        s2.to_hit({"paperId": "zzz", "title": "Sleep Spindles Predict Memory Consolidation in Adults!"}),
        found_by="title-only",
    )
    assert not created and late is record  # matched on the normalised title alone


def test_polite_http_retries_caches_and_redacts(tmp_path: Path) -> None:
    calls: list[str] = []
    replies = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={"ok": 1})])
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return next(replies)

    http = PoliteHttp(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=slept.append,
        cache_path=tmp_path / "cache.sqlite",
        default_interval_s=0.0,
    )
    assert http.get_json("https://api.example.org/x", params={"api_key": "SECRET", "q": "a"}) == {"ok": 1}
    assert http.get_json("https://api.example.org/x", params={"api_key": "SECRET", "q": "a"}) == {"ok": 1}
    assert len(calls) == 2 and 7.0 in slept  # one retry honouring Retry-After; second call served from cache
    assert "SECRET" not in (tmp_path / "cache.sqlite").read_bytes().decode("latin-1")
    assert redact("https://h/p?api_key=abc&email=a@b&q=1") == "https://h/p?api_key=%2A%2A%2A&email=%2A%2A%2A&q=1"
    with pytest.raises(HttpError, match="HTTP 404"):
        http_with(lambda r: httpx.Response(404)).get_json("https://h/missing")


def test_downloads_are_capped_and_typed(tmp_path: Path) -> None:
    pdf = b"%PDF-1.7 " + b"x" * 100

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/big.pdf":
            return httpx.Response(200, content=b"%PDF" + b"x" * 5000, headers={"content-type": "application/pdf"})
        if request.url.path == "/page":
            return httpx.Response(200, content=b"zip", headers={"content-type": "application/zip"})
        return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf; charset=binary"})

    http = http_with(handler)
    out = tmp_path / "downloads"
    result = http.download("https://h/a.pdf", out / "a.pdf")
    assert result.bytes == len(pdf) and result.sha256.startswith("sha256:") and (out / "a.pdf").read_bytes() == pdf
    with pytest.raises(HttpError, match="larger than"):
        http.download("https://h/big.pdf", out / "b.pdf", max_bytes=1000)
    with pytest.raises(HttpError, match="unexpected content type"):
        http.download("https://h/page", out / "c.pdf")
    assert sorted(p.name for p in out.iterdir()) == ["a.pdf"]  # no partial files left behind


def test_rate_limiter_spaces_requests() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(3.0, clock=lambda: now[0], sleep=sleep)
    limiter.wait()
    limiter.wait()
    now[0] += 1.0
    limiter.wait()
    assert sleeps == [3.0, 2.0]


def test_arxiv_dois_give_the_arxiv_id() -> None:
    """OpenAlex often knows a preprint only by its DataCite DOI; without the id, arxiv.org was never tried."""
    assert arxiv_from_doi("https://doi.org/10.48550/arXiv.2510.18515") == "2510.18515"
    assert arxiv_from_doi("10.1109/access.2026.3734747") is None and arxiv_from_doi(None) is None
    hit = PaperHit(source="openalex", source_id="W1", title="T", doi="10.48550/arxiv.2510.18515v2")
    assert hit.arxiv_id == "2510.18515"
    kept = PaperHit(source="arxiv", source_id="1", title="T", doi="10.48550/arxiv.1", arxiv_id="2101.00001")
    assert kept.arxiv_id == "2101.00001"
