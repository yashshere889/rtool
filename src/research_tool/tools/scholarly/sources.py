"""OpenAlex, Semantic Scholar, arXiv and Unpaywall clients.

All of them go through ``PoliteHttp`` (rate limits, retries, response cache)
and return ``PaperHit`` objects. Keys come from the environment (see
``core/secrets.py``); without a key a source still works but is slower or
more tightly limited.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from ..http import HttpError, PoliteHttp
from .base import PaperHit, normalize_arxiv, normalize_doi

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_QUERY_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_ARXIV_STOP = {"a", "an", "the", "of", "on", "in", "for", "and", "or", "to", "with", "by", "via", "using"}


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def openalex_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as an inverted index {word: [positions]}."""
    if not inverted:
        return ""
    positions = [(pos, word) for word, places in inverted.items() for pos in places]
    return " ".join(word for _, word in sorted(positions))


class OpenAlexSource:
    name = "openalex"
    base = "https://api.openalex.org"

    def __init__(self, http: PoliteHttp, api_key: str | None = None) -> None:
        self.http = http
        self.api_key = api_key

    def _get(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        if self.api_key:
            params["api_key"] = self.api_key
        return self.http.get_json(f"{self.base}/works", params=params).get("results", [])

    @staticmethod
    def _short(openalex_url: str | None) -> str | None:
        return openalex_url.rsplit("/", 1)[-1] if openalex_url else None

    def to_hit(self, work: dict[str, Any]) -> PaperHit:
        work_id = self._short(work.get("id")) or ""
        locations = [work.get("best_oa_location") or {}, work.get("primary_location") or {}]
        oa_urls = [u for loc in locations for u in (loc.get("pdf_url"),) if u]
        if (work.get("open_access") or {}).get("oa_url"):
            oa_urls.append(work["open_access"]["oa_url"])
        arxiv = None
        for loc in work.get("locations") or []:
            landing = (loc or {}).get("landing_page_url") or ""
            if "arxiv.org/abs/" in landing:
                arxiv = normalize_arxiv(landing)
                break
        source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        return PaperHit(
            source=self.name,
            source_id=work_id,
            title=_clean(work.get("display_name") or work.get("title")),
            abstract=_clean(openalex_abstract(work.get("abstract_inverted_index"))),
            year=work.get("publication_year"),
            authors=[a["author"]["display_name"] for a in work.get("authorships") or [] if a.get("author")],
            venue=source,
            doi=normalize_doi(work.get("doi")),
            arxiv_id=arxiv,
            openalex_id=work_id or None,
            url=work.get("id"),
            oa_urls=list(dict.fromkeys(oa_urls)),
            citation_count=work.get("cited_by_count"),
            reference_ids=[r for r in (self._short(x) for x in work.get("referenced_works") or []) if r],
        )

    def search(self, query: str, limit: int, *, year_from: int | None = None) -> list[PaperHit]:
        params: dict[str, Any] = {"search": query, "per-page": min(limit, 100)}
        if year_from:
            params["filter"] = f"from_publication_date:{year_from}-01-01"
        return [self.to_hit(w) for w in self._get(params) if w.get("display_name") or w.get("title")]

    def by_ids(self, work_ids: list[str]) -> list[PaperHit]:
        hits: list[PaperHit] = []
        for start in range(0, len(work_ids), 50):
            batch = work_ids[start : start + 50]
            hits += [self.to_hit(w) for w in self._get({"filter": "openalex_id:" + "|".join(batch), "per-page": 50})]
        return hits

    def citing(self, work_id: str, limit: int) -> list[PaperHit]:
        return [self.to_hit(w) for w in self._get({"filter": f"cites:{work_id}", "per-page": min(limit, 100)})]


class SemanticScholarSource:
    name = "semantic_scholar"
    base = "https://api.semanticscholar.org/graph/v1"
    fields = "paperId,title,abstract,year,authors,venue,externalIds,openAccessPdf,citationCount"

    def __init__(self, http: PoliteHttp, api_key: str | None = None) -> None:
        self.http = http
        self.headers = {"x-api-key": api_key} if api_key else {}

    def to_hit(self, paper: dict[str, Any]) -> PaperHit:
        external = paper.get("externalIds") or {}
        pdf = (paper.get("openAccessPdf") or {}).get("url")
        return PaperHit(
            source=self.name,
            source_id=paper.get("paperId") or "",
            title=_clean(paper.get("title")),
            abstract=_clean(paper.get("abstract")),
            year=paper.get("year"),
            authors=[a.get("name", "") for a in paper.get("authors") or [] if a.get("name")],
            venue=paper.get("venue") or "",
            doi=normalize_doi(external.get("DOI")),
            arxiv_id=normalize_arxiv(external.get("ArXiv")),
            s2_id=paper.get("paperId"),
            url=f"https://www.semanticscholar.org/paper/{paper['paperId']}" if paper.get("paperId") else None,
            oa_urls=[pdf] if pdf else [],
            citation_count=paper.get("citationCount"),
        )

    def search(self, query: str, limit: int, *, year_from: int | None = None) -> list[PaperHit]:
        params: dict[str, Any] = {"query": query, "limit": min(limit, 100), "fields": self.fields}
        if year_from:
            params["year"] = f"{year_from}-"
        data = self.http.get_json(f"{self.base}/paper/search", params=params, headers=self.headers)
        return [self.to_hit(p) for p in data.get("data") or [] if p.get("title")]

    def _linked(self, ref: str, kind: str, key: str, limit: int) -> list[PaperHit]:
        url = f"{self.base}/paper/{ref}/{kind}"
        data = self.http.get_json(url, params={"fields": self.fields, "limit": min(limit, 1000)}, headers=self.headers)
        return [self.to_hit(item[key]) for item in data.get("data") or [] if (item.get(key) or {}).get("title")]

    @staticmethod
    def reference_for(hit_or_record: Any) -> str | None:
        if getattr(hit_or_record, "s2_id", None):
            return hit_or_record.s2_id
        if getattr(hit_or_record, "doi", None):
            return f"DOI:{hit_or_record.doi}"
        if getattr(hit_or_record, "arxiv_id", None):
            return f"ARXIV:{hit_or_record.arxiv_id}"
        return None

    def references(self, ref: str, limit: int) -> list[PaperHit]:
        return self._linked(ref, "references", "citedPaper", limit)

    def citations(self, ref: str, limit: int) -> list[PaperHit]:
        return self._linked(ref, "citations", "citingPaper", limit)


class ArxivSource:
    """arXiv's legacy API: Atom XML; their terms ask for one request every three seconds."""

    name = "arxiv"
    base = "https://export.arxiv.org/api/query"

    def __init__(self, http: PoliteHttp) -> None:
        self.http = http

    @staticmethod
    def build_query(query: str) -> str:
        words = [w for w in _QUERY_WORD.findall(query) if w.lower() not in _ARXIV_STOP][:8]
        return " AND ".join(f"all:{w}" for w in words) or f"all:{query}"

    @staticmethod
    def parse(feed: str) -> list[PaperHit]:
        try:
            root = ET.fromstring(feed)
        except ET.ParseError as exc:
            raise HttpError(f"invalid arXiv feed: {exc}") from None
        hits = []
        for entry in root.findall("a:entry", _ATOM):
            entry_id = entry.findtext("a:id", default="", namespaces=_ATOM)
            arxiv_id = normalize_arxiv(entry_id)
            if not arxiv_id or "api/errors" in entry_id:
                continue
            published = entry.findtext("a:published", default="", namespaces=_ATOM)
            pdf = next(
                (link.get("href") for link in entry.findall("a:link", _ATOM) if link.get("title") == "pdf"), None
            )
            hits.append(
                PaperHit(
                    source="arxiv",
                    source_id=arxiv_id,
                    title=_clean(entry.findtext("a:title", default="", namespaces=_ATOM)),
                    abstract=_clean(entry.findtext("a:summary", default="", namespaces=_ATOM)),
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    authors=[
                        _clean(a.findtext("a:name", default="", namespaces=_ATOM))
                        for a in entry.findall("a:author", _ATOM)
                    ],
                    doi=normalize_doi(entry.findtext("arxiv:doi", default="", namespaces=_ATOM)),
                    arxiv_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    oa_urls=[u for u in (pdf,) if u],
                )
            )
        return hits

    def search(self, query: str, limit: int, *, year_from: int | None = None) -> list[PaperHit]:
        params = {
            "search_query": self.build_query(query),
            "start": 0,
            "max_results": min(limit, 100),
            "sortBy": "relevance",
        }
        hits = self.parse(self.http.get_text(self.base, params=params))
        return [h for h in hits if year_from is None or h.year is None or h.year >= year_from]


class UnpaywallResolver:
    """Finds a legal open-access copy for a DOI (Unpaywall asks for a contact email on every call)."""

    base = "https://api.unpaywall.org/v2"

    def __init__(self, http: PoliteHttp, email: str | None) -> None:
        self.http = http
        self.email = email

    def oa_urls(self, doi: str) -> list[str]:
        if not self.email:
            return []
        try:
            data = self.http.get_json(f"{self.base}/{doi}", params={"email": self.email})
        except HttpError as exc:
            if exc.status == 404:
                return []
            raise
        urls = []
        for location in [data.get("best_oa_location") or {}, *(data.get("oa_locations") or [])]:
            for key in ("url_for_pdf", "url"):
                if location.get(key):
                    urls.append(location[key])
        return list(dict.fromkeys(urls))
