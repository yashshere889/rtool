"""Common paper representation and identifier normalisation for all sources."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Protocol

from pydantic import BaseModel, Field

_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.I)
_ARXIV_PREFIX = re.compile(r"^(arxiv:|https?://arxiv\.org/(abs|pdf)/)", re.I)
_ARXIV_VERSION = re.compile(r"v\d+$")
_WORD = re.compile(r"[a-z0-9]+")
_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")
_STOPWORDS = {"a", "an", "the", "of", "on", "in", "for", "and", "to", "with", "via", "towards", "toward", "is", "are"}


class PaperHit(BaseModel):
    """One search result from one source."""

    source: str
    source_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    oa_urls: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    reference_ids: list[str] = Field(default_factory=list)


class ScholarlySource(Protocol):
    name: str

    def search(self, query: str, limit: int, *, year_from: int | None = None) -> list[PaperHit]: ...


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = _DOI_PREFIX.sub("", doi.strip()).strip().lower()
    return doi or None


def normalize_arxiv(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    value = _ARXIV_PREFIX.sub("", arxiv_id.strip()).removesuffix(".pdf")
    value = _ARXIV_VERSION.sub("", value)
    return value or None


def title_key(title: str) -> str:
    """Lowercase ASCII words; accents folded, any other character (dashes, colons...) separates words."""
    decomposed = unicodedata.normalize("NFKD", title)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_WORD.findall(_NON_ALNUM.sub(" ", folded).lower()))


def paper_id(*, doi: str | None, arxiv_id: str | None, openalex_id: str | None, s2_id: str | None, title: str) -> str:
    if doi:
        return f"doi:{doi}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if openalex_id:
        return f"openalex:{openalex_id}"
    if s2_id:
        return f"s2:{s2_id}"
    return "title:" + hashlib.sha1(title_key(title).encode()).hexdigest()[:12]


def bib_key(authors: list[str], year: int | None, title: str, taken: set[str]) -> str:
    surname = title_key(authors[0].split(",")[0].split()[-1]) if authors and authors[0].split() else "anon"
    surname = surname.replace(" ", "") or "anon"
    word = next((w for w in title_key(title).split() if w not in _STOPWORDS), "paper")
    base = f"{surname}{year or 'nd'}{word}"
    key, n = base, 0
    while key in taken:
        n += 1
        key = f"{base}{chr(ord('a') + n - 1)}" if n <= 26 else f"{base}{n}"
    taken.add(key)
    return key
