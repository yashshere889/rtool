"""Merge search hits from several sources into one record per paper.

The same paper arrives from OpenAlex, Semantic Scholar and arXiv with different
subsets of identifiers (DOI, arXiv id, source ids) and slightly different
metadata. ``PaperIndex`` matches on any shared identifier (or a long enough
normalised title) and merges incrementally, so snowballing rounds keep adding
to the same records. A record's id never changes once assigned.
"""

from __future__ import annotations

from collections.abc import Iterable

from ...schemas.literature import PaperRecord
from .base import PaperHit, paper_id, title_key

MIN_TITLE_KEY = 25  # shorter titles ("Introduction", "Deep learning") are too generic to merge on


def _keys(
    *, doi: str | None, arxiv_id: str | None, openalex_id: str | None, s2_id: str | None, title: str
) -> list[str]:
    keys = []
    if doi:
        keys.append(f"doi:{doi}")
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")
    if openalex_id:
        keys.append(f"openalex:{openalex_id}")
    if s2_id:
        keys.append(f"s2:{s2_id}")
    normalized = title_key(title)
    if len(normalized) >= MIN_TITLE_KEY:
        keys.append(f"title:{normalized}")
    return keys


def _record_keys(record: PaperRecord) -> list[str]:
    return _keys(
        doi=record.doi, arxiv_id=record.arxiv_id, openalex_id=record.openalex_id, s2_id=record.s2_id, title=record.title
    )


class PaperIndex:
    def __init__(self, records: Iterable[PaperRecord] = ()) -> None:
        self.records: dict[str, PaperRecord] = {}
        self._by_key: dict[str, str] = {}
        for record in records:
            self.records[record.id] = record
            self._index(record)

    def _index(self, record: PaperRecord) -> None:
        for key in _record_keys(record):
            self._by_key.setdefault(key, record.id)

    def find(self, hit: PaperHit) -> PaperRecord | None:
        for key in _keys(
            doi=hit.doi, arxiv_id=hit.arxiv_id, openalex_id=hit.openalex_id, s2_id=hit.s2_id, title=hit.title
        ):
            if key in self._by_key:
                return self.records[self._by_key[key]]
        return None

    def add(self, hit: PaperHit, *, found_by: str, round_number: int = 0) -> tuple[PaperRecord, bool]:
        """Merge ``hit`` in; returns (record, created)."""
        record = self.find(hit)
        created = record is None
        if record is None:
            record = PaperRecord(
                id=paper_id(
                    doi=hit.doi, arxiv_id=hit.arxiv_id, openalex_id=hit.openalex_id, s2_id=hit.s2_id, title=hit.title
                ),
                title=hit.title,
                round=round_number,
            )
            self.records[record.id] = record
        self._absorb(record, hit)
        if found_by and found_by not in record.found_by:
            record.found_by.append(found_by)
        self._index(record)
        return record, created

    @staticmethod
    def _absorb(record: PaperRecord, hit: PaperHit) -> None:
        record.doi = record.doi or hit.doi
        record.arxiv_id = record.arxiv_id or hit.arxiv_id
        record.openalex_id = record.openalex_id or hit.openalex_id
        record.s2_id = record.s2_id or hit.s2_id
        if len(hit.abstract) > len(record.abstract):
            record.abstract = hit.abstract
        record.authors = record.authors or hit.authors
        record.venue = record.venue or hit.venue
        record.url = record.url or hit.url
        if hit.year and (record.year is None or hit.year < record.year):
            record.year = hit.year  # the preprint year when both exist
        if hit.citation_count is not None:
            record.citation_count = max(record.citation_count or 0, hit.citation_count)
        record.oa_urls = list(dict.fromkeys([*record.oa_urls, *hit.oa_urls]))
        record.reference_ids = list(dict.fromkeys([*record.reference_ids, *hit.reference_ids]))
        if hit.source not in record.sources:
            record.sources.append(hit.source)

    def __len__(self) -> int:
        return len(self.records)

    def all(self) -> list[PaperRecord]:
        return list(self.records.values())
