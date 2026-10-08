"""Deterministic parts of the knowledge map: entities, claims, coverage, gaps, Markdown, BibTeX.

The LLM only labels topics, synthesises open problems and writes the
overview; everything here is computed, so it is reproducible and cheap.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime

from ...knowledge.verify import DEFAULT_THRESHOLD, quote_support
from ...schemas.literature import (
    Claim,
    CoverageCell,
    Entity,
    Gap,
    KnowledgeMap,
    PaperExtraction,
    PaperRecord,
    Topic,
)

_KEY = re.compile(r"[^a-z0-9]+")
_CITATION = re.compile(r"\[([A-Za-z0-9_:\-]+)\]")


def entity_key(name: str) -> str:
    return _KEY.sub("", name.lower())


def build_claims(
    extractions: dict[str, PaperExtraction], texts: dict[str, str], threshold: float = DEFAULT_THRESHOLD
) -> list[Claim]:
    from ...knowledge.verify import normalize

    claims: list[Claim] = []
    for paper_id, extraction in extractions.items():
        normalized = normalize(texts.get(paper_id, ""))

        def support(quote: str) -> float:
            return quote_support(quote, "", normalized_source=normalized)  # noqa: B023

        for result in extraction.key_results:
            s = support(result.quote)
            claims.append(
                Claim(
                    paper_id=paper_id,
                    kind="result",
                    text=result.claim,
                    quote=result.quote,
                    verified=s >= threshold,
                    support=s,
                    method=result.method,
                    dataset=result.dataset,
                    metric=result.metric,
                    value=result.value,
                )
            )
        for kind, items in (("limitation", extraction.limitations), ("future_work", extraction.future_work)):
            for item in items:
                s = support(item.quote)
                claims.append(
                    Claim(
                        paper_id=paper_id,
                        kind=kind,
                        text=item.statement,
                        quote=item.quote,  # type: ignore[arg-type]
                        verified=s >= threshold,
                        support=s,
                    )
                )
    return claims


def build_entities(extractions: dict[str, PaperExtraction]) -> list[Entity]:
    names: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    papers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for paper_id, extraction in extractions.items():
        for kind, values in (
            ("method", extraction.methods),
            ("dataset", extraction.datasets),
            ("metric", extraction.metrics),
        ):
            for value in values:
                key = entity_key(value)
                if len(key) < 2:
                    continue
                names[(kind, key)][value.strip()] += 1
                papers[(kind, key)].add(paper_id)
    entities = []
    for (kind, key), counter in names.items():
        display = counter.most_common(1)[0][0]
        entities.append(
            Entity(
                kind=kind,
                name=display,
                aliases=sorted(n for n in counter if n != display),  # type: ignore[arg-type]
                paper_ids=sorted(papers[(kind, key)]),
            )
        )
    return sorted(entities, key=lambda e: (-len(e.paper_ids), e.kind, e.name.lower()))


def build_coverage(extractions: dict[str, PaperExtraction], entities: list[Entity]) -> list[CoverageCell]:
    display = {(e.kind, entity_key(n)): e.name for e in entities for n in (e.name, *e.aliases)}
    cells: dict[tuple[str, str], CoverageCell] = {}
    for paper_id, extraction in extractions.items():
        pairs = {(m, d) for m in extraction.methods for d in extraction.datasets}
        results = [(r.method, r.dataset, r) for r in extraction.key_results if r.method and r.dataset]
        pairs |= {(m, d) for m, d, _ in results}
        for method, dataset in pairs:
            m = display.get(("method", entity_key(method)), method)
            d = display.get(("dataset", entity_key(dataset)), dataset)
            cell = cells.setdefault((m, d), CoverageCell(method=m, dataset=d, paper_ids=[]))
            if paper_id not in cell.paper_ids:
                cell.paper_ids.append(paper_id)
        for method, dataset, result in results:
            m = display.get(("method", entity_key(method)), method)
            d = display.get(("dataset", entity_key(dataset)), dataset)
            if result.metric or result.value:
                cells[(m, d)].results.append(f"{result.metric}={result.value} [{paper_id}]".strip())
    return sorted(cells.values(), key=lambda c: (-len(c.paper_ids), c.method, c.dataset))


def find_gaps(
    entities: list[Entity],
    coverage: list[CoverageCell],
    topics: list[Topic],
    papers: dict[str, PaperRecord],
    *,
    now_year: int | None = None,
) -> list[Gap]:
    now_year = now_year or datetime.now(UTC).year
    gaps: list[Gap] = []
    covered = {(c.method, c.dataset) for c in coverage}
    methods = [e for e in entities if e.kind == "method" and len(e.paper_ids) >= 2][:8]
    datasets = [e for e in entities if e.kind == "dataset" and len(e.paper_ids) >= 2][:8]
    for method in methods:
        for dataset in datasets:
            if (method.name, dataset.name) not in covered:
                gaps.append(
                    Gap(
                        kind="untested_combination",
                        description=f"{method.name} has not been evaluated on {dataset.name} in this corpus",
                        evidence=[*method.paper_ids[:3], *dataset.paper_ids[:3]],
                    )
                )
    for topic in topics:
        years = [papers[p].year for p in topic.paper_ids if p in papers and papers[p].year]
        if len(topic.paper_ids) <= 2:
            gaps.append(
                Gap(
                    kind="thin_topic",
                    description=f"Only {len(topic.paper_ids)} paper(s) on '{topic.label}'",
                    evidence=topic.paper_ids,
                )
            )
        elif years and max(years) <= now_year - 5:
            gaps.append(
                Gap(
                    kind="stale_topic",
                    description=f"No paper on '{topic.label}' since {max(years)}",
                    evidence=topic.paper_ids,
                )
            )
    return gaps[:20]


def check_citations(text: str, known_keys: set[str]) -> tuple[str, int]:
    """Replace citations of unknown keys with [?]; returns (text, number replaced)."""
    replaced = 0

    def fix(match: re.Match[str]) -> str:
        nonlocal replaced
        if match.group(1) in known_keys:
            return match.group(0)
        replaced += 1
        return "[?]"

    return _CITATION.sub(fix, text), replaced


def timeline(papers: Iterable[PaperRecord]) -> dict[str, int]:
    counts = Counter(str(p.year) for p in papers if p.year)
    return dict(sorted(counts.items()))


def _escape_bib(value: str) -> str:
    return value.replace("\\", "").replace("{", "(").replace("}", ")")


def bibtex(papers: Iterable[PaperRecord]) -> str:
    entries = []
    for p in papers:
        fields = {
            "title": p.title,
            "author": " and ".join(p.authors[:20]),
            "year": str(p.year or ""),
            "journal": p.venue,
            "doi": p.doi or "",
            "eprint": p.arxiv_id or "",
            "url": p.url or "",
        }
        body = ",\n".join(f"  {k} = {{{_escape_bib(v)}}}" for k, v in fields.items() if v)
        entries.append(f"@article{{{p.bib_key},\n{body}\n}}")
    return "\n\n".join(entries) + "\n"


def render_markdown(km: KnowledgeMap, papers: dict[str, PaperRecord]) -> str:
    def cite(paper_id: str) -> str:
        paper = papers.get(paper_id)
        return f"[{paper.bib_key}]" if paper and paper.bib_key else f"[{paper_id}]"

    lines = [f"# Knowledge map: {km.topic}", ""]
    stats = ", ".join(f"{k}: {v}" for k, v in km.stats.items() if not isinstance(v, dict))
    lines += [f"_{stats}_", "", "## Overview", "", km.overview or "(no overview)", ""]
    lines += ["## Topics", ""]
    for topic in km.topics:
        lines.append(f"- **{topic.label}** — {topic.summary} " + " ".join(cite(p) for p in topic.paper_ids))
    lines += ["", "## Open problems", ""]
    for problem in km.open_problems:
        lines.append(f"- {problem.statement} ({problem.why_open}) " + " ".join(cite(p) for p in problem.paper_ids))
    lines += ["", "## Gaps found automatically", ""]
    lines += [f"- {gap.description}" for gap in km.gaps] or ["- none"]
    lines += ["", "## Most common methods, datasets and metrics", ""]
    for kind in ("method", "dataset", "metric"):
        top = [e for e in km.entities if e.kind == kind][:10]
        lines.append(f"- **{kind}s:** " + (", ".join(f"{e.name} ({len(e.paper_ids)})" for e in top) or "none"))
    verified = [c for c in km.claims if c.verified and c.kind == "result"]
    lines += ["", f"## Verified results ({len(verified)} of {sum(c.kind == 'result' for c in km.claims)})", ""]
    for claim in verified[:40]:
        lines.append(f"- {claim.text} {cite(claim.paper_id)} — “{claim.quote}”")
    lines += ["", "## Papers", "", "| key | year | title | status |", "|---|---|---|---|"]
    for paper in sorted(papers.values(), key=lambda p: (p.year or 0, p.title)):
        title = paper.title.replace("|", "/")
        lines.append(f"| {paper.bib_key} | {paper.year or ''} | {title} | {paper.status} |")
    return "\n".join(lines) + "\n"
