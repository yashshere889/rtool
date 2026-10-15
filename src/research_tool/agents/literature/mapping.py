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
# a number standing on its own: not part of a name like "GPT-4", "A5" or "v2", nor a path or a decimal's tail
_NUMBER = re.compile(r"(?<![\w.@/])(?<![A-Za-z]-)(\d+(?:,\d{3})*(?:\.\d+)?)")
MAX_NAME_WORDS = 8  # a method, dataset or metric longer than this is a description, not a name


def entity_key(name: str) -> str:
    return _KEY.sub("", name.lower())


def is_name(value: str) -> bool:
    """A method, dataset or metric name, rather than a description of a step (or nothing)."""
    return len(entity_key(value)) >= 2 and len(value.split()) <= MAX_NAME_WORDS


def numbers_in(text: str) -> list[str]:
    """Stand-alone numbers in ``text`` (thousands separators removed), years 1900-2099 left out."""
    out = []
    for match in _NUMBER.finditer(text):
        number = match.group(1).replace(",", "")
        if not (len(number) == 4 and number.isdigit() and 1900 <= int(number) <= 2099):
            out.append(number)
    return out


def _value_set(text: str) -> set[float]:
    return {float(n) for n in numbers_in(text)}


def _missing(numbers: list[str], present: set[float]) -> list[str]:
    return list(dict.fromkeys(n for n in numbers if not any(abs(float(n) - v) <= 1e-9 for v in present)))


def build_claims(
    extractions: dict[str, PaperExtraction], texts: dict[str, str], threshold: float = DEFAULT_THRESHOLD
) -> list[Claim]:
    """Each extracted finding as a claim, verified when its quote is in the paper and every number it states is too.

    A real quote does not make the claim around it true: on 2026-10-12, 14 of 585 results stated numbers that
    appear nowhere in their paper (e.g. "reduces drawdown by 50%"). Numbers in the claim but only elsewhere in the
    paper are kept as ``unquoted_numbers``, without unverifying the claim.
    """
    from ...knowledge.verify import normalize

    claims: list[Claim] = []
    for paper_id, extraction in extractions.items():
        raw = texts.get(paper_id, "")
        normalized = normalize(raw)
        paper_numbers = _value_set(raw)

        def support(quote: str) -> float:
            return quote_support(quote, "", normalized_source=normalized)  # noqa: B023

        for result in extraction.key_results:
            s = support(result.quote)
            stated = numbers_in(f"{result.claim} {result.value}")
            unquoted = _missing(stated, _value_set(result.quote))
            invented = _missing(unquoted, paper_numbers)
            claims.append(
                Claim(
                    paper_id=paper_id,
                    kind="result",
                    text=result.claim,
                    quote=result.quote,
                    verified=s >= threshold and not invented,
                    support=s,
                    method=result.method,
                    dataset=result.dataset,
                    metric=result.metric,
                    value=result.value,
                    unquoted_numbers=unquoted,
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
                if not is_name(value):
                    continue  # "Applying temporal clustering with HDBSCAN to ..." describes a step, it names nothing
                key = entity_key(value)
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
        methods = [m for m in extraction.methods if is_name(m)]
        datasets = [d for d in extraction.datasets if is_name(d)]
        pairs = {(m, d) for m in methods for d in datasets}
        results = [(r.method, r.dataset, r) for r in extraction.key_results if is_name(r.method) and is_name(r.dataset)]
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


def subtopic_papers(must_cover: list[str], papers: Iterable[PaperRecord]) -> dict[str, list[str]]:
    """Sub-topic of the search plan -> included papers that screening found to cover it."""
    found: dict[str, list[str]] = {name: [] for name in must_cover}
    for paper in papers:
        for name in paper.subtopics:
            if name in found:
                found[name].append(paper.id)
    return found


MAX_COMBINATION_GAPS = 5


def untested_combinations(
    entities: list[Entity], coverage: list[CoverageCell], limit: int = MAX_COMBINATION_GAPS
) -> list[Gap]:
    """A method never tested on a dataset that papers in the corpus keep testing alongside the method's datasets.

    Every pair nobody tried is not a gap: on 2026-10-14 the plain rule gave twenty like "GPT-4o has not been
    evaluated on UK Biobank". A dataset that at least two papers use together with one the method was tested on
    belongs to the same benchmark family; leaving the method out of it is a gap someone might care about.
    """
    method_datasets: dict[str, set[str]] = defaultdict(set)
    dataset_papers: dict[str, set[str]] = defaultdict(set)
    for cell in coverage:
        method_datasets[cell.method].add(cell.dataset)
        dataset_papers[cell.dataset].update(cell.paper_ids)
    methods = [e.name for e in entities if e.kind == "method" and len(e.paper_ids) >= 2]
    datasets = [e.name for e in entities if e.kind == "dataset" and len(e.paper_ids) >= 2]
    found: list[tuple[int, Gap]] = []
    for method in methods:
        tested = method_datasets.get(method, set())
        for dataset in datasets:
            if dataset in tested:
                continue
            together = {d: dataset_papers[d] & dataset_papers[dataset] for d in tested}
            companion, papers = max(together.items(), key=lambda kv: len(kv[1]), default=("", set()))
            if len(papers) < 2:
                continue
            found.append(
                (
                    len(papers),
                    Gap(
                        kind="untested_combination",
                        description=(
                            f"{method} was evaluated on {companion} but not on {dataset}, which {len(papers)} papers "
                            f"in this corpus test alongside {companion}"
                        ),
                        evidence=sorted(papers)[:6],
                    ),
                )
            )
    found.sort(key=lambda item: -item[0])
    return [gap for _, gap in found[:limit]]


def find_gaps(
    entities: list[Entity],
    coverage: list[CoverageCell],
    topics: list[Topic],
    papers: dict[str, PaperRecord],
    *,
    subtopics: dict[str, list[str]] | None = None,
    now_year: int | None = None,
) -> list[Gap]:
    """Gaps found by counting, each with the papers it rests on.

    * a sub-topic the search plan said matters, with at most one paper covering it;
    * a method left out of a dataset that the corpus usually tests alongside the method's own datasets
      (see ``untested_combinations``);
    * a topic of three or more papers with nothing recent.

    Small clusters are not reported as thin topics: they are outlier papers, not under-studied areas.
    """
    now_year = now_year or datetime.now(UTC).year
    gaps: list[Gap] = []
    for name, ids in (subtopics or {}).items():
        if len(ids) <= 1:
            gaps.append(
                Gap(
                    kind="uncovered_subtopic",
                    description=(
                        f"{'Only one paper' if ids else 'No paper'} in the collection covers '{name}', a sub-topic "
                        "the search plan said matters: either the search missed work on it, or little exists"
                    ),
                    evidence=list(ids),
                )
            )
    gaps += untested_combinations(entities, coverage)
    for topic in topics:
        years = [papers[p].year for p in topic.paper_ids if p in papers and papers[p].year]
        if len(topic.paper_ids) >= 3 and years and max(years) <= now_year - 5:
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
    lines += [f"_{stats}_", ""]
    if km.scope is not None:
        lines += ["## How the topic was read", "", km.scope.interpretation, ""]
        lines.append("- **Key concepts** (each paper was checked against these): " + "; ".join(km.scope.concepts))
        if km.scope.out_of_scope:
            lines.append("- **Treated as out of scope:** " + "; ".join(km.scope.out_of_scope))
        if km.scope.other_readings:
            lines.append(
                "- **Other possible readings** (request changes with a comment to switch): "
                + "; ".join(km.scope.other_readings)
            )
        lines.append("")
    if km.quality:
        lines += ["## Quality checks", ""]
        for check in km.quality:
            mark = "ok" if check.ok else "**check**"
            lines.append(f"- {mark} — {check.check}: {check.value}" + ("" if check.ok else f". {check.note}"))
        lines.append("")
    lines += ["## Overview", "", km.overview or "(no overview)", ""]
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
