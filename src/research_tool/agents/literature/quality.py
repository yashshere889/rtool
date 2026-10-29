"""Measured properties of the Literature stage's output, shown at the literature review gate.

These are the numbers that exposed the first real run (2026-10-12): every
candidate scored 6-10 at screening, 88 of 98 papers had no named method, no
paper came from citation following. Each check says what a failing value
means, so the researcher knows what to look at before approving. Nothing here
calls the model.
To measure a finished run (also one made before these checks existed), see ``measure.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

from ...schemas.literature import Claim, PaperExtraction, PaperRecord, QualityCheck
from .mapping import is_name

LENIENT_SHARE = 0.8  # screening that keeps more than this share of candidates is not telling them apart
MIN_FULL_TEXT = 0.6
MIN_QUOTES_FOUND = 0.8
MAX_INVENTED_NUMBERS = 0.05
MIN_WITH_METHODS = 0.7


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole}" + (f" ({part / whole:.0%})" if whole else "")


def _named(values: Iterable[str]) -> bool:
    return any(is_name(v) for v in values)


def checks(
    *,
    candidates: list[PaperRecord],
    papers: dict[str, PaperRecord],
    extractions: dict[str, PaperExtraction],
    claims: list[Claim],
    subtopics: dict[str, list[str]],
    threshold: float,
    snowball_rounds: int,
) -> list[QualityCheck]:
    out: list[QualityCheck] = []
    scored = [p for p in candidates if p.relevance is not None and "manual" not in p.found_by]
    if scored:
        kept = sum((p.relevance or 0) >= threshold for p in scored)
        lowest = min(p.relevance or 0 for p in scored)
        out.append(
            QualityCheck(
                check="screening",
                value=f"{_share(kept, len(scored))} candidates passed; lowest score {lowest:g}",
                ok=kept <= LENIENT_SHARE * len(scored) or len(scored) < 20,
                note=(
                    "Screening kept almost everything, so it is probably not telling relevant papers from ones that "
                    "only share words with the topic. Check how the topic was read, and the excluded papers"
                ),
            )
        )
    read = [pid for pid in papers if pid in extractions]
    full = sum(papers[pid].status == "extracted" for pid in papers)
    out.append(
        QualityCheck(
            check="full text",
            value=f"{_share(full, len(papers))} papers read in full",
            ok=not papers or full >= MIN_FULL_TEXT * len(papers),
            note="Many papers were read from the abstract only: upload PDFs of the important ones",
        )
    )
    quotes = [c for c in claims if c.quote]
    found = sum(c.support >= 0.85 for c in quotes)
    out.append(
        QualityCheck(
            check="quotes found in the papers",
            value=_share(found, len(quotes)),
            ok=not quotes or found >= MIN_QUOTES_FOUND * len(quotes),
            note="Many quotes the model gave are not in the papers; claims without a found quote are not used",
        )
    )
    results = [c for c in claims if c.kind == "result" and c.support >= 0.85]
    invented = sum(not c.verified for c in results)  # quote found, but a number in the claim is not in the paper
    out.append(
        QualityCheck(
            check="results with numbers not in the paper",
            value=_share(invented, len(results)),
            ok=not results or invented <= MAX_INVENTED_NUMBERS * len(results),
            note="The model put numbers in its findings that the papers do not contain; those findings are not used",
        )
    )
    with_methods = sum(_named(extractions[pid].methods) for pid in read)
    with_datasets = sum(_named(extractions[pid].datasets) for pid in read)
    out.append(
        QualityCheck(
            check="papers with named methods",
            value=f"{_share(with_methods, len(read))}; with named data: {with_datasets}",
            ok=not read or with_methods >= MIN_WITH_METHODS * len(read),
            note="Without method and dataset names, the coverage table and the untested-combination gaps are empty",
        )
    )
    if subtopics:
        thin = [name for name, ids in subtopics.items() if len(ids) <= 1]
        out.append(
            QualityCheck(
                check="sub-topics covered",
                value=f"{len(subtopics) - len(thin)} of {len(subtopics)} have two or more papers",
                ok=not thin,
                note="Hardly any papers on: " + "; ".join(thin) + ". Request changes with a comment to search for them",
            )
        )
    followed = sum(any(f.startswith(("refs:", "cites:")) for f in p.found_by) for p in papers.values())
    out.append(
        QualityCheck(
            check="papers found through citations",
            value=f"{followed} of {len(papers)}",
            ok=snowball_rounds == 0 or followed > 0,
            note="No paper came from following references and citations, which finds what keyword search misses",
        )
    )
    return out
