"""The literature quality checks, measured on a finished run: also one made before the checks existed.

Claims are re-checked from the knowledge base with today's rules, so an old run and a new one are
compared on the same terms. Read-only; nothing calls the model::

    python -m research_tool.agents.literature.measure runs/<project>/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...core.artifacts import ArtifactStore
from ...knowledge.store import KnowledgeStore
from ...schemas.literature import Corpus, PaperExtraction, QualityCheck
from .mapping import build_claims, subtopic_papers
from .quality import checks


def measure_run(run_root: Path, *, threshold: float = 6, snowball_rounds: int = 2) -> list[QualityCheck]:
    """The checks for a run directory, from its corpus artifact and knowledge base (read-only)."""
    corpus = ArtifactStore(run_root / "artifacts").get("corpus", model=Corpus)
    kb_path = run_root / corpus.kb_path
    if not kb_path.is_file():  # e.g. a copy of the run without its snapshots
        kb_path = run_root / "work" / "literature" / "kb.sqlite"
    kb = KnowledgeStore(kb_path, readonly=True)
    try:
        papers = {p.id: p for p in corpus.included()}
        extractions = {
            pid: PaperExtraction.model_validate(data) for pid, data in kb.extractions().items() if pid in papers
        }
        texts = {pid: kb.full_text(pid) or papers[pid].abstract for pid in extractions}
    finally:
        kb.close()
    return checks(
        candidates=corpus.papers,
        papers=papers,
        extractions=extractions,
        claims=build_claims(extractions, texts),
        subtopics=subtopic_papers(corpus.must_cover, papers.values()),
        threshold=threshold,
        snowball_rounds=snowball_rounds,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quality checks of a run's literature stage.")
    parser.add_argument("run", type=Path, help="The run directory (it has artifacts/ and work/).")
    parser.add_argument("--threshold", type=float, default=6, help="The relevance threshold the run used.")
    parser.add_argument("--json", action="store_true", help="Print the checks as JSON.")
    args = parser.parse_args(argv)
    found = measure_run(args.run, threshold=args.threshold)
    if args.json:
        print(json.dumps([c.model_dump() for c in found], indent=2))
        return
    for check in found:
        print(f"{'ok   ' if check.ok else 'CHECK'}  {check.check}: {check.value}")
        if not check.ok:
            print(f"       {check.note}.")


if __name__ == "__main__":
    main()
