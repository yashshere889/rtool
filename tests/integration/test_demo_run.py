"""The demo runs the web UI is developed against: real pipeline, fictional papers, every check passing."""

from __future__ import annotations

import re
from pathlib import Path

from research_tool.core.artifacts import ArtifactStore
from research_tool.knowledge.verify import normalize
from research_tool.service import run_status
from research_tool.testing import demo_data
from research_tool.testing.demo_run import build_demo_run, demo_pdf_to_text, is_demo, text_pdf


def test_every_demo_quote_is_in_its_paper() -> None:
    for paper in demo_data.PAPERS:
        source = normalize(paper.text if paper.sections else paper.abstract)
        quotes = [f.quote for f in paper.results] + [q for _, q in paper.limitations + paper.future_work]
        assert all(normalize(q) in source for q in quotes), paper.key


def test_demo_pdfs_are_real_pdfs(tmp_path: Path) -> None:
    paper = demo_data.BY_KEY["morgan2019"]
    pdf = text_pdf(paper.title, paper.text * 3, marker=paper.key)
    assert pdf.startswith(b"%PDF-1.4\n") and pdf.rstrip().endswith(b"%%EOF")
    assert int(re.search(rb"/Count (\d+)", pdf).group(1)) > 1  # type: ignore[union-attr]  # several pages
    xref = int(pdf.rsplit(b"startxref\n", 1)[1].split(b"\n", 1)[0])
    assert pdf[xref:].startswith(b"xref")
    offsets = [int(line[:10]) for line in pdf[xref:].split(b"\n")[3:] if line.endswith(b" n ")]
    assert all(pdf[o:].startswith(b"%d 0 obj" % n) for n, o in enumerate(offsets, 1))  # a valid cross-reference table
    path = tmp_path / "paper.pdf"
    path.write_bytes(pdf)
    assert demo_pdf_to_text(path) == paper.text


def test_a_finished_demo_run(runs_root: Path) -> None:
    run = build_demo_run(runs_root, "done")
    assert is_demo(run) and run_status(run).last_outcome == "done"
    store = ArtifactStore(run.artifacts_dir)
    corpus, km = store.get("corpus"), store.get("knowledge_map")
    assert corpus.stats["full_text"] == 9 and corpus.stats["excluded"] == 2 and len(corpus.needs_manual) == 1
    assert km.claims and all(c.verified for c in km.claims)
    files = [p.file for p in corpus.papers if p.file]
    assert len(set(files)) == len(files) == 9 and all((run.root / f).exists() for f in files)
    hypotheses = store.get("hypothesis_set").hypotheses
    assert len(hypotheses) == 3 and all(e.verified for h in hypotheses for e in h.evidence)
    results = store.get("experiment_results")
    assert results.complete and len(results.runs) == 12 and results.figures
    review = store.get("review")
    assert review.verdict == "accept" and review.edited_sections == ["results"]
    report = (run.root / store.get("final_report").markdown_path).read_text()
    assert "{{" not in report  # every result placeholder was filled from the results
    assert "[@osei2021slow]" in report and "**[osei2021slow]**" in report  # cited, and in the reference list


def test_a_demo_run_stopped_at_the_hypothesis_checkpoint(runs_root: Path) -> None:
    run = build_demo_run(runs_root, "hypothesis_selection")
    gate = run_status(run).open_gates[0]
    assert gate.gate == "hypothesis_selection" and gate.kind == "select" and len(gate.summary["options"]) == 3
