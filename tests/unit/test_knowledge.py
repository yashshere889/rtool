"""Document conversion, sectioning, quote verification, vectors/clustering and the knowledge store."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.knowledge.store import KnowledgeStore, fts_query
from research_tool.knowledge.vectors import HashingEmbedder, agglomerative, suggested_clusters, top_k
from research_tool.knowledge.verify import is_supported, quote_support
from research_tool.schemas.literature import PaperRecord
from research_tool.tools.documents.convert import ConversionError, document_to_text, html_to_text
from research_tool.tools.documents.sections import budgeted_text, chunk_sections, main_text, split_sections

ARXIV_HTML = """<!DOCTYPE html><html><head><title>t</title><style>p{}</style><script>var x=1;</script></head>
<body><nav>menu</nav><h1>Sleep Spindles Predict Recall</h1>
<h2>Abstract</h2><p>We show that spindle density predicts recall (<math alttext="r = 0.41">...</math>).</p>
<h2>1 Introduction</h2><p>Sleep matters &amp; spindles matter.</p>
<h2>References</h2><ul><li>[1] Someone. A paper. 2019.</li></ul></body></html>"""

PDF_STYLE_TEXT = """Title of the Paper
Ana Lopez

Abstract
We study spindles.

1 Introduction
Sleep is important for memory consolidation in adults.

2. Methods
We recorded EEG in 64 adults during N2 sleep and measured spindle density.

Results
Spindle density correlated with overnight recall improvement (r = 0.41).

References
[1] A. Person. Something. 2001.
"""


def test_html_to_text_keeps_headings_and_math_drops_chrome() -> None:
    text = html_to_text(ARXIV_HTML)
    assert "# Sleep Spindles Predict Recall" in text and "## 1 Introduction" in text
    assert "r = 0.41" in text and "Sleep matters & spindles matter." in text
    assert "var x" not in text and "menu" not in text


def test_unclosed_headings_do_not_swallow_the_document() -> None:
    text = html_to_text("<html><body><h2>Methods<p>We recorded EEG.</p><h2>Results<p>It worked.</p></body></html>")
    assert [s.title for s in split_sections(text)] == ["Methods", "Results"]


def test_document_to_text_by_content(tmp_path: Path) -> None:
    (tmp_path / "a.html").write_text(ARXIV_HTML)
    assert "Abstract" in document_to_text(tmp_path / "a.html")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.7 fake")
    assert document_to_text(tmp_path / "b.pdf", pdf_converter=lambda p: "converted") == "converted"
    (tmp_path / "c.bin").write_bytes(b"\xff\xfe\x00\x81")
    with pytest.raises(ConversionError):
        document_to_text(tmp_path / "c.bin")


def test_sections_and_references_split() -> None:
    sections = split_sections(PDF_STYLE_TEXT)
    titles = [s.title for s in sections]
    assert titles == ["front matter", "Abstract", "Introduction", "Methods", "Results", "References"]
    body, refs = main_text(sections)
    assert [s.title for s in refs] == ["References"] and "References" not in [s.title for s in body]
    html_sections = split_sections(html_to_text(ARXIV_HTML))
    assert [s.title for s in html_sections][-1] == "References"


def test_budgeted_text_prefers_core_sections_in_paper_order() -> None:
    sections = split_sections(PDF_STYLE_TEXT + "\nAcknowledgements\nThanks to everyone " * 50)
    body, _ = main_text(sections)
    text = budgeted_text(body, max_chars=400)
    assert len(text) <= 400 and "Thanks" not in text
    assert text.index("Abstract") < text.index("Introduction")


def test_chunking_respects_size_and_boundaries() -> None:
    sections = split_sections("# Long\n" + ("This is a sentence about spindles. " * 400))
    chunks = chunk_sections(sections, max_chars=1000, overlap_chars=100)
    assert len(chunks) > 10 and all(len(c.text) <= 1000 for c in chunks)
    assert chunks[0].text.endswith(".") and [c.seq for c in chunks] == list(range(len(chunks)))


def test_quote_verification_tolerates_pdf_artefacts_not_invention() -> None:
    source = (
        "In 64 adults, spindle density during N2 sleep corre-\nlated with overnight recall improve­ment "
        "(r = 0.41). Pharmacologically suppressing spindles abolished the eﬀect."
    )
    assert quote_support("spindle density during N2 sleep correlated with overnight recall", source) == 1.0
    assert is_supported("Pharmacologically suppressing spindles abolished the effect.", source)
    assert is_supported("spindle density during N2 sleep correlated with nightly recall improvement", source)
    assert not is_supported("Spindles caused a 50% improvement in long-term memory for all participants.", source)
    assert quote_support("short", source) == 0.0  # too short to verify


def test_embeddings_similarity_and_clustering() -> None:
    texts = [
        "sleep spindles memory consolidation EEG",
        "spindle density predicts memory recall during sleep",
        "graph neural network ocean heat transport surrogate",
        "neural surrogate model for ocean circulation",
    ]
    vectors = HashingEmbedder().embed(texts)
    assert top_k(vectors[0], vectors, 2)[1][0] == 1  # most similar other text
    assert agglomerative(vectors, 2) == [[0, 1], [2, 3]] or sorted(agglomerative(vectors, 2)) == [[0, 1], [2, 3]]
    assert suggested_clusters(100) == 7 and suggested_clusters(1) == 1


def test_knowledge_store_roundtrip_search_and_snapshot(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.sqlite"
    with KnowledgeStore(kb_path) as kb:
        kb.upsert_papers(
            [
                PaperRecord(id="arxiv:1", title="Spindles", status="included"),
                PaperRecord(id="doi:2", title="Oceans", status="excluded"),
            ]
        )
        kb.set_chunks("arxiv:1", chunk_sections(split_sections(PDF_STYLE_TEXT)))
        kb.put_extraction("arxiv:1", {"methods": ["EEG"]}, {"prompt_hash": "sha256:x"})
        kb.put_vectors("paper", "hashing-512", {"arxiv:1": [0.5, 0.25]})
        kb.add_citations([("arxiv:1", "doi:2")])
        kb.set_meta("round", 2)
        hits = kb.search("spindle density recall", k=3)
        assert hits and hits[0]["paper_id"] == "arxiv:1"
        assert kb.search("") == []
        kb.search('"; drop table papers; -- NEAR( OR AND *')  # hostile input is just words, never syntax
        assert len(kb.papers()) == 2
        assert [p.id for p in kb.papers(["included"])] == ["arxiv:1"]
        snapshot_sha = kb.copy_to(tmp_path / "snap" / "kb-v1.sqlite")
    assert snapshot_sha.startswith("sha256:")
    readonly = KnowledgeStore(tmp_path / "snap" / "kb-v1.sqlite", readonly=True)
    assert readonly.extraction("arxiv:1")[0] == {"methods": ["EEG"]}  # type: ignore[index]
    assert readonly.vectors("paper", "hashing-512")["arxiv:1"] == [0.5, 0.25]
    assert readonly.citations() == [("arxiv:1", "doi:2")] and readonly.meta("round") == 2
    assert "Spindle density correlated" in readonly.paper_text("arxiv:1")
    readonly.close()
    assert fts_query('spindle "density" (r=0.41)') == '"spindle" OR "density" OR "41"'


def test_working_copy_writes_back(tmp_path: Path) -> None:
    home = tmp_path / "shared" / "kb.sqlite"
    with KnowledgeStore.working_copy(home, tmp_path / "local") as kb:
        kb.upsert_papers([PaperRecord(id="arxiv:9", title="T")])
        kb.sync()
        assert home.exists()
    assert KnowledgeStore(home).paper("arxiv:9") is not None
    assert not list((tmp_path / "local").glob("*.sqlite"))  # local copy cleaned up


def test_locate_quote_points_into_the_original_text() -> None:
    from research_tool.knowledge.verify import locate_quote

    text = "Intro.\nSpindle density predicted\nrecall (r = 0.41) across\nparticipants. The effect remained."
    start, end, score = locate_quote("spindle density predicted recall (r = 0.41) across participants", text)
    assert score == 1.0 and text[start:end] == "Spindle density predicted\nrecall (r = 0.41) across\nparticipants"
    start, end, score = locate_quote("Spindle density predicted the recall (r = 0.41) across participants", text)
    assert 0.85 <= score < 1 and text[start:end].startswith("Spindle") and text[start:end].endswith("participants")
    assert locate_quote("Sleep deprivation impairs working memory in rats", text) is None
