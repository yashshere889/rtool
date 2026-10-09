"""The Literature page's API on the demo corpus: cards, per-paper facts, highlighted text, search, uploads."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research_tool.core.run import RunHandle  # noqa: E402
from research_tool.testing.web import WRITE, web_client  # noqa: E402

BASE_URL = "/api/runs/sleep-demo/1-literature/literature"
MORGAN = "doi:10.5555/demo.2019.001"
HADDAD = "doi:10.5555/demo.2024.003"  # paywalled: abstract only, needs an upload


def test_the_corpus_at_a_glance(runs_root: Path) -> None:
    view = web_client(runs_root).get(BASE_URL).json()
    assert view["stage"] == "literature" and view["corpus_version"] == 1
    papers = {p["id"]: p for p in view["papers"]}
    assert len(papers) == 12 and sum(p["included"] for p in papers.values()) == 10
    morgan = papers[MORGAN]
    assert morgan["read"] == "full_text" and morgan["file_kind"] == "pdf" and morgan["file"].endswith(".pdf")
    assert morgan["claims"] == {"result": 3, "limitation": 2, "future_work": 1}
    assert morgan["links"]["doi"] == "https://doi.org/10.5555/demo.2019.001" and morgan["topics"]
    haddad = papers[HADDAD]
    assert haddad["needs_upload"] and haddad["read"] == "abstract" and haddad["file"] is None
    html = next(p for p in papers.values() if p["file_kind"] == "html")
    assert html["links"]["arxiv"].startswith("https://arxiv.org/abs/")
    assert view["topics"] and view["entities"] and view["coverage"] and view["gaps"] and view["open_problems"]
    assert view["keys"]["morgan2019phase"] == MORGAN and "[morgan2019phase]" in view["overview"]


def test_one_paper_in_detail(runs_root: Path) -> None:
    detail = web_client(runs_root).get(f"{BASE_URL}/paper", params={"id": MORGAN}).json()
    assert detail["problem"].startswith("Whether strengthening slow oscillations")
    assert "closed-loop auditory stimulation" in detail["methods"] and detail["datasets"]
    assert "Twenty-four healthy adults" in detail["methods_section"]  # the paper's own methods text
    kinds = [c["kind"] for c in detail["claims"]]
    assert kinds.count("result") == 3 and all(c["verified"] for c in detail["claims"])
    assert [r["bib_key"] for r in detail["cited_by"]][:1] and any(
        r["bib_key"].startswith("mensah2018") for r in detail["cites"]
    )
    assert detail["bibtex"].startswith("@") and "morgan2019phase" in detail["bibtex"]
    assert any(MORGAN in p["paper_ids"] for p in detail["open_problems"])
    assert web_client(runs_root).get(f"{BASE_URL}/paper", params={"id": "doi:nope"}).status_code == 404


def test_the_text_with_claims_highlighted(runs_root: Path) -> None:
    c = web_client(runs_root)
    text = c.get(f"{BASE_URL}/paper/text", params={"id": MORGAN}).json()
    detail = c.get(f"{BASE_URL}/paper", params={"id": MORGAN}).json()
    titles = [s["title"] for s in text["sections"]]
    assert text["source"] == "full_text" and {"Abstract", "Introduction", "Methods", "Results"} <= set(titles)
    assert len(text["highlights"]) == len(detail["claims"])  # every verified quote found in the text
    for h in text["highlights"]:
        span = text["sections"][h["section"]]["text"][h["start"] : h["end"]]
        assert " ".join(span.split()) == " ".join(detail["claims"][h["claim"]]["quote"].split())
    abstract_only = c.get(f"{BASE_URL}/paper/text", params={"id": HADDAD}).json()
    assert abstract_only["source"] == "abstract" and abstract_only["highlights"]


def test_full_text_search(runs_root: Path) -> None:
    hits = web_client(runs_root).get(f"{BASE_URL}/search", params={"q": "spindle coupling"}).json()
    assert hits and hits[0]["paper_id"].startswith(("arxiv:", "doi:")) and "pindle" in hits[0]["snippet"]
    assert web_client(runs_root).get(f"{BASE_URL}/search", params={"q": "?!"}).json() == []


def test_uploading_a_paper_you_have(runs_root: Path) -> None:
    c = web_client(runs_root)
    url = f"{BASE_URL}/inbox"
    params = {"title": "Closed-loop stimulation in mild cognitive impairment: a pilot study", "paper_id": HADDAD}
    pdf = b"%PDF-1.4\n% my library copy\n"
    assert c.put(url, params=params, content=pdf, headers={"content-type": "application/pdf"}).status_code == 403
    sent = c.put(url, params=params, content=pdf, headers={**WRITE, "content-type": "application/pdf"})
    assert sent.status_code == 201, sent.text
    run = RunHandle(runs_root / "sleep-demo" / "1-literature")
    inbox = run.work_dir("literature") / "inbox"
    stored = run.root / sent.json()["file"]
    assert stored.parent == inbox and stored.read_bytes() == pdf
    assert '"paper_id"' in stored.with_suffix(".json").read_text()
    assert not [p for p in inbox.iterdir() if p.name.startswith(".")]  # no half-written temp files left
    wrong = c.put(url, params=params, content=b"hello", headers={**WRITE, "content-type": "application/pdf"})
    assert wrong.status_code == 422
    other = c.put(url, params=params, content=b"x", headers={**WRITE, "content-type": "image/png"})
    assert other.status_code == 415
