"""The Literature agent end to end, against fake APIs, fake documents and a schema-aware mock LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from research_tool.agents.literature.toolkit import LiteratureToolkit
from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore
from research_tool.core.engine import Engine, Outcome, load_status
from research_tool.core.events import EventLog
from research_tool.core.services import Services
from research_tool.knowledge.store import KnowledgeStore
from research_tool.knowledge.vectors import HashingEmbedder
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.types import ChatRequest
from research_tool.tools.http import PoliteHttp
from research_tool.tools.scholarly.base import PaperHit

PAPER_TEXT = """# {title}

## Abstract
{title} is studied here. Spindle density during N2 sleep correlated with overnight recall improvement in adults.

## 1 Introduction
Sleep supports memory consolidation through coordinated thalamocortical oscillations.
Prior work measured spindles in small samples.

## 2 Methods
We recorded EEG from 64 adults and detected spindles automatically with a validated algorithm on the Sleep-EDF dataset.

## 3 Results
Spindle density predicted recall with a correlation of r = 0.41 across participants.
The effect remained after controlling for age.

## 4 Discussion
These findings extend earlier reports that sleep oscillations support the stabilisation of new memories. The association
between spindle density and recall was robust across the sensitivity analyses we ran, including alternative detection
thresholds and the exclusion of participants with fragmented sleep. Together with work on targeted memory reactivation,
the results suggest that spindles index a process that transfers information from hippocampus to neocortex during sleep.
Future studies should test whether manipulating spindles changes memory, rather than only observing the correlation.
Spindle density is known to vary with age and sex, so we modelled both as covariates in every analysis we report here.
We also checked that the automated detector agreed with expert scoring on a random subset of recordings from the cohort.
The broader literature on sleep and memory has grown quickly; our design follows recommendations from recent reviews.

## 5 Limitations
Our sample was small and drawn from a single site, which limits generalisation to other populations.

## References
[1] Someone. An older paper. 2001.
"""


def hit(n: int, title: str, **ids: Any) -> PaperHit:
    return PaperHit(
        source="fake",
        source_id=str(n),
        title=title,
        abstract=f"Abstract of {title}.",
        year=2018 + n,
        authors=[f"Author{n} Lastname{n}"],
        **ids,
    )


HITS = [
    hit(1, "Sleep spindles predict memory consolidation in adults", arxiv_id="2101.00001", s2_id="S1"),
    hit(2, "Targeted memory reactivation during slow wave sleep", arxiv_id="2102.00002"),
    hit(3, "Deep learning spindle detection in sleep EEG", doi="10.1/sp3", oa_urls=["https://repo.example/sp3.pdf"]),
    hit(4, "Sleep deprivation impairs working memory", doi="10.1/sp4"),
    hit(5, "Graph surrogates for ocean heat transport", doi="10.1/ocean"),
    hit(6, "Crop yield prediction from satellite images", doi="10.1/crop"),
]
REFERENCE = hit(
    7,
    "Thalamocortical spindles and synaptic plasticity in sleep",
    doi="10.1/thal",
    oa_urls=["https://repo.example/thal.pdf"],
)


class FakeSearch:
    def __init__(self, hits: list[PaperHit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, limit: int, *, year_from: int | None = None) -> list[PaperHit]:
        self.queries.append(query)
        return self.hits[:limit]


class FakeS2(FakeSearch):
    @staticmethod
    def reference_for(paper: Any) -> str | None:
        return paper.s2_id

    def references(self, ref: str, limit: int) -> list[PaperHit]:
        return [REFERENCE, HITS[4]] if ref == "S1" else []

    def citations(self, ref: str, limit: int) -> list[PaperHit]:
        return []


class Documents:
    """Serves arXiv HTML for paper 1, a PDF for everything else that is open access."""

    def __init__(self) -> None:
        self.downloads: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.downloads.append(url)
        if url == "https://arxiv.org/html/2101.00001":
            return httpx.Response(
                200, text=to_html(PAPER_TEXT.format(title=HITS[0].title)), headers={"content-type": "text/html"}
            )
        if url.startswith("https://arxiv.org/html/"):
            return httpx.Response(404)
        if url.startswith(("https://arxiv.org/pdf/", "https://repo.example/")):
            return httpx.Response(200, content=b"%PDF-1.7 " + url.encode(), headers={"content-type": "application/pdf"})
        return httpx.Response(404)


def fake_pdf_to_text(path: Path) -> str:
    return PAPER_TEXT.format(title=path.stem)


def to_html(markdown: str) -> str:
    """The fixture text as a real (well-formed) arXiv-style HTML page."""
    body = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            body.append(f"<h{level}>{line.lstrip('#').strip()}</h{level}>")
        elif line.strip():
            body.append(f"<p>{line}</p>")
    return "<!DOCTYPE html><html><body><nav>arXiv menu</nav>" + "\n".join(body) + "</body></html>"


def schema(request: ChatRequest) -> str:
    return str((request.json_schema or {}).get("title", ""))


def last(request: ChatRequest) -> str:
    return request.messages[-1].content


def relevance(request: ChatRequest) -> dict[str, Any]:
    title = re.search(r"Paper: (.+?) \(", last(request)).group(1).lower()  # type: ignore[union-attr]
    relevant = any(word in title for word in ("sleep", "spindle", "memory", "thalamo"))
    return {"score": 9 if relevant else 1, "reason": "on topic" if relevant else "different field"}


def extraction(request: ChatRequest) -> dict[str, Any]:
    text = last(request).split("<<<", 1)[1].split(">>>", 1)[0]
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", text) if len(s.strip()) > 30 and "#" not in s]
    return {
        "problem": "How spindles relate to memory.",
        "approach": "EEG recordings.",
        "methods": ["EEG spindle detection"],
        "datasets": ["Sleep-EDF"],
        "metrics": ["correlation"],
        "key_results": [
            {
                "claim": "Spindles predict recall",
                "quote": sentences[0],
                "method": "EEG spindle detection",
                "dataset": "Sleep-EDF",
                "metric": "r",
                "value": "0.41",
            },
            {"claim": "Invented finding", "quote": "Spindles tripled the IQ of every participant in all conditions."},
        ],
        "limitations": [{"statement": "Small single-site sample", "quote": sentences[-1]}],
    }


def open_problems(request: ChatRequest) -> dict[str, Any]:
    first_id = re.search(r"^\[([^\]]+)\]", last(request), re.M).group(1)  # type: ignore[union-attr]
    return {
        "problems": [
            {
                "statement": "Whether spindles cause better memory",
                "paper_ids": [first_id, "made-up:1"],
                "why_open": "evidence is correlational",
            }
        ]
    }


def overview(request: ChatRequest) -> dict[str, Any]:
    key = re.search(r"^\[([a-z0-9]+)\]", last(request), re.M).group(1)  # type: ignore[union-attr]
    return {"text": f"Spindles track memory [{key}], though causality is unproven [nobody1999fake]."}


def make_llm(calls: dict[str, int]) -> MockLLM:
    def counted(name: str, reply: Any) -> Any:
        def handler(request: ChatRequest) -> Any:
            calls[name] = calls.get(name, 0) + 1
            return reply(request) if callable(reply) else reply

        return handler

    llm = MockLLM()
    llm.add(
        lambda r: schema(r) == "SearchPlan",
        counted(
            "plan", {"queries": ["sleep spindles memory", "memory reactivation sleep"], "must_cover": ["spindles"]}
        ),
    )
    llm.add(lambda r: schema(r) == "RelevanceJudgment", counted("relevance", relevance))
    llm.add(lambda r: schema(r) == "PaperExtraction", counted("extract", extraction))
    llm.add(
        lambda r: schema(r) == "TopicLabel", counted("topic", {"label": "Sleep and memory", "summary": "Spindles."})
    )
    llm.add(lambda r: schema(r) == "OpenProblems", counted("problems", open_problems))
    llm.add(lambda r: schema(r) == "Overview", counted("overview", overview))
    digest = {
        "summary": "Spindles were measured overnight; recall rose in 73 percent of sleepers.",  # 73 is made up
        "design": "Correlational sleep study.",
        "sample": "Healthy adults.",
        "procedure": "Sleep was recorded and recall tested the next morning.",
        "gaps": ["Causality is not shown."],
    }
    llm.add(lambda r: schema(r) == "PaperDigest", counted("digest", digest))
    return llm


@pytest.fixture
def setup(runs_root: Path):
    from research_tool.testing.harness import make_run

    run = make_run(
        runs_root,
        "research",
        topic="Sleep spindles and memory consolidation",
        overrides={"llm": {"structured_with_reasoning": True}},
    )
    documents = Documents()
    sources = {"openalex": FakeSearch(HITS), "semantic_scholar": FakeS2([])}
    calls: dict[str, int] = {}

    def toolkit(config: Any, cache_path: Path) -> LiteratureToolkit:
        http = PoliteHttp(
            client=httpx.Client(transport=httpx.MockTransport(documents)), sleep=lambda _s: None, default_interval_s=0.0
        )
        return LiteratureToolkit(
            sources=sources, unpaywall=None, http=http, pdf_converter=fake_pdf_to_text, embedder=HashingEmbedder()
        )

    services = Services(llm=LLMClient(make_llm(calls), run.config.llm), tools={"literature_toolkit": toolkit})
    return run, services, documents, calls, sources


def test_literature_agent_builds_a_verified_corpus_and_map(setup) -> None:  # noqa: ANN001
    run, services, documents, calls, _ = setup
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.WAITING_GATE and result.open_gates[0]["gate"] == "literature_review"
    assert any(path.endswith("knowledge_map.md") for path in result.open_gates[0]["summary"]["review_files"])

    store = ArtifactStore(run.artifacts_dir)
    corpus = store.get("corpus")
    by_title = {p.title: p for p in corpus.papers}
    assert by_title[HITS[4].title].status == "excluded" and by_title[HITS[5].title].status == "excluded"
    assert by_title[REFERENCE.title].found_by == ["refs:arxiv:2101.00001"]  # found by snowballing
    assert by_title[HITS[0].title].file.endswith(".html")  # arXiv HTML preferred
    assert by_title[HITS[1].title].file.endswith(".pdf")  # HTML missing -> PDF
    assert by_title[HITS[3].title].status == "unavailable" and by_title[HITS[3].title].id in corpus.needs_manual
    assert sum(p.status == "extracted" for p in corpus.papers) == 4
    assert corpus.stats["included"] == 5 and corpus.stats["excluded"] == 2

    km = store.get("knowledge_map")
    results = [c for c in km.claims if c.kind == "result"]
    assert any(c.verified for c in results)
    invented = [c for c in results if c.text == "Invented finding"]
    assert invented and not any(c.verified for c in invented)  # fabricated quotes are caught
    assert km.stats["unknown_citations_removed"] == 1 and "[nobody1999fake]" not in km.overview
    assert km.open_problems and all(pid in {p.id for p in corpus.papers} for pid in km.open_problems[0].paper_ids)
    assert km.topics and km.entities and km.coverage[0].method == "EEG spindle detection"
    assert (run.root / km.markdown_path).read_text().startswith("# Knowledge map")

    snapshot = run.root / corpus.kb_path
    kb = KnowledgeStore(snapshot, readonly=True)
    assert kb.search("spindle density recall", k=3)  # downstream agents can search full text
    digests = [d["digest"] for d in kb.extractions().values() if "digest" in d]
    assert len(digests) == len(kb.extractions()) == calls["digest"]  # one digest per paper that was read
    assert digests[0]["gaps"] == ["Causality is not shown."]
    assert any("73" in n for n in digests[0]["unverified_numbers"])  # a number the paper does not contain
    assert ("arxiv:2101.00001", "doi:10.1/thal") in kb.citations()
    kb.close()
    assert "literature_phase_done" in [e["kind"] for e in EventLog(run.events_path, run.run_id)]


def test_every_paper_gets_its_own_file(setup) -> None:  # noqa: ANN001
    run, services, _, _, _ = setup
    Engine(run, services).advance()
    corpus = ArtifactStore(run.artifacts_dir).get("corpus")
    copies = [p for p in corpus.papers if p.file]
    assert len(copies) >= 4
    assert len({p.file for p in copies}) == len(copies)  # doi:10.1/sp3 and doi:10.1/thal used to share doi_10.pdf
    for paper in copies:
        content = (run.root / paper.file).read_bytes()
        if paper.file.endswith(".pdf"):  # the fake server puts the URL it served into the PDF
            assert any(marker.encode() in content for marker in [paper.arxiv_id or "", *paper.oa_urls] if marker)


def test_paper_stems_never_collide_on_dotted_ids() -> None:
    from research_tool.agents.literature.agent import paper_stem

    ids = ["doi:10.1038/s41586-020-2649-2", "doi:10.1038/nature14539", "arxiv:2401.12345", "arxiv:2401.67890"]
    stems = [paper_stem(i) for i in ids]
    assert len(set(stems)) == len(ids)
    assert all("/" not in s and ":" not in s for s in stems)
    assert stems[0].startswith("doi_10.1038_s41586-020-2649-2-")


def test_rejecting_the_gate_reuses_downloads_and_reading(setup) -> None:  # noqa: ANN001
    run, services, documents, calls, sources = setup
    Engine(run, services).advance()
    downloads, extractions, digests = len(documents.downloads), calls["extract"], calls["digest"]
    from research_tool.testing.harness import queue

    queue(run, ActionKind.REJECT, payload={"feedback": "Also cover targeted memory reactivation."})
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.WAITING_GATE
    assert len(documents.downloads) == downloads  # nothing downloaded twice
    assert calls["extract"] == extractions and calls["digest"] == digests  # nothing read twice
    assert calls["plan"] == 2 and calls["relevance"] > 7  # replanned and re-screened with the feedback
    _, stages, _ = load_status(run)
    assert stages["literature"].attempt == 2
    queue(run, ActionKind.APPROVE)
    result = Engine(run, services).advance()
    _, stages, _ = load_status(run)
    assert stages["literature"].status.value == "done"
    assert result.ran == ["hypotheses"]  # the run moved on to the next stage


def test_papers_dropped_in_the_inbox_are_ingested(setup) -> None:  # noqa: ANN001
    run, services, _, _, _ = setup
    inbox = run.work_dir("literature") / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "mine.pdf").write_bytes(b"%PDF-1.7 my paywalled paper")
    (inbox / "mine.json").write_text(json.dumps({"title": "My paywalled sleep paper", "doi": "10.9/mine"}))
    Engine(run, services).advance()
    corpus = ArtifactStore(run.artifacts_dir).get("corpus")
    mine = next(p for p in corpus.papers if p.doi == "10.9/mine")
    assert mine.found_by == ["manual"] and mine.status == "extracted" and mine.note == "added by hand"
    assert not list(inbox.iterdir())  # moved into papers/


def test_papers_marked_to_keep_or_drop_are_applied_as_marked(setup) -> None:  # noqa: ANN001
    from research_tool.testing.harness import queue

    run, services, _, calls, _ = setup
    Engine(run, services).advance()
    corpus = ArtifactStore(run.artifacts_dir).get("corpus")
    by_title = {p.title: p for p in corpus.papers}
    drop, keep = by_title[HITS[0].title].id, by_title[HITS[4].title].id  # read in full / excluded at screening
    screened = calls["relevance"]
    queue(run, ActionKind.REJECT, payload={"include": [keep], "exclude": [drop, "doi:not-in-the-corpus"]})
    assert Engine(run, services).advance().outcome is Outcome.WAITING_GATE
    assert calls["relevance"] == screened  # choices only: nothing re-screened by the model
    papers = {p.id: p for p in ArtifactStore(run.artifacts_dir).get("corpus").papers}
    assert (papers[drop].status, papers[drop].override, papers[drop].note) == ("excluded", "exclude", "dropped by you")
    assert papers[keep].status != "excluded" and papers[keep].override == "include"
    assert "literature_choices_applied" in [e["kind"] for e in EventLog(run.events_path, run.run_id)]

    queue(run, ActionKind.REJECT, payload={"feedback": "Also cover targeted memory reactivation."})
    Engine(run, services).advance()  # re-screened with the comment: the researcher's choices still stand
    papers = {p.id: p for p in ArtifactStore(run.artifacts_dir).get("corpus").papers}
    assert calls["relevance"] > screened
    assert papers[drop].status == "excluded" and papers[keep].status != "excluded"
