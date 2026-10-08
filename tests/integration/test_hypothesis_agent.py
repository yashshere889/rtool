"""The Hypothesis agent against a hand-built corpus and knowledge map, with a schema-aware mock LLM."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.engine import Engine, Outcome
from research_tool.core.run import RunHandle
from research_tool.core.services import Services
from research_tool.knowledge.store import KnowledgeStore
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.types import ChatRequest
from research_tool.schemas.literature import Claim, Corpus, Gap, KnowledgeMap, OpenProblem, PaperRecord
from research_tool.testing.harness import make_run, queue
from research_tool.tools.documents.sections import Section

PIPELINE = {
    "name": "h",
    "initial_artifacts": ["research_brief", "corpus", "knowledge_map"],
    "gates": {
        "pick": {
            "kind": "select",
            "options_artifact": "hypothesis_set",
            "options_path": "hypotheses",
            "selection_artifact": "selected_hypothesis",
        }
    },
    "stages": [{"id": "hypotheses", "agent": "hypothesis", "gate": "pick"}],
}
TEXTS = {
    "p1": "Spindle density during N2 sleep correlated with overnight recall improvement in adults. "
    "Our sample was small and drawn from a single site.",
    "p2": "Closed-loop auditory stimulation increased slow oscillation amplitude in older adults. "
    "Whether stimulation improves memory in young adults remains untested.",
    "p3": "Deep learning detectors matched expert spindle scoring on the Sleep-EDF dataset. "
    "Detectors were not evaluated on home EEG recordings.",
}


def seed_run(runs_root: Path, km: KnowledgeMap | None = None, overrides: dict | None = None) -> RunHandle:
    run = make_run(runs_root, PIPELINE, topic="Sleep spindles and memory", overrides=overrides)
    papers = [
        PaperRecord(id=pid, title=f"Paper {pid}", abstract=text[:80], year=2020 + i, status="extracted", relevance=9)
        for i, (pid, text) in enumerate(TEXTS.items())
    ]
    with KnowledgeStore(run.root / "kb.sqlite") as kb:
        kb.upsert_papers(papers)
        for pid, text in TEXTS.items():
            kb.set_sections(pid, [Section("Results", text)])
    store = ArtifactStore(run.artifacts_dir)
    user = Producer(stage="literature", agent="test")
    store.put(
        "corpus",
        Corpus(topic="t", queries=[], papers=papers, kb_path="kb.sqlite", kb_sha256="x", bibtex_path="refs.bib"),
        producer=user,
    )
    if km is None:
        sentences = {pid: re.split(r"(?<=\.)\s+", text) for pid, text in TEXTS.items()}
        km = KnowledgeMap(
            topic="t",
            overview="",
            claims=[
                Claim(
                    paper_id="p1",
                    kind="limitation",
                    text="small sample",
                    quote=sentences["p1"][1],
                    verified=True,
                    support=1.0,
                ),
                Claim(
                    paper_id="p2",
                    kind="future_work",
                    text="young adults",
                    quote=sentences["p2"][1],
                    verified=True,
                    support=1.0,
                ),
                Claim(
                    paper_id="p3",
                    kind="limitation",
                    text="no home EEG",
                    quote=sentences["p3"][1],
                    verified=True,
                    support=1.0,
                ),
                Claim(
                    paper_id="p1",
                    kind="result",
                    text="spindles predict",
                    quote=sentences["p1"][0],
                    verified=True,
                    support=1.0,
                ),
            ],
            open_problems=[
                OpenProblem(statement="Does stimulation help young adults?", paper_ids=["p2"], why_open="untested")
            ],
            gaps=[Gap(kind="untested_combination", description="Detector X never tested on home EEG", evidence=["p3"])],
        )
    store.put("knowledge_map", km, producer=user)
    return run


def schema(request: ChatRequest) -> str:
    return str((request.json_schema or {}).get("title", ""))


def draft(title: str, statement: str, paper_id: str, quote: str, compute: str = "small_gpu") -> dict[str, Any]:
    return {
        "title": title,
        "statement": statement,
        "null_hypothesis": "No effect.",
        "rationale": "Follows.",
        "evidence": [{"paper_id": paper_id, "quote": quote}],
        "independent_variables": ["stimulation"],
        "dependent_variables": ["recall"],
        "datasets": ["Sleep-EDF"],
        "methods": ["EEG"],
        "success_criteria": "recall improves by 5%",
        "compute": compute,
    }


def drafts(request: ChatRequest) -> dict[str, Any]:
    prompt = request.messages[-1].content
    pid, quote = re.search(r"^\[(p\d)\] (.+)$", prompt, re.M).groups()  # type: ignore[union-attr]
    if "Open problem" in prompt:
        return {
            "hypotheses": [
                draft(
                    "Stimulation in young adults",
                    "If closed-loop audio stimulation targets slow oscillations in "
                    "young adults then overnight recall improves",
                    pid,
                    quote,
                ),
                draft(
                    "Already done idea",
                    "Spindle density already correlates with recall in adults as reported",
                    pid,
                    quote,
                ),
            ]
        }
    if "Gap" in prompt:
        return {
            "hypotheses": [
                draft(
                    "Home EEG detectors",
                    "If deep spindle detectors are applied to home EEG then their agreement "
                    "with experts drops below laboratory levels",
                    pid,
                    quote,
                    compute="large_gpu",
                ),
                draft(
                    "Invented support",
                    "Home EEG makes spindles twice as dense in every population studied",
                    pid,
                    "This sentence was invented by the model and is not in any paper.",
                ),
            ]
        }
    return {
        "hypotheses": [
            draft(
                "Bigger multisite sample",
                "If spindle density is measured across several sites with larger samples "
                "then its correlation with recall replicates",
                pid,
                quote,
                compute="cpu_only",
            ),
            draft(
                "Stimulation copy",
                "If closed-loop audio stimulation targets slow oscillations in young adults then "
                "overnight recall improves",
                pid,
                quote,
            ),
        ]
    }


def novelty(request: ChatRequest) -> dict[str, Any]:
    prompt = request.messages[-1].content
    first = re.search(r"^\[(p\d)\]", prompt, re.M).group(1)  # type: ignore[union-attr]
    repeated = "already correlates" in prompt
    return {
        "novel": not repeated,
        "closest_paper_id": first,
        "overlap": "differs in population",
        "score": 1 if repeated else 8,
    }


def rubric(request: ChatRequest) -> dict[str, Any]:
    prompt = request.messages[-1].content
    best = "closed-loop" in prompt
    value = 5 if best else 3
    return {
        "novelty": value,
        "feasibility": 4,
        "testability": value,
        "significance": value,
        "clarity": 4,
        "comment": "ok",
    }


@pytest.fixture
def llm() -> MockLLM:
    mock = MockLLM()
    mock.add(lambda r: schema(r) == "HypothesisDrafts", drafts)
    mock.add(lambda r: schema(r) == "NoveltyJudgment", novelty)
    mock.add(lambda r: schema(r) == "RubricScores", rubric)
    return mock


def test_hypotheses_are_grounded_deduplicated_checked_and_ranked(runs_root: Path, llm: MockLLM) -> None:
    run = seed_run(runs_root, overrides={"budgets": {"run": {"max_gpu_hours": 100}}})
    result = Engine(run, Services(llm=LLMClient(llm, run.config.llm))).advance()
    assert result.outcome is Outcome.WAITING_GATE
    options = result.open_gates[0]["summary"]["options"]
    assert len(options) == 3 and options[0]["title"] == "Stimulation in young adults"  # best score first

    hs = ArtifactStore(run.artifacts_dir).get("hypothesis_set")
    reasons = {h.title: h.rejected_reason for h in hs.candidates}
    assert reasons["Invented support"] == "no evidence quote could be found in the cited papers"
    assert reasons["Stimulation copy"].startswith("near-duplicate of")
    assert reasons["Already done idea"].startswith("already tested by")
    shortlisted = {h.title for h in hs.hypotheses}
    assert shortlisted == {"Stimulation in young adults", "Home EEG detectors", "Bigger multisite sample"}
    home = next(h for h in hs.hypotheses if h.title == "Home EEG detectors")
    assert home.scores["feasibility"] == 2.0  # large_gpu exceeds the 100 GPU-hour budget
    assert all(e.verified for h in hs.hypotheses for e in h.evidence)
    assert (run.root / hs.markdown_path).read_text().startswith("# Hypotheses")
    rubric_calls = [r for r in llm.calls if schema(r) == "RubricScores"]
    assert len(rubric_calls) == 3 * len(hs.hypotheses)  # self-consistency: median of 3 samples each

    queue(run, ActionKind.SELECT, payload={"index": 1})
    assert Engine(run, Services(llm=LLMClient(llm, run.config.llm))).advance().outcome is Outcome.DONE
    chosen = ArtifactStore(run.artifacts_dir).get("selected_hypothesis")
    assert chosen.index == 1 and chosen.item["title"] == hs.hypotheses[1].title


def test_no_evidence_means_a_human_must_widen_the_corpus(runs_root: Path, llm: MockLLM) -> None:
    run = seed_run(runs_root, km=KnowledgeMap(topic="t", overview=""))
    result = Engine(run, Services(llm=LLMClient(llm, run.config.llm))).advance()
    assert result.outcome is Outcome.NEEDS_HUMAN
    assert "no verified evidence" in result.failed[0]["error"]
