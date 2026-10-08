"""Writer and Reviewer on hand-made inputs: repairs, findings, change requests, consensus, verdicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_tool.agents.coder.results import aggregate
from research_tool.agents.reviewer.agent import ReviewerConfig, Reviewing
from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.engine import Engine, Outcome
from research_tool.core.events import EventLog
from research_tool.core.run import RunHandle
from research_tool.core.services import Services
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.types import ChatRequest
from research_tool.schemas.code import CodeBundle, ExperimentResults
from research_tool.schemas.literature import Claim, Corpus, KnowledgeMap, PaperRecord, Topic
from research_tool.schemas.report import ReviewDraft, ReviewIssue, Rubric
from research_tool.testing.harness import make_run, queue
from research_tool.testing.samples import sample_plan as plan
from research_tool.testing.samples import sample_records as records

INITIAL = ["research_brief", "corpus", "knowledge_map", "experiment_plan", "experiment_results", "code_bundle"]


def seed(runs_root: Path, *, runs: list | None = None, stages: list[dict] | None = None) -> RunHandle:
    run = make_run(
        runs_root,
        {
            "name": "write",
            "initial_artifacts": INITIAL,
            "gates": {"final_report": {"kind": "approve", "prompt": "ok?"}},
            "stages": stages
            or [{"id": "report", "agent": "writer"}, {"id": "review", "agent": "reviewer", "gate": "final_report"}],
        },
    )
    (run.root / "refs.bib").write_text("@article{smith2020,\n  title={Spindles}\n}\n")
    (run.root / "README.md").write_text("code")
    (run.root / "results.json").write_text("{}")
    p = plan()
    runs = runs if runs is not None else records()
    summaries, comparisons = aggregate(p, runs)
    corpus = Corpus(
        topic="t",
        queries=["q"],
        papers=[PaperRecord(id="doi:1", bib_key="smith2020", title="Spindles", year=2020, status="extracted")],
        kb_path="kb.sqlite",
        kb_sha256="",
        bibtex_path="refs.bib",
    )
    km = KnowledgeMap(
        topic="t",
        overview="Spindles matter for memory [smith2020].",
        topics=[Topic(label="Spindles", summary="memory", paper_ids=["doi:1"])],
        claims=[
            Claim(
                paper_id="doi:1",
                kind="result",
                text="Spindle density predicted recall (r = 0.41)",
                quote="spindle density correlated with recall (r = 0.41)",
                verified=True,
                support=0.95,
            ),
        ],
    )
    results = ExperimentResults(
        runs=runs,
        summaries=summaries,
        comparisons=comparisons,
        complete=all(r.status == "ok" for r in runs),
        results_path="results.json",
    )
    bundle = CodeBundle(root="code", files=[], pack="generic", readme_path="README.md")
    store = ArtifactStore(run.artifacts_dir)
    for name, value in [
        ("corpus", corpus),
        ("knowledge_map", km),
        ("experiment_plan", p),
        ("experiment_results", results),
        ("code_bundle", bundle),
    ]:
        store.put(name, value, producer=Producer(stage="user", agent="test"))
    return run


def honest(request: ChatRequest) -> str:
    prompt = request.messages[-1].content
    text = "Spindles matter for memory [@smith2020]." if "@smith2020" in prompt else "Spindles matter."
    if "{{mean:acc:new}}" in prompt:
        text += " The new method reached {{mean:acc:new}} ({{p:acc:new}})."
    if '"Discussion"' in prompt:
        text += "\n\nLimitations. Three seeds only."
    return text


def writer_llm(section: Any = honest, fix: Any = honest, issues: list[dict] | None = None) -> MockLLM:
    review = {
        "scores": {"soundness": 4, "evidence": 4, "clarity": 4, "reproducibility": 4, "limitations": 4},
        "summary": "fine",
        "issues": issues or [],
    }
    return (
        MockLLM()
        .add(lambda r: r.messages[-1].content.startswith("Rewrite this"), fix)
        .add(lambda r: "You are editing the" in r.messages[-1].content, lambda r: honest(r) + " Edited.")
        .add(lambda r: "section of a research report" in r.messages[-1].content, section)
        .add(
            lambda r: "Write the title and the abstract" in r.messages[-1].content,
            {"title": "New beats base", "abstract": "Across {{runs_ok}} runs the new method won."},
        )
        .add(lambda r: "Check the report sections below" in r.messages[-1].content, {"items": []})
        .add(lambda r: "Review this research report" in r.messages[-1].content, review)
    )


def services(run: RunHandle, llm: MockLLM) -> Services:
    return Services(llm=LLMClient(llm, run.config.llm))


def test_writer_repairs_typed_numbers_and_unknown_citations(runs_root: Path) -> None:
    run = seed(runs_root, stages=[{"id": "report", "agent": "writer"}])
    sloppy = lambda r: honest(r) + " It reached 0.713 accuracy [@nobody2024]."  # noqa: E731
    llm = writer_llm(section=sloppy)
    result = Engine(run, services(run, llm)).advance()
    assert result.outcome is Outcome.DONE, result.failed
    report = ArtifactStore(run.artifacts_dir).get("report")
    assert not report.findings  # every problem was repaired
    fixes = [r for r in llm.calls if r.messages[-1].content.startswith("Rewrite this")]
    assert fixes and "number typed instead of a placeholder" in fixes[0].messages[-1].content
    text = (run.root / report.markdown_path).read_text()
    assert "nobody2024" not in text and "reached 0.713 (p = 0.010)" in text  # the number came from the results
    assert report.citations == ["smith2020"] and "r = 0.41" not in text  # nothing invented


def test_writer_strips_what_repairs_cannot_fix(runs_root: Path) -> None:
    run = seed(runs_root, stages=[{"id": "report", "agent": "writer"}])
    stubborn = lambda r: honest(r) + " It reached 0.99."  # noqa: E731
    result = Engine(run, services(run, writer_llm(section=stubborn, fix=stubborn))).advance()
    assert result.outcome is Outcome.DONE
    report = ArtifactStore(run.artifacts_dir).get("report")
    assert {f.kind for f in report.findings} == {"number"} and all(f.severity == "minor" for f in report.findings)
    assert "0.99" not in (run.root / report.markdown_path).read_text()


def test_writer_asks_for_missing_conditions(runs_root: Path) -> None:
    only_base = [  # every run of `new` failed: the coder records them as missing
        r if r.condition == "base" else r.model_copy(update={"status": "missing", "metrics": {}}) for r in records()
    ]
    run = seed(runs_root, runs=only_base, stages=[{"id": "report", "agent": "writer"}])
    Engine(run, services(run, writer_llm())).advance()
    report = ArtifactStore(run.artifacts_dir).get("report")
    assert {"missing_condition", "incomplete_runs"} <= {f.kind for f in report.findings}
    raised = [e for e in EventLog(run.events_path, run.run_id) if e["kind"] == "change_request_raised"]
    assert raised and raised[0]["data"]["artifact"] == "experiment_results"


def test_review_accepts_edits_and_waits_for_the_human(runs_root: Path) -> None:
    run = seed(runs_root)
    issue = {"section": "results", "problem": "Name the test", "suggestion": "say paired t-test", "target": "report"}
    result = Engine(run, services(run, writer_llm(issues=[issue]))).advance()
    assert result.outcome is Outcome.WAITING_GATE and result.open_gates[0]["gate"] == "final_report"
    store = ArtifactStore(run.artifacts_dir)
    review = store.get("review")
    assert review.verdict == "accept" and review.edited_sections == ["results"]
    assert "Edited." in (run.root / store.get("final_report").markdown_path).read_text()
    queue(run, ActionKind.APPROVE)
    assert Engine(run, services(run, writer_llm())).advance().outcome is Outcome.DONE


def test_major_experiment_problems_go_back_upstream(runs_root: Path) -> None:
    run = seed(runs_root)
    issue = {
        "section": "results",
        "problem": "The baseline was trained for fewer epochs",
        "suggestion": "train both for the same number of epochs",
        "target": "experiment_results",
        "severity": "major",
    }
    Engine(run, services(run, writer_llm(issues=[issue]))).advance()
    review = ArtifactStore(run.artifacts_dir).get("review")
    assert review.verdict == "needs_human" and review.change_requests
    raised = [e["data"] for e in EventLog(run.events_path, run.run_id) if e["kind"] == "change_request_raised"]
    assert raised[-1]["artifact"] == "experiment_results" and "fewer epochs" in raised[-1]["reason"]


def test_failed_checks_need_a_human(runs_root: Path) -> None:
    runs = records()
    runs[0] = runs[0].model_copy(update={"status": "failed", "metrics": {}})
    run = seed(runs_root, runs=runs)
    Engine(run, services(run, writer_llm())).advance()
    review = ArtifactStore(run.artifacts_dir).get("review")
    failed = [c.name for c in review.checks if not c.ok]
    assert review.verdict == "needs_human" and failed == ["all planned runs succeeded"]


def test_issues_need_a_majority() -> None:
    def draft(*problems: str, score: int = 4) -> ReviewDraft:
        scores = Rubric(soundness=score, evidence=4, clarity=4, reproducibility=4, limitations=4)
        issues = [ReviewIssue(section="results", problem=p, suggestion="fix it") for p in problems]
        return ReviewDraft(scores=scores, summary=f"score {score}", issues=issues)

    reviewing = Reviewing.__new__(Reviewing)  # consensus() only needs the config
    reviewing.cfg = ReviewerConfig(agreement=0.5)
    drafts = [
        draft("The test is not named", "Typo in line two", score=2),
        draft("The statistical test is not named", score=4),
        draft("Figures lack units", score=5),
    ]
    scores, issues, summary = reviewing.consensus(drafts)
    assert scores["soundness"] == 4 and summary == "score 4"  # median, and the median review's summary
    assert [i.problem for i in issues] == ["The test is not named"]  # raised by two of three
