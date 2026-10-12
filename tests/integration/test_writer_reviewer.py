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
from research_tool.schemas.report import ReviewDraft, ReviewIssueDraft, Rubric
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
    numbers = [f for f in report.findings if f.kind == "number"]
    assert numbers and all(f.severity == "minor" for f in numbers)
    # what is left in mid-sentence stays visible, and is a major finding until someone fixes it
    unresolved = [f for f in report.findings if f.kind == "placeholder"]
    assert unresolved and all(f.severity == "major" and "show as [?]" in f.detail for f in unresolved)
    text = (run.root / report.markdown_path).read_text()
    assert "0.99" not in text and "It reached [?]." in text


def test_an_unchecked_number_in_brackets_is_left_out(runs_root: Path) -> None:
    run = seed(runs_root, stages=[{"id": "report", "agent": "writer"}])
    aside = lambda r: honest(r) + " The effect was clear (p < 0.001)."  # noqa: E731
    Engine(run, services(run, writer_llm(section=aside, fix=aside))).advance()
    report = ArtifactStore(run.artifacts_dir).get("report")
    assert all(f.severity == "minor" for f in report.findings)
    text = (run.root / report.markdown_path).read_text()
    assert "The effect was clear." in text and "[?]" not in text and "0.001" not in text


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
    issue = {
        "section": "results",
        "problem": "Name the test",
        "suggestion": "say paired t-test",
        "target": "report",
        "severity": "minor",
    }
    result = Engine(run, services(run, writer_llm(issues=[issue]))).advance()
    assert result.outcome is Outcome.WAITING_GATE and result.open_gates[0]["gate"] == "final_report"
    store = ArtifactStore(run.artifacts_dir)
    review = store.get("review")
    assert review.verdict == "accept" and review.edited_sections == ["results"]
    assert "Edited." in (run.root / store.get("final_report").markdown_path).read_text()
    queue(run, ActionKind.APPROVE)
    assert Engine(run, services(run, writer_llm())).advance().outcome is Outcome.DONE


def test_rejecting_the_final_report_sends_it_back_to_the_writer(runs_root: Path) -> None:
    run = seed(runs_root)
    assert Engine(run, services(run, writer_llm())).advance().outcome is Outcome.WAITING_GATE
    queue(run, ActionKind.REJECT, payload={"feedback": "Shorten the discussion"})
    llm = writer_llm()
    result = Engine(run, services(run, llm)).advance()
    assert result.outcome is Outcome.WAITING_GATE and result.open_gates[0]["gate"] == "final_report"
    from research_tool.core.engine import load_status

    _, stages, _ = load_status(run)
    assert stages["report"].attempt == 2 and stages["review"].attempt == 3  # placeholder round, then a real review
    drafts = [r for r in llm.calls if "section of a research report" in r.messages[-1].content]
    assert drafts and all("Shorten the discussion" in r.messages[-1].content for r in drafts)  # the Writer heard it
    reviews = [r for r in llm.calls if "Review this research report" in r.messages[-1].content]
    assert len(reviews) == 3  # one full review (3 samples), none wasted on the rejected version
    review = ArtifactStore(run.artifacts_dir).get("review")
    assert review.verdict == "accept" and not review.summary.startswith("Sent back")
    assert "review_returned_to_writer" in [e["kind"] for e in EventLog(run.events_path, run.run_id)]


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
        issues = [
            ReviewIssueDraft(section="results", problem=p, suggestion="fix it", severity="minor") for p in problems
        ]
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


def section_value(report: Any, section_id: str, template: str) -> dict:
    value = report.model_dump(mode="json")
    next(s for s in value["sections"] if s["id"] == section_id)["template"] = template
    return value


def test_hand_edits_are_checked_reviewed_again_and_kept_by_the_writer(runs_root: Path) -> None:
    from research_tool.core.actions import ActionInbox
    from research_tool.core.engine import load_status

    run = seed(runs_root)
    assert Engine(run, services(run, writer_llm())).advance().outcome is Outcome.WAITING_GATE
    store = ArtifactStore(run.artifacts_dir)
    report = store.get("report")
    mine = "In our hands {{mean:acc:new}} beat the baseline, as expected [@smith2020].\n\nLimitations. Only 12 nights."
    edit = {"artifact": "report", "base_version": 1, "value": section_value(report, "discussion", mine)}
    queue(run, ActionKind.EDIT, payload={**edit, "value": section_value(report, "methods", "My methods")})
    queue(run, ActionKind.EDIT, payload={**edit, "value": section_value(report, "discussion", "See [@nobody].")})
    queue(run, ActionKind.EDIT, payload=edit)
    assert Engine(run, services(run, writer_llm())).advance().outcome is Outcome.WAITING_GATE
    outcomes = [p["outcome"] for p in ActionInbox(run.actions_dir).processed()]
    assert "written by rtool" in outcomes[0]["error"] and "unknown citation key @nobody" in outcomes[1]["error"]
    assert outcomes[2]["status"] == "applied" and outcomes[2]["stale"] == ["review"]

    edited = store.get("report")
    discussion = edited.section("discussion")
    assert discussion is not None and discussion.origin == "human" and discussion.template == mine
    assert any(f.kind == "number" and "12" in f.detail for f in edited.findings)
    markdown = (run.root / edited.markdown_path).read_text()
    assert "In our hands" in markdown and "{{" not in markdown  # rendered again, numbers filled in
    _, stages, _ = load_status(run)
    assert stages["review"].attempt == 2  # the edited report was reviewed again
    assert not next(c for c in store.get("review").checks if c.name == "numbers trace to the results").ok

    final = store.get("final_report")  # the researcher's last word, at the open checkpoint
    value = section_value(final, "conclusion", "We conclude that {{mean:acc:new}} is better.")
    queue(run, ActionKind.EDIT, payload={"artifact": "final_report", "base_version": 2, "value": value})
    Engine(run, services(run, writer_llm())).advance()
    _, stages, _ = load_status(run)
    assert stages["review"].gate.summary["artifacts"]["final_report"] == 3  # type: ignore[union-attr]

    queue(run, ActionKind.REJECT, payload={"feedback": "Tighten the introduction"})  # back to the Writer
    llm = writer_llm()
    assert Engine(run, services(run, llm)).advance().outcome is Outcome.WAITING_GATE
    again = store.get("report")
    assert again.section("discussion").template == mine and again.section("discussion").origin == "human"  # type: ignore[union-attr]
    drafted = [r.messages[-1].content for r in llm.calls if "section of a research report" in r.messages[-1].content]
    assert drafted and not any('"Discussion"' in text for text in drafted)  # the researcher's section was not redrafted
    assert "report_human_sections_kept" in [e["kind"] for e in EventLog(run.events_path, run.run_id)]


def test_the_reviewer_rewrites_sections_with_unchecked_values(runs_root: Path) -> None:
    run = seed(runs_root)
    stubborn = lambda r: honest(r) + " It reached 0.99."  # noqa: E731
    Engine(run, services(run, writer_llm(section=stubborn, fix=stubborn))).advance()
    store = ArtifactStore(run.artifacts_dir)
    assert "[?]" in (run.root / store.get("report").markdown_path).read_text()
    review = store.get("review")
    unresolved = [i for i in review.issues if "shown as [?]" in i.problem]
    assert unresolved and all(i.severity == "major" for i in unresolved)
    assert set(review.edited_sections) >= {i.section for i in unresolved}  # rewritten: "... Edited."
    final = (run.root / store.get("final_report").markdown_path).read_text()
    assert "[?]" not in final
    check = next(c for c in review.checks if c.name == "no unresolved values ([?])")
    assert check.ok  # judged on the final report, after the edits
