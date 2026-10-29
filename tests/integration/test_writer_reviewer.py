"""Writer and Reviewer on hand-made inputs: repairs, findings, change requests, consensus, verdicts."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

from research_tool.agents.coder.results import aggregate
from research_tool.agents.reviewer.agent import ReviewerConfig, Reviewing, shown_code
from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.engine import Engine, Outcome
from research_tool.core.events import EventLog
from research_tool.core.run import RunHandle
from research_tool.core.services import Services
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.types import ChatRequest
from research_tool.schemas.code import CodeBundle, CodeFile, ExperimentResults, FixRecord
from research_tool.schemas.literature import Claim, Corpus, KnowledgeMap, PaperRecord, Topic
from research_tool.schemas.report import AuditAnswer, CodeAuditDraft, ReviewDraft, ReviewIssueDraft, Rubric
from research_tool.testing.harness import make_run, queue
from research_tool.testing.samples import sample_plan as plan
from research_tool.testing.samples import sample_records as records

INITIAL = ["research_brief", "corpus", "knowledge_map", "experiment_plan", "experiment_results", "code_bundle"]
EXPERIMENT = """import random

ACCURACY = {"base": 0.65, "new": 0.72}


def run(condition, seed, resources, smoke, workdir):
    acc = ACCURACY[condition] + random.Random(seed).uniform(-0.02, 0.02)
    return {"acc": acc, "loss": 1 - acc}
"""
QUESTION = re.compile(r'^\d+\. aspect "(\w+)"(?:, subject "([^"]*)")?', re.M)


def auditor(problems: dict[tuple[str, str], dict] | None = None, *, skip: tuple = ()) -> Any:
    """Mock code audits answering every question ok, except ``problems``: each is raised by ``votes`` of 3 audits."""
    calls = itertools.count()

    def reply(request: ChatRequest) -> dict:
        index, answers = next(calls) % 3, []
        for key in QUESTION.findall(request.messages[-1].content):
            problem = dict((problems or {}).get(key, {"votes": 0}))
            raised = index < problem.pop("votes", 3)
            if key not in skip:
                answers.append({"aspect": key[0], "subject": key[1], "ok": not raised, **(problem if raised else {})})
        return {"answers": answers}

    return reply


def is_audit(request: ChatRequest) -> bool:
    return request.messages[-1].content.startswith("You are auditing the code")


def seed(
    runs_root: Path,
    *,
    runs: list | None = None,
    stages: list[dict] | None = None,
    fixes: list | None = None,
    primary: str = "acc",
    files: list | None = None,
) -> RunHandle:
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
    (run.root / "code").mkdir()
    (run.root / "code" / "experiment.py").write_text(EXPERIMENT)
    (run.root / "code" / "config.json").write_text(json.dumps({"seeds": [0, 1, 2], "resources": {"data": "/x"}}))
    p = plan(primary=primary)
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
    bundle = CodeBundle(root="code", files=files or [], pack="generic", readme_path="README.md", fixes=fixes or [])
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


VERDICT = {
    "checks": ["The criterion asks for a gain of at least 20%; the gain was +10.3%: not met"],
    "outcome": "refuted",
}


def writer_llm(
    section: Any = honest,
    fix: Any = honest,
    issues: list[dict] | None = None,
    audit: Any = None,
    verdict: dict = VERDICT,
) -> MockLLM:
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
        .add(lambda r: r.messages[-1].content.startswith("Decide what the experiment says"), verdict)
        .add(lambda r: "Check the report sections below" in r.messages[-1].content, {"items": []})
        .add(lambda r: "Review this research report" in r.messages[-1].content, review)
        .add(is_audit, audit or auditor())
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


def test_the_verdict_comes_from_the_success_criterion_not_from_significance(runs_root: Path) -> None:
    """Seen on Barkla: a significant 7.5% cut was written up as supporting a hypothesis that asked for 30%."""
    run = seed(runs_root)
    llm = writer_llm()
    Engine(run, services(run, llm)).advance()
    store = ArtifactStore(run.artifacts_dir)
    report = store.get("report")
    assert report.hypothesis_outcome == "refuted" and report.hypothesis_checks == VERDICT["checks"]
    prompts = [r.messages[-1].content for r in llm.calls]
    judge = next(t for t in prompts if t.startswith("Decide what the experiment says"))
    # the criterion and the size of the change, with numbers: significance alone cannot decide it
    assert "## Success criterion" in judge and "relative change +10.3%" in judge and "i.e. +6.7 percentage" in judge
    assert prompts.index(judge) == 0  # before any prose, so every section states the same verdict
    for title in ("Results", "Discussion", "Conclusion"):  # each can state the verdict and the size of the change
        prompt = next(t for t in prompts if t.startswith(f'Write the "{title}" section'))
        assert "Verdict on the hypothesis, from its success criterion: REFUTED" in prompt
        assert "{{rel:acc:new}}" in prompt and "Success criterion (fixed before the experiment ran)" in prompt
    review = next(t for t in prompts if t.startswith("Review this research report"))
    assert "Verdict on the hypothesis, from its success criterion: REFUTED" in review


def test_code_written_by_the_researcher_is_disclosed(runs_root: Path) -> None:
    hand = CodeFile(path="experiment.py", sha256="x", lines=9, origin="human")
    fixes = [FixRecord(round=1, phase="smoke", signature="traceback", action="rewrote experiment.py (LLM)")]
    run = seed(runs_root, stages=[{"id": "report", "agent": "writer"}], files=[hand], fixes=fixes)
    llm = writer_llm()
    Engine(run, services(run, llm)).advance()
    report = ArtifactStore(run.artifacts_dir).get("report")
    repro = report.section("reproducibility").template
    assert "Written or edited by the researcher: `experiment.py`" in repro
    assert "Automatic repairs before the researcher's edits: 1" in repro
    discussion = next(r.messages[-1].content for r in llm.calls if '"Discussion"' in r.messages[-1].content)
    assert "The researcher wrote or edited the experiment code by hand (experiment.py)" in discussion


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
    llm = writer_llm(issues=[issue])
    result = Engine(run, services(run, llm)).advance()
    assert result.outcome is Outcome.WAITING_GATE and result.open_gates[0]["gate"] == "final_report"
    store = ArtifactStore(run.artifacts_dir)
    review = store.get("review")
    assert review.verdict == "accept" and review.edited_sections == ["results"]
    assert "Edited." in (run.root / store.get("final_report").markdown_path).read_text()
    # the code audit: three samples saw the code, config.json and the plan, and every answer was ok
    audits = [r.messages[-1].content for r in llm.calls if is_audit(r)]
    assert len(audits) == 3 and len(set(audits)) == 1
    prompt = audits[0]
    assert "   7      acc = ACCURACY[condition]" in prompt and '"resources": {"data": "/x"}' in prompt
    assert "- `acc` (higher is better): accuracy" in prompt and "- `base` (baseline): b" in prompt
    assert "## Automatic repairs made while the code was being tested\nNone." in prompt
    asked = QUESTION.findall(prompt)
    assert asked == [("condition", "base"), ("condition", "new"), ("distinct", ""), ("held_out", "")] + [
        ("metric", "acc"),
        ("metric", "loss"),
    ]  # no repairs, so no question about them
    assert {f.state for f in review.code_audit} == {"ok"} and len(review.code_audit) == 6
    assert next(c for c in review.checks if c.name == "code matches the plan").ok
    markdown = (run.root / review.review_path).read_text()
    assert "## Code audit" in markdown and "- ok: condition `new`" in markdown and "- ok: metric `loss`" in markdown
    assert "code_audit.j2" in store.envelope("review").meta["prompts"]
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


def test_code_audit_sends_major_problems_to_the_coder(runs_root: Path) -> None:
    repair = FixRecord(round=1, phase="smoke", signature="traceback", action="rewrote experiment.py (LLM)", detail="E")
    run = seed(runs_root, fixes=[repair])
    leak = {"problem": "Line 7 tunes on the test split", "fix": "tune on a validation split", "severity": "major"}
    llm = writer_llm(audit=auditor({("held_out", ""): {**leak, "votes": 2}}))
    Engine(run, services(run, llm)).advance()
    review = ArtifactStore(run.artifacts_dir).get("review")
    prompt = next(r.messages[-1].content for r in llm.calls if is_audit(r))
    assert "1. smoke · traceback: rewrote experiment.py (LLM) — E" in prompt
    assert ("repairs", "") in QUESTION.findall(prompt)  # the repair history is audited too
    held_out = next(f for f in review.code_audit if f.aspect == "held_out")
    assert (held_out.state, held_out.severity, held_out.raised_by) == ("problem", "major", 2)
    check = next(c for c in review.checks if c.name == "code matches the plan")
    assert not check.ok and "tunes on the test split" in check.detail
    assert review.verdict == "needs_human"
    assert review.change_requests == ["code_bundle: code audit — held-out evaluation"]
    raised = [e["data"] for e in EventLog(run.events_path, run.run_id) if e["kind"] == "change_request_raised"]
    assert [r["artifact"] for r in raised] == ["code_bundle"] and "tunes on the test split" in raised[0]["reason"]
    rubric = next(r.messages[-1].content for r in llm.calls if "Review this research report" in r.messages[-1].content)
    assert "code matches the plan: PROBLEM" in rubric  # the rubric reviews know what the audit found
    markdown = (run.root / review.review_path).read_text()
    assert (
        "- PROBLEM [major] held-out evaluation: Line 7 tunes on the test split — tune on a validation split "
        "(raised by 2 of 3 audits; sent back to code_bundle)" in markdown
    )
    assert "- ok: automatic repairs" in markdown


def test_code_audit_needs_a_majority_and_minor_problems_stay_open(runs_root: Path) -> None:
    run = seed(runs_root)
    loss = {"problem": "loss is 1 - acc, not the model's loss", "fix": "compute the loss", "severity": "minor"}
    leak = {"problem": "might tune on test data", "severity": "major", "votes": 1}
    Engine(run, services(run, writer_llm(audit=auditor({("metric", "loss"): loss, ("held_out", ""): leak})))).advance()
    review = ArtifactStore(run.artifacts_dir).get("review")
    assert review.verdict == "accept" and not review.change_requests  # nothing major was confirmed
    check = next(c for c in review.checks if c.name == "code matches the plan")
    assert check.ok and "not the model's loss" in check.detail
    markdown = (run.root / review.review_path).read_text()
    assert "- PROBLEM [minor] metric `loss`: loss is 1 - acc, not the model's loss — compute the loss" in markdown
    assert "(raised by 3 of 3 audits; open)" in markdown
    assert "- not confirmed: held-out evaluation: might tune on test data (raised by 1 of 3 audits)" in markdown


def test_code_audit_fails_closed(runs_root: Path) -> None:
    run = seed(runs_root)  # most audits skip a question: it was not checked
    Engine(run, services(run, writer_llm(audit=auditor(skip=(("held_out", ""),))))).advance()
    review = ArtifactStore(run.artifacts_dir).get("review")
    check = next(c for c in review.checks if c.name == "code matches the plan")
    assert review.verdict == "needs_human" and not check.ok and check.detail == "held-out evaluation: unanswered"
    assert (
        "- unanswered: held-out evaluation (answered by 0 of 3 audits)" in (run.root / review.review_path).read_text()
    )

    run = seed(runs_root / "no-code")
    (run.root / "code" / "experiment.py").unlink()
    llm = writer_llm()
    Engine(run, services(run, llm)).advance()
    review = ArtifactStore(run.artifacts_dir).get("review")
    assert review.verdict == "needs_human" and not any(is_audit(r) for r in llm.calls)
    markdown = (run.root / review.review_path).read_text()
    assert "## Code audit\n\nThe code audit did not run: no experiment.py in code." in markdown

    def down(request: ChatRequest) -> str:
        raise ConnectionError("the LLM server went away")

    run = seed(runs_root / "no-llm")  # every audit fails
    Engine(run, services(run, writer_llm(audit=down))).advance()
    review = ArtifactStore(run.artifacts_dir).get("review")
    check = next(c for c in review.checks if c.name == "code matches the plan")
    assert review.verdict == "needs_human" and not check.ok and "no audit returned a valid answer" in check.detail


def test_the_report_leads_with_the_primary_outcome(runs_root: Path) -> None:
    run = seed(runs_root, primary="loss")
    llm = writer_llm()
    Engine(run, services(run, llm)).advance()
    prompts = [r.messages[-1].content for r in llm.calls]
    results = next(p for p in prompts if p.startswith('Write the "Results" section'))
    assert results.index("Primary outcome (pre-registered) `loss`") < results.index("Exploratory metric `acc`")
    assert "`new` vs `base` (confirmatory)" in results and "`new` vs `base` (exploratory)" in results
    assert "Holm-corrected {{p_adj:loss:new}}" in results  # two comparisons: corrected by default
    store = ArtifactStore(run.artifacts_dir)
    text = (run.root / store.get("report").markdown_path).read_text()
    assert text.index("**`loss`** (lower is better; primary outcome)") < text.index("**`acc`** (higher is better;")
    assert "| `new` vs `base` | confirmatory |" in text and "The primary outcome, `loss`, was fixed" in text
    review = next(p for p in prompts if p.startswith("Review this research report"))
    facts = review.split("## Facts from the experiment")[1]
    assert facts.index("Confirmatory: new vs base on loss") < facts.index("Exploratory: new vs base on acc")


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


def test_audit_answers_are_matched_to_questions_and_voted_on() -> None:
    def answer(aspect: str, subject: str = "", problem: str = "") -> AuditAnswer:
        return AuditAnswer(aspect=aspect, subject=subject, ok=not problem, problem=problem, severity="major")

    reviewing = Reviewing.__new__(Reviewing)  # audit_consensus() needs the config, the plan and the bundle
    reviewing.cfg, reviewing.plan = ReviewerConfig(agreement=0.5), plan()
    reviewing.bundle = CodeBundle(root="code", files=[], pack="generic")
    drafts = [
        CodeAuditDraft(
            answers=[
                answer("condition", "`New`", "ignores the condition"),
                answer("condition", "new"),  # a second answer from the same audit: the problem stands
                answer("metric", "acc"),
                answer("condition", "everything", "odd"),  # no such question
            ]
        ),
        CodeAuditDraft(answers=[answer("condition", "NEW", "falls back to base"), answer("metric", "acc")]),
        CodeAuditDraft(answers=[answer("condition", "new"), answer("distinct", "x")]),
    ]
    findings = reviewing.audit_consensus(reviewing.audit_questions(), drafts)
    state = {(f.aspect, f.subject): (f.state, f.answered, f.raised_by) for f in findings}
    assert state == {
        ("condition", "base"): ("unanswered", 0, 0),
        ("condition", "new"): ("problem", 3, 2),  # names are matched despite case and backticks
        ("distinct", ""): ("unanswered", 1, 0),
        ("held_out", ""): ("unanswered", 0, 0),
        ("metric", "acc"): ("ok", 2, 0),
        ("metric", "loss"): ("unanswered", 0, 0),
        ("condition", "everything"): ("unconfirmed", 1, 1),
    }
    new = findings[1]
    assert new.severity == "major" and new.problem == "ignores the condition" and new.label == "condition `new`"


def test_long_code_keeps_every_signature_and_run() -> None:
    helper = 'def helper_{i}(x):\n    """Doc {i}."""\n' + "    x += 1\n" * 40 + "    return x\n"
    source = "import random\n\n\n" + "\n\n".join(helper.format(i=i) for i in range(5))
    source += "\n\ndef run(condition, seed, resources, smoke, workdir):\n    return {'acc': helper_0(1)}\n"
    text, omitted = shown_code(source, 2000)
    assert omitted and len(text) <= 2200 and "# … lines" in text
    assert all(f"def helper_{i}(x):" in text and f'"""Doc {i}."""' in text for i in range(5))
    line = source.splitlines().index("def run(condition, seed, resources, smoke, workdir):") + 1
    assert f"{line:>4}  def run(" in text and "return {'acc': helper_0(1)}" in text  # original line numbers
    broken, cut = shown_code("x = (\n" + "y = 1\n" * 2000, 2000)  # does not parse: the middle goes
    assert cut and broken.startswith("   1  x = (") and broken.endswith("2001  y = 1") and len(broken) <= 2100
    assert shown_code(EXPERIMENT, 32000) == (
        "\n".join(f"{i:>4}  {t}" for i, t in enumerate(EXPERIMENT.splitlines(), 1)),
        0,
    )


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
