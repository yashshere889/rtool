"""Scoring a validity run: which levels it reached, read from what is on disk, with no model asked.

    L0 runs             every stage finished and nobody had to act on the run
    L1 code consistent  every planned run of the pinned conditions succeeded, on the code that was
                        audited, with no sign of a fake experiment
    L2 analysis sound   the plan has no open issues and uses the brief's seeds and primary metric, and the
                        comparison (or summary) that decides the case was computed
    L3 right answer     that comparison or summary matches the truth, and so does the final report

Levels are checked in order and scoring stops at the first that fails: its first failed check names
the failure category. The Reviewer's code audit is counted next to the levels but decides nothing,
because it is a model's judgement.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from ..core.artifacts import ArtifactStore
from ..core.engine import load_status
from ..core.errors import RToolError
from ..core.fs import read_json
from ..core.run import RunHandle
from ..core.state import RunState, StageState, StageStatus
from ..schemas.code import CodeBundle, Comparison, ExperimentResults, MetricSummary
from ..schemas.plan import ExperimentPlan
from ..schemas.report import Report, Review
from .cases import Case, CaseBrief, CaseKind, Expectation
from .runs import MARKER

LEVELS = ("L0 runs", "L1 code consistent", "L2 analysis sound", "L3 right answer")
_COMPARISON_FACTS = {"difference", "ci_low", "ci_high", "test", "p_value", "p_adjusted", "significant", "better", "n"}


class Check(BaseModel):
    level: int
    name: str
    ok: bool
    detail: str = ""
    category: str = ""  # what kind of failure this is, when not ok


class RunScore(BaseModel):
    run_id: str
    case: str
    kind: CaseKind
    variant: str
    repeat: int
    commit: str
    finished: bool  # False: the run is still going (or stopped without failing); nothing is scored
    max_level: int = 3  # open cases fix no answer: only L0 and L1 are scored
    confirmed: bool = True  # the case's expected outcome was confirmed (cards report unconfirmed cases apart)
    level: int = -1  # the highest level passed together with every level below it; -1: not even L0
    category: str = ""  # the first failed check's category; "" if every scored level passed
    false_claim: bool = False  # a null case whose final report says the hypothesis is supported
    checks: list[Check] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)  # the numbers behind the verdict, for the card
    audit: dict[str, int] = Field(default_factory=dict)  # the Reviewer's code audit, by state


def score_run(run: RunHandle) -> RunScore:
    """Score one run of a validity case against the case as it was when the run was created."""
    path = run.root / MARKER
    if not path.is_file():
        raise RToolError(f"{run.root} is not a validity run (no {MARKER})")
    marker = read_json(path)
    # shortcut: the frozen case is checked again by today's rules, so a run whose case a later commit's
    # rules reject cannot be scored; give such runs a category of their own once PR 3 scores across commits.
    case = Case.model_validate(marker["case"])
    expect = case.expectation(marker["variant"])
    brief = case.brief_for(marker["variant"])  # the variant's conditions, metric and seeds
    run_state, stages, _ = load_status(run)
    score = RunScore(
        run_id=run.run_id,
        case=case.id,
        kind=case.kind,
        variant=marker["variant"],
        repeat=marker["repeat"],
        commit=marker["commit"],
        finished=_at_rest(stages),
        max_level=1 if case.kind == "open" else 3,
        confirmed=case.confirmed,
    )
    if not score.finished:
        return score
    store = ArtifactStore(run.artifacts_dir)

    def head(name: str) -> Any:
        return store.get(name) if store.head(name) is not None else None

    plan: ExperimentPlan | None = head("experiment_plan")
    results: ExperimentResults | None = head("experiment_results")
    code: CodeBundle | None = head("code_bundle")
    final: Report | None = head("final_report")
    review: Review | None = head("review")
    comparison = _comparison(brief, expect, results)
    summary = _summary(brief, expect, results)
    score.facts = _facts(plan, comparison, summary, final, stages)
    if review is not None:
        score.audit = dict(Counter(f.state for f in review.code_audit))
    score.false_claim = case.kind == "null" and final is not None and final.hypothesis_outcome == "supported"

    levels = [
        lambda: _l0(stages, run_state),
        lambda: _l1(brief, plan, results, code),
        lambda: _l2(brief, expect, plan, results, comparison, summary),
        lambda: _l3(expect, comparison, summary, final),
    ]
    if case.kind == "synthesis":  # a literature review: its own three levels after L0
        review, corpus = head("synthesis"), head("corpus")
        score.facts.update(_synthesis_facts(review, expect, corpus))
        levels = [levels[0], *(lambda n=n: _synthesis_level(n, review, expect, corpus) for n in (1, 2, 3))]
    for level, checks in enumerate(levels[: score.max_level + 1]):
        found = checks()
        score.checks += found
        failed = next((c for c in found if not c.ok), None)
        if failed is not None:
            score.category = failed.category
            break
        score.level = level
    return score


# ------------------------------------------------------------------ lookups
def _at_rest(stages: dict[str, StageState]) -> bool:
    """Done, or stopped where only a person could move it on; anything else is still going."""
    statuses = {st.status for st in stages.values()}
    return statuses == {StageStatus.DONE} or bool(statuses & {StageStatus.FAILED, StageStatus.AWAITING_GATE})


def _baseline(brief: CaseBrief) -> str | None:
    return next((c.name for c in brief.conditions if c.baseline), None)


def _comparison(brief: CaseBrief, expect: Expectation, results: ExperimentResults | None) -> Comparison | None:
    """The confirmatory comparison that decides the case: its condition against the pinned baseline."""
    metric = brief.primary_metric
    if results is None or metric is None or expect.condition is None:
        return None
    return next(
        (
            c
            for c in results.comparisons
            if c.metric == metric.name
            and c.condition == expect.condition
            and c.baseline == _baseline(brief)
            and c.analysis == "confirmatory"
        ),
        None,
    )


def _summary(brief: CaseBrief, expect: Expectation, results: ExperimentResults | None) -> MetricSummary | None:
    metric = brief.primary_metric
    if results is None or metric is None or expect.value is None:
        return None
    return results.summary(expect.value.condition, metric.name)


def _facts(
    plan: ExperimentPlan | None,
    comparison: Comparison | None,
    summary: MetricSummary | None,
    final: Report | None,
    stages: dict[str, StageState],
) -> dict[str, Any]:
    facts: dict[str, Any] = {"failures": {sid: st.failures for sid, st in stages.items() if st.failures}}
    if plan is not None:
        facts["conditions"] = [c.name for c in plan.run_conditions()]
        facts["primary_metric"] = plan.primary_metric().name
        facts["seeds"] = len(plan.seeds)
    if comparison is not None:
        facts["comparison"] = comparison.model_dump(include=_COMPARISON_FACTS)
    if summary is not None:
        facts["mean"] = summary.mean
    if final is not None:
        facts["outcome"] = final.hypothesis_outcome
    return facts


def _found(expect: Expectation, review: Any, corpus: Any) -> list[bool]:
    """For each expected finding, whether a kept sentence citing a paper of that title states its terms."""
    titles = {p.bib_key: p.title.lower() for p in corpus.papers if p.bib_key} if corpus is not None else {}
    found = []
    for finding in expect.findings:
        papers = {key for key, title in titles.items() if finding.paper.lower() in title}  # "" matches every title
        found.append(
            any(
                set(s.papers) & papers and all(term.lower() in s.text.lower() for term in finding.terms)
                for s in review.kept()
            )
        )
    return found


def _synthesis_facts(review: Any, expect: Expectation, corpus: Any) -> dict[str, Any]:
    if review is None:
        return {}
    found = _found(expect, review, corpus)
    return {
        "kept_sentences": len(review.kept()),
        "dropped_sentences": len(review.dropped),
        "findings_used": len({c for s in review.kept() for c in s.claims}),
        "findings_offered": len(review.claims),
        "recall": round(sum(found) / len(found), 4) if found else 0.0,
    }


def _synthesis_level(level: int, review: Any, expect: Expectation, corpus: Any) -> list[Check]:
    """L1: every sentence rests on findings the review lists; L2: its numbers and papers match those findings
    (checked again here, not taken from the agent); L3: it states the known findings."""
    from ..agents.synthesis.agent import numbers  # noqa: PLC0415  (agents import the scorer's neighbours)

    if review is None:
        return [Check(level=level, name="the review was written", ok=False, category="unsupported_review")]
    kept = review.kept()
    if level == 1:
        loose = [s.text for s in kept if not s.claims or any(c not in review.claims for c in s.claims)]
        return [
            Check(level=1, name="the review has sentences", ok=bool(kept), category="unsupported_review"),
            Check(
                level=1,
                name="every sentence rests on listed findings",
                ok=not loose,
                detail=loose[0][:150] if loose else "",
                category="unsupported_review",
            ),
        ]
    if level == 2:
        wrong = []
        for s in kept:
            cited = [review.claims[c] for c in s.claims]
            extra = numbers(s.text) - numbers(" ".join(f"{c.text} {c.quote}" for c in cited))
            if extra or set(s.papers) != {c.bib_key for c in cited}:
                wrong.append(s.text)
        return [
            Check(
                level=2,
                name="numbers and citations match the findings",
                ok=not wrong,
                detail=wrong[0][:150] if wrong else "",
                category="unfaithful_review",
            )
        ]
    found = _found(expect, review, corpus)
    missed = [f.paper for f, ok in zip(expect.findings, found, strict=True) if not ok]
    recall = sum(found) / len(found) if found else 0.0
    return [
        Check(
            level=3,
            name=f"the review states at least {expect.min_recall:.0%} of the known findings",
            ok=recall >= expect.min_recall,
            detail=f"{sum(found)} of {len(found)}" + (f"; missed: {', '.join(missed[:3])}" if missed else ""),
            category="missed_findings",
        )
    ]


def _first_line(text: str | None) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0][:200] if lines else ""


# ------------------------------------------------------------------- levels
def _l0(stages: dict[str, StageState], run_state: RunState | None) -> list[Check]:
    stopped = [(sid, st) for sid, st in stages.items() if st.status is not StageStatus.DONE]
    detail, category = "", ""
    if stopped:
        sid, st = next(((s, t) for s, t in stopped if t.status is StageStatus.FAILED), stopped[0])
        if st.status is StageStatus.FAILED:
            detail, category = f"{sid} failed: {_first_line(st.last_error)}", f"stage_failed:{sid}"
        else:
            gate = st.gate.gate_id if st.gate is not None else "?"
            detail, category = f"{sid} waits at its {gate} checkpoint", f"stalled_at_gate:{gate}"
    acted = run_state.processed_actions if run_state is not None else []
    return [
        Check(level=0, name="every stage finished", ok=not stopped, detail=detail, category=category),
        Check(
            level=0,
            name="nobody acted on the run",
            ok=not acted,
            detail=f"{len(acted)} human action(s) applied" if acted else "",
            category="human_intervened",
        ),
    ]


def _l1(
    brief: CaseBrief, plan: ExperimentPlan | None, results: ExperimentResults | None, code: CodeBundle | None
) -> list[Check]:
    if plan is None or results is None or code is None:
        return [Check(level=1, name="the experiment left results", ok=False, category="incomplete_runs")]
    failed = [f"{r.condition}/seed {r.seed}: {r.status}" for r in results.runs if r.status != "ok"]
    checks = [
        Check(
            level=1,
            name="every planned run succeeded",
            ok=results.complete,
            detail="; ".join(failed[:5]),
            category="incomplete_runs",
        )
    ]
    planned = [c.name for c in plan.run_conditions()]
    units = plan.replicates == "units"
    for pinned in brief.conditions:
        ok_runs = sum(1 for r in results.runs if r.condition == pinned.name and r.status == "ok")
        if pinned.name not in planned:
            checks.append(
                Check(
                    level=1,
                    name=f"condition {pinned.name} was run",
                    ok=False,
                    detail=f"the plan's conditions are {planned}",
                    category="missing_condition",
                )
            )
        elif units:  # a data analysis: one run per condition, results per unit
            checks.append(
                Check(
                    level=1,
                    name=f"condition {pinned.name} has results for at least two units",
                    ok=ok_runs >= 2,
                    detail=f"{ok_runs} {plan.unit or 'unit'}s",
                    category="incomplete_runs",
                )
            )
        else:
            checks.append(
                Check(
                    level=1,
                    name=f"condition {pinned.name} ran on every seed",
                    ok=ok_runs == len(plan.seeds),
                    detail=f"{ok_runs} of {len(plan.seeds)} seeds",
                    category="incomplete_runs",
                )
            )
    warnings = [*code.warnings, *results.warnings]
    checks += [
        Check(
            level=1,
            name="the results come from the audited code",
            ok=bool(code.fingerprint) and results.code_fingerprint == code.fingerprint,
            category="stale_results",
        ),
        Check(
            level=1,
            name="no sign of a fake experiment",
            ok=not warnings,
            detail=warnings[0] if warnings else "",
            category="fake_experiment",
        ),
    ]
    return checks


def _l2(
    brief: CaseBrief,
    expect: Expectation,
    plan: ExperimentPlan | None,
    results: ExperimentResults | None,
    comparison: Comparison | None,
    summary: MetricSummary | None,
) -> list[Check]:
    assert plan is not None and results is not None  # L1 passed, so there are both
    units = plan.replicates == "units"
    pinned = brief.primary_metric
    primary = plan.primary_metric()
    checks = [
        Check(
            level=2,
            name="the plan has no open issues",
            ok=not plan.issues,
            detail=plan.issues[0] if plan.issues else "",
            category="plan_issue",
        ),
        Check(
            level=2,
            name="the plan uses the brief's seeds",
            ok=len(plan.seeds) == (1 if units else brief.seeds),
            detail=f"{len(plan.seeds)} seeds, the brief asks for {brief.seeds}",
            category="plan_issue",
        ),
        Check(
            level=2,
            name="the primary metric is the pinned one",
            ok=pinned is not None
            and primary.name == pinned.name
            and primary.higher_is_better == pinned.higher_is_better,
            detail=f"the plan's primary metric is {primary.name} "
            f"({'higher' if primary.higher_is_better else 'lower'} is better)",
            category="plan_issue",
        ),
    ]
    if expect.significant is not None or expect.direction is not None or expect.difference is not None:
        checks.append(
            Check(
                level=2,
                name=f"{expect.condition} was tested against {_baseline(brief)}",
                ok=comparison is not None and comparison.p_value is not None,
                detail="" if comparison is None else comparison.verdict(),
                category="missing_comparison",
            )
        )
    if expect.value is not None:
        checks.append(
            Check(
                level=2,
                name=f"{expect.value.condition} was summarised on every seed",
                ok=summary is not None and (summary.n >= 2 if units else summary.n == brief.seeds),
                detail="" if summary is None else f"{summary.n} {'units' if units else 'seeds'}",
                category="missing_comparison",
            )
        )
    checks.append(
        Check(
            level=2,
            name="the data show no design problem",
            ok=not results.issues,
            detail=results.issues[0] if results.issues else "",
            category="design_problem",
        )
    )
    return checks


def _l3(
    expect: Expectation, comparison: Comparison | None, summary: MetricSummary | None, final: Report | None
) -> list[Check]:
    checks = []
    if expect.significant is not None:
        assert comparison is not None  # L2 checked that it exists
        checks.append(
            Check(
                level=3,
                name="significant" if expect.significant else "not significant",
                ok=comparison.significant == expect.significant,
                detail=comparison.verdict(),
                category="wrong_analysis",
            )
        )
    if expect.direction is not None:
        assert comparison is not None
        checks.append(
            Check(
                level=3,
                name=f"the condition is {expect.direction} than the baseline",
                ok=comparison.better is (expect.direction == "better"),
                detail=comparison.verdict(),
                category="wrong_analysis",
            )
        )
    if expect.difference is not None:
        assert comparison is not None
        band = expect.difference
        checks.append(
            Check(
                level=3,
                name=f"the difference is between {band.low:g} and {band.high:g}",
                ok=band.low <= comparison.difference <= band.high,
                detail=f"difference {comparison.difference:.4g}",
                category="wrong_analysis",
            )
        )
    if expect.value is not None:
        assert summary is not None
        value = expect.value
        checks.append(
            Check(
                level=3,
                name=f"the mean of {value.condition} is between {value.low:g} and {value.high:g}",
                ok=value.low <= summary.mean <= value.high,
                detail=f"mean {summary.mean:.4g}",
                category="wrong_analysis",
            )
        )
    if expect.outcome:
        said = final.hypothesis_outcome if final is not None else ""
        checks.append(
            Check(
                level=3,
                name=f"the report concludes {' or '.join(expect.outcome)}",
                ok=said in expect.outcome,
                detail=f"it concludes {said or 'nothing'}",
                category="misread_results",
            )
        )
    return checks
