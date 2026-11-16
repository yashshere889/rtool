"""Writer and Reviewer artifacts (report, final_report, review) and the small schemas their LLM calls fill in.

A report is stored twice: as section *templates* — Markdown in which every
result is a placeholder such as ``{{mean:acc:new}}`` and every citation is
``[@bib_key]`` — and rendered. The model never writes a digit of a result;
the renderer fills them in from experiment_results, so every number traces back.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..core.artifacts import artifact_type, register_artifact_type

Origin = Literal["rtool", "llm", "reviewer", "human"]
Severity = Literal["minor", "major"]
Target = Literal["report", "experiment_results", "code_bundle", "experiment_plan"]
# What the code audit asks: each condition does what the plan says, the conditions differ, evaluation is
# on held-out data, each metric is computed as defined, automatic repairs did not weaken the experiment.
AuditAspect = Literal["condition", "distinct", "held_out", "metric", "repairs"]
AuditState = Literal["ok", "problem", "unconfirmed", "unanswered"]


class ReportSection(BaseModel):
    id: str  # abstract, introduction, related_work, methods, results, discussion, conclusion, reproducibility
    title: str
    template: str  # Markdown with {{placeholders}} and [@citations]
    origin: Origin = "llm"


class Finding(BaseModel):
    """Something the Writer's audit or the Reviewer's checks found."""

    kind: str  # missing_condition, missing_metric, incomplete_runs, number, citation, contradiction, ...
    detail: str
    section: str = ""
    severity: Severity = "major"
    target: Target = "report"  # where it must be fixed; anything but "report" becomes a change request


@artifact_type("report")
class Report(BaseModel):
    title: str
    title_origin: Origin = "llm"  # "human" once the researcher retitles it (the Writer then keeps it)
    sections: list[ReportSection]
    markdown_path: str  # rendered report, run-relative
    bibtex_path: str  # entries for the cited papers only
    citations: list[str] = Field(default_factory=list)  # bib keys, in order of first use
    figures: list[str] = Field(default_factory=list)  # run-relative
    findings: list[Finding] = Field(default_factory=list)
    numbers: int = 0  # placeholders filled from the results
    # the hypothesis's success criterion applied to the results ("" if there was no criterion to apply)
    hypothesis_outcome: Literal["", "supported", "refuted", "inconclusive"] = ""
    hypothesis_checks: list[str] = Field(default_factory=list)  # each condition of the criterion, measured

    def section(self, section_id: str) -> ReportSection | None:
        return next((s for s in self.sections if s.id == section_id), None)


register_artifact_type("final_report", Report)  # the report after the Reviewer's edits


class Check(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class ReviewIssue(BaseModel):
    section: str = Field(description="Section id, e.g. 'results' or 'discussion'")
    problem: str
    suggestion: str = Field(description="What to change, concretely")
    target: Target = Field(
        default="report",
        description="'report' if editing the text fixes it; otherwise the artifact that must change",
    )
    severity: Severity = "minor"


class AuditFinding(BaseModel):
    """One question of the code audit after the vote of its samples."""

    aspect: AuditAspect
    subject: str = ""  # the condition or metric asked about
    label: str  # e.g. "metric `acc`"
    state: AuditState  # problem: raised by most audits; unconfirmed: by some; unanswered: most gave no answer
    answered: int = 0  # audits that answered this question
    raised_by: int = 0  # audits that answered "not ok"
    problem: str = ""
    fix: str = ""
    severity: Severity = "minor"


@artifact_type("review")
class Review(BaseModel):
    verdict: Literal["accept", "needs_human"]
    summary: str
    scores: dict[str, float] = Field(default_factory=dict)  # median over the judge samples, 1-5
    checks: list[Check] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)
    code_audit: list[AuditFinding] = Field(default_factory=list)
    edited_sections: list[str] = Field(default_factory=list)
    change_requests: list[str] = Field(default_factory=list)
    review_path: str = ""  # review.md, run-relative
    diff_path: str = ""  # report -> final report, unified diff


# ------------------------------------------------------------------ LLM outputs
class TitleAbstract(BaseModel):
    title: str = Field(description="Informative title, at most 15 words")
    abstract: str = Field(description="One paragraph; results only as {{placeholders}}")


class HypothesisVerdict(BaseModel):
    """The success criterion, condition by condition, against the computed results (reasons before the verdict)."""

    checks: list[str] = Field(
        min_length=1,
        description="One per condition in the success criterion: the threshold, the measured value it applies to, "
        "and whether it holds",
    )
    outcome: Literal["inconclusive", "supported", "refuted"]

    @field_validator("checks")
    @classmethod
    def _sentences(cls, checks: list[str]) -> list[str]:
        # seen on Barkla: checks == [":{"] after a reasoning pass; a check must say something
        if any(len(c.split()) < 4 for c in checks):
            raise ValueError("each check must be a sentence: the threshold, the measured value, whether it holds")
        return checks


class Contradiction(BaseModel):
    section: str
    statement: str = Field(description="The sentence that is wrong, copied")
    problem: str = Field(description="What the results actually show")


class Contradictions(BaseModel):
    items: list[Contradiction] = Field(default_factory=list, max_length=8)


class Rubric(BaseModel):
    soundness: int = Field(ge=1, le=5, description="Do the conclusions follow from the results and statistics?")
    evidence: int = Field(ge=1, le=5, description="Are claims about prior work supported by the cited papers?")
    clarity: int = Field(ge=1, le=5)
    reproducibility: int = Field(ge=1, le=5, description="Could someone rerun this from the report and code?")
    limitations: int = Field(ge=1, le=5, description="Are limitations and threats to validity discussed honestly?")


class ReviewIssueDraft(ReviewIssue):
    """What a reviewer must say about an issue: no default severity, so "minor" is never just left in."""

    severity: Severity = Field(
        description="'major' if a reader would be misled: a claim contradicts the facts (e.g. a result reported "
        "in the wrong direction), the statistics shown are invalid, values are missing, or the results may not "
        "come from a real experiment; 'minor' for wording, clarity and small omissions"
    )


class ReviewDraft(BaseModel):
    scores: Rubric
    summary: str = Field(description="Two or three sentences for the researcher")
    issues: list[ReviewIssueDraft] = Field(default_factory=list, max_length=10)


class AuditAnswer(BaseModel):
    aspect: AuditAspect
    subject: str = Field(default="", description="The condition or metric name the question gives; else empty")
    ok: bool = Field(description="false only for a real problem you can point to in the code")
    problem: str = Field(default="", description="If not ok: what the code does wrong, naming the lines")
    fix: str = Field(default="", description="If not ok: what to change in experiment.py")
    severity: Severity = Field(default="minor", description="'major' only if the results cannot be trusted as is")


class CodeAuditDraft(BaseModel):
    answers: list[AuditAnswer] = Field(default_factory=list, max_length=30, description="One per question")
