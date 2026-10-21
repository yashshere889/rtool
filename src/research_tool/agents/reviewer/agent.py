"""Reviewer: automated checks, a code audit, a rubric review by independent samples, edits, change requests.

1. Automated checks — citations exist and were read, results only enter the
   text through placeholders, every planned condition is reported, runs
   completed, limitations are discussed, the run is reproducible.
2. Code audit — ``samples`` independent audits (the ``judge`` profile) read
   experiment.py, config.json, the plan and the repair history and answer
   narrow questions: does each condition do what the plan says, do the
   conditions differ, is evaluation on held-out data, is each metric computed
   as defined, did an automatic repair weaken the experiment. A problem counts
   only if most audits raise it. Leakage or a wrong metric shows in the code,
   rarely in the report, which is all the rubric review sees.
3. Rubric review — ``samples`` independent reviews (the ``judge`` profile);
   scores are their median, and an issue counts only if most reviews raise it.
4. Edits — issues with the text are fixed by rewriting the prose sections
   under the same rules as the Writer (facts sections are never edited by
   the model). Problems with the experiment itself (major audit problems too)
   become change requests to the stage that owns it; only major ones, since
   they can cost GPU hours.
5. Verdict — ``accept`` only if every check passes (the code audit's
   included), every median score reaches ``min_score`` and nothing major is
   left open; otherwise a human decides at the final gate. The edited report
   is ``final_report``.
"""

from __future__ import annotations

import ast
import difflib
import math
import shutil
import statistics
from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...core.context import AgentContext
from ...core.fs import atomic_write_text
from ...core.registry import register_agent
from ...llm.client import LLMOutputError
from ...llm.prompts import PromptLibrary
from ...schemas.code import CodeBundle, ExperimentResults
from ...schemas.common import ResearchBrief
from ...schemas.literature import Corpus, KnowledgeMap
from ...schemas.plan import ExperimentPlan
from ...schemas.report import (
    AuditAnswer,
    AuditAspect,
    AuditFinding,
    AuditState,
    Check,
    CodeAuditDraft,
    Report,
    ReportSection,
    Review,
    ReviewDraft,
    ReviewIssue,
    Rubric,
    Severity,
)
from ...tools.report.render import (
    assemble,
    bib_keys,
    bibtex_subset,
    citation_keys,
    problems,
    reference_list,
    render,
    value_table,
)
from ..common import SYSTEM

EDITABLE = ("abstract", "introduction", "related_work", "results", "discussion", "conclusion")
RESULT_SECTIONS = ("abstract", "results", "conclusion")
AUDIT_CHECK = "code matches the plan"


class ReviewerConfig(BaseModel):
    samples: int = Field(default=3, ge=1, le=7)  # independent reviews
    agreement: float = Field(default=0.5, gt=0, le=1)  # share of reviews that must raise an issue
    min_score: float = Field(default=3.0, ge=1, le=5)  # every median rubric score must reach this to accept
    repair_rounds: int = Field(default=1, ge=0, le=3)
    max_code_chars: int = Field(default=32000, ge=2000)  # experiment.py shown to the code audit (~8k tokens)


def similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.45


def audit_label(aspect: str, subject: str) -> str:
    return {
        "condition": f"condition `{subject}`",
        "distinct": "conditions differ",
        "held_out": "held-out evaluation",
        "metric": f"metric `{subject}`",
        "repairs": "automatic repairs",
    }[aspect]


def _bodies(source: str) -> list[tuple[int, int]]:
    """Line slices of the bodies of top-level functions and classes other than ``run``, longest first."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) or node.name == "run":
            continue
        body = node.body[1:] if ast.get_docstring(node) is not None and len(node.body) > 1 else node.body
        first = body[0]
        start = min([first.lineno, *(d.lineno for d in getattr(first, "decorator_list", []))]) - 1
        spans.append((start, node.end_lineno or first.lineno))
    return sorted(spans, key=lambda span: span[0] - span[1])


def shown_code(source: str, limit: int) -> tuple[str, int]:
    """``source`` with line numbers, cut to about ``limit`` characters, and the number of lines left out.

    The bodies of the longest functions and classes go first, so every signature (and ``run``, which the
    harness calls) stays visible; if that is not enough, the middle of the file goes too.
    """
    lines = source.splitlines()
    numbered = [f"{i:>4}  {text}" for i, text in enumerate(lines, 1)]
    keep = [True] * len(lines)
    size = sum(len(n) + 1 for n in numbered)
    for start, end in _bodies(source) if size > limit else []:
        size -= sum(len(numbered[i]) + 1 for i in range(start, end) if keep[i])
        keep[start:end] = [False] * (end - start)
        if size <= limit:
            break
    if size > limit:
        kept = [i for i, k in enumerate(keep) if k]
        used, head, tail = 0, 0, len(kept)
        while head < tail and used + len(numbered[kept[head]]) < limit * 2 // 3:
            used += len(numbered[kept[head]]) + 1
            head += 1
        while tail > head and used + len(numbered[kept[tail - 1]]) < limit:
            tail -= 1
            used += len(numbered[kept[tail]]) + 1
        for i in kept[head:tail]:
            keep[i] = False
    out, omitted, i = [], 0, 0
    while i < len(lines):
        if keep[i]:
            out.append(numbered[i])
            i += 1
            continue
        j = next((k for k in range(i, len(lines)) if keep[k]), len(lines))
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        out.append(f"{'':>4}  {indent}# … lines {i + 1}-{j} omitted …")
        omitted, i = omitted + j - i, j
    return "\n".join(out), omitted


@register_agent("reviewer")
class ReviewerAgent(Agent):
    description = (
        "Checks and reviews the report, audits the experiment code, edits the text, "
        "and sends other problems back as change requests."
    )
    consumes = (
        "research_brief",
        "report",
        "experiment_plan",
        "experiment_results",
        "corpus",
        "knowledge_map",
        "code_bundle",
    )
    produces = ("review", "final_report")
    config_model = ReviewerConfig

    def run(self, ctx: AgentContext) -> None:
        Reviewing(ctx, self.config).review()


class Reviewing:
    def __init__(self, ctx: AgentContext, cfg: ReviewerConfig) -> None:
        self.ctx, self.cfg = ctx, cfg
        brief: ResearchBrief = ctx.inputs.get("research_brief")
        self.report: Report = ctx.inputs.get("report")
        self.plan: ExperimentPlan = ctx.inputs.get("experiment_plan")
        self.results: ExperimentResults = ctx.inputs.get("experiment_results")
        self.corpus: Corpus = ctx.inputs.get("corpus")
        self.km: KnowledgeMap = ctx.inputs.get("knowledge_map")
        self.bundle: CodeBundle = ctx.inputs.get("code_bundle")
        self.values = value_table(self.plan, self.results)
        self.keys = set(bib_keys(self.corpus))
        self.paper_key = {p.id: p.bib_key for p in self.corpus.papers if p.bib_key}
        self.prompts = PromptLibrary.for_agent(__package__, "reviewer", pack=brief.domain_pack)
        self.provenance: dict[str, str] = {}
        self.audit_omitted = 0  # lines of experiment.py the code audit did not see

    def render_prompt(self, name: str, **variables: Any) -> str:
        prompt = self.prompts.render(name, **variables)
        self.provenance[name] = prompt.template_hash
        return prompt.text

    # ------------------------------------------------------------ material
    def plan_text(self) -> str:
        p = self.plan
        return "\n".join([p.hypothesis, p.null_hypothesis, p.design, p.expected_results, *self.results.notes])

    def claims_text(self) -> str:
        return "\n".join(f"{c.text} {c.quote}" for c in self.km.claims if c.verified)

    def allowed_numbers(self, section: ReportSection, original: str = "") -> str:
        material = [self.plan_text(), original]
        if section.id not in RESULT_SECTIONS:
            material.append(self.claims_text())
        return "\n".join(material)

    def section_problems(self, section: ReportSection, text: str, original: str = "") -> list[str]:
        keys = self.keys if section.id != "abstract" else set()
        return problems(text, self.values, keys, allowed_numbers=self.allowed_numbers(section, original))

    def facts(self) -> str:
        lines = []
        for c in self.results.comparisons:
            verdict = (
                "not significant"
                if not c.significant
                else ("significantly better" if c.better else "significantly worse")
            )
            key = f"{c.metric}:{c.condition}"
            lines.append(
                f"- {c.condition} vs {c.baseline} on {c.metric}: difference {self.values[f'diff:{key}'].text} "
                f"({self.values[f'diff_ci:{key}'].text}), {self.values[f'p:{key}'].text} ({c.test}) — {verdict}"
            )
        ok = sum(1 for r in self.results.runs if r.status == "ok")
        lines.append(f"- {ok} of {len(self.results.runs)} planned runs produced results")
        return "\n".join(lines)

    # -------------------------------------------------------------- checks
    def checks(self) -> list[Check]:
        report, found = self.report, []
        unknown = [k for k in report.citations if k not in self.keys]
        found.append(
            Check(
                name="citations exist",
                ok=not unknown,
                detail=f"unknown keys: {unknown}" if unknown else f"{len(report.citations)} papers cited",
            )
        )
        status = {p.bib_key: p.status for p in self.corpus.papers if p.bib_key}
        grounded = {self.paper_key[c.paper_id] for c in self.km.claims if c.verified and c.paper_id in self.paper_key}
        unread = [
            k
            for k in report.citations
            if k in self.keys and k not in grounded and status.get(k) not in ("downloaded", "parsed", "extracted")
        ]
        found.append(
            Check(
                name="cited papers were read",
                ok=not unread,
                detail=f"cited from the abstract only: {unread}" if unread else "",
            )
        )
        typed = [
            f"{s.id}: {p}" for s in report.sections if s.origin != "rtool" for p in self.section_problems(s, s.template)
        ]
        found.append(Check(name="numbers trace to the results", ok=not typed, detail="; ".join(typed[:4])))
        ok = sum(1 for r in self.results.runs if r.status == "ok")
        found.append(
            Check(
                name="all planned runs succeeded",
                ok=self.results.complete,
                detail=f"{ok} of {len(self.results.runs)} runs produced results",
            )
        )
        tables = report.section("results_tables")
        missing = [c.name for c in self.plan.run_conditions() if tables is None or f"`{c.name}`" not in tables.template]
        found.append(
            Check(name="every condition is reported", ok=not missing, detail=f"missing: {missing}" if missing else "")
        )
        discussion = report.section("discussion")
        found.append(
            Check(
                name="limitations are discussed",
                ok=discussion is not None and "limitation" in discussion.template.lower(),
            )
        )
        readme = self.ctx.absolute(self.bundle.readme_path) if self.bundle.readme_path else None
        results_file = self.ctx.absolute(self.results.results_path)
        repro = [
            ("code README", readme is not None and readme.exists()),
            ("results.json", results_file.exists()),
            ("at least two seeds", len(self.plan.seeds) >= 2),
        ]
        absent = [name for name, present in repro if not present]
        found.append(Check(name="reproducible", ok=not absent, detail=f"missing: {absent}" if absent else ""))
        majors = [f.detail for f in report.findings if f.severity == "major"]
        found.append(Check(name="no major findings left by the writer", ok=not majors, detail="; ".join(majors[:3])))
        return found

    # ---------------------------------------------------------- code audit
    def audit_questions(self) -> list[tuple[AuditAspect, str]]:
        """(aspect, subject) of every question the audit must answer."""
        questions: list[tuple[AuditAspect, str]] = [("condition", c.name) for c in self.plan.run_conditions()]
        questions += [("distinct", ""), ("held_out", "")]
        questions += [("metric", m.name) for m in self.plan.metrics]
        if self.bundle.fixes:
            questions.append(("repairs", ""))
        return questions

    def audit(self) -> tuple[list[AuditFinding], Check, int]:
        """The audit's findings, its check and how many audits replied."""
        root = self.ctx.absolute(self.bundle.root)
        code = root / "experiment.py"
        if not code.exists():
            return [], Check(name=AUDIT_CHECK, ok=False, detail=f"no experiment.py in {self.bundle.root}"), 0
        config = root / "config.json"
        config_text = config.read_text(encoding="utf-8") if config.exists() else "(missing)"
        source = code.read_text(encoding="utf-8", errors="replace")
        shown, self.audit_omitted = shown_code(source, self.cfg.max_code_chars)
        questions = self.audit_questions()
        prompt = self.render_prompt(
            "code_audit.j2",
            design=self.plan.design,
            conditions=self.plan.run_conditions(),
            metrics=self.plan.metrics,
            procedure=self.plan.procedure,
            config=config_text if len(config_text) <= 4000 else config_text[:4000] + "\n… (cut)",
            fixes=self.bundle.fixes,
            code=shown,
            omitted=self.audit_omitted,
            questions=[{"aspect": a, "subject": s} for a, s in questions],
        )
        replies = self.ctx.llm.parallel(
            lambda _i: self.ctx.llm.structured(CodeAuditDraft, prompt, profile="judge", system=SYSTEM),
            list(range(self.cfg.samples)),
        )
        drafts = [r for r in replies if isinstance(r, CodeAuditDraft)]
        if not drafts:
            failed = "no audit returned a valid answer; read experiment.py yourself"
            return [], Check(name=AUDIT_CHECK, ok=False, detail=failed), 0
        findings = self.audit_consensus(questions, drafts)
        blocking = [f for f in findings if (f.state == "problem" and f.severity == "major") or f.state == "unanswered"]
        noted = [f for f in findings if f.state == "problem" and f.severity == "minor"]
        detail = "; ".join(
            f"{f.label}: {f.problem}" if f.state == "problem" else f"{f.label}: unanswered" for f in blocking + noted
        )
        return findings, Check(name=AUDIT_CHECK, ok=not blocking, detail=detail), len(drafts)

    def audit_consensus(
        self, questions: list[tuple[AuditAspect, str]], drafts: list[CodeAuditDraft]
    ) -> list[AuditFinding]:
        """One finding per question: a problem only if most audits raise it."""
        named = [*(c.name for c in self.plan.run_conditions()), *(m.name for m in self.plan.metrics)]
        names = {n.lower(): n for n in named}
        votes: dict[tuple[AuditAspect, str], dict[int, AuditAnswer]] = {}
        for index, draft in enumerate(drafts):
            for answer in draft.answers:
                subject = answer.subject.strip(" `'\"") if answer.aspect in ("condition", "metric") else ""
                ballot = votes.setdefault((answer.aspect, names.get(subject.lower(), subject)), {})
                if index not in ballot or (ballot[index].ok and not answer.ok):  # one vote per audit
                    ballot[index] = answer
        need = max(1, math.ceil(self.cfg.agreement * len(drafts)))
        extra = [k for k, b in votes.items() if k not in questions and any(not a.ok for a in b.values())]
        findings = []
        for aspect, subject in [*questions, *extra]:
            answers = list(votes.get((aspect, subject), {}).values())
            raised = [a for a in answers if not a.ok]
            majors = sum(1 for a in raised if a.severity == "major")
            severity: Severity = "major" if majors * 2 > len(raised) else "minor"
            example = next((a for a in sorted(raised, key=lambda a: not a.problem) if a.severity == severity), None)
            state: AuditState
            if len(raised) >= need:
                state = "problem"
            elif len(answers) < need and (aspect, subject) in questions:
                state = "unanswered"
            else:
                state = "unconfirmed" if raised else "ok"
            findings.append(
                AuditFinding(
                    aspect=aspect,
                    subject=subject,
                    label=audit_label(aspect, subject),
                    state=state,
                    answered=len(answers),
                    raised_by=len(raised),
                    problem=example.problem if example else "",
                    fix=example.fix if example else "",
                    severity=severity,
                )
            )
        return findings

    # -------------------------------------------------------------- review
    def drafts(self, checks: list[Check]) -> list[ReviewDraft]:
        prompt = self.render_prompt(
            "review.j2",
            checks=checks,
            facts=self.facts(),
            section_ids=[s.id for s in self.report.sections],
            report=render("\n\n".join(f"### {s.id}\n{s.template}" for s in self.report.sections), self.values),
        )
        replies = self.ctx.llm.parallel(
            lambda _i: self.ctx.llm.structured(ReviewDraft, prompt, profile="judge", system=SYSTEM),
            list(range(self.cfg.samples)),
        )
        return [r for r in replies if isinstance(r, ReviewDraft)]

    def consensus(self, drafts: list[ReviewDraft]) -> tuple[dict[str, float], list[ReviewIssue], str]:
        if not drafts:
            return {}, [], "The automatic review failed; please review the report yourself."
        scores = {
            name: float(statistics.median(getattr(d.scores, name) for d in drafts)) for name in Rubric.model_fields
        }
        need = max(1, math.ceil(self.cfg.agreement * len(drafts)))
        clusters: list[tuple[ReviewIssue, set[int], list[ReviewIssue]]] = []
        for index, draft in enumerate(drafts):
            for issue in draft.issues:
                for representative, raised_by, members in clusters:
                    if representative.section == issue.section and similar(representative.problem, issue.problem):
                        raised_by.add(index)
                        members.append(issue)
                        break
                else:
                    clusters.append((issue, {index}, [issue]))
        issues = []
        for representative, raised_by, members in clusters:
            if len(raised_by) >= need:
                majors = sum(1 for m in members if m.severity == "major")
                severity = "major" if majors * 2 > len(members) else "minor"
                issues.append(representative.model_copy(update={"severity": severity}))
        middle = min(drafts, key=lambda d: sum(abs(getattr(d.scores, n) - scores[n]) for n in scores))
        return scores, issues, middle.summary

    def edit(self, section: ReportSection, points: list[ReviewIssue]) -> str | None:
        """The section rewritten for the review points, or None if no rewrite passes the checks."""
        keys = sorted(self.keys) if section.id != "abstract" else []
        prompt = self.render_prompt(
            "edit_section.j2",
            title=section.title,
            points=[f"{p.problem} — {p.suggestion}" for p in points],
            keys=keys,
            values=sorted(self.values.items()) if section.id not in ("introduction", "related_work") else [],
            text=section.template,
        )
        for _ in range(self.cfg.repair_rounds + 1):
            text = self.ctx.llm.chat(prompt, profile="write", system=SYSTEM).text.strip()
            text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("#", "```"))).strip()
            unchanged = " ".join(text.split()) == " ".join(section.template.split())
            if text and not unchanged and not self.section_problems(section, text, original=section.template):
                return text
        return None  # no acceptable rewrite: the issue stays open for the human

    # ---------------------------------------------------------------- main
    def review(self) -> None:
        ctx = self.ctx
        checks = self.checks()
        audit, audit_check, audits = self.audit()
        checks.append(audit_check)
        try:
            drafts = self.drafts(checks)
        except LLMOutputError:
            drafts = []
        scores, issues, summary = self.consensus(drafts)

        sections = {s.id: s for s in self.report.sections}
        textual = [i for i in issues if i.target == "report"]
        to_edit = sorted({i.section for i in textual if i.section in EDITABLE and i.section in sections})
        rewrites = ctx.llm.parallel(
            lambda sid: self.edit(sections[sid], [i for i in textual if i.section == sid]), to_edit
        )
        edited, open_issues = [], [i for i in textual if i.section not in to_edit]
        for sid, new in zip(to_edit, rewrites, strict=True):
            if isinstance(new, str):
                sections[sid] = sections[sid].model_copy(update={"template": new, "origin": "reviewer"})
                edited.append(sid)
            else:
                open_issues += [i for i in textual if i.section == sid]

        requests = []
        for issue in issues:
            if issue.target != "report" and issue.severity == "major":
                ctx.request_change(
                    artifact=issue.target, reason=f"reviewer: {issue.problem}", required_changes=[issue.suggestion]
                )
                requests.append(f"{issue.target}: {issue.problem}")
            elif issue.target != "report":
                open_issues.append(issue)
        sent = [f for f in audit if f.state == "problem" and f.severity == "major"]
        if sent:  # one request, so the Coder fixes them in one revision round
            ctx.request_change(
                artifact="code_bundle",
                reason="reviewer (code audit): " + "; ".join(f"{f.label}: {f.problem}" for f in sent),
                required_changes=[f.fix or f.problem for f in sent],
            )
            requests.append("code_bundle: code audit — " + ", ".join(f.label for f in sent))

        final_sections = [sections[s.id] for s in self.report.sections]
        out = ctx.workspace.mkdir("final")
        self.copy_figures(out)
        cited = citation_keys("\n".join(s.template for s in final_sections))
        markdown = assemble(self.report.title, final_sections, self.values, reference_list(self.corpus, cited))
        final_path, bib_path = out / "final_report.md", out / "refs.bib"
        atomic_write_text(final_path, markdown)
        source_bib = ctx.absolute(self.corpus.bibtex_path) if self.corpus.bibtex_path else None
        bibtex = source_bib.read_text(encoding="utf-8") if source_bib is not None and source_bib.exists() else ""
        atomic_write_text(bib_path, bibtex_subset(bibtex, cited))
        before = ctx.absolute(self.report.markdown_path).read_text(encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True), markdown.splitlines(keepends=True), "report.md", "final_report.md"
            )
        )
        diff_path = out / "report.diff"
        atomic_write_text(diff_path, diff or "(no edits)\n")

        accept = (
            bool(drafts)
            and all(c.ok for c in checks)
            and all(v >= self.cfg.min_score for v in scores.values())
            and not requests
            and not any(i.severity == "major" for i in open_issues)
            and not sent
        )
        review = Review(
            verdict="accept" if accept else "needs_human",
            summary=summary,
            scores=scores,
            checks=checks,
            issues=issues,
            code_audit=audit,
            edited_sections=edited,
            change_requests=requests,
            diff_path=ctx.relative(diff_path),
        )
        review_path = out / "review.md"
        review.review_path = ctx.relative(review_path)
        atomic_write_text(review_path, self.review_markdown(review, len(drafts), open_issues, audits))
        ctx.outputs.put("review", review, meta={"review": review.review_path, "prompts": self.provenance})
        final = self.report.model_copy(
            update={
                "sections": final_sections,
                "markdown_path": ctx.relative(final_path),
                "bibtex_path": ctx.relative(bib_path),
                "citations": cited,
                "figures": [ctx.relative(p) for p in sorted((out / "figures").glob("*.svg"))],
            }
        )
        ctx.outputs.put("final_report", final, meta={"review": ctx.relative(final_path)})

    def copy_figures(self, out: Any) -> None:
        target = out / "figures"
        target.mkdir(parents=True, exist_ok=True)
        for figure in self.report.figures:
            source = self.ctx.absolute(figure)
            if source.exists():
                shutil.copyfile(source, target / source.name)

    def review_markdown(self, review: Review, samples: int, open_issues: list[ReviewIssue], audits: int) -> str:
        verdict = "accept" if review.verdict == "accept" else "needs a human decision"
        lines = [f"# Review: {self.report.title}", "", f"**Verdict: {verdict}.** {review.summary}", ""]
        if review.scores:
            lines += [
                f"## Scores (median of {samples} independent reviews, 1-5)",
                "",
                "| criterion | score |",
                "|---|---|",
            ]
            lines += [f"| {name} | {value:g} |" for name, value in review.scores.items()]
            lines.append("")
        lines += ["## Automated checks", ""]
        lines += [
            f"- {'ok' if c.ok else 'PROBLEM'}: {c.name}" + (f" — {c.detail}" if c.detail else "") for c in review.checks
        ]
        lines += ["", "## Code audit", "", *self.audit_markdown(review, audits)]
        if review.issues:
            lines += ["", "## Issues raised by most reviews", ""]
            for issue in review.issues:
                if issue.section in review.edited_sections and issue.target == "report":
                    state = "edited"
                elif any(issue.problem in r for r in review.change_requests):
                    state = f"sent back to {issue.target}"
                else:
                    state = "open" if issue in open_issues else "noted"
                lines.append(f"- [{issue.severity}] {issue.section}: {issue.problem} — {issue.suggestion} ({state})")
        if review.edited_sections:
            lines += ["", f"Edited sections: {', '.join(review.edited_sections)} (diff: `{review.diff_path}`)."]
        return "\n".join(lines) + "\n"

    def audit_markdown(self, review: Review, audits: int) -> list[str]:
        if not review.code_audit:
            check = next(c for c in review.checks if c.name == AUDIT_CHECK)
            return [f"The code audit did not run: {check.detail}."]
        need = max(1, math.ceil(self.cfg.agreement * audits))
        lines = [
            f"`experiment.py` checked against the plan by {audits} independent audits; a problem counts when {need} "
            "or more raise it." + (f" {self.audit_omitted} lines were left out to fit." if self.audit_omitted else ""),
            "",
        ]
        for f in review.code_audit:
            of = f"of {audits} audits"
            if f.state == "ok":
                lines.append(f"- ok: {f.label}")
            elif f.state == "problem":
                where = "sent back to code_bundle" if f.severity == "major" else "open"
                fix = f" — {f.fix}" if f.fix else ""
                lines.append(
                    f"- PROBLEM [{f.severity}] {f.label}: {f.problem}{fix} (raised by {f.raised_by} {of}; {where})"
                )
            elif f.state == "unconfirmed":
                lines.append(f"- not confirmed: {f.label}: {f.problem} (raised by {f.raised_by} {of})")
            else:
                raised = f"; raised: {f.problem}" if f.problem else ""
                lines.append(f"- unanswered: {f.label} (answered by {f.answered} {of}{raised})")
        return lines
