"""Writer: the full report, written from the plan, the results and the literature, with every number traceable.

Facts are written by rtool (methods, result tables, reproducibility). The model
writes the prose — introduction, related work, results, discussion, conclusion,
then title and abstract — each from narrow material and under three checks:
placeholders must exist, citations must be corpus keys, and no number may be
typed that is not in the material. Failed checks get repair rounds; whatever
still fails is removed and listed as a finding. Finally an audit compares the
report with the plan: planned conditions or metrics without results become
change requests to the stage that produced the results, and a consistency check
rewrites sentences that contradict the statistics.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...core.context import AgentContext
from ...core.editing import EditContext, EditResult, artifact_editor
from ...core.fs import atomic_write_text
from ...core.registry import register_agent
from ...llm.client import CONCISE, LOOP_BREAKER, LLMOutputError
from ...llm.prompts import PromptLibrary
from ...schemas.code import CodeBundle, ExperimentResults
from ...schemas.common import ResearchBrief, SelectedItem
from ...schemas.literature import Corpus, KnowledgeMap
from ...schemas.plan import ExperimentPlan
from ...schemas.report import Contradictions, Finding, Report, ReportSection, TitleAbstract
from ...tools.report.editing import edit_report
from ...tools.report.numbers import fmt, fmt_ci, fmt_p
from ...tools.report.render import (
    Value,
    assemble,
    bib_keys,
    bibtex_subset,
    citation_keys,
    placeholders,
    problems,
    reference_list,
    render,
    strip_problems,
    to_pandoc_citations,
    value_table,
)
from ..coder.results import translated_tests
from ..common import SYSTEM

PROSE = ("introduction", "related_work", "results", "discussion", "conclusion")
TITLES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related_work": "Related work",
    "methods": "Methods",
    "results": "Results",
    "results_tables": "Result tables and figures",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "reproducibility": "Reproducibility",
}
ORDER = list(TITLES)
INSTRUCTIONS = {
    "introduction": (
        "Two or three paragraphs: the problem and why it matters, what is known and what is still open (cite "
        "the papers), and the hypothesis this study tests. Do not describe our results here."
    ),
    "related_work": (
        "Three or four paragraphs grouped by theme: what prior work found (cite every claim) and how our "
        "hypothesis relates to it. Use only the claims and topics in the material."
    ),
    "results": (
        "Two to four paragraphs reporting what the experiment found, metric by metric, with the placeholders: "
        "means with their confidence intervals, the difference to the baseline, its p-value and whether it is "
        "significant. Report non-significant and negative results as plainly as positive ones."
    ),
    "discussion": (
        "Three or four paragraphs: what the results mean for the hypothesis, how they relate to prior work "
        "(cite it), then a paragraph that starts with 'Limitations.' covering threats to validity, failed or "
        "missing runs and the small number of seeds, and finally future work."
    ),
    "conclusion": "One paragraph: the hypothesis, the main result (placeholders) and the take-away.",
}


class WriterConfig(BaseModel):
    repair_rounds: int = Field(default=2, ge=0, le=5)
    max_claims: int = Field(default=30, ge=0)  # verified claims shown to the model
    consistency_check: bool = True


def settle(
    text: str, check: Callable[[str], list[str]], fix: Callable[[str, list[str]], str | None], rounds: int
) -> tuple[str, list[str]]:
    """Check, repair, re-check (bounded). Returns the text and whatever problems remain."""
    found = check(text)
    for _ in range(rounds):
        if not found:
            break
        fixed = fix(text, found)
        if not fixed:
            break
        text, found = fixed, check(fixed)
    return text, found


def clean_prose(text: str) -> str:
    """Drop headings or fences the model added around a section."""
    lines = [line for line in text.strip().splitlines() if not line.lstrip().startswith(("#", "```"))]
    return "\n".join(lines).strip()


@register_agent("writer")
class WriterAgent(Agent):
    description = "Writes the full report with checked citations and numbers filled in from the results."
    consumes = ("research_brief", "corpus", "knowledge_map", "experiment_plan", "experiment_results", "code_bundle")
    # its own last report: sections the researcher wrote (origin "human") are kept when it writes again
    optional_inputs = ("selected_hypothesis", "report")
    produces = ("report",)
    config_model = WriterConfig

    def run(self, ctx: AgentContext) -> None:
        Drafting(ctx, self.config).write()


@artifact_editor("report")
def edit(value: dict[str, Any], ctx: EditContext) -> EditResult:
    return edit_report(value, ctx)


class Drafting:
    def __init__(self, ctx: AgentContext, cfg: WriterConfig) -> None:
        self.ctx, self.cfg = ctx, cfg
        self.brief: ResearchBrief = ctx.inputs.get("research_brief")
        self.corpus: Corpus = ctx.inputs.get("corpus")
        self.km: KnowledgeMap = ctx.inputs.get("knowledge_map")
        self.plan: ExperimentPlan = ctx.inputs.get("experiment_plan")
        self.results: ExperimentResults = ctx.inputs.get("experiment_results")
        self.bundle: CodeBundle = ctx.inputs.get("code_bundle")
        self.selected: SelectedItem | None = (
            ctx.inputs.get("selected_hypothesis") if ctx.inputs.has("selected_hypothesis") else None
        )
        self.values = value_table(self.plan, self.results)
        self.paper_key = {p.id: p.bib_key for p in self.corpus.papers if p.bib_key}
        self.keys = set(bib_keys(self.corpus))
        self.prompts = PromptLibrary.for_agent(__package__, "writer", pack=self.brief.domain_pack)
        self.provenance: dict[str, str] = {}
        self.findings: list[Finding] = []
        previous: Report | None = ctx.inputs.get("report") if ctx.inputs.has("report") else None
        self.kept = {s.id: s for s in previous.sections if s.origin == "human"} if previous else {}
        self.kept_title = previous.title if previous is not None and previous.title_origin == "human" else None

    # --------------------------------------------------------------- llm
    def render_prompt(self, name: str, **variables: Any) -> str:
        prompt = self.prompts.render(name, **variables)
        self.provenance[name] = prompt.template_hash
        return prompt.text

    def prose(self, prompt: str) -> str:
        reply = self.ctx.llm.chat(prompt, profile="write", system=SYSTEM)
        if reply.finish_reason == "length":  # greedy decoding looping: ask again, briefly, sampled
            reply = self.ctx.llm.chat(f"{prompt}\n\n{CONCISE}", profile="write", system=SYSTEM, **LOOP_BREAKER)
        return clean_prose(reply.text)

    def placeholders_for(self, section: str) -> dict[str, Value]:
        return {} if section in ("introduction", "related_work") else self.values

    def material_for(self, section: str) -> str:
        if section == "introduction":
            return "\n\n".join([self.brief_text(), self.hypothesis_text(), self.overview(), self.open_problems()])
        if section == "related_work":
            return "\n\n".join([self.topics(), self.claims()])
        if section == "discussion":
            return "\n\n".join([self.hypothesis_text(), self.facts(), self.limitations(), self.claims(limit=15)])
        return "\n\n".join([self.hypothesis_text(), self.facts()])

    def check(self, section: str, text: str) -> list[str]:
        keys = self.keys if section != "abstract" else set()
        return problems(text, self.placeholders_for(section), keys, allowed_numbers=self.material_for(section))

    def fix(self, section: str, text: str, found: list[str]) -> str | None:
        prompt = self.render_prompt(
            "fix_section.j2",
            title=TITLES[section],
            problems=found,
            keys=sorted(self.keys) if section != "abstract" else [],
            values=sorted(self.placeholders_for(section).items()),
            text=text,
        )
        fixed = self.prose(prompt)
        return fixed or None

    def draft(self, section: str) -> str:
        values = sorted(self.placeholders_for(section).items())
        example = next((f"{{{{{k}}}}}" for k, _ in values if k.startswith("mean:")), "{{runs_ok}}")
        feedback = [f.text for f in self.ctx.feedback]
        prompt = self.render_prompt(
            "section.j2",
            title=TITLES[section],
            instructions=INSTRUCTIONS[section],
            values=values,
            example=example,
            material=self.material_for(section),
            feedback=feedback,
        )
        text, remaining = settle(
            self.prose(prompt),
            lambda t: self.check(section, t),
            lambda t, found: self.fix(section, t, found),
            self.cfg.repair_rounds,
        )
        return self.give_up_on(section, text, remaining)

    def give_up_on(self, section: str, text: str, remaining: list[str]) -> str:
        """Repairs did not fix everything: remove what cannot be trusted and say so."""
        if not remaining:
            return text
        for problem in remaining:
            kind = "citation" if "citation" in problem else "number" if "number" in problem else "placeholder"
            self.findings.append(Finding(kind=kind, detail=problem, section=section, severity="minor"))
        keys = self.keys if section != "abstract" else set()
        return strip_problems(text, self.placeholders_for(section), keys, self.material_for(section))

    # ------------------------------------------------------------ material
    def brief_text(self) -> str:
        lines = [f"Research topic: {self.brief.topic}"]
        lines += [f"Question: {q}" for q in self.brief.questions]
        return "\n".join(lines)

    def hypothesis_text(self) -> str:
        lines = [f"Hypothesis tested: {self.plan.hypothesis}", f"Null hypothesis: {self.plan.null_hypothesis}"]
        if self.selected is not None:
            item = self.selected.item
            if item.get("rationale"):
                lines.append(f"Rationale: {item['rationale']}")
            for ev in item.get("evidence", [])[:4]:
                key = self.paper_key.get(ev.get("paper_id", ""))
                if key and ev.get("verified", True):
                    lines.append(f'Evidence [@{key}]: "{ev.get("quote", "")[:300]}"')
        return "\n".join(lines)

    def overview(self) -> str:
        return "State of the art (from the literature review):\n" + to_pandoc_citations(self.km.overview, self.keys)

    def open_problems(self) -> str:
        lines = ["Open problems in the literature:"]
        for problem in self.km.open_problems[:6]:
            cites = "; ".join(f"@{self.paper_key[i]}" for i in problem.paper_ids if i in self.paper_key)
            lines.append(f"- {problem.statement}" + (f" [{cites}]" if cites else ""))
        return "\n".join(lines)

    def topics(self) -> str:
        lines = ["Themes in the literature:"]
        for topic in self.km.topics:
            cites = "; ".join(f"@{self.paper_key[i]}" for i in topic.paper_ids if i in self.paper_key)
            lines.append(f"- {topic.label}: {topic.summary}" + (f" [{cites}]" if cites else ""))
        return "\n".join(lines)

    def claims(self, limit: int | None = None) -> str:
        limit = self.cfg.max_claims if limit is None else limit
        per_paper: dict[str, int] = {}
        lines = ["Verified findings from the papers (each with its citation key):"]
        for claim in sorted(self.km.claims, key=lambda c: -c.support):
            key = self.paper_key.get(claim.paper_id)
            if not claim.verified or key is None or per_paper.get(key, 0) >= 3 or len(lines) > limit:
                continue
            per_paper[key] = per_paper.get(key, 0) + 1
            kind = "" if claim.kind == "result" else f" ({claim.kind.replace('_', ' ')})"
            lines.append(f'- [@{key}]{kind} {claim.text} — quote: "{claim.quote[:240]}"')
        return "\n".join(lines)

    def facts(self) -> str:
        """The results in words, with placeholders: what the prose may say about our experiment."""
        baseline = next((c.name for c in self.plan.conditions if c.is_baseline), "")
        lines = ["Our results (placeholders render to the numbers):"]
        lines.append(
            "- Runs: {{runs_ok}} of {{runs_total}} planned runs produced results ({{seeds}} seeds per condition)."
        )
        for metric in self.plan.metrics:
            direction = "higher" if metric.higher_is_better else "lower"
            lines.append(f"- Metric `{metric.name}` ({direction} is better): {metric.definition}")
            for s in (s for s in self.results.summaries if s.metric == metric.name):
                tag = " (baseline)" if s.condition == baseline else ""
                lines.append(
                    f"  - `{s.condition}`{tag}: mean {{{{mean:{metric.name}:{s.condition}}}}}, "
                    f"{{{{ci:{metric.name}:{s.condition}}}}}"
                )
            for c in (c for c in self.results.comparisons if c.metric == metric.name):
                key = f"{metric.name}:{c.condition}"
                test = f"the {c.test}" + (", corrected for multiple comparisons" if c.p_adjusted is not None else "")
                lines.append(
                    f"  - `{c.condition}` vs `{c.baseline}`: difference {{{{diff:{key}}}}} ({{{{diff_ci:{key}}}}}), "
                    f"{{{{{c.p_key}:{key}}}}} with {test} — {c.verdict()}."
                )
        lines.append(f"- What the plan expected: {self.plan.expected_results}")
        lines += [f"- Note: {note}" for note in self.results.notes if not note.startswith("Warning: ")]
        return "\n".join(lines)

    def limitations(self) -> str:
        lines = ["Known limitations and risks:"]
        lines += [f"- Warning about the experiment itself (say so plainly): {w}" for w in self.results.warnings]
        lines += [f"- {risk}" for risk in self.plan.risks]
        failed = [r for r in self.results.runs if r.status != "ok"]
        if failed:
            lines.append(
                f"- {len(failed)} planned runs produced no results, e.g. `{failed[0].condition}` seed "
                f"{failed[0].seed}: {failed[0].error[:150]}"
            )
        if len(self.plan.seeds) < 5:
            lines.append("- Few seeds per condition: confidence intervals are wide and tests have little power.")
        return "\n".join(lines)

    # ---------------------------------------------------- rtool's sections
    def methods(self) -> str:
        p, c = self.plan, self.plan.compute
        comparisons = self.results.comparisons
        test = " / ".join(sorted({x.test for x in comparisons})) if comparisons else p.statistical_tests[0].test
        corrected = sum(1 for x in comparisons if x.p_adjusted is not None)
        method = {"holm": "Holm's method", "bonferroni": "Bonferroni"}.get(self.results.correction, "")
        translated = translated_tests(p) if comparisons else []
        lines = [
            f"**Design.** {p.design}",
            "",
            "**Conditions.** "
            + "; ".join(f"`{x.name}`{' (baseline)' if x.is_baseline else ''}: {x.description}" for x in p.conditions)
            + ".",
        ]
        if p.ablations:
            lines += ["", "**Ablations.** " + "; ".join(f"`{x.name}`: {x.description}" for x in p.ablations) + "."]
        lines += [
            "",
            "**Measures.** "
            + "; ".join(
                f"`{m.name}` ({'higher' if m.higher_is_better else 'lower'} is better): {m.definition}"
                for m in p.metrics
            )
            + ".",
            "",
            "**Procedure.** " + " ".join(f"({i}) {step}" for i, step in enumerate(p.procedure, 1)),
            "",
            f"**Statistics.** Every condition ran with {len(p.seeds)} seeds ({', '.join(map(str, p.seeds))}). Each "
            f"condition is compared with the baseline using the {test} (paired by seed where all seeds succeeded); "
            "means are reported with 95% confidence intervals from the t distribution and effect sizes as Cohen's d"
            + (
                f"; p-values are corrected for the {corrected} comparisons with {method}"
                if corrected and method
                else ""
            )
            + "."
            + (
                f" The plan named {', '.join(translated)}; these were run as pairwise comparisons with the baseline."
                if translated
                else ""
            )
            + f" Significance level: {p.statistical_tests[0].alpha:g}.",
        ]
        if p.resources:
            lines += [
                "",
                "**Data and models.** "
                + "; ".join(
                    f"{r.name} ({r.source}: `{r.identifier}`{f', licence {r.license}' if r.license else ''})"
                    for r in p.resources
                )
                + ".",
            ]
        gpu = f"{c.gpus_per_run}x {c.gpu_type} GPU" if c.gpus_per_run else "CPU only"
        lines += [
            "",
            f"**Compute.** {c.runs} runs on the `{c.partition}` partition ({gpu}, time limit {c.time_limit} each).",
        ]
        return "\n".join(lines)

    def result_tables(self, figures: dict[str, str]) -> str:
        baseline = next((c.name for c in self.plan.conditions if c.is_baseline), "")
        lines: list[str] = []
        for metric in self.plan.metrics:
            direction = "higher" if metric.higher_is_better else "lower"
            lines += [f"**`{metric.name}`** ({direction} is better)", ""]
            lines += ["| condition | n | mean | 95% CI | std |", "|---|---|---|---|---|"]
            for s in (s for s in self.results.summaries if s.metric == metric.name):
                name = f"`{s.condition}` (baseline)" if s.condition == baseline else f"`{s.condition}`"
                lines.append(f"| {name} | {s.n} | {fmt(s.mean)} | {fmt_ci(s.ci_low, s.ci_high)} | {fmt(s.std)} |")
            rows = [c for c in self.results.comparisons if c.metric == metric.name]
            if rows:
                lines += [
                    "",
                    "| comparison | difference | 95% CI | test | p | effect size (d) |",
                    "|---|---|---|---|---|---|",
                ]
                for c in rows:
                    p = fmt_p(c.p_adjusted if c.p_adjusted is not None else c.p_value)
                    lines.append(
                        f"| `{c.condition}` vs `{c.baseline}` | {fmt(c.difference)} | {fmt_ci(c.ci_low, c.ci_high)} "
                        f"| {c.test} | {p} | {fmt(c.effect_size)} |"
                    )
            if metric.name in figures:
                lines += ["", f"![{metric.name}: mean and 95% CI per condition]({figures[metric.name]})"]
            lines.append("")
        return "\n".join(lines).strip()

    def reproducibility(self) -> str:
        b, r = self.bundle, self.results
        seeds = ", ".join(map(str, self.plan.seeds))
        lines = [
            f"- Code: `{b.root}` (entry point `{b.entry_point}`, domain pack `{b.pack}`, "
            f"fingerprint `{b.fingerprint[:12]}`).",
            f"- Environment: {f'`{b.container}`' if b.container else 'local Python'}"
            + (f"; extra packages: {', '.join(b.requirements)}" if b.requirements else "")
            + ".",
            f"- Seeds: {seeds}; every run writes its metrics to `runs/<condition>__<seed>.json`.",
            f"- Results: `{r.results_path}` (validated against the plan's results schema); tables: `{r.tables_path}`.",
            f"- Slurm jobs: {', '.join(r.jobs) or 'none'}; GPU-hours requested: {r.gpu_hours_requested:g}.",
            f"- Smoke tests before the full runs: {', '.join(s.condition for s in b.smoke if s.ok) or 'none'}.",
        ]
        if b.fixes:
            lines.append(f"- Automatic repairs before success: {len(b.fixes)} (listed in `{b.readme_path}`).")
        return "\n".join(lines)

    # ---------------------------------------------------------------- audit
    def audit(self) -> None:
        summarised = {(s.condition, s.metric) for s in self.results.summaries}
        conditions = [c.name for c in self.plan.run_conditions()]
        missing_conditions = [c for c in conditions if not any(s[0] == c for s in summarised)]
        missing_metrics = [m.name for m in self.plan.metrics if not any(s[1] == m.name for s in summarised)]
        if missing_conditions:
            self.findings.append(
                Finding(
                    kind="missing_condition",
                    target="experiment_results",
                    detail=f"planned conditions without any results: {missing_conditions}",
                )
            )
        if missing_metrics:
            self.findings.append(
                Finding(
                    kind="missing_metric",
                    target="experiment_results",
                    detail=f"planned metrics without results: {missing_metrics}",
                )
            )
        if not self.results.complete:
            failed = sum(1 for r in self.results.runs if r.status != "ok")
            self.findings.append(
                Finding(
                    kind="incomplete_runs",
                    severity="minor" if failed * 2 < len(self.results.runs) else "major",
                    detail=f"{failed} of {len(self.results.runs)} runs have no results (reported as a limitation)",
                )
            )
        upstream = [f for f in self.findings if f.target != "report"]
        if upstream:
            self.ctx.request_change(
                artifact="experiment_results",
                reason="the report cannot cover the whole plan: " + "; ".join(f.detail for f in upstream),
                required_changes=[f.detail for f in upstream],
            )

    def consistency(self, sections: dict[str, str]) -> dict[str, str]:
        """Rewrite sentences that contradict the computed results (one round; the researcher's sections are left)."""
        if not self.cfg.consistency_check:
            return sections
        checked = [
            s for s in ("abstract", "results", "discussion", "conclusion") if s in sections and s not in self.kept
        ]
        prompt = self.render_prompt(
            "consistency.j2",
            facts=render(self.facts(), self.values),
            sections=[{"id": s, "text": render(sections[s], self.values)} for s in checked],
        )
        try:
            found = self.ctx.llm.structured(Contradictions, prompt, profile="judge", system=SYSTEM)
        except LLMOutputError:
            return sections
        for section in checked:
            items = [i for i in found.items if i.section == section]
            if not items:
                continue
            issues = [f'"{i.statement}" is wrong: {i.problem}' for i in items]
            fixed = self.fix(section, sections[section], issues)
            if fixed and not self.check(section, fixed):
                sections[section] = fixed
                self.ctx.emit("report_contradiction_fixed", section=section, count=len(items))
            else:
                self.findings += [
                    Finding(kind="contradiction", section=section, detail=i, severity="major") for i in issues
                ]
        return sections

    # ------------------------------------------------------------------ main
    def keep_human_sections(self) -> dict[str, str]:
        """The researcher's sections, unchanged; what no longer fits the results is reported, not rewritten."""
        kept = {}
        for sid, section in self.kept.items():
            keys = self.keys if sid != "abstract" else set()
            for problem in problems(section.template, self.values, keys, allowed_numbers=None):
                self.findings.append(
                    Finding(kind="placeholder", section=sid, severity="minor", detail=f"in your text: {problem}")
                )
            kept[sid] = section.template
        if kept:
            self.ctx.emit("report_human_sections_kept", sections=sorted(kept))
        return kept

    def write(self) -> None:
        ctx = self.ctx
        prose: dict[str, str] = self.keep_human_sections()
        todo = [s for s in PROSE if s not in prose]
        drafts = ctx.llm.parallel(self.draft, todo)
        for section, text in zip(todo, drafts, strict=True):
            if isinstance(text, Exception):
                raise text
            prose[section] = text
        report_text = "\n\n".join(f"## {TITLES[s]}\n{prose[s]}" for s in PROSE)
        if "abstract" in prose:  # written by the researcher: keep it (and any title they chose)
            title = self.kept_title or self.plan.hypothesis
        else:
            try:
                head = ctx.llm.structured(
                    TitleAbstract,
                    self.render_prompt("abstract.j2", report=report_text),
                    profile="write",
                    system=SYSTEM,
                )
                title = self.kept_title or head.title.strip().strip("#").strip() or self.plan.hypothesis
                abstract, remaining = settle(
                    clean_prose(head.abstract),
                    lambda t: self.check("abstract", t),
                    lambda t, found: self.fix("abstract", t, found),
                    self.cfg.repair_rounds,
                )
                prose["abstract"] = self.give_up_on("abstract", abstract, remaining)
            except LLMOutputError:
                title, prose["abstract"] = self.kept_title or self.plan.hypothesis, ""
                self.findings.append(Finding(kind="abstract", detail="no abstract could be written", severity="minor"))
        prose = self.consistency(prose)
        self.audit()

        out = ctx.workspace.mkdir("report")
        figures = self.copy_figures(out / "figures")
        texts = {
            **prose,
            "methods": self.methods(),
            "results_tables": self.result_tables(figures),
            "reproducibility": self.reproducibility(),
        }
        origin = {"methods": "rtool", "results_tables": "rtool", "reproducibility": "rtool"}
        origin |= {s: "human" for s in self.kept}
        sections = [
            ReportSection(
                id=s,
                title=self.kept[s].title if s in self.kept else TITLES[s],
                template=texts[s],
                origin=origin.get(s, "llm"),  # type: ignore[arg-type]
            )
            for s in ORDER
            if texts.get(s)
        ]
        cited = citation_keys("\n".join(s.template for s in sections))
        markdown = assemble(title, sections, self.values, reference_list(self.corpus, cited))
        report_path = out / "report.md"
        atomic_write_text(report_path, markdown)
        bib_path = out / "refs.bib"
        source_bib = ctx.absolute(self.corpus.bibtex_path) if self.corpus.bibtex_path else None
        bibtex = source_bib.read_text(encoding="utf-8") if source_bib is not None and source_bib.exists() else ""
        atomic_write_text(bib_path, bibtex_subset(bibtex, cited))
        used = sum(len(set(self.values) & set(placeholders(s.template))) for s in sections)
        ctx.outputs.put(
            "report",
            Report(
                title=title,
                title_origin="human" if self.kept_title else "llm",
                sections=sections,
                markdown_path=ctx.relative(report_path),
                bibtex_path=ctx.relative(bib_path),
                citations=cited,
                figures=[ctx.relative(out / f) for f in figures.values()],
                findings=self.findings,
                numbers=used,
            ),
            meta={"review": ctx.relative(report_path), "prompts": self.provenance},
        )

    def copy_figures(self, target: Path) -> dict[str, str]:
        """The results' SVG figures, next to the report (paths relative to report.md)."""
        target.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        for figure in self.results.figures:
            source = self.ctx.absolute(figure)
            if source.exists():
                shutil.copyfile(source, target / source.name)
                copied[source.stem] = f"figures/{source.name}"
        return copied
