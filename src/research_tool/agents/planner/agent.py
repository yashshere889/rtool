"""Experiment Planner: design → resources → compute, each checked mechanically, with one repair round.

Three narrow LLM calls (design, resources, compute) are followed by
deterministic checks: is there a baseline, enough seeds, do the datasets and
models really exist (Hugging Face Hub API), does one run fit an allowed
partition and the GPU budget. Problems are fed back once for repair;
whatever remains is listed in ``plan.issues`` for the human at the plan gate.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...core.context import AgentContext
from ...core.registry import register_agent
from ...llm.prompts import PromptLibrary
from ...schemas.common import ResearchBrief, SelectedItem
from ...schemas.hypothesis import Hypothesis
from ...schemas.literature import Corpus, KnowledgeMap
from ...schemas.plan import ComputeEstimate, ExperimentPlan, PlanCore, ResourcePlan, ResourceSpec
from ...tools.http import PoliteHttp
from ...tools.resources.checks import HubChecker, ResourceChecker
from ..common import SYSTEM, compute_profile, open_corpus_kb
from .checks import design_issues, fit_compute, normalise_core, one_primary, resource_issues, results_schema


class PlannerConfig(BaseModel):
    min_seeds: int = Field(default=3, ge=1)
    repair_rounds: int = Field(default=1, ge=0, le=3)
    evidence_snippets: int = Field(default=8, ge=0)
    check_resources: bool = True
    max_dataset_gb: float = Field(default=200, gt=0)


@register_agent("planner")
class PlannerAgent(Agent):
    description = "Designs the experiment for the chosen hypothesis, checks resources and fits compute to Barkla."
    consumes = ("research_brief", "selected_hypothesis", "corpus", "knowledge_map")
    produces = ("experiment_plan",)
    config_model = PlannerConfig

    def run(self, ctx: AgentContext) -> None:
        cfg: PlannerConfig = self.config
        brief: ResearchBrief = ctx.inputs.get("research_brief")
        selected: SelectedItem = ctx.inputs.get("selected_hypothesis")
        corpus: Corpus = ctx.inputs.get("corpus")
        km: KnowledgeMap = ctx.inputs.get("knowledge_map")
        hypothesis = Hypothesis.model_validate(selected.item)
        prompts = PromptLibrary.for_agent(__package__, "planner", pack=brief.domain_pack)
        provenance: dict[str, str] = {}

        def render(name: str, **variables: Any) -> str:
            prompt = prompts.render(name, **variables)
            provenance[name] = prompt.template_hash
            return prompt.text

        snippets = self.evidence(ctx, corpus, hypothesis, cfg.evidence_snippets)
        known = [e.name for e in km.entities if e.kind == "dataset"][:15]
        compute = compute_profile(ctx.app_config)
        feedback = [f.text for f in ctx.feedback]
        checker = self.checker(ctx)

        core_issues: list[str] = []
        res_issues: list[str] = []
        cmp_issues: list[str] = []
        core = resources = estimate = None
        for _round in range(cfg.repair_rounds + 1):
            if core is None or core_issues:
                core = normalise_core(
                    ctx.llm.structured(
                        PlanCore,
                        render(
                            "plan_core.j2",
                            h=hypothesis,
                            snippets=snippets,
                            compute=compute,
                            feedback=feedback,
                            issues=core_issues,
                            min_seeds=cfg.min_seeds,
                        ),
                        profile="reason",
                        system=SYSTEM,
                    )
                )
                core_issues = design_issues(core, cfg.min_seeds)
            if resources is None or res_issues:
                plan = ctx.llm.structured(
                    ResourcePlan,
                    render("resources.j2", h=hypothesis, core=core, known_datasets=known, issues=res_issues),
                    profile="reason",
                    system=SYSTEM,
                )
                resources = [self.verify(checker, choice, cfg.check_resources) for choice in plan.resources]
                res_issues = resource_issues(resources, cfg.max_dataset_gb)
            runs = len(core.conditions + core.ablations) * core.n_seeds
            if estimate is None or cmp_issues:
                estimate = ctx.llm.structured(
                    ComputeEstimate,
                    render("compute.j2", core=core, resources=resources, compute=compute, runs=runs, issues=cmp_issues),
                    profile="reason",
                    system=SYSTEM,
                )
            request, cmp_issues = fit_compute(estimate, runs, ctx.app_config)
            if not (core_issues or res_issues or cmp_issues):
                break
            ctx.emit("plan_repair", design=core_issues, resources=res_issues, compute=cmp_issues)
        assert core is not None and resources is not None

        seeds = list(range(core.n_seeds))
        conditions = [*core.conditions, *core.ablations]
        plan_artifact = ExperimentPlan(
            hypothesis_id=hypothesis.id,
            hypothesis=hypothesis.statement,
            null_hypothesis=hypothesis.null_hypothesis,
            objective=core.objective,
            design=core.design,
            independent_variables=core.independent_variables,
            dependent_variables=core.dependent_variables,
            controls=core.controls,
            conditions=core.conditions,
            ablations=core.ablations,
            metrics=one_primary(core.metrics),
            statistical_tests=core.statistical_tests,
            seeds=seeds,
            procedure=core.procedure,
            resources=resources,
            compute=request,
            expected_results=core.expected_results,
            risks=core.risks,
            results_schema=results_schema(conditions, core.metrics, seeds),
            issues=core_issues + res_issues + cmp_issues,
        )
        markdown = ctx.workspace.write_text("experiment_plan.md", render_markdown(plan_artifact, hypothesis))
        plan_artifact.markdown_path = ctx.relative(markdown)
        ctx.outputs.put(
            "experiment_plan", plan_artifact, meta={"review": plan_artifact.markdown_path, "prompts": provenance}
        )

    @staticmethod
    def evidence(ctx: AgentContext, corpus: Corpus, h: Hypothesis, k: int) -> list[dict[str, str]]:
        kb = open_corpus_kb(ctx, corpus)
        if kb is None or k == 0:
            return []
        titles = {p.id: p.title for p in corpus.papers}
        try:
            query = " ".join([h.statement, *h.methods, *h.datasets])
            return [
                {
                    "title": titles.get(hit["paper_id"], hit["paper_id"]),
                    "section": hit["section"],
                    "text": hit["text"][:700],
                }
                for hit in kb.search(query, k=k)
            ]
        finally:
            kb.close()

    @staticmethod
    def checker(ctx: AgentContext) -> ResourceChecker:
        injected = ctx.tools.get("resource_checker")
        if injected is not None:
            return injected
        http = PoliteHttp(cache_path=ctx.workspace.path("http_cache.sqlite"), limits={"huggingface.co": 0.3})
        return HubChecker(http, token=os.environ.get("HF_TOKEN"))

    @staticmethod
    def verify(checker: ResourceChecker, choice: Any, enabled: bool) -> ResourceSpec:
        spec = ResourceSpec(**choice.model_dump())
        if not enabled:
            spec.note = "not checked"
            spec.verified = True
            return spec
        result = checker.check(choice)
        spec.verified, spec.license, spec.size_gb, spec.note = (
            result.exists,
            result.license,
            result.size_gb,
            result.note,
        )
        return spec


def render_markdown(plan: ExperimentPlan, h: Hypothesis) -> str:
    c, primary = plan.compute, plan.primary_metric()
    lines = [
        f"# Experiment plan: {h.title}",
        "",
        "Approve with `rtool approve <run>`, or reject with feedback.",
        "",
    ]
    if plan.issues:
        lines += ["## Unresolved issues (please check)", "", *[f"- {i}" for i in plan.issues], ""]
    lines += [
        f"**Hypothesis.** {plan.hypothesis}",
        "",
        f"**Null.** {plan.null_hypothesis}",
        "",
        f"**Objective.** {plan.objective}",
        "",
        f"**Design.** {plan.design}",
        "",
        "## Conditions",
        "",
        *[f"- `{x.name}`{' (baseline)' if x.is_baseline else ''}: {x.description}" for x in plan.conditions],
        *[f"- `{x.name}` (ablation): {x.description}" for x in plan.ablations],
        "",
        "## Primary outcome (pre-registered)",
        "",
        f"`{primary.name}` ({'higher' if primary.higher_is_better else 'lower'} is better): {primary.definition}",
        "",
        "Approving this plan fixes it: the comparisons on this metric are the confirmatory test of the "
        "hypothesis, and the report presents them first."
        + (" Every other metric is reported as exploratory." if len(plan.metrics) > 1 else ""),
        "",
        "## Metrics",
        "",
        *[
            f"- `{m.name}` ({'higher' if m.higher_is_better else 'lower'} is better"
            f"{'; primary' if m is primary else '; exploratory'}): {m.definition}"
            for m in plan.metrics
        ],
        "",
        "## Statistics",
        "",
        *[f"- {t.test}: {t.comparison} (alpha {t.alpha}, correction: {t.correction})" for t in plan.statistical_tests],
        "- multiple comparisons: p-values are adjusted (Holm, or Bonferroni if a test above asks for it) "
        "within the confirmatory and within the exploratory comparisons",
        f"- seeds per condition: {len(plan.seeds)}",
        "",
        "## Procedure",
        "",
        *[f"{n}. {step}" for n, step in enumerate(plan.procedure, 1)],
        "",
        "## Resources",
        "",
        *[
            f"- {r.name} — {r.source}:`{r.identifier}` ({r.kind}; {'found' if r.verified else 'NOT FOUND'}"
            f"{f', {r.size_gb:g} GB' if r.size_gb else ''}{f', licence {r.license}' if r.license else ''})"
            f"{f' — {r.note}' if r.note else ''}"
            for r in plan.resources
        ],
        "",
        "## Compute",
        "",
        f"- {c.runs} runs on `{c.partition}`"
        + (f" with {c.gpus_per_run}x {c.gpu_type}" if c.gpus_per_run else " (CPU)"),
        f"- per run: ~{c.hours_per_run:g} h (time limit {c.time_limit}); total ~{c.gpu_hours_total:g} GPU-hours",
        f"- why: {c.justification}",
        "",
        f"**Expected results.** {plan.expected_results}",
        "",
        "## Risks",
        "",
        *[f"- {r}" for r in plan.risks],
    ]
    return "\n".join(lines) + "\n"
