"""Protocol: a study a person carries out, with its sample size, randomised allocation and checklist.

One model call designs the study (conditions, the unit of analysis, the primary outcome, the smallest
effect worth detecting, materials, blinding); everything else is computed. The sample size comes from
the usual normal approximation with a small-sample correction (64 per group for d = 0.5, alpha 0.05,
power 0.8; 34 units for a paired design), with alpha split across the comparisons with the control.
The allocation is a seeded, block-randomised table. The checklist is shown at the protocol checkpoint
and its failures become plan issues.
"""

from __future__ import annotations

import csv
import io
import math
import random
from statistics import NormalDist
from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...core.context import AgentContext
from ...core.registry import register_agent
from ...llm.prompts import PromptLibrary
from ...packs import load_pack, pack_plan_issues
from ...packs.units import pseudoreplication_issues
from ...schemas.common import ResearchBrief, SelectedItem
from ...schemas.hypothesis import Hypothesis
from ...schemas.plan import ComputeEstimate, ExperimentPlan, Metric, ResourceSpec
from ...schemas.report import Check
from ..coder.results import family_of_tests
from ..common import SYSTEM, refuse
from ..planner.checks import (
    brief_issues,
    design_issues,
    fit_compute,
    follow_brief,
    normalise_core,
    one_primary,
    results_schema,
)
from .schemas import DATA_FILE, Protocol, ProtocolDraft

ALPHA, POWER = 0.05, 0.8


def sample_size(
    effect: float, *, paired: bool, comparisons: int = 1, alpha: float = ALPHA, power: float = POWER
) -> int:
    """Units per condition (paired: units in all) to detect Cohen's d = ``effect`` in a two-sided test."""
    a = alpha / max(1, comparisons)  # Bonferroni across the comparisons with the control: safe for Holm too
    z_a, z_b = NormalDist().inv_cdf(1 - a / 2), NormalDist().inv_cdf(power)
    n = ((z_a + z_b) / effect) ** 2 + z_a**2 / 2 if paired else 2 * ((z_a + z_b) / effect) ** 2 + z_a**2 / 4
    return max(2, math.ceil(n))


def allocation(conditions: list[str], n: int, *, paired: bool, seed: int) -> list[dict[str, str]]:
    """Who gets which condition: permuted blocks of every condition once (paired: each unit's order)."""
    rng = random.Random(seed)
    rows = []
    if paired:
        for i in range(n):
            order = list(conditions)
            rng.shuffle(order)
            rows.append({"unit": f"U{i + 1:03d}", "order": " then ".join(order)})
        return rows
    units = iter(range(1, n * len(conditions) + 1))
    for _ in range(n):
        block = list(conditions)
        rng.shuffle(block)
        rows += [{"unit": f"U{next(units):03d}", "condition": c} for c in block]
    return rows


class ProtocolConfig(BaseModel):
    repair_rounds: int = Field(default=1, ge=0, le=3)
    allocation_seed: int = 20261121  # fixed, so the allocation can be reproduced from the protocol


@register_agent("protocol")
class ProtocolAgent(Agent):
    description = "Designs a study a person carries out: sample size, randomised allocation, materials, checklist."
    consumes = ("research_brief", "selected_hypothesis", "corpus", "knowledge_map")
    produces = ("experiment_plan", "protocol")
    config_model = ProtocolConfig

    def run(self, ctx: AgentContext) -> None:
        cfg: ProtocolConfig = self.config
        brief: ResearchBrief = ctx.inputs.get("research_brief")
        selected: SelectedItem = ctx.inputs.get("selected_hypothesis")
        hypothesis = Hypothesis.model_validate(selected.item)
        pack = load_pack(brief.domain_pack)
        refuse(pack, "design a study of this hypothesis", hypothesis.statement)
        prompts = PromptLibrary.for_agent(__package__, "study", pack=brief.domain_pack)
        feedback = [f.text for f in ctx.feedback]
        issues: list[str] = []
        provenance: dict[str, str] = {}
        draft: ProtocolDraft | None = None
        for _round in range(cfg.repair_rounds + 1):
            prompt = prompts.render("protocol.j2", h=hypothesis, brief=brief, feedback=feedback, issues=issues)
            provenance["protocol.j2"] = prompt.template_hash
            draft = normalise_core(ctx.llm.structured(ProtocolDraft, prompt.text, profile="reason", system=SYSTEM))
            draft = follow_brief(draft, brief)
            refuse(pack, "write this protocol", draft.objective)
            draft.n_seeds = 1  # the units are the repetitions
            issues = design_issues(draft, 1, units=True) + brief_issues(
                brief, conditions=draft.conditions, metrics=draft.metrics, units=True
            )
            if not issues:
                break
            ctx.emit("protocol_repair", issues=issues)
        assert draft is not None
        plan = self.plan(ctx, hypothesis, draft, issues)
        protocol = self.protocol(ctx, plan, draft, cfg.allocation_seed)
        plan.issues += [
            f"protocol checklist: {c.name}" + (f" ({c.detail})" if c.detail else "")
            for c in protocol.checklist
            if not c.ok
        ]
        plan.issues += pack_plan_issues(pack, plan, brief)
        ctx.outputs.put("experiment_plan", plan, meta={"review": protocol.markdown_path, "prompts": provenance})
        ctx.outputs.put("protocol", protocol, meta={"review": protocol.markdown_path})

    @staticmethod
    def plan(ctx: AgentContext, hypothesis: Hypothesis, draft: ProtocolDraft, issues: list[str]) -> ExperimentPlan:
        estimate = ComputeEstimate(gpu_memory_gb=0, gpus_per_run=0, hours_per_run=0.05, justification="a data table")
        conditions = [*draft.conditions, *draft.ablations]
        request, compute_issues = fit_compute(estimate, len(conditions), ctx.app_config)
        metrics = one_primary([Metric.model_validate(m.model_dump()) for m in draft.metrics])
        data = ResourceSpec(
            kind="dataset",
            name="study_data",
            source="local",
            identifier="study_data",
            purpose="the data the study records, one row per unit and condition",
            note="recorded by the researcher and uploaded at the data checkpoint",
        )
        return ExperimentPlan(
            hypothesis_id=hypothesis.id,
            hypothesis=hypothesis.statement,
            null_hypothesis=hypothesis.null_hypothesis,
            objective=draft.objective,
            design=draft.design,
            independent_variables=draft.independent_variables,
            dependent_variables=draft.dependent_variables,
            controls=draft.controls,
            conditions=draft.conditions,
            ablations=draft.ablations,
            metrics=metrics,
            statistical_tests=draft.statistical_tests,
            seeds=[0],
            procedure=draft.procedure,
            resources=[data],
            compute=request,
            expected_results=draft.expected_results,
            risks=draft.risks,
            results_schema=results_schema(conditions, metrics, [0], units=True),
            issues=issues + compute_issues,
            replicates="units",
            unit=draft.unit_of_analysis,
        )

    @staticmethod
    def protocol(ctx: AgentContext, plan: ExperimentPlan, draft: ProtocolDraft, seed: int) -> Protocol:
        names = [c.name for c in plan.run_conditions()]
        baseline = next((c.name for c in plan.conditions if c.is_baseline), names[0])
        family = family_of_tests(plan.statistical_tests)[0]
        paired = family == "wilcoxon" or any("paired" in t.test.lower() for t in plan.statistical_tests)
        n = sample_size(draft.expected_effect_size, paired=paired, comparisons=len(names) - 1)
        rows = allocation(names, n, paired=paired, seed=seed)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        allocation_path = ctx.workspace.write_text("allocation.csv", buffer.getvalue())
        metric_names = [m.name for m in plan.metrics]
        data_format = (
            f"{DATA_FILE}: a CSV file with a header row and one row per unit and condition; columns unit (the "
            f"{plan.unit or 'unit'} id), condition (one of {', '.join(names)}), {', '.join(metric_names)} (numbers), "
            "and optionally batch (site, run or plate)"
        )
        primary = plan.primary_metric()
        unit_problems = pseudoreplication_issues(plan, None)
        checklist = [
            Check(name="a control condition", ok=any(c.is_baseline for c in plan.conditions), detail=baseline),
            Check(name="the primary outcome is fixed before any data", ok=True, detail=primary.name),
            Check(
                name="the sample size is computed",
                ok=n >= 2,
                detail=f"{n} {'units in all, each under every condition' if paired else 'units per condition'} for "
                f"d = {draft.expected_effect_size:g}, alpha {ALPHA:g}, power {POWER:g}",
            ),
            Check(
                name="the unit of analysis is independent",
                ok=not unit_problems,
                detail=unit_problems[0] if unit_problems else plan.unit,
            ),
            Check(name="materials are listed", ok=bool(draft.materials), detail=", ".join(draft.materials[:5])),
            Check(name="blinding is stated", ok=bool(draft.blinding.strip()), detail=draft.blinding[:120]),
        ]
        protocol = Protocol(
            unit=plan.unit,
            conditions=names,
            baseline=baseline,
            primary_metric=primary.name,
            effect_size=draft.expected_effect_size,
            alpha=ALPHA,
            power=POWER,
            paired=paired,
            n_per_condition=n,
            allocation_path=ctx.relative(allocation_path),
            materials=draft.materials,
            blinding=draft.blinding,
            data_format=data_format,
            checklist=checklist,
        )
        markdown = ctx.workspace.write_text("protocol.md", render_protocol(plan, protocol))
        protocol.markdown_path = ctx.relative(markdown)
        return protocol


def render_protocol(plan: ExperimentPlan, protocol: Protocol, **_: Any) -> str:
    each = "units, each under every condition" if protocol.paired else "units per condition"
    lines = [
        f"# Study protocol: {plan.hypothesis}",
        "",
        f"**Design.** {plan.design}",
        "",
        f"**Unit of analysis.** {protocol.unit}. **Conditions.** "
        + "; ".join(
            f"`{c.name}`{' (control)' if c.is_baseline else ''}: {c.description}" for c in plan.run_conditions()
        ),
        "",
        f"**Primary outcome** (fixed now, before any data): `{protocol.primary_metric}`.",
        "",
        f"**Sample size.** {protocol.n_per_condition} {each} to detect d = {protocol.effect_size:g} with alpha "
        f"{protocol.alpha:g} and power {protocol.power:g}.",
        "",
        f"**Allocation.** `{protocol.allocation_path}` (seeded, block-randomised).",
        "",
        f"**Blinding.** {protocol.blinding}",
        "",
        "**Materials.** " + "; ".join(protocol.materials),
        "",
        "**Procedure.** " + " ".join(f"({i}) {step}" for i, step in enumerate(plan.procedure, 1)),
        "",
        f"**Data to bring back.** {protocol.data_format}. Upload it with `rtool add-data <run> data.csv`.",
        "",
        "## Checklist",
        "",
        *(f"- [{'x' if c.ok else ' '}] {c.name}" + (f": {c.detail}" if c.detail else "") for c in protocol.checklist),
        "",
    ]
    return "\n".join(lines)
