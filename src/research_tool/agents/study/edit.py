"""Hand edits of a study: the protocol (effect size, materials, blinding) and the design behind it.

Both edits change the protocol and the plan together, because each follows from the other: the sample size, the
allocation table, the data format and the checklist are worked out again from the design and the three protocol
settings (the same code the protocol agent uses), and the plan's list of problems carries the checklist's failures.
The engine stores the main artifact and the other as one edit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ...core.editing import EditContext, EditResult, artifact_editor
from ...core.errors import ActionError, ArtifactError
from ...packs import DEFAULT_PACK, load_pack, pack_plan_issues
from ...schemas.common import ResearchBrief
from ...schemas.plan import ExperimentPlan
from ..planner.checks import brief_issues, compute_issues, design_issues, results_schema, slug
from .protocol import ProtocolConfig, make_protocol, render_protocol
from .schemas import Protocol

EDITABLE = ("effect_size", "materials", "blinding")
# the plan fields a researcher edits as the design; everything else is the stage's or follows from these
FIXED = ("hypothesis_id", "hypothesis", "null_hypothesis", "markdown_path", "resources", "seeds", "replicates")
CHECKLIST = "protocol checklist: "


def _seed(ctx: EditContext) -> int:
    return ctx.config.allocation_seed if isinstance(ctx.config, ProtocolConfig) else ProtocolConfig().allocation_seed


def _plan(ctx: EditContext) -> ExperimentPlan:
    try:
        return ExperimentPlan.model_validate(ctx.store.envelope("experiment_plan").payload)
    except ArtifactError:
        raise ActionError("there is no experiment plan to compute the protocol from") from None


def _with_checklist(plan: ExperimentPlan, protocol: Protocol) -> ExperimentPlan:
    """The plan with its problems list carrying this protocol's failed checks (and no older ones)."""
    issues = [i for i in plan.issues if not i.startswith(CHECKLIST)]
    issues += [f"{CHECKLIST}{c.name}" + (f" ({c.detail})" if c.detail else "") for c in protocol.checklist if not c.ok]
    return plan.model_copy(update={"issues": issues})


def _files(previous: Protocol, protocol: Protocol, allocation_csv: str, plan: ExperimentPlan) -> dict[str, str]:
    files = {previous.allocation_path.rsplit("/", 1)[-1]: allocation_csv}
    if previous.markdown_path:
        files[previous.markdown_path.rsplit("/", 1)[-1]] = render_protocol(plan, protocol)
    return files


@artifact_editor("protocol", agent="protocol")
def edit_protocol(value: dict[str, Any], ctx: EditContext) -> EditResult:
    previous = Protocol.model_validate(ctx.previous.payload)
    wanted = Protocol.model_validate({**previous.model_dump(), **{k: value[k] for k in EDITABLE if k in value}})
    if not 0 < wanted.effect_size <= 5:
        raise ActionError("the effect size (Cohen's d) must be above 0 and at most 5")
    materials = [m.strip() for m in wanted.materials if m.strip()]
    plan = _plan(ctx)
    protocol, allocation_csv = make_protocol(plan, wanted.effect_size, materials, wanted.blinding.strip(), _seed(ctx))
    protocol.allocation_path, protocol.markdown_path = previous.allocation_path, previous.markdown_path
    return EditResult(
        protocol,
        files=_files(previous, protocol, allocation_csv, plan),
        extra={"experiment_plan": _with_checklist(plan, protocol)},
    )


@artifact_editor("experiment_plan", agent="protocol")
def edit_design(value: dict[str, Any], ctx: EditContext) -> EditResult:
    """The design (objective, variables, conditions, metrics, tests, procedure, unit): the protocol follows."""
    previous = ExperimentPlan.model_validate(ctx.previous.payload)
    plan = ExperimentPlan.model_validate({**value, **previous.model_dump(include=set(FIXED))})
    for condition in plan.run_conditions():
        condition.name = slug(condition.name) if condition.name.strip() else ""
    for metric in plan.metrics:
        metric.name = slug(metric.name) if metric.name.strip() else ""
    if len(plan.conditions) < 2:
        raise ActionError("a study compares at least two conditions: a control and one more")
    if not plan.metrics:
        raise ActionError("the study needs at least one outcome (metric)")
    if not plan.unit.strip():
        raise ActionError("say what one unit of analysis is (for example participant)")
    for names, what in (
        ([c.name for c in plan.run_conditions()], "condition"),
        ([m.name for m in plan.metrics], "metric"),
    ):
        if any(not n for n in names):
            raise ActionError(f"every {what} needs a name")
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ActionError(f"{what} names must be different (they become columns and keys): {repeated}")
    plan.seeds = [0]  # the units are the repetitions
    plan.compute = previous.compute.model_copy(update={"runs": len(plan.run_conditions())})
    plan.results_schema = results_schema(plan.run_conditions(), plan.metrics, plan.seeds, units=True)
    try:
        brief = ResearchBrief.model_validate(
            ctx.store.envelope("research_brief", ctx.previous.inputs.get("research_brief")).payload
        )
    except ArtifactError:
        brief = None
    # the problems the protocol agent looks for (not resource checks: the table is uploaded later)
    core = SimpleNamespace(
        conditions=plan.conditions,
        ablations=plan.ablations,
        n_seeds=1,
        metrics=plan.metrics,
        statistical_tests=plan.statistical_tests,
    )
    plan.issues = (
        design_issues(core, 1, units=True)  # type: ignore[arg-type]
        + brief_issues(brief, conditions=plan.conditions, metrics=plan.metrics, units=True)
        + compute_issues(plan.compute, ctx.app_config)
        + pack_plan_issues(load_pack(brief.domain_pack if brief else DEFAULT_PACK), plan, brief)
    )
    kept = Protocol.model_validate(ctx.store.envelope("protocol").payload)
    protocol, allocation_csv = make_protocol(plan, kept.effect_size, kept.materials, kept.blinding, _seed(ctx))
    protocol.allocation_path, protocol.markdown_path = kept.allocation_path, kept.markdown_path
    return EditResult(
        _with_checklist(plan, protocol),
        files=_files(kept, protocol, allocation_csv, plan),
        extra={"protocol": protocol},
    )
