"""Hand edits of the experiment plan: keep what the researcher wrote, recompute what follows from it.

The researcher may change the design, conditions, metrics, statistics, seeds,
procedure, resources and compute. Derived fields are recomputed (results.json
schema, number of runs, GPU-hours, the time limit when only the hours changed),
resources whose identity changed lose their "found" mark until the download
step checks them, and the remaining problems are listed in ``issues`` as the
planner would. The hypothesis is not editable here: it comes from the choice
on the Hypotheses page. The partition is never re-picked (``fit_compute`` is
not used), so a deliberate choice stays.
"""

from __future__ import annotations

from typing import Any

from ...core.editing import EditContext, EditResult, artifact_editor
from ...core.errors import ActionError, ArtifactError
from ...hpc.jobspec import parse_time_limit
from ...schemas.plan import ComputeRequest, ExperimentPlan
from .agent import PlannerConfig, render_markdown
from .checks import SAFETY_FACTOR, UNCHECKED, format_time, plan_issues, results_schema, slug

FIXED = ("hypothesis_id", "hypothesis", "null_hypothesis", "markdown_path")


def _unique(names: list[str], what: str) -> None:
    if any(not n for n in names):
        raise ActionError(f"every {what} needs a name")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ActionError(f"{what} names must be different (they become keys in the results): {repeated}")


def _compute(plan: ExperimentPlan, previous: ExperimentPlan, ctx: EditContext) -> ComputeRequest:
    c = plan.compute
    spec = ctx.app_config.slurm.partitions.get(c.partition)
    gpu_type = c.gpu_type
    if c.partition != previous.compute.partition:
        gpu_type = spec.gpu_type if spec is not None else None
    time_limit = c.time_limit
    if time_limit == previous.compute.time_limit and (
        c.hours_per_run != previous.compute.hours_per_run or c.partition != previous.compute.partition
    ):
        seconds = max(c.hours_per_run * 3600 * SAFETY_FACTOR, 600)
        if spec is not None:
            seconds = min(seconds, parse_time_limit(spec.max_time))
        time_limit = format_time(seconds)
    try:
        parse_time_limit(time_limit)
    except ValueError as exc:
        raise ActionError(str(exc)) from None
    if c.hours_per_run <= 0 or c.gpus_per_run < 0 or c.gpu_memory_gb < 0:
        raise ActionError("hours per run must be positive, and GPUs and GPU memory cannot be negative")
    runs = len(plan.run_conditions()) * len(plan.seeds)
    return c.model_copy(
        update={
            "gpu_type": gpu_type if c.gpus_per_run else None,
            "time_limit": time_limit,
            "runs": runs,
            "gpu_hours_total": round(runs * c.gpus_per_run * c.hours_per_run, 2),
        }
    )


def _hypothesis_title(ctx: EditContext, plan: ExperimentPlan) -> str:
    version = ctx.previous.inputs.get("selected_hypothesis")
    try:
        item = ctx.store.envelope("selected_hypothesis", version).payload.get("item") or {}
    except ArtifactError:
        item = {}
    return str(item.get("title") or plan.hypothesis[:80])


@artifact_editor("experiment_plan")
def edit_plan(value: dict[str, Any], ctx: EditContext) -> EditResult:
    previous = ExperimentPlan.model_validate(ctx.previous.payload)
    plan = ExperimentPlan.model_validate({**value, **previous.model_dump(include=set(FIXED))})

    for condition in plan.run_conditions():
        condition.name = slug(condition.name) if condition.name.strip() else ""
    for metric in plan.metrics:
        metric.name = slug(metric.name) if metric.name.strip() else ""
    if not plan.conditions:
        raise ActionError("the plan needs at least one condition (and a baseline to compare with)")
    if not plan.metrics:
        raise ActionError("the plan needs at least one metric")
    _unique([c.name for c in plan.run_conditions()], "condition")
    _unique([m.name for m in plan.metrics], "metric")
    plan.seeds = sorted(set(plan.seeds))
    if not plan.seeds or plan.seeds[0] < 0:
        raise ActionError("the plan needs at least one seed, and seeds cannot be negative")

    known = {(r.source, r.identifier, r.subset, r.split): r for r in previous.resources}
    for resource in plan.resources:
        found = known.get((resource.source, resource.identifier, resource.subset, resource.split))
        if found is None:  # new, or pointing somewhere else: nobody has checked that it exists
            resource.verified, resource.license, resource.size_gb, resource.note = False, "", None, UNCHECKED
        else:
            resource.verified, resource.license, resource.size_gb, resource.note = (
                found.verified,
                found.license,
                found.size_gb,
                found.note,
            )

    plan.compute = _compute(plan, previous, ctx)
    plan.results_schema = results_schema(plan.run_conditions(), plan.metrics, plan.seeds)
    cfg = ctx.config if isinstance(ctx.config, PlannerConfig) else PlannerConfig()
    plan.issues = plan_issues(plan, ctx.app_config, cfg.min_seeds, cfg.max_dataset_gb)
    files = {}
    if plan.markdown_path:
        files[plan.markdown_path.rsplit("/", 1)[-1]] = render_markdown(plan, _hypothesis_title(ctx, plan))
    return EditResult(plan, files=files)
