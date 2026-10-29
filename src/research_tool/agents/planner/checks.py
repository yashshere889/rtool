"""Deterministic plan checks: fit compute to the cluster, validate the design, define results.json."""

from __future__ import annotations

import re
from typing import Any

from ...core.config import AppConfig
from ...hpc.jobspec import format_time_limit, parse_time_limit
from ...schemas.plan import ComputeEstimate, ComputeRequest, Condition, ExperimentPlan, Metric, PlanCore, ResourceSpec

SAFETY_FACTOR = 1.3  # runs take longer than estimated
_SLUG = re.compile(r"[^a-z0-9]+")
UNCHECKED = "changed by hand; checked when it is downloaded"  # ResourceSpec.note of an edited resource


def slug(name: str) -> str:
    return _SLUG.sub("_", name.lower()).strip("_") or "unnamed"


format_time = format_time_limit  # whole minutes, Slurm syntax


def fit_compute(estimate: ComputeEstimate, runs: int, config: AppConfig) -> tuple[ComputeRequest, list[str]]:
    """Pick the first allowed partition (in configured preference order) that fits one run.

    Also checks the partition's time limit and the run's GPU-hour budget.
    """
    slurm = config.slurm
    issues: list[str] = []
    gpu_type: str | None = None
    if estimate.gpus_per_run == 0:
        partition = slurm.episode.cpu_partition
    elif not slurm.partitions:
        # No cluster profile configured (e.g. local development): nothing to validate against.
        partition, gpu_type = slurm.episode.partition, slurm.episode.gpu_type
    else:
        allowed = slurm.experiment_partitions or [p for p, s in slurm.partitions.items() if s.gpus_per_node]
        fits = [
            (name, spec)
            for name in allowed
            if (spec := slurm.partitions.get(name)) is not None
            and spec.gpus_per_node >= estimate.gpus_per_run
            and (spec.gpu_memory_gb or 0) >= estimate.gpu_memory_gb
        ]
        if fits:
            partition, spec = fits[0]  # the user's order: e.g. plentiful L40S before scarce H100
            gpu_type = spec.gpu_type
        else:
            issues.append(
                f"no allowed partition has {estimate.gpus_per_run} GPU(s) with {estimate.gpu_memory_gb:g} GB each; "
                "use a smaller model or batch size, or more GPUs per run"
            )
            known = [(n, slurm.partitions[n]) for n in allowed if n in slurm.partitions]
            if known:
                partition, spec = max(known, key=lambda item: item[1].gpu_memory_gb or 0)
                gpu_type = spec.gpu_type
            else:
                partition = allowed[0] if allowed else "gpu"
    spec = slurm.partitions.get(partition)
    wanted = estimate.hours_per_run * 3600 * SAFETY_FACTOR
    if spec is not None and wanted > parse_time_limit(spec.max_time):
        issues.append(
            f"one run needs about {estimate.hours_per_run * SAFETY_FACTOR:.1f} h but {partition} allows "
            f"{spec.max_time}; split runs into resumable chunks or reduce the work per run"
        )
        wanted = parse_time_limit(spec.max_time)
    total = round(runs * estimate.gpus_per_run * estimate.hours_per_run, 2)
    budget = config.budgets.run.max_gpu_hours
    if budget is not None and total > budget:
        issues.append(f"the plan needs about {total:g} GPU-hours; the budget is {budget:g}")
    request = ComputeRequest(
        partition=partition,
        gpu_type=gpu_type,
        gpus_per_run=estimate.gpus_per_run,
        time_limit=format_time(max(wanted, 600)),
        runs=runs,
        hours_per_run=estimate.hours_per_run,
        gpu_hours_total=total,
        gpu_memory_gb=estimate.gpu_memory_gb,
        justification=estimate.justification,
    )
    return request, issues


def normalise_core(core: PlanCore) -> PlanCore:
    """Snake_case names, so they can be used as keys in results.json; the first metric is primary if none is."""
    for condition in [*core.conditions, *core.ablations]:
        condition.name = slug(condition.name)
    for metric in core.metrics:
        metric.name = slug(metric.name)
    if not any(m.primary for m in core.metrics):
        core.metrics[0].primary = True
    return core


def one_primary(metrics: list[Metric]) -> list[Metric]:
    """Exactly one primary outcome: the first metric marked primary (repair rounds may not have settled it)."""
    primary = next((m for m in metrics if m.primary), metrics[0])
    for metric in metrics:
        metric.primary = metric is primary
    return metrics


def primary_issues(metrics: list[Metric]) -> list[str]:
    primary = [m.name for m in metrics if m.primary]
    if len(primary) > 1:
        return [
            f"mark exactly one metric as the primary outcome (primary = true), not {len(primary)}: {primary}; "
            "the others are reported as exploratory"
        ]
    return []


# Words in a metric's name that usually say which way is better. A name with words of both kinds
# ("error_reduction") is left alone.
_LOWER = {
    "error", "errors", "loss", "losses", "collision", "collisions", "crash", "crashes", "latency", "delay",
    "duration", "runtime", "cost", "regret", "violation", "violations", "failure", "failures", "mse", "rmse",
    "mae", "perplexity", "overhead", "distance", "steps", "wer", "cer",
}  # fmt: skip
_HIGHER = {
    "accuracy", "acc", "success", "reward", "return", "score", "f1", "precision", "recall", "auc", "auroc",
    "efficiency", "throughput", "gain", "reduction", "improvement", "saved", "coverage", "bleu", "rouge",
}  # fmt: skip


def metric_issues(metrics: list[Metric]) -> list[str]:
    """Metrics whose direction contradicts their name (seen on Barkla: collision_frequency, 'higher is better')."""
    issues = []
    for metric in metrics:
        words = set(slug(metric.name).split("_"))
        lower, higher = sorted(words & _LOWER), sorted(words & _HIGHER)
        if metric.higher_is_better and lower and not higher:
            issues.append(
                f"metric {metric.name} says higher is better, but a higher {lower[0]} is usually worse: "
                "set higher_is_better to false if smaller values are better"
            )
        elif not metric.higher_is_better and higher and not lower:
            issues.append(
                f"metric {metric.name} says lower is better, but a higher {higher[0]} is usually better: "
                "set higher_is_better to true if larger values are better"
            )
    return issues


def design_issues(core: PlanCore, min_seeds: int) -> list[str]:
    return (
        _design_issues(core.conditions, core.ablations, core.n_seeds, min_seeds)
        + primary_issues(core.metrics)
        + metric_issues(core.metrics)
    )


def _design_issues(conditions: list[Condition], ablations: list[Condition], n_seeds: int, min_seeds: int) -> list[str]:
    issues = []
    if not any(c.is_baseline for c in conditions):
        issues.append("no condition is marked as the baseline (is_baseline = true)")
    names = [c.name for c in [*conditions, *ablations]]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        issues.append(f"condition names must be unique: {duplicates}")
    if n_seeds < min_seeds:
        issues.append(f"use at least {min_seeds} seeds (independent runs) per condition, not {n_seeds}")
    return issues


def resource_issues(resources: list[ResourceSpec], max_dataset_gb: float) -> list[str]:
    issues = []
    if not any(r.kind == "dataset" for r in resources):
        issues.append("no dataset is listed")
    for r in resources:
        if r.source == "manual":
            issues.append(
                f"{r.name} is marked manual, so nothing will be downloaded for it: use a public resource "
                "(Hugging Face, a URL, torchvision or a pip package) if one exists; otherwise someone must "
                "provide its files, or the experiment code must build it"
            )
        elif not r.verified and r.note == UNCHECKED:
            issues.append(f"{r.name} ({r.source}: {r.identifier}) was changed by hand and is not checked yet")
        elif not r.verified:
            issues.append(
                f"{r.name} ({r.source}: {r.identifier}) could not be found: {r.note or 'check the identifier'}"
            )
        elif r.size_gb is not None and r.size_gb > max_dataset_gb:
            issues.append(f"{r.name} is {r.size_gb:g} GB, above the {max_dataset_gb:g} GB limit; use a subset")
    return issues


def results_schema(conditions: list[Condition], metrics: list[Metric], seeds: list[int]) -> dict[str, Any]:
    """The JSON Schema the Coder's results.json must satisfy (one record per condition and seed)."""
    names = [m.name for m in metrics]
    return {
        "title": "ExperimentResults",
        "type": "object",
        "required": ["runs"],
        "properties": {
            "runs": {
                "type": "array",
                "minItems": len(conditions) * len(seeds),
                "items": {
                    "type": "object",
                    "required": ["condition", "seed", "metrics"],
                    "properties": {
                        "condition": {"enum": [c.name for c in conditions]},
                        "seed": {"enum": seeds},
                        "metrics": {
                            "type": "object",
                            "required": names,
                            "properties": {name: {"type": "number"} for name in names},
                        },
                    },
                },
            },
            "notes": {"type": "string"},
        },
    }


def compute_issues(request: ComputeRequest, config: AppConfig) -> list[str]:
    """Does a compute request (e.g. edited by hand) fit the cluster and the budget? ``fit_compute`` is not
    used for that: it would pick a partition again instead of checking the one chosen."""
    slurm = config.slurm
    issues = []
    try:
        wanted = parse_time_limit(request.time_limit)
    except ValueError as exc:
        return [str(exc)]
    spec = slurm.partitions.get(request.partition)
    if slurm.partitions and spec is None:
        issues.append(f"{request.partition} is not a known partition")
    if request.gpus_per_run and slurm.experiment_partitions and request.partition not in slurm.experiment_partitions:
        issues.append(
            f"experiments may not run on {request.partition}; allowed: {', '.join(slurm.experiment_partitions)}"
        )
    if spec is not None:
        if request.gpus_per_run > spec.gpus_per_node:
            issues.append(f"{request.partition} has {spec.gpus_per_node} GPU(s) per node, not {request.gpus_per_run}")
        elif request.gpus_per_run and request.gpu_memory_gb > (spec.gpu_memory_gb or 0):
            issues.append(
                f"one run needs {request.gpu_memory_gb:g} GB per GPU; {request.partition} has "
                f"{spec.gpu_memory_gb or 0:g} GB"
            )
        if wanted > parse_time_limit(spec.max_time):
            issues.append(
                f"the time limit {request.time_limit} is above the {spec.max_time} {request.partition} allows"
            )
    if request.hours_per_run * 3600 > wanted:
        issues.append(
            f"runs are expected to take {request.hours_per_run:g} h but the time limit is {request.time_limit}"
        )
    budget = config.budgets.run.max_gpu_hours
    if budget is not None and request.gpu_hours_total > budget:
        issues.append(f"the plan needs about {request.gpu_hours_total:g} GPU-hours; the budget is {budget:g}")
    return issues


def plan_issues(plan: ExperimentPlan, config: AppConfig, min_seeds: int, max_dataset_gb: float) -> list[str]:
    """Everything the planner would warn about, for a whole plan."""
    return (
        _design_issues(plan.conditions, plan.ablations, len(plan.seeds), min_seeds)
        + primary_issues(plan.metrics)
        + metric_issues(plan.metrics)
        + resource_issues(plan.resources, max_dataset_gb)
        + compute_issues(plan.compute, config)
    )
