"""Deterministic plan checks: fit compute to the cluster, validate the design, define results.json."""

from __future__ import annotations

import re
from typing import Any

from ...core.config import AppConfig
from ...hpc.jobspec import format_time_limit, parse_time_limit
from ...schemas.plan import ComputeEstimate, ComputeRequest, Condition, Metric, PlanCore, ResourceSpec

SAFETY_FACTOR = 1.3  # runs take longer than estimated
_SLUG = re.compile(r"[^a-z0-9]+")


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
    """Snake_case names, so they can be used as keys in results.json."""
    for condition in [*core.conditions, *core.ablations]:
        condition.name = slug(condition.name)
    for metric in core.metrics:
        metric.name = slug(metric.name)
    return core


def design_issues(core: PlanCore, min_seeds: int) -> list[str]:
    issues = []
    if not any(c.is_baseline for c in core.conditions):
        issues.append("no condition is marked as the baseline (is_baseline = true)")
    names = [c.name for c in [*core.conditions, *core.ablations]]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        issues.append(f"condition names must be unique: {duplicates}")
    if core.n_seeds < min_seeds:
        issues.append(f"use at least {min_seeds} seeds (independent runs) per condition, not {core.n_seeds}")
    return issues


def resource_issues(resources: list[ResourceSpec], max_dataset_gb: float) -> list[str]:
    issues = []
    if not any(r.kind == "dataset" for r in resources):
        issues.append("no dataset is listed")
    for r in resources:
        if not r.verified:
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
