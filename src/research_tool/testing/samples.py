"""Small, realistic sample artifacts for tests and for trying out new agents."""

from __future__ import annotations

from ..agents.planner.checks import results_schema
from ..schemas.code import RunRecord
from ..schemas.plan import ComputeRequest, Condition, ExperimentPlan, Metric, StatisticalTest, Variable

ACCURACY = {"base": [0.65, 0.66, 0.63], "new": [0.71, 0.74, 0.69]}


def sample_plan(test: str = "paired t-test", correction: str = "none", primary: str = "acc") -> ExperimentPlan:
    """Two conditions (``base`` is the baseline), metrics ``acc`` (higher is better) and ``loss``, seeds 0-2.

    ``primary`` names the pre-registered primary metric ("" marks none, like plans written before it existed).
    """
    conditions = [Condition(name="base", description="b", is_baseline=True), Condition(name="new", description="n")]
    metrics = [
        Metric(name="acc", definition="accuracy", primary=primary == "acc"),
        Metric(name="loss", definition="l", higher_is_better=False, primary=primary == "loss"),
    ]
    return ExperimentPlan(
        hypothesis_id="h1",
        hypothesis="new beats base",
        null_hypothesis="no difference",
        objective="o",
        design="d",
        independent_variables=[Variable(name="method", description="m")],
        dependent_variables=[Variable(name="acc", description="a")],
        conditions=conditions,
        metrics=metrics,
        statistical_tests=[StatisticalTest(test=test, comparison="new vs base", correction=correction)],
        seeds=[0, 1, 2],
        procedure=["a", "b", "c"],
        resources=[],
        compute=ComputeRequest(
            partition="gpu-l40s",
            gpu_type="l40s",
            gpus_per_run=1,
            time_limit="01:00:00",
            runs=6,
            hours_per_run=1,
            gpu_hours_total=6,
        ),
        expected_results="new is better",
        results_schema=results_schema(conditions, metrics, [0, 1, 2]),
    )


def sample_records() -> list[RunRecord]:
    """Six successful runs of :func:`sample_plan` (``new`` is clearly better)."""
    return [
        RunRecord(
            condition=c,
            seed=s,
            status="ok",
            metrics={"acc": ACCURACY[c][s], "loss": 1 - ACCURACY[c][s]},
        )
        for c in ("base", "new")
        for s in (0, 1, 2)
    ]
