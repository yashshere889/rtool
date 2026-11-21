"""Data analyses (units mode): one run per condition reports a value per donor or sample, compared across units."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from research_tool.agents.coder.results import aggregate, collect, confounding_issues
from research_tool.agents.planner.checks import design_issues, results_schema
from research_tool.packs import load_pack
from research_tool.packs.units import pseudoreplication_issues
from research_tool.schemas.code import RunRecord
from research_tool.schemas.plan import Condition, ExperimentPlan, Metric
from research_tool.testing.samples import sample_plan


def units_plan(test: str = "Welch t-test") -> ExperimentPlan:
    plan = sample_plan(test=test)
    conditions = [
        Condition(name="control", description="c", is_baseline=True),
        Condition(name="disease", description="d"),
    ]
    metrics = [Metric(name="expr", definition="log CPM", primary=True)]
    return plan.model_copy(
        update={
            "conditions": conditions,
            "metrics": metrics,
            "seeds": [0],
            "replicates": "units",
            "unit": "donor",
            "results_schema": results_schema(conditions, metrics, [0], units=True),
        }
    )


def records(values: dict[str, dict[str, float]], batches: dict[str, str] | None = None) -> list[RunRecord]:
    return [
        RunRecord(condition=c, seed=0, status="ok", unit=u, batch=(batches or {}).get(u, ""), metrics={"expr": v})
        for c, units in values.items()
        for u, v in units.items()
    ]


def test_the_harness_takes_one_entry_per_unit(tmp_path: Path) -> None:
    harness = resources.files("research_tool.agents.coder").joinpath("harness")
    (tmp_path / "run.py").write_text(harness.joinpath("run.py").read_text())
    config = {"conditions": {"a": {}}, "metrics": {"expr": {}}, "resources": {}, "seeds": [0], "replicates": "units"}
    (tmp_path / "config.json").write_text(json.dumps(config))

    def run_with(body: str) -> subprocess.CompletedProcess[str]:
        (tmp_path / "experiment.py").write_text(f"def run(condition, seed, resources, smoke, workdir):\n    {body}\n")
        cmd = [sys.executable, "run.py", "--condition", "a", "--seed", "0", "--output", str(tmp_path / "o.json")]
        return subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)

    ok = run_with("return {'units': [{'unit': 'd1', 'expr': 1.0, 'batch': 'A'}, {'unit': 2, 'expr': 3.0}]}")
    assert ok.returncode == 0, ok.stdout
    record = json.loads((tmp_path / "o.json").read_text())
    assert record["metrics"] == {"expr": 2.0}  # the mean, where one number per run is shown
    assert record["units"] == [
        {"unit": "d1", "batch": "A", "metrics": {"expr": 1.0}},
        {"unit": "2", "batch": "", "metrics": {"expr": 3.0}},
    ]
    for body, problem in (
        ("return {'expr': 1.0}", "one entry per independent unit"),
        ("return {'units': [{'unit': 'd1', 'expr': 1.0}]}", "at least two"),
        ("return {'units': [{'unit': 'd1', 'expr': 1.0}, {'unit': 'd1', 'expr': 2.0}]}", "appears twice"),
        ("return {'units': [{'unit': 'd1', 'expr': 1.0}, {'unit': 'd2'}]}", "units[1] (d2): missing metric 'expr'"),
    ):
        broken = run_with(body)
        assert broken.returncode == 3 and problem in broken.stdout, broken.stdout


def test_collect_expands_a_conditions_run_into_its_units(tmp_path: Path) -> None:
    plan = units_plan()
    units = [{"unit": f"d{i}", "batch": "A", "metrics": {"expr": float(i)}} for i in range(3)]
    (tmp_path / "control__0.json").write_text(json.dumps({"metrics": {"expr": 1.0}, "units": units, "seconds": 2}))
    (tmp_path / "disease__0.json").write_text(json.dumps({"metrics": {"expr": 1.0}, "seconds": 2}))  # no units
    found = collect(plan, tmp_path)
    assert [(r.condition, r.unit, r.status) for r in found] == [
        ("control", "d0", "ok"), ("control", "d1", "ok"), ("control", "d2", "ok"), ("disease", "", "failed"),
    ]  # fmt: skip
    assert "no per-unit results" in found[-1].error


def test_independent_groups_are_compared_unpaired_and_the_same_units_paired() -> None:
    plan = units_plan()
    groups = {"control": {"c1": 1.0, "c2": 1.2, "c3": 0.9, "c4": 1.1}, "disease": {"d1": 2.0, "d2": 2.2, "d3": 1.8}}
    summaries, comparisons = aggregate(plan, records(groups))  # dispatches to the units analysis
    assert [(s.condition, s.n) for s in summaries] == [("control", 4), ("disease", 3)]
    (comparison,) = comparisons
    assert comparison.test == "Welch t-test" and comparison.n == 3 and comparison.significant and comparison.better
    same = {"control": {"p1": 1.0, "p2": 2.0, "p3": 3.0}, "disease": {"p1": 1.5, "p2": 2.4, "p3": 3.6}}
    (paired,) = aggregate(plan, records(same))[1]
    assert paired.test == "paired t-test" and paired.n == 3  # each donor measured under both conditions
    (ranked,) = aggregate(units_plan("Mann-Whitney U"), records(groups))[1]
    assert ranked.test == "Mann-Whitney U"


def test_a_condition_confounded_with_batch_is_a_design_problem() -> None:
    plan = units_plan()
    groups = {"control": {"c1": 1.0, "c2": 1.2}, "disease": {"d1": 2.0, "d2": 2.2}}
    split = records(groups, {"c1": "A", "c2": "A", "d1": "B", "d2": "B"})
    (issue,) = confounding_issues(plan, split)
    assert issue.startswith("`disease` and the baseline `control` share no batch (B against A)")
    mixed = records(groups, {"c1": "A", "c2": "B", "d1": "A", "d2": "B"})
    assert confounding_issues(plan, mixed) == [] and confounding_issues(plan, records(groups)) == []
    assert confounding_issues(sample_plan(), split) == []  # seeds mode has no batches


@pytest.mark.parametrize(
    ("unit", "flagged"),
    [("donor", False), ("patient sample", False), ("cell", True), ("cells per donor", True), ("", True)],
)
def test_cells_are_not_independent_units(unit: str, flagged: bool) -> None:
    plan = units_plan().model_copy(update={"unit": unit})
    assert bool(pseudoreplication_issues(plan, None)) == flagged
    assert pseudoreplication_issues(sample_plan(), None) == []  # simulations repeat seeds, not units


def test_data_packs_compare_units_and_need_no_seeds() -> None:
    omics, tabular = load_pack("omics"), load_pack("tabular")
    assert omics.replicates == tabular.replicates == "units" and load_pack("generic").replicates == "seeds"
    assert omics.review_checks and "pubmed" in omics.literature_sources and "local" in omics.resource_sources
    from research_tool.agents.planner.checks import normalise_core  # noqa: PLC0415
    from research_tool.schemas.plan import PlanCoreUnits  # noqa: PLC0415

    core = normalise_core(
        PlanCoreUnits(
            objective="o", design="d", independent_variables=[{"name": "x", "description": "x"}],
            dependent_variables=[{"name": "y", "description": "y"}],
            conditions=[
                {"name": "control", "description": "c", "is_baseline": True},
                {"name": "disease", "description": "d"},
            ],
            metrics=[{"name": "expr", "definition": "e", "higher_is_better": True}],
            statistical_tests=[{"test": "Mann-Whitney U", "comparison": "disease vs control"}],
            n_seeds=1, procedure=["1", "2", "3"], expected_results="r", unit_of_analysis="donor",
        )
    )  # fmt: skip
    # one analysis per condition: one "seed" is right, and a rank test's limit is about units, not seeds
    assert design_issues(core, min_seeds=3, units=True) == []
    assert any("seeds" in i for i in design_issues(core, min_seeds=3))
