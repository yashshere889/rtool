"""A data analysis through the real pipeline: the shipped omics case, its generated cells, a pseudobulk per donor.

The demo model answers as usual, except that the planner gets a data-analysis design and the Coder an
analysis of the case's cells (both written here). Everything else (engine, harness, fetch, statistics,
Writer, Reviewer, scoring) is the real thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_tool.core.artifacts import ArtifactStore
from research_tool.core.config import deep_merge, load_config
from research_tool.core.engine import Outcome
from research_tool.core.run import RunHandle
from research_tool.llm.types import ChatRequest
from research_tool.testing.demo_run import CLUSTER, advance, demo_services
from research_tool.validity.cases import load_case
from research_tool.validity.runs import create_case_run
from research_tool.validity.score import score_run

CASES = Path(__file__).resolve().parents[2] / "benchmarks" / "omics"

DESIGN = {
    "objective": "Compare IFI27 pseudobulk expression between disease and control donors",
    "design": "Sum each donor's cells into a pseudobulk profile, normalise to CPM, compare log2(CPM + 1) of IFI27",
    "independent_variables": [
        {"name": "condition", "description": "disease or control", "values": ["control", "disease"]}
    ],
    "dependent_variables": [{"name": "ifi27_log_cpm", "description": "log2(CPM + 1) of IFI27 per donor"}],
    "controls": ["the same normalisation for every donor"],
    "conditions": [
        {"name": "control", "description": "control donors", "is_baseline": True},
        {"name": "disease", "description": "disease donors"},
    ],
    "ablations": [],
    "metrics": [
        {"name": "ifi27_log_cpm", "definition": "log2(CPM + 1) of IFI27", "higher_is_better": True, "primary": True},
        {"name": "library_size", "definition": "counts in the donor's pseudobulk", "higher_is_better": True},
    ],
    "statistical_tests": [{"test": "Welch t-test", "comparison": "disease vs control on ifi27_log_cpm"}],
    "n_seeds": 1,
    "procedure": ["Read cells.csv", "Sum each donor's counts", "Normalise to CPM", "Return one entry per donor"],
    "expected_results": "IFI27 is higher in disease donors",
    "risks": ["synthetic data"],
    "unit_of_analysis": "donor",
}
RESOURCES = {
    "resources": [{"kind": "dataset", "name": "cells", "source": "local", "identifier": "cells", "purpose": "counts"}]
}
COMPUTE = {"gpu_memory_gb": 0, "gpus_per_run": 0, "hours_per_run": 0.05, "justification": "a small table"}
ANALYSIS = '''"""Pseudobulk analysis of IFI27 (written for this test)."""
# requirements:
import csv
import math
from pathlib import Path


def run(condition, seed, resources, smoke, workdir):
    sums, totals, batch = {}, {}, {}
    with open(Path(resources["cells"]) / "cells.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != condition:
                continue
            counts = [int(v) for k, v in row.items() if k not in ("cell", "donor", "condition", "batch")]
            donor = row["donor"]
            sums[donor] = sums.get(donor, 0) + int(row["IFI27"])
            totals[donor] = totals.get(donor, 0) + sum(counts)
            batch[donor] = row["batch"]
    donors = sorted(sums)[: 3 if smoke else None]
    return {
        "units": [
            {
                "unit": d,
                "batch": batch[d],
                "ifi27_log_cpm": math.log2(1e6 * sums[d] / totals[d] + 1),
                "library_size": float(totals[d]),
            }
            for d in donors
        ]
    }
'''


def schema_is(title: str) -> Any:
    return lambda request: (request.json_schema or {}).get("title") == title


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> RunHandle:
    runs_root = tmp_path_factory.mktemp("units") / "runs"
    config = load_config(overrides=deep_merge(CLUSTER, {"paths": {"runs_root": str(runs_root)}}), env={})
    run = create_case_run(config, load_case(CASES / "pseudobulk-planted"), commit="0" * 12, variant="fold_2")
    services = demo_services(run)
    backend = services.llm.backend
    for rule in (
        (schema_is("PlanCoreUnits"), DESIGN),
        (schema_is("ResourcePlan"), RESOURCES),
        (schema_is("ComputeEstimate"), COMPUTE),
        ("Write the complete Python file `experiment.py`", f"```python\n{ANALYSIS}```\n"),
    ):
        backend.rules.insert(0, rule)
    result = advance(run, services, max_episodes=60)
    assert result.outcome is Outcome.DONE, f"{result.outcome.value}: {result.message}"
    run.llm_calls = list(backend.calls)  # type: ignore[attr-defined]  # what the agents asked the model
    return run


def test_the_plan_is_a_data_analysis_across_donors(finished: RunHandle) -> None:
    plan = ArtifactStore(finished.artifacts_dir).get("experiment_plan")
    assert (plan.replicates, plan.unit, plan.seeds) == ("units", "donor", [0])
    assert plan.issues == [] and plan.compute.gpus_per_run == 0
    prompts: list[ChatRequest] = finished.llm_calls  # type: ignore[attr-defined]
    coder = next(r.messages[-1].content for r in prompts if "Write the complete Python file" in r.messages[-1].content)
    assert '{"units": [{"unit": <id>' in coder and "never return one entry per cell" in coder
    reviewer = next(
        r.messages[-1].content for r in prompts if "careful, fair scientific reviewer" in r.messages[-1].content
    )
    assert "Questions this field asks of every study" in reviewer and "pseudobulk per donor" in reviewer


def test_the_results_are_per_donor_and_compared_across_donors(finished: RunHandle) -> None:
    store = ArtifactStore(finished.artifacts_dir)
    results = store.get("experiment_results")
    assert {(r.condition, r.unit[0]) for r in results.runs} == {("control", "c"), ("disease", "d")}
    assert len(results.runs) == 12 and results.complete and results.issues == [] and not results.warnings
    primary = results.comparison("disease", "ifi27_log_cpm")
    assert primary.test == "Welch t-test" and primary.n == 6 and primary.significant and primary.better
    assert 0.6 <= primary.difference <= 1.6  # the reference's donor-level difference is 1.106
    report = (finished.root / store.get("final_report").markdown_path).read_text()
    assert "Each condition was analysed once, with one value per donor (6 under control, 6 under disease)" in report
    assert "across donors using the Welch t-test" in report and "seeds" not in report


def test_the_case_scores_the_right_answer(finished: RunHandle) -> None:
    score = score_run(finished)
    assert (score.level, score.category) == (3, ""), [c for c in score.checks if not c.ok]
    assert any(c.name == "condition disease has results for at least two units" and c.ok for c in score.checks)
