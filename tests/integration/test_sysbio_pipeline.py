"""A model simulation through the real pipeline: a BioModels SBML file fetched, simulated, checked for mass balance.

The demo model answers, except that the planner gets this design and the Coder a simulation of the
fetched model (both written here); BioModels is a mock server answering in its documented format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_tool.core.artifacts import ArtifactStore
from research_tool.core.config import deep_merge, load_config
from research_tool.core.engine import Outcome
from research_tool.core.run import RunHandle
from research_tool.testing.demo_run import CLUSTER, advance, demo_services
from research_tool.testing.samples import SBML, biomodels_mock
from research_tool.tools.resources.fetch import Fetcher
from research_tool.validity.cases import Case
from research_tool.validity.runs import create_case_run
from research_tool.validity.score import score_run

CASE = {
    "id": "faster-conversion",
    "kind": "planted",
    "pack": "sysbio",
    "confirmed_by": "analytic: B(1) = A0 (1 - exp(-k)) grows with k",
    "brief": {
        "topic": "Does a faster rate constant convert more A into B within one time unit?",
        "seeds": 6,
        "conditions": [
            {"name": "base_rate", "baseline": True, "description": "k as in the model"},
            {"name": "double_rate", "description": "k doubled"},
        ],
        "primary_metric": {"name": "b_at_t1", "higher_is_better": True},
    },
    "hypothesis": {
        "statement": "Doubling k increases the amount of B at t = 1.",
        "null_hypothesis": "B at t = 1 does not depend on k.",
        "success_criteria": "double_rate has a higher mean b_at_t1 than base_rate, p < 0.05 in a paired t-test.",
        "independent_variables": ["rate constant"],
        "dependent_variables": ["b_at_t1"],
    },
    "expect": {"significant": True, "direction": "better"},
}
DESIGN = {
    "objective": "Compare B at t = 1 for the model's k and for 2k",
    "design": "Analytic solution of A -> B with k drawn per seed within ±20 % of the model's value",
    "independent_variables": [{"name": "rate", "description": "k or 2k", "values": ["k", "2k"]}],
    "dependent_variables": [{"name": "b_at_t1", "description": "amount of B at t = 1"}],
    "controls": ["the same initial amounts"],
    "conditions": [
        {"name": "base_rate", "description": "k", "is_baseline": True},
        {"name": "double_rate", "description": "2k"},
    ],
    "ablations": [],
    "metrics": [
        {"name": "b_at_t1", "definition": "amount of B at t = 1", "higher_is_better": True, "primary": True},
        {"name": "mass_balance_error", "definition": "relative change of A + B", "higher_is_better": False},
    ],
    "statistical_tests": [{"test": "paired t-test", "comparison": "double_rate vs base_rate"}],
    "n_seeds": 6,
    "procedure": ["Read the SBML model", "Draw k within ±20 %", "Solve to t = 1", "Report B and the mass balance"],
    "expected_results": "More B with 2k",
    "risks": ["a toy model"],
}
RESOURCES = {
    "resources": [
        {
            "kind": "model",
            "name": "conversion",
            "source": "biomodels",
            "identifier": "BIOMD0000000999",
            "purpose": "the model",
        },
        {"kind": "dataset", "name": "none", "source": "pip", "identifier": "numpy", "purpose": "maths"},
    ]
}
COMPUTE = {"gpu_memory_gb": 0, "gpus_per_run": 0, "hours_per_run": 0.05, "justification": "an analytic solution"}
SIMULATION = '''"""A -> B with k from the fetched SBML model (written for this test)."""
# requirements:
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

LEAK = {leak}  # share of the total lost per run: 0 conserves mass


def run(condition, seed, resources, smoke, workdir):
    model = next(Path(resources["conversion"]).glob("*.xml"))
    root = ET.parse(model).getroot()
    amounts = {{e.get("id"): float(e.get("initialAmount")) for e in root.iter() if e.tag.endswith("species")}}
    k = next(float(e.get("value")) for e in root.iter() if e.tag.endswith("parameter") and e.get("id") == "k")
    k *= random.Random(seed).uniform(0.8, 1.2) * (2 if condition == "double_rate" else 1)
    total = amounts["A"] + amounts["B"]
    a = amounts["A"] * math.exp(-k)
    b = (total - a) * (1 - LEAK)
    return {{"b_at_t1": b, "mass_balance_error": abs(a + b - total) / total}}
'''


def schema_is(title: str) -> Any:
    return lambda request: (request.json_schema or {}).get("title") == title


def run_case(root: Path, leak: float) -> RunHandle:
    config = load_config(overrides=deep_merge(CLUSTER, {"paths": {"runs_root": str(root)}}), env={})
    run = create_case_run(config, Case.model_validate(CASE), commit="0" * 12)
    services = demo_services(run)
    services.tools["resource_fetcher"] = lambda target: Fetcher(
        target, http=biomodels_mock({"model.xml": SBML.encode()})
    )
    for rule in (
        (schema_is("PlanCore"), DESIGN),
        (schema_is("ResourcePlan"), RESOURCES),
        (schema_is("ComputeEstimate"), COMPUTE),
        ("Write the complete Python file `experiment.py`", f"```python\n{SIMULATION.format(leak=leak)}```\n"),
    ):
        services.llm.backend.rules.insert(0, rule)
    result = advance(run, services, max_episodes=60)
    assert result.outcome is Outcome.DONE, f"{result.outcome.value}: {result.message}"
    return run


@pytest.fixture(scope="module")
def conserving(tmp_path_factory: pytest.TempPathFactory) -> RunHandle:
    return run_case(tmp_path_factory.mktemp("sysbio") / "runs", leak=0.0)


def test_the_model_is_fetched_from_biomodels_and_simulated(conserving: RunHandle) -> None:
    store = ArtifactStore(conserving.artifacts_dir)
    model = next(r for r in store.get("resource_manifest").resources if r.name == "conversion")
    assert model.status == "ready" and model.note == "BioModels BIOMD0000000999"
    assert (Path(model.path) / "model.xml").read_text() == SBML  # type: ignore[arg-type]
    results = store.get("experiment_results")
    assert results.issues == [] and results.comparison("double_rate", "b_at_t1").significant
    assert (score_run(conserving).level, score_run(conserving).category) == (3, "")


def test_a_simulation_that_loses_mass_is_caught(tmp_path: Path) -> None:
    leaky = run_case(tmp_path / "runs", leak=0.01)
    results = ArtifactStore(leaky.artifacts_dir).get("experiment_results")
    # 1 % of B is lost: up to 0.7 % of A + B, depending on how much A has turned into B
    assert results.issues and results.issues[0].startswith("`mass_balance_error` reaches 0.00")
    assert any(note.startswith("Design problem: `mass_balance_error`") for note in results.notes)  # the Writer sees it
    score = score_run(leaky)
    assert (score.level, score.category) == (1, "design_problem")
