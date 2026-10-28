"""Signs that an experiment measured nothing (agents/coder/sanity.py), on the code from a real Barkla run."""

from __future__ import annotations

from research_tool.agents.coder import sanity
from research_tool.schemas.code import RunRecord
from research_tool.testing.samples import sample_plan as plan
from research_tool.testing.samples import sample_records as records

# What the Coder wrote on 2026-10-12 (shortened): resources looked up but never opened, the seed's
# generator never used, and the metrics a formula of the condition.
MOCK = '''"""Experiment: more agents degrade coordination (a mock run would be wrong here)."""

import random
from pathlib import Path

CONDITIONS = {"n_2": "Baseline condition", "n_3": "Treatment with a placeholder description"}


def run(condition: str, seed: int, resources: dict[str, str], smoke: bool, workdir: str) -> dict[str, float]:
    """Run one condition with one seed."""
    rng = random.Random(seed)
    n_agents = int(condition.split("_")[1])
    env_path = Path(resources["CS:GO multi-agent environment"])
    episodes = 1 if smoke else 100

    # Mock metric calculation based on agent count (hypothetical trend)
    task_success_rate = 1.0 - (n_agents - 2) * 0.1
    return {"task_success_rate": float(task_success_rate)}
'''

REAL = '''"""Experiment: a small simulation."""

import random


def run(condition: str, seed: int, resources: dict[str, str], smoke: bool, workdir: str) -> dict[str, float]:
    rng = random.Random(seed)
    steps = 2 if smoke else 1000
    total = 0.0
    for _ in range(steps):
        total += rng.gauss(0.5 if condition == "new" else 0.4, 0.1)
    return {"acc": total / steps}
'''


def test_the_mock_from_the_barkla_run_is_flagged() -> None:
    admissions = sanity.mock_admissions(MOCK)
    # the comment counts; the plan's words in the module docstring and CONDITIONS do not
    assert [line for line, _text in admissions] == [16]
    assert sanity.unused_names(MOCK) == ["rng", "env_path", "episodes"]
    warnings = sanity.code_warnings(MOCK, has_resources=True)
    assert len(warnings) == 2
    assert "mock, dummy or placeholder" in warnings[0] and "line 16" in warnings[0] and "hypothetical" in warnings[0]
    assert "`rng`, `env_path`, `episodes`" in warnings[1] and "never uses them" in warnings[1]


def test_honest_code_has_no_warnings() -> None:
    assert sanity.code_warnings(REAL, has_resources=False) == []
    # data was downloaded, yet run() never looks at it
    assert sanity.code_warnings(REAL, has_resources=True) == [
        "run() never reads `resources`, although data or models were downloaded for it"
    ]
    assert sanity.code_warnings("def run(:\n", has_resources=True) == []  # syntax errors are reported elsewhere


def test_seed_and_missing_resource_warnings() -> None:
    same = sanity.seed_warning("n_2", (0, 1), {"a": 0.6, "b": 1.0}, {"a": 0.6, "b": 1.0})
    assert "exactly the same metrics with seeds 0 and 1" in same
    assert sanity.seed_warning("n_2", (0, 1), {"a": 0.6}, {"a": 0.61}) == ""
    assert sanity.missing_resources_warning([]) == ""
    assert "env was never provided" in sanity.missing_resources_warning(["env"])
    assert "a, b were never provided" in sanity.missing_resources_warning(["a", "b"])


def test_results_that_ignore_the_seed_and_take_no_time() -> None:
    p = plan()  # expects 1 h per run
    assert sanity.results_warnings(p, records()) == []
    flat = [
        RunRecord(condition=c, seed=s, status="ok", metrics={"acc": v, "loss": 0.3 + 0.01 * s}, seconds=0.0)
        for c, v in (("base", 0.6), ("new", 0.7))
        for s in (0, 1, 2)
    ]
    warnings = sanity.results_warnings(p, flat)
    assert len(warnings) == 2
    assert "exactly the same `acc` in every condition" in warnings[0] and "`loss`" not in warnings[0]
    assert "under 1 s" in warnings[1] and "about 1 h per run" in warnings[1]
    quick = p.model_copy(update={"compute": p.compute.model_copy(update={"hours_per_run": 0.05})})
    assert len(sanity.results_warnings(quick, flat)) == 1  # a 3-minute plan may well finish in a second


def test_identical_planned_metrics_are_flagged_even_when_timings_differ() -> None:
    first = {"miss_rate": 1.0, "false_alarm_rate": 0.03, "seconds_per_article": 0.0041}
    second = {"miss_rate": 1.0, "false_alarm_rate": 0.03, "seconds_per_article": 0.0040}
    assert sanity.seed_warning("baseline", (0, 1), first, second) == ""  # a timing hides it...
    planned = ["miss_rate", "false_alarm_rate"]
    assert "exactly the same metrics" in sanity.seed_warning("baseline", (0, 1), first, second, planned)
