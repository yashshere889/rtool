"""The analysis on simulated results with a known truth: false positives at alpha, power as theory says.

Pipeline runs are too few to measure how often rtool reports an effect that is not there, so the
deterministic part, the analysis every report's numbers come from (``results.aggregate``), is
measured here on thousands of simulated experiments instead (docs/stage0_design.md, section 7).
Each simulated seed shares one random component across conditions, as seeded runs on the same data
split do, so paired tests have something to gain from the pairing.
"""

from __future__ import annotations

import functools
import math
import random

import pytest

from research_tool.agents.coder import stats
from research_tool.agents.coder.results import aggregate
from research_tool.schemas.code import Comparison, RunRecord
from research_tool.schemas.plan import Condition, ExperimentPlan, Metric
from research_tool.testing.samples import sample_plan

# Noncentral t with 7 degrees of freedom and noncentrality sqrt(8): the power of a two-sided paired t-test
# at alpha 0.05 for 8 seeds and an effect of d_z = 1. Computed by integrating over the chi-square
# distribution (the same integration gives 0.0500 for d_z = 0).
POWER_8_SEEDS_DZ_1 = 0.681


@pytest.fixture(autouse=True)
def _cached_t_quantiles(monkeypatch: pytest.MonkeyPatch) -> None:
    # t_critical searches for the quantile on every call, and the same few degrees of freedom repeat here
    monkeypatch.setattr(stats, "t_critical", functools.lru_cache(maxsize=None)(stats.t_critical))


def plan(test: str, *, seeds: int = 8, treatments: int = 1) -> ExperimentPlan:
    conditions = [Condition(name="base", description="b", is_baseline=True)]
    conditions += [Condition(name=f"new{i}", description="n") for i in range(treatments)]
    metric = Metric(name="acc", definition="accuracy", primary=True)
    return sample_plan(test=test).model_copy(
        update={"conditions": conditions, "metrics": [metric], "seeds": list(range(seeds))}
    )


def experiment(
    p: ExperimentPlan, rng: random.Random, effect: float = 0.0, failed: tuple[str, int] | None = None
) -> list[Comparison]:
    """One simulated experiment, analysed: every condition but the baseline is shifted by ``effect``."""
    records = []
    for seed in p.seeds:
        shared = rng.gauss(0.0, 1.0)  # what the seed does to every condition alike
        for c in p.conditions:
            if (c.name, seed) == failed:
                continue
            value = shared + (0.0 if c.is_baseline else effect) + rng.gauss(0.0, 1.0)
            records.append(RunRecord(condition=c.name, seed=seed, status="ok", metrics={"acc": value}))
    return aggregate(p, records)[1]


def share(p: ExperimentPlan, reps: int, seed: int, **kwargs: object) -> float:
    """How often at least one condition came out significantly different from the baseline."""
    rng = random.Random(seed)
    return sum(any(c.significant for c in experiment(p, rng, **kwargs)) for _ in range(reps)) / reps  # type: ignore[arg-type]


def test_the_paired_t_test_finds_false_effects_at_alpha() -> None:
    assert 0.035 <= share(plan("paired t-test"), 2000, seed=1) <= 0.065  # binomial sd at 0.05: 0.005


def test_the_paired_t_test_has_the_power_theory_gives() -> None:
    # the difference of two conditions has sd sqrt(2) here, so a shift of sqrt(2) is d_z = 1
    power = share(plan("paired t-test"), 2000, seed=2, effect=math.sqrt(2))
    assert abs(power - POWER_8_SEEDS_DZ_1) <= 0.04  # binomial sd at 0.68: 0.010


def test_a_failed_run_makes_the_test_unpaired_without_extra_false_effects() -> None:
    p = plan("paired t-test")
    assert {c.test for c in experiment(p, random.Random(3), failed=("new0", 0))} == {"Welch t-test"}
    # ignoring the pairing only makes it more cautious
    assert share(p, 1000, seed=3, failed=("new0", 0)) <= 0.065


@pytest.mark.parametrize(("test", "seeds", "reps"), [("Wilcoxon signed-rank", 8, 1000), ("Mann-Whitney U", 6, 600)])
def test_the_exact_rank_tests_never_exceed_alpha(test: str, seeds: int, reps: int) -> None:
    p = plan(test, seeds=seeds)
    assert {c.test for c in experiment(p, random.Random(4))} == {test}
    assert share(p, reps, seed=4) <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / reps)


def test_with_five_seeds_the_wilcoxon_test_can_never_be_significant() -> None:
    # why validity cases use at least six seeds (validity.cases.MIN_SEEDS), even for huge effects
    rng = random.Random(5)
    found = [c for _ in range(50) for c in experiment(plan("Wilcoxon signed-rank", seeds=5), rng, effect=10.0)]
    assert min(c.p_value for c in found if c.p_value is not None) == 2 / 32
    assert not any(c.significant for c in found)


def test_holm_keeps_the_family_wise_rate_at_alpha() -> None:
    p, rng = plan("paired t-test", treatments=3), random.Random(6)
    raw = corrected = 0
    for _ in range(1000):
        comparisons = experiment(p, rng)
        raw += any(c.p_value is not None and c.p_value < stats.ALPHA for c in comparisons)
        corrected += any(c.significant for c in comparisons)
    assert all(c.correction == "holm" for c in comparisons)
    assert corrected / 1000 <= 0.065  # binomial sd at 0.05: 0.007
    assert raw / 1000 >= 0.09  # three uncorrected tests against one baseline would report false effects twice as often
