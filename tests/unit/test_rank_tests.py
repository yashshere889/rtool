"""The rank tests give exact p-values by counting, and say so when they have to approximate."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from research_tool.agents.coder import stats


def brute_wilcoxon(a: list[float], b: list[float]) -> float:
    d = [x - y for x, y in zip(a, b, strict=True) if x != y]
    ranks = stats._ranks([abs(x) for x in d])
    centre = sum(ranks) / 2
    observed = abs(sum(r for r, x in zip(ranks, d, strict=True) if x > 0) - centre)
    patterns = list(itertools.product((0, 1), repeat=len(d)))
    hits = sum(
        1 for p in patterns if abs(sum(r for r, s in zip(ranks, p, strict=True) if s) - centre) >= observed - 1e-12
    )
    return hits / len(patterns)


def brute_mann_whitney(a: list[float], b: list[float]) -> float:
    ranks = stats._ranks([*a, *b])
    na = len(a)
    centre = na * (na + 1) / 2 + na * len(b) / 2
    observed = abs(sum(ranks[:na]) - centre)
    combos = list(itertools.combinations(range(len(ranks)), na))
    hits = sum(1 for c in combos if abs(sum(ranks[i] for i in c) - centre) >= observed - 1e-12)
    return hits / len(combos)


@pytest.mark.parametrize("seed", range(20))
def test_counting_matches_enumeration_with_ties(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(3, 8)
    a = [float(rng.randint(0, 4)) for _ in range(n)]  # few distinct values: many ties
    b = [float(rng.randint(0, 4)) for _ in range(n)]
    wilcoxon = stats.wilcoxon_signed_rank(a, b)
    counted = not (wilcoxon.no_variation or wilcoxon.constant_difference) and a != b  # else no rank test runs
    if counted:
        assert wilcoxon.p_value == pytest.approx(brute_wilcoxon(a, b))
    if not stats.welch_t(a, b).no_variation:
        assert stats.mann_whitney(a, b).p_value == pytest.approx(brute_mann_whitney(a, b))


def test_mann_whitney_is_exact_with_eleven_per_group() -> None:
    a = [float(i) for i in range(11)]
    b = [float(i) + 100 for i in range(11)]  # complete separation
    result = stats.mann_whitney(a, b)
    assert result.test == "Mann-Whitney U"
    assert result.p_value == pytest.approx(2 / math.comb(22, 11))  # not the Welch t-test's 1.8e-25


def test_wilcoxon_is_exact_with_21_pairs() -> None:
    a = [float(i) + 1 for i in range(21)]
    b = [0.0] * 21  # every difference positive
    result = stats.wilcoxon_signed_rank(a, b)
    assert result.test == "Wilcoxon signed-rank"
    assert result.p_value == pytest.approx(2 / 2**21)  # not the paired t-test's 9.2e-8
    assert result.statistic == sum(range(1, 22))


def test_too_big_to_count_is_named_an_approximation() -> None:
    rng = random.Random(1)
    a = [rng.gauss(0, 1) for _ in range(60)]
    b = [rng.gauss(0.5, 1) for _ in range(60)]
    unpaired = stats.mann_whitney(a, b)
    assert unpaired.test == "Mann-Whitney U" + stats.APPROXIMATE
    assert 0 < unpaired.p_value < 1
    paired = stats.wilcoxon_signed_rank(a * 5, b * 5)  # 300 pairs
    assert paired.test == "Wilcoxon signed-rank" + stats.APPROXIMATE
    assert 0 < paired.p_value < 1
