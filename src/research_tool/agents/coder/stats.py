"""Small, dependency-free statistics for experiment results (no scipy on the orchestrator).

Means with t-based 95 % confidence intervals, paired and Welch t-tests, exact
Wilcoxon signed-rank and Mann-Whitney U tests for the small samples typical of
seeds-per-condition, Cohen's d, and Holm/Bonferroni corrections. Every number
the Writer reports comes from here, deterministically. When nothing varies
across seeds there is no noise to test against, and no difference is significant;
the same holds when both conditions vary but differ by the same amount every seed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

ALPHA = 0.05
FLAT = 1e-9  # a spread below this fraction of the values' size is rounding, not seed-to-seed variation


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation (n - 1); 0 for fewer than two values."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def flat(xs: Sequence[float], scale: float | None = None) -> bool:
    """At least two values, all the same up to floating-point rounding.

    Seeds that all give the same value mean the experiment ignores the seed or
    returns fixed numbers; there is no noise to test a difference against.
    Rounding is judged against ``scale``, by default the values' own size; pass
    the size of the inputs when ``xs`` are differences between larger numbers.
    """
    if len(xs) < 2:
        return False
    size = max(abs(x) for x in xs) if scale is None else scale
    return max(xs) - min(xs) <= FLAT * size


# ------------------------------------------------------- t distribution
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with ``df`` degrees of freedom."""
    if math.isinf(t):
        return 0.0
    return min(1.0, betainc(df / 2.0, 0.5, df / (df + t * t)))


def t_critical(df: float, alpha: float = ALPHA) -> float:
    """Two-sided critical value: P(|T| >= t) = alpha."""
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean_ci(xs: Sequence[float], alpha: float = ALPHA) -> tuple[float, float]:
    m = mean(xs)
    if len(xs) < 2:
        return m, m
    half = t_critical(len(xs) - 1, alpha) * stdev(xs) / math.sqrt(len(xs))
    return m - half, m + half


# -------------------------------------------------------------- tests
@dataclass
class TestResult:
    test: str
    difference: float  # mean(a) - mean(b)
    ci_low: float
    ci_high: float
    statistic: float | None
    p_value: float | None
    effect_size: float | None
    n: int
    no_variation: bool = False  # neither side varied across seeds: the test is undefined (see _degenerate)
    constant_difference: bool = False  # both varied, but a - b was the same for every seed (see _degenerate)


def _like(base: TestResult, test: str, *, p_value: float | None) -> TestResult:
    """A rank test's result: its own statistic and p-value, the t-based CI, effect size and flags of ``base``."""
    return replace(base, test=test, statistic=None, p_value=p_value)


def _degenerate(test: str, diff: float, n: int, *, constant_difference: bool = False) -> TestResult:
    """No noise estimate, so a difference is never significant.

    Either neither side varies across seeds, or (``constant_difference``) both do
    but differ by the same amount every seed, which is what a hardcoded offset
    plus shared seeded noise gives. Identical values (difference 0) keep p = 1;
    any other difference gets no p-value at all instead of the p = 0 that an
    infinite t would give.
    """
    p = 1.0 if diff == 0 else None
    return TestResult(test, diff, diff, diff, None, p, None, n, not constant_difference, constant_difference)


def paired_t(a: Sequence[float], b: Sequence[float], alpha: float = ALPHA) -> TestResult:
    d = [x - y for x, y in zip(a, b, strict=True)]
    n, md = len(d), mean(d)
    if n < 2:
        return TestResult("paired t-test", md, md, md, None, None, None, n)
    if flat(a) and flat(b):
        return _degenerate("paired t-test", md, n)
    if flat(d, scale=max(abs(x) for x in (*a, *b))):  # both vary, yet differ by the same amount for every seed
        return _degenerate("paired t-test", md, n, constant_difference=True)
    sd = stdev(d)
    se = sd / math.sqrt(n)
    t = md / se
    half = t_critical(n - 1, alpha) * se
    return TestResult("paired t-test", md, md - half, md + half, t, t_two_sided_p(t, n - 1), md / sd, n)


def welch_t(a: Sequence[float], b: Sequence[float], alpha: float = ALPHA) -> TestResult:
    na, nb = len(a), len(b)
    diff = mean(a) - mean(b)
    n = min(na, nb)
    if na < 2 or nb < 2:
        return TestResult("Welch t-test", diff, diff, diff, None, None, None, n)
    va, vb = stdev(a) ** 2 / na, stdev(b) ** 2 / nb
    se = math.sqrt(va + vb)
    pooled = math.sqrt(((na - 1) * stdev(a) ** 2 + (nb - 1) * stdev(b) ** 2) / (na + nb - 2))
    if se == 0 or (flat(a) and flat(b)):  # one fixed group is fine (a one-sample test against it); two are not
        return _degenerate("Welch t-test", diff, n)
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    t = diff / se
    half = t_critical(df, alpha) * se
    effect = diff / pooled if pooled > 0 else None
    return TestResult("Welch t-test", diff, diff - half, diff + half, t, t_two_sided_p(t, df), effect, n)


def _ranks(values: Sequence[float]) -> list[float]:
    """Ranks starting at 1, ties get their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float], alpha: float = ALPHA) -> TestResult:
    """Exact two-sided test over all sign flips (fine for the handful of seeds per condition)."""
    paired = paired_t(a, b, alpha)  # the CI and effect size reported alongside
    result = _like(paired, "Wilcoxon signed-rank", p_value=None)
    if paired.no_variation or paired.constant_difference:  # no noise: six equal differences would give p = 0.031
        result.p_value = paired.p_value
        return result
    d = [x - y for x, y in zip(a, b, strict=True) if x != y]
    if not d:
        result.p_value = 1.0
        return result
    if len(d) > 20:  # too many sign patterns to enumerate; the t-test is a fine approximation there
        result.p_value = paired.p_value
        return result
    ranks = _ranks([abs(x) for x in d])
    w_plus = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    centre = sum(ranks) / 2
    observed = abs(w_plus - centre)
    extreme = sum(
        1
        for signs in itertools.product((0, 1), repeat=len(d))
        if abs(sum(r for r, s in zip(ranks, signs, strict=True) if s) - centre) >= observed - 1e-12
    )
    result.statistic, result.p_value = w_plus, extreme / 2 ** len(d)
    return result


def mann_whitney(a: Sequence[float], b: Sequence[float], alpha: float = ALPHA) -> TestResult:
    """Exact two-sided permutation version of the Mann-Whitney U test."""
    welch = welch_t(a, b, alpha)
    na, nb = len(a), len(b)
    result = _like(welch, "Mann-Whitney U", p_value=welch.p_value)
    if welch.no_variation:  # two fixed values: complete separation would look significant from 4 seeds each
        return result
    if na == 0 or nb == 0 or math.comb(na + nb, na) > 200_000:
        return result  # too many arrangements to enumerate; keep the Welch p-value
    ranks = _ranks([*a, *b])
    centre = na * (na + 1) / 2 + na * nb / 2  # expected rank sum of group a

    def rank_sum(indices: Sequence[int]) -> float:
        return sum(ranks[i] for i in indices)

    observed = abs(rank_sum(range(na)) - centre)
    combos = list(itertools.combinations(range(na + nb), na))
    extreme = sum(1 for combo in combos if abs(rank_sum(combo) - centre) >= observed - 1e-12)
    result.statistic = rank_sum(range(na)) - na * (na + 1) / 2  # U for group a
    result.p_value = extreme / len(combos)
    return result


# --------------------------------------------------------- corrections
def holm(p_values: Sequence[float | None]) -> list[float | None]:
    present = [(p, i) for i, p in enumerate(p_values) if p is not None]
    out: list[float | None] = [None] * len(p_values)
    m, running = len(present), 0.0
    for rank, (p, i) in enumerate(sorted(present)):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def bonferroni(p_values: Sequence[float | None]) -> list[float | None]:
    m = sum(1 for p in p_values if p is not None)
    return [None if p is None else min(1.0, p * m) for p in p_values]
