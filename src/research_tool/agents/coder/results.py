"""From per-run output files to results.json, statistics, a Markdown table and figures.

Everything here is deterministic: the LLM never computes or copies a number.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from ...schemas.code import Comparison, MetricSummary, RunRecord
from ...schemas.plan import ExperimentPlan
from ...tools.report.numbers import fmt, fmt_ci, fmt_p
from ...tools.report.svg import bar_chart
from . import stats


def planned_runs(plan: ExperimentPlan) -> list[tuple[str, int]]:
    return [(c.name, seed) for c in plan.run_conditions() for seed in plan.seeds]


def output_name(condition: str, seed: int) -> str:
    return f"{condition}__{seed}.json"


def read_output(path: Path, metrics: list[str]) -> tuple[dict[str, Any] | None, str]:
    """A run's output file, if it exists and holds every metric as a finite number."""
    if not path.exists():
        return None, "no output file"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        values = {k: float(v) for k, v in data["metrics"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return None, f"unreadable output: {exc}"
    missing = [m for m in metrics if m not in values or not math.isfinite(values[m])]
    if missing:
        return None, f"missing or non-finite metrics {missing}"
    return {"metrics": values, "seconds": data.get("seconds"), "job_id": str(data.get("job_id") or "")}, ""


def collect(
    plan: ExperimentPlan,
    runs_dir: Path,
    *,
    job_of: dict[str, str] | None = None,
    failure_of: Callable[[str, int], str] | None = None,
) -> list[RunRecord]:
    """One record per planned (condition, seed); ``failure_of`` explains runs without a valid output."""
    names = [m.name for m in plan.metrics]
    records = []
    for condition, seed in planned_runs(plan):
        key = output_name(condition, seed)
        data, problem = read_output(runs_dir / key, names)
        job = (job_of or {}).get(key, "")
        if data is not None:  # the job the run file names beats the one rtool submitted
            data["job_id"] = data["job_id"] or job
            records.append(RunRecord(condition=condition, seed=seed, status="ok", **data))
            continue
        status = "missing" if problem == "no output file" and not job else "failed"
        error = (failure_of(condition, seed) if failure_of else "") or problem
        records.append(RunRecord(condition=condition, seed=seed, status=status, job_id=job, error=error[:500]))
    return records


# --------------------------------------------------------- results.json
def results_payload(records: list[RunRecord], notes: list[str]) -> dict[str, Any]:
    runs = [{"condition": r.condition, "seed": r.seed, "metrics": r.metrics} for r in records if r.status == "ok"]
    return {"runs": runs, "notes": "; ".join(notes)}


def validate(data: Any, schema: dict[str, Any], where: str = "$") -> list[str]:
    """The JSON-Schema subset results_schema uses: type, required, properties, items, enum, minItems."""
    problems: list[str] = []
    kind = schema.get("type")
    checks: dict[str, Callable[[Any], bool]] = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "string": lambda v: isinstance(v, str),
    }
    if kind in checks and not checks[kind](data):
        return [f"{where}: expected {kind}"]
    if "enum" in schema and data not in schema["enum"]:
        problems.append(f"{where}: {data!r} is not one of {schema['enum']}")
    if isinstance(data, dict):
        problems += [f"{where}: missing {key!r}" for key in schema.get("required", []) if key not in data]
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                problems += validate(data[key], sub, f"{where}.{key}")
    if isinstance(data, list):
        if len(data) < schema.get("minItems", 0):
            problems.append(f"{where}: {len(data)} items, at least {schema['minItems']} required")
        if "items" in schema:
            for i, item in enumerate(data):
                problems += validate(item, schema["items"], f"{where}[{i}]")
    return problems


# ------------------------------------------------------------ statistics
# What rtool can run for each condition against the baseline, recognised in the plan's free text. ANOVA and
# Tukey's HSD become t-tests, Kruskal-Wallis becomes Mann-Whitney U: the same question asked pairwise.
_FAMILIES = (
    ("wilcoxon", re.compile(r"wilcoxon|signed[- ]rank")),
    ("mann-whitney", re.compile(r"mann|whitney|kruskal|rank[- ]sum|\bu[- ]test")),
    ("t", re.compile(r"t[- ]test|student|welch|anova|tukey|\bt\b")),
)
_RUN_AS = {"t": "t-tests", "wilcoxon": "Wilcoxon signed-rank tests", "mann-whitney": "Mann-Whitney U tests"}
_SAME = {  # a plan's test that names exactly what rtool runs (not something translated into it)
    "t": re.compile(r"t[- ]test|student|welch|\bt\b"),
    "wilcoxon": _FAMILIES[0][1],
    "mann-whitney": re.compile(r"mann|whitney|rank[- ]sum|\bu[- ]test"),
}


def _family_of(text: str) -> str | None:
    return next((family for family, pattern in _FAMILIES if pattern.search(text.lower())), None)


def _test_family(plan: ExperimentPlan) -> tuple[str, Literal["holm", "bonferroni"]]:
    """(test, correction) chosen from the plan's statistical tests (free text written by the planner).

    The first test rtool recognises decides (the plan's main analysis; later ones are often robustness checks).
    The correction is Holm unless the plan asks for Bonferroni (and not Holm); ``aggregate`` applies it
    whenever more than one comparison has a p-value. A plan that says "none" (or names a correction we do not
    implement) still gets Holm: "none" is the field's default, so it cannot be told apart from a planner
    that never considered the question, and an uncorrected family of metrics x conditions is how false
    positives get reported. The raw p-values stay in the results next to the adjusted ones.
    """
    test = next((f for t in plan.statistical_tests if (f := _family_of(t.test))), None)
    test = test or _family_of(" ".join(t.comparison for t in plan.statistical_tests)) or "t"
    corrections = " ".join(t.correction for t in plan.statistical_tests).lower()
    return test, "bonferroni" if "bonferroni" in corrections and "holm" not in corrections else "holm"


def translated_tests(plan: ExperimentPlan) -> list[str]:
    """The plan's tests that rtool does not run as named (e.g. ANOVA, run as pairwise t-tests)."""
    family, _ = _test_family(plan)
    return [t.test for t in plan.statistical_tests if not _SAME[family].search(t.test.lower())]


def correction_used(plan: ExperimentPlan, comparisons: list[Comparison]) -> str:
    """'holm' or 'bonferroni' if the comparisons' p-values were corrected, else ''."""
    return next((c.correction for c in comparisons if c.correction != "none"), "")


def analysis_notes(plan: ExperimentPlan, comparisons: list[Comparison]) -> list[str]:
    """What the analysis did, in words, where it differs from what the plan says."""
    if not comparisons:
        return []
    family, _ = _test_family(plan)
    tests = sorted({c.test for c in comparisons})
    notes = [f"each condition is compared with the baseline using the {' / '.join(tests)}"]
    translated = translated_tests(plan)
    if translated:
        notes.append(
            f"the plan named {', '.join(translated)}; rtool runs pairwise {_RUN_AS[family]} against the baseline "
            "instead"
        )
    adjusted = [c for c in comparisons if c.p_adjusted is not None]
    if adjusted:
        method = "Bonferroni" if correction_used(plan, comparisons) == "bonferroni" else "Holm's method"
        notes.append(
            f"p-values are corrected with {method} within the confirmatory and within the exploratory comparisons "
            f"({len(adjusted)} corrected in all)"
        )
    untestable = sorted({c.metric for c in comparisons if c.p_value is None and c.n >= 2})
    if untestable:
        notes.append(
            f"no test was possible for {', '.join(untestable)}: the results (or their differences from the "
            "baseline) did not vary across seeds"
        )
    return notes


def aggregate(plan: ExperimentPlan, records: list[RunRecord]) -> tuple[list[MetricSummary], list[Comparison]]:
    """Summaries per (condition, metric) and comparisons of every condition with the baseline.

    Comparisons on the plan's primary metric are confirmatory (the pre-registered test of the hypothesis),
    all others exploratory. The correction is applied within each of the two analyses, so exploratory
    metrics never dilute the primary test, and exploratory results are still corrected among themselves.
    """
    ok = [r for r in records if r.status == "ok"]
    conditions = [c.name for c in plan.run_conditions()]
    summaries: list[MetricSummary] = []
    for metric in plan.metrics:
        for condition in conditions:
            values = [r.metrics[metric.name] for r in ok if r.condition == condition]
            if not values:
                continue
            low, high = stats.mean_ci(values)
            summaries.append(
                MetricSummary(
                    condition=condition,
                    metric=metric.name,
                    n=len(values),
                    mean=stats.mean(values),
                    std=stats.stdev(values),
                    ci_low=low,
                    ci_high=high,
                )
            )
    baseline = next((c.name for c in plan.conditions if c.is_baseline), plan.conditions[0].name)
    family, correction = _test_family(plan)
    primary = plan.primary_metric().name
    comparisons: list[Comparison] = []
    for metric in plan.metrics:
        base = {r.seed: r.metrics[metric.name] for r in ok if r.condition == baseline}
        for condition in conditions:
            if condition == baseline:
                continue
            other = {r.seed: r.metrics[metric.name] for r in ok if r.condition == condition}
            if not other or not base:
                continue
            seeds = sorted(set(base) & set(other))
            paired = len(seeds) >= 2 and len(seeds) == len(base) == len(other)
            a = [other[s] for s in seeds] if paired else list(other.values())
            b = [base[s] for s in seeds] if paired else list(base.values())
            if family == "wilcoxon" and paired:
                result = stats.wilcoxon_signed_rank(a, b)
            elif family in ("wilcoxon", "mann-whitney"):
                result = stats.mann_whitney(a, b)
            else:
                result = stats.paired_t(a, b) if paired else stats.welch_t(a, b)
            comparisons.append(
                Comparison(
                    metric=metric.name,
                    condition=condition,
                    baseline=baseline,
                    difference=result.difference,
                    ci_low=result.ci_low,
                    ci_high=result.ci_high,
                    test=result.test,
                    statistic=result.statistic,
                    p_value=result.p_value,
                    effect_size=result.effect_size,
                    n=result.n,
                    no_seed_variation=result.no_variation,
                    constant_difference=result.constant_difference,
                    analysis="confirmatory" if metric.name == primary else "exploratory",
                )
            )
    tested = sum(1 for c in comparisons if c.p_value is not None)
    adjust = {"holm": stats.holm, "bonferroni": stats.bonferroni}[correction] if tested > 1 else None
    direction = {m.name: m.higher_is_better for m in plan.metrics}
    for analysis in ("confirmatory", "exploratory"):
        members = [c for c in comparisons if c.analysis == analysis]
        adjusted = adjust([c.p_value for c in members]) if adjust else [c.p_value for c in members]
        for comparison, p in zip(members, adjusted, strict=True):
            if adjust and p is not None:
                comparison.p_adjusted, comparison.correction = p, correction
            comparison.significant = p is not None and p < stats.ALPHA
            if comparison.significant:
                comparison.better = (comparison.difference > 0) == direction[comparison.metric]
    return summaries, comparisons


def variation_notes(plan: ExperimentPlan, records: list[RunRecord], comparisons: list[Comparison]) -> list[str]:
    """Notes on results, or differences, that do not vary across seeds (they reach the Writer and the Reviewer)."""
    ok = [r for r in records if r.status == "ok"]
    fixed = []
    for condition in (c.name for c in plan.run_conditions()):
        same = [m.name for m in plan.metrics if stats.flat([r.metrics[m.name] for r in ok if r.condition == condition])]
        if same:
            fixed.append(f"`{condition}` ({', '.join(same)})")
    notes = []
    if fixed:
        notes.append(
            f"no variation across seeds for {', '.join(fixed)}: the experiment may ignore the seed or return "
            "fixed values"
        )
    untested = [f"`{c.condition}` vs `{c.baseline}` on {c.metric}" for c in comparisons if c.no_seed_variation]
    if untested:
        notes.append(
            f"not testable, so not significant: {', '.join(untested)} (both sides give the same value for every "
            "seed, so there is no noise to test a difference against)"
        )
    offsets = [
        f"`{c.condition}` vs `{c.baseline}` on {c.metric} (difference {fmt(c.difference)})"
        for c in comparisons
        if c.constant_difference
    ]
    if offsets:
        notes.append(
            f"not testable, so not significant: {', '.join(offsets)} (the difference is identical for every seed: "
            "it may be a fixed offset rather than a measured effect)"
        )
    return notes


# --------------------------------------------------------- tables, figures
def render_tables(
    plan: ExperimentPlan,
    records: list[RunRecord],
    summaries: list[MetricSummary],
    comparisons: list[Comparison],
    figures: dict[str, str],
    *,
    warnings: Sequence[str] = (),
) -> str:
    ok = sum(1 for r in records if r.status == "ok")
    baseline = next((c.name for c in plan.conditions if c.is_baseline), "")
    primary = plan.primary_metric()
    correction = next((c.correction for c in comparisons if c.correction != "none"), "none")
    lines = [
        "# Experiment results",
        "",
        f"{ok} of {len(records)} planned runs succeeded "
        f"({len(plan.run_conditions())} conditions x {len(plan.seeds)} seeds).",
        "Means with 95 % confidence intervals (t distribution); comparisons are against the baseline "
        f"`{baseline}` (difference = condition minus baseline).",
        f"The primary outcome, fixed when the plan was approved, is `{primary.name}`: its comparisons are "
        "confirmatory, all others exploratory."
        + (
            f" p-values are adjusted ({correction.capitalize()}) within the confirmatory and within the "
            "exploratory comparisons."
            if correction != "none"
            else ""
        ),
        "",
    ]
    if warnings:
        lines += ["## Warnings: check before trusting these numbers", "", *[f"- {w}" for w in warnings], ""]
    for metric in plan.metrics_primary_first():
        direction = "higher" if metric.higher_is_better else "lower"
        role = "primary outcome" if metric is primary else "exploratory"
        lines += [f"## `{metric.name}` ({direction} is better; {role})", "", metric.definition, ""]
        lines += ["| condition | n | mean | 95 % CI | std |", "|---|---|---|---|---|"]
        for s in (s for s in summaries if s.metric == metric.name):
            label = f"{s.condition} (baseline)" if s.condition == baseline else s.condition
            lines.append(f"| {label} | {s.n} | {fmt(s.mean)} | {fmt_ci(s.ci_low, s.ci_high)} | {fmt(s.std)} |")
        rows = [c for c in comparisons if c.metric == metric.name]
        if rows:
            lines += [
                "",
                "| comparison | analysis | difference | 95 % CI | test | p | p (adjusted) | effect size |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for c in rows:
                why = "no variation across seeds" if c.no_seed_variation else "same difference for every seed"
                p = fmt_p(c.p_value) + (f" ({why})" if c.no_seed_variation or c.constant_difference else "")
                lines.append(
                    f"| {c.condition} vs {c.baseline} | {c.analysis} | {fmt(c.difference)} "
                    f"| {fmt_ci(c.ci_low, c.ci_high)} | {c.test} | {p} "
                    f"| {fmt_p(c.p_adjusted) if c.p_adjusted is not None else '-'} | {fmt(c.effect_size)} |"
                )
        if metric.name in figures:
            lines += ["", f"![{metric.name}]({figures[metric.name]})"]
        lines.append("")
    failed = [r for r in records if r.status != "ok"]
    if failed:
        lines += ["## Runs without results", ""]
        lines += [f"- `{r.condition}` seed {r.seed}: {r.status} — {r.error}" for r in failed]
        lines.append("")
    return "\n".join(lines)


def render_figures(plan: ExperimentPlan, summaries: list[MetricSummary], out_dir: Path) -> dict[str, Path]:
    """One bar chart per metric (mean and 95 % CI per condition)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = next((c.name for c in plan.conditions if c.is_baseline), None)
    figures: dict[str, Path] = {}
    for metric in plan.metrics:
        rows = [s for s in summaries if s.metric == metric.name]
        if not rows:
            continue
        labels = [s.condition for s in rows]
        svg = bar_chart(
            metric.name,
            labels,
            [s.mean for s in rows],
            [s.ci_low for s in rows],
            [s.ci_high for s in rows],
            ylabel=metric.name,
            highlight=labels.index(baseline) if baseline in labels else None,
            note=f"mean and 95% CI over seeds; grey = baseline; {'higher' if metric.higher_is_better else 'lower'}"
            " is better",
        )
        path = out_dir / f"{metric.name}.svg"
        path.write_text(svg, encoding="utf-8")
        figures[metric.name] = path
    return figures
