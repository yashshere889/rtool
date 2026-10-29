"""Coder outputs: the experiment code (code_bundle) and what running it produced (experiment_results)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..core.artifacts import artifact_type

FixPhase = Literal["check", "smoke", "full"]


class CodeFile(BaseModel):
    path: str  # relative to the code directory
    sha256: str
    lines: int
    origin: Literal["harness", "template", "llm", "human"]


class SmokeResult(BaseModel):
    condition: str
    seed: int | None = None
    ok: bool
    seconds: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str = ""


class FixRecord(BaseModel):
    round: int
    phase: FixPhase
    signature: str  # e.g. missing_module, cuda_oom, traceback
    action: str  # what was done: "added requirement scikit-learn", "LLM rewrite", ...
    detail: str = ""


@artifact_type("code_bundle")
class CodeBundle(BaseModel):
    root: str  # code directory, relative to the run directory
    entry_point: str = "run.py"
    files: list[CodeFile]
    pack: str
    container: str | None = None
    requirements: list[str] = Field(default_factory=list)  # pip packages installed next to the code
    smoke: list[SmokeResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # smoke runs that ignored the seed or condition
    fixes: list[FixRecord] = Field(default_factory=list)
    readme_path: str = ""  # run-relative
    fingerprint: str = ""  # hash over all files: results belong to exactly this code


# ------------------------------------------------------------------ results
class RunRecord(BaseModel):
    condition: str
    seed: int
    status: Literal["ok", "failed", "missing"]
    metrics: dict[str, float] = Field(default_factory=dict)
    seconds: float | None = None
    job_id: str = ""
    error: str = ""


class MetricSummary(BaseModel):
    condition: str
    metric: str
    n: int
    mean: float
    std: float
    ci_low: float  # 95 % confidence interval of the mean (t distribution)
    ci_high: float


class Comparison(BaseModel):
    """One condition against the baseline on one metric."""

    metric: str
    condition: str
    baseline: str
    difference: float  # condition minus baseline (mean)
    ci_low: float
    ci_high: float
    test: str  # "paired t-test" (same seeds) or "Welch t-test"
    statistic: float | None = None
    p_value: float | None = None  # None: no test possible (too few runs, or no variation across seeds)
    p_adjusted: float | None = None  # after the multiple-comparison correction within its analysis, if any
    correction: Literal["holm", "bonferroni", "none"] = "none"  # how p_adjusted was computed
    effect_size: float | None = None  # Cohen's d (d_z for paired data)
    n: int
    significant: bool = False
    better: bool | None = None  # in the metric's preferred direction (None if not significant)
    no_seed_variation: bool = False  # both sides gave one fixed value for every seed: no p-value, never significant
    constant_difference: bool = False  # both varied, but differed by the same amount every seed: likewise untested
    analysis: Literal["confirmatory", "exploratory"] = "exploratory"  # confirmatory: on the primary metric

    @property
    def p_key(self) -> str:
        """The report placeholder of the p-value that decides significance: the corrected one if there is one."""
        return "p_adj" if self.p_adjusted is not None else "p"

    def verdict(self) -> str:
        """The result in words, for the Writer and the Reviewer."""
        if self.p_value is None:
            if self.n < 2:
                return "no test possible (too few runs)"
            if self.constant_difference:
                return (
                    "no test possible: the difference was identical for every seed, so it may be a fixed offset "
                    "rather than a measured effect"
                )
            return "no test possible: the results did not vary across seeds, so there is nothing to test against"
        if not self.significant:
            return "not statistically significant"
        return "a significant improvement" if self.better else "a significant deterioration"


@artifact_type("experiment_results")
class ExperimentResults(BaseModel):
    runs: list[RunRecord]
    summaries: list[MetricSummary] = Field(default_factory=list)
    comparisons: list[Comparison] = Field(default_factory=list)
    complete: bool  # every planned (condition, seed) run succeeded
    results_path: str  # results.json, matching the plan's results_schema (run-relative)
    tables_path: str = ""  # results.md (run-relative)
    figures: list[str] = Field(default_factory=list)  # run-relative SVG files
    jobs: list[str] = Field(default_factory=list)
    gpu_hours_requested: float = 0.0
    code_fingerprint: str = ""
    notes: list[str] = Field(default_factory=list)
    correction: str = ""  # "holm" or "bonferroni" if the comparisons' p-values were corrected
    # signs that nothing real was measured (e.g. every seed gave the same value): for people to check
    warnings: list[str] = Field(default_factory=list)

    def summary(self, condition: str, metric: str) -> MetricSummary | None:
        return next((s for s in self.summaries if s.condition == condition and s.metric == metric), None)

    def comparison(self, condition: str, metric: str) -> Comparison | None:
        return next((c for c in self.comparisons if c.condition == condition and c.metric == metric), None)
