"""Experiment Planner outputs (the plan) and fetch_resources outputs (the manifest)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.artifacts import artifact_type

ResourceSource = Literal["huggingface", "url", "torchvision", "pip", "manual"]


class Variable(BaseModel):
    name: str
    description: str
    values: list[str] = Field(default_factory=list, description="Levels or settings, for independent variables")


class Condition(BaseModel):
    name: str = Field(description="Short identifier, e.g. 'baseline' or 'with_stimulation'")
    description: str
    is_baseline: bool = False


class Metric(BaseModel):
    name: str = Field(description="Short identifier, e.g. 'accuracy' or 'recall_gain'")
    definition: str
    higher_is_better: bool = True
    primary: bool = Field(
        default=False,
        description="True for exactly one metric: the primary outcome that decides the hypothesis (pre-registered)",
    )


class StatisticalTest(BaseModel):
    test: str = Field(description="e.g. 'paired t-test', 'Wilcoxon signed-rank', 'bootstrap 95% CI'")
    comparison: str = Field(description="Which conditions or groups are compared, on which metric")
    alpha: float = 0.05
    correction: str = Field(
        default="none", description="Multiple-comparison correction: 'holm' (used for several tests) or 'bonferroni'"
    )


# ------------------------------------------------------------- LLM outputs
class PlanCore(BaseModel):
    objective: str
    design: str = Field(description="One paragraph describing the experimental design")
    independent_variables: list[Variable] = Field(min_length=1)
    dependent_variables: list[Variable] = Field(min_length=1)
    controls: list[str] = Field(default_factory=list)
    conditions: list[Condition] = Field(min_length=2, description="Include at least one baseline")
    ablations: list[Condition] = Field(default_factory=list)
    metrics: list[Metric] = Field(min_length=1)
    statistical_tests: list[StatisticalTest] = Field(min_length=1)
    n_seeds: int = Field(ge=1, le=10, description="Independent repetitions per condition")
    procedure: list[str] = Field(min_length=3, description="Ordered steps a programmer can follow")
    expected_results: str
    risks: list[str] = Field(default_factory=list)


class ResourceChoice(BaseModel):
    kind: Literal["dataset", "model", "code", "other"]
    name: str
    source: ResourceSource
    identifier: str = Field(description="Hugging Face repo id, URL, torchvision name or pip package")
    subset: str | None = Field(default=None, description="Hugging Face config name, if any")
    split: str | None = None
    purpose: str


class ResourcePlan(BaseModel):
    resources: list[ResourceChoice] = Field(min_length=1)


class ComputeEstimate(BaseModel):
    gpu_memory_gb: float = Field(ge=0, description="GPU memory one run needs; 0 if no GPU")
    gpus_per_run: int = Field(ge=0, le=8)
    hours_per_run: float = Field(gt=0, le=72, description="Wall-clock hours for one run of one condition/seed")
    justification: str


# ---------------------------------------------------------- plan artifact
class ResourceSpec(ResourceChoice):
    verified: bool = False
    license: str = ""
    size_gb: float | None = None
    note: str = ""


class ComputeRequest(BaseModel):
    partition: str
    gpu_type: str | None = None
    gpus_per_run: int = 0
    time_limit: str  # Slurm format, per run
    runs: int
    hours_per_run: float
    gpu_hours_total: float
    gpu_memory_gb: float = 0
    justification: str = ""


@artifact_type("experiment_plan")
class ExperimentPlan(BaseModel):
    hypothesis_id: str
    hypothesis: str
    null_hypothesis: str
    objective: str
    design: str
    independent_variables: list[Variable]
    dependent_variables: list[Variable]
    controls: list[str] = Field(default_factory=list)
    conditions: list[Condition]
    ablations: list[Condition] = Field(default_factory=list)
    metrics: list[Metric]
    statistical_tests: list[StatisticalTest]
    seeds: list[int]
    procedure: list[str]
    resources: list[ResourceSpec]
    compute: ComputeRequest
    expected_results: str
    risks: list[str] = Field(default_factory=list)
    results_schema: dict[str, Any]  # JSON Schema the Coder's results.json must satisfy
    issues: list[str] = Field(default_factory=list)  # problems the checks could not fix (shown at the gate)
    markdown_path: str = ""

    def run_conditions(self) -> list[Condition]:
        return [*self.conditions, *self.ablations]

    def primary_metric(self) -> Metric:
        """The pre-registered primary outcome (the first metric in plans written before metrics were marked)."""
        return next((m for m in self.metrics if m.primary), self.metrics[0])

    def metrics_primary_first(self) -> list[Metric]:
        primary = self.primary_metric()
        return [primary, *(m for m in self.metrics if m is not primary)]


# -------------------------------------------------------- fetch artifact
class FetchedResource(BaseModel):
    name: str
    kind: str
    source: str
    identifier: str
    status: Literal["ready", "deferred", "manual", "failed"]
    path: str | None = None  # absolute path on the cluster (shared resources may live outside the run)
    bytes: int = 0
    files: int = 0
    note: str = ""


@artifact_type("resource_manifest")
class ResourceManifest(BaseModel):
    resources: list[FetchedResource]
    root: str

    def ready(self) -> list[FetchedResource]:
        return [r for r in self.resources if r.status == "ready"]
