"""Validity cases: a question with a known answer, run through rtool to measure whether it gets it right.

A case is a folder ``benchmarks/<pack>/<case>/`` holding ``case.yaml``: the brief, the hypothesis it
fixes, the data it provides and the outcome it expects (docs/stage0_design.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..core.config import deep_merge
from ..core.errors import ConfigError, RToolError
from ..core.run import is_valid_name
from ..packs import load_pack
from ..schemas.common import ResearchBrief
from ..schemas.hypothesis import ComputeClass, Hypothesis

CASE_FILE = "case.yaml"
MAIN_VARIANT = "main"  # the name of a case's only variant when it lists none
# The exact Wilcoxon signed-rank test, which the planner may choose, cannot reach p < 0.05 with fewer
# paired seeds: with 5 its smallest two-sided p-value is 2/32 = 0.0625.
MIN_SEEDS = 6

CaseKind = Literal["planted", "null", "replication", "open"]
Outcome = Literal["supported", "refuted", "inconclusive"]
DEFAULT_OUTCOMES: dict[str, list[Outcome]] = {"planted": ["supported"], "null": ["refuted", "inconclusive"]}
_SLUG = r"^[a-z][a-z0-9_]*$"  # how the planner names conditions and metrics


class CaseError(RToolError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt key in case.yaml must fail, not be ignored


class PinnedCondition(_Strict):
    name: str = Field(pattern=_SLUG)
    baseline: bool = False
    description: str = ""


class PinnedMetric(_Strict):
    name: str = Field(pattern=_SLUG)
    higher_is_better: bool


class CaseBrief(_Strict):
    topic: str
    questions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    notes: str = ""
    seeds: int | None = Field(default=None, ge=1)
    max_minutes_per_run: float | None = Field(default=None, gt=0)
    datasets: list[str] = Field(default_factory=list)
    conditions: list[PinnedCondition] = Field(default_factory=list)
    primary_metric: PinnedMetric | None = None

    def research_brief(self, pack: str) -> ResearchBrief:
        """The brief a run of this case starts from."""
        # shortcut: the pinned conditions and metric travel as requirement lines the plan checks do not
        # enforce; Stage 0 PR 2 adds them to ResearchBrief, checked like seeds and datasets.
        lines = list(self.constraints)
        if self.conditions:
            arms = "; ".join(
                f"`{c.name}`{' (the baseline)' if c.baseline else ''}{': ' + c.description if c.description else ''}"
                for c in self.conditions
            )
            lines.append(f"Use exactly these conditions, with exactly these names: {arms}.")
        if self.primary_metric:
            direction = "higher is better" if self.primary_metric.higher_is_better else "lower is better"
            lines.append(f"The primary metric is `{self.primary_metric.name}` ({direction}).")
        return ResearchBrief(
            topic=self.topic,
            questions=self.questions,
            constraints=lines,
            notes=self.notes,
            domain_pack=pack,
            seeds=self.seeds,
            max_minutes_per_run=self.max_minutes_per_run,
            datasets=self.datasets,
        )


class CaseHypothesis(_Strict):
    """The hypothesis a case fixes: the fields of a :class:`Hypothesis` an author has to write."""

    statement: str
    null_hypothesis: str
    success_criteria: str = Field(min_length=1)
    independent_variables: list[str] = Field(min_length=1)
    dependent_variables: list[str] = Field(min_length=1)
    title: str = ""
    rationale: str = ""
    controls: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    compute: ComputeClass = "cpu_only"

    def hypothesis(self, case_id: str) -> Hypothesis:
        fields = self.model_dump()
        fields["title"] = self.title or self.statement
        fields["rationale"] = self.rationale or "Fixed by a validity case, which knows the answer."
        return Hypothesis(id=case_id, evidence=[], **fields)


class DataSpec(_Strict):
    name: str
    generator: str  # a stdlib-only Python script in the case folder
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class Band(_Strict):
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> Band:
        if self.low > self.high:
            raise ValueError(f"low ({self.low:g}) is above high ({self.high:g})")
        return self


class ValueBand(Band):
    condition: str = Field(pattern=_SLUG)


class Expectation(_Strict):
    condition: str | None = Field(default=None, pattern=_SLUG)  # compared with the baseline; default: the only one
    significant: bool | None = None
    direction: Literal["better", "worse"] | None = None  # in the primary metric's preferred direction
    difference: Band | None = None  # the estimated difference (condition minus baseline) must fall in it
    value: ValueBand | None = None  # the mean of the primary metric for one condition must fall in it
    outcome: list[Outcome] = Field(default_factory=list)  # what the final report may conclude; empty: not checked


class Variant(_Strict):
    name: str
    data_args: dict[str, Any] = Field(default_factory=dict)
    expect: dict[str, Any] = Field(default_factory=dict)  # merged over the case's expect


class Case(_Strict):
    id: str
    kind: CaseKind
    pack: str
    start_at: str = ""  # default: "literature" for open cases, else "plan"
    brief: CaseBrief
    hypothesis: CaseHypothesis | None = None
    data: list[DataSpec] = Field(default_factory=list)
    expect: Expectation = Field(default_factory=Expectation)
    variants: list[Variant] = Field(default_factory=list)
    repeats: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> Case:
        problems = self._problems()
        if problems:
            raise ValueError("; ".join(problems))
        return self

    # ------------------------------------------------------------ queries
    @property
    def start(self) -> str:
        return self.start_at or ("literature" if self.kind == "open" else "plan")

    @property
    def baseline(self) -> str | None:
        return next((c.name for c in self.brief.conditions if c.baseline), None)

    def variant_names(self) -> list[str]:
        return [v.name for v in self.variants] or [MAIN_VARIANT]

    def expectation(self, variant: str = MAIN_VARIANT) -> Expectation:
        """What this variant must show: the case's expectation with the variant's changes and the defaults."""
        found = next((v for v in self.variants if v.name == variant), None)
        if found is None and variant != MAIN_VARIANT:
            raise CaseError(f"case {self.id!r} has no variant {variant!r}; it has {self.variant_names()}")
        merged = deep_merge(self.expect.model_dump(exclude_none=True), found.expect if found else {})
        expect = Expectation.model_validate(merged)
        updates: dict[str, Any] = {}
        if not expect.outcome and self.kind in DEFAULT_OUTCOMES:
            updates["outcome"] = DEFAULT_OUTCOMES[self.kind]
        if expect.condition is None:
            treatments = [c.name for c in self.brief.conditions if not c.baseline]
            if len(treatments) == 1:
                updates["condition"] = treatments[0]
        return expect.model_copy(update=updates) if updates else expect

    def research_brief(self) -> ResearchBrief:
        return self.brief.research_brief(self.pack)

    # ---------------------------------------------------------- validation
    def _problems(self) -> list[str]:
        problems: list[str] = []
        if not is_valid_name(self.id):
            problems.append(f"id {self.id!r} may use only letters, digits, '.', '_' and '-'")
        names = [v.name for v in self.variants]
        if len(names) != len(set(names)) or not all(is_valid_name(n) for n in names):
            problems.append("variant names must be unique and use only letters, digits, '.', '_' and '-'")
        data = [d.name for d in self.data]
        if len(data) != len(set(data)):
            problems.append("data names must be unique")
        unlisted = [n for n in data if n not in self.brief.datasets]
        if unlisted:
            problems.append(f"brief.datasets must list the data the case provides: {unlisted}")
        if self.kind == "open":
            if self.hypothesis or self.brief.conditions or self.brief.primary_metric:
                problems.append("an open case fixes nothing: no hypothesis, conditions or primary metric")
            return problems
        problems += self._design_problems()
        for variant in self.variant_names():
            try:
                expect = self.expectation(variant)
            except (ValidationError, CaseError) as exc:
                problems.append(f"variant {variant!r}: {exc}")
                continue
            problems += [f"{p} (variant {variant!r})" if self.variants else p for p in self._expect_problems(expect)]
        return problems

    def _design_problems(self) -> list[str]:
        problems = []
        if self.hypothesis is None:
            problems.append(f"a {self.kind} case needs a hypothesis")
        conditions = [c.name for c in self.brief.conditions]
        if len(conditions) < 2 or len(set(conditions)) != len(conditions):
            problems.append("brief.conditions needs at least two conditions with different names")
        if sum(c.baseline for c in self.brief.conditions) != 1:
            problems.append("exactly one of brief.conditions must be the baseline")
        if self.brief.primary_metric is None:
            problems.append("brief.primary_metric is required")
        if self.brief.seeds is None:
            problems.append("brief.seeds is required, so every run of the case has the same number")
        elif self.kind in ("planted", "null") and self.brief.seeds < MIN_SEEDS:
            problems.append(f"use at least {MIN_SEEDS} seeds: with fewer, an exact rank test cannot be significant")
        return problems

    def _expect_problems(self, expect: Expectation) -> list[str]:
        problems = []
        treatments = [c.name for c in self.brief.conditions if not c.baseline]
        compares = expect.significant is not None or expect.direction is not None or expect.difference is not None
        if expect.condition is not None and expect.condition not in treatments:
            problems.append(f"expect.condition {expect.condition!r} is not a non-baseline condition {treatments}")
        elif compares and expect.condition is None:
            problems.append(f"expect.condition must name the condition compared with the baseline, one of {treatments}")
        if self.kind == "planted" and not (expect.significant is True and expect.direction):
            problems.append("a planted case expects a significant effect: significant: true and a direction")
        if self.kind == "null" and (expect.significant is not False or expect.direction or expect.difference):
            problems.append("a null case expects no effect: significant: false, and no direction or difference")
        if self.kind == "replication":
            names = [c.name for c in self.brief.conditions]
            if expect.value is None:
                problems.append("a replication case needs expect.value: the known value and its condition")
            elif expect.value.condition not in names:
                problems.append(f"expect.value.condition {expect.value.condition!r} is not one of {names}")
        return problems


# ------------------------------------------------------------------ loading
def load_case(folder: Path) -> Case:
    """The case in ``folder`` (its ``case.yaml``), checked; a mistake names the file and the problem."""
    folder = Path(folder)
    path = folder / CASE_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CaseError(f"no {CASE_FILE} in {folder}") from None
    except yaml.YAMLError as exc:
        raise CaseError(f"invalid YAML in {path}: {exc}") from None
    if not isinstance(data, dict):
        raise CaseError(f"{path} must contain a mapping")
    try:
        case = Case.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'case'}: {e['msg'].removeprefix('Value error, ')}"
            for e in exc.errors()
        )
        raise CaseError(f"invalid case {path}: {problems}") from None
    if case.id != folder.name:
        raise CaseError(f"{path}: the case id {case.id!r} must be its folder's name {folder.name!r}")
    try:
        load_pack(case.pack)
    except ConfigError as exc:
        raise CaseError(f"{path}: {exc}") from None
    root = folder.resolve()
    for spec in case.data:
        script = (folder / spec.generator).resolve()
        if not script.is_relative_to(root) or not script.is_file():
            raise CaseError(f"{path}: data {spec.name!r}: no generator {spec.generator!r} in the case folder")
    return case


def load_cases(root: Path, pack: str) -> list[Case]:
    """Every case of ``pack`` under ``root`` (``<root>/<pack>/<case>/case.yaml``), in folder order."""
    cases = [load_case(p.parent) for p in sorted((Path(root) / pack).glob(f"*/{CASE_FILE}"))]
    wrong = [c.id for c in cases if c.pack != pack]
    if wrong:
        raise CaseError(f"cases under {Path(root) / pack} must use the {pack!r} pack: {wrong}")
    return cases
