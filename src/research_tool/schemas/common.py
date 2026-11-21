"""Artifacts shared by every pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.artifacts import artifact_type

_NAME = r"^[a-z][a-z0-9_]*$"  # how plans name conditions and metrics


class BriefCondition(BaseModel):
    """A condition the researcher fixes: the plan must have it, under this name."""

    name: str = Field(pattern=_NAME)
    baseline: bool = False
    description: str = ""


class BriefMetric(BaseModel):
    """The primary outcome the researcher fixes, and which way is better."""

    name: str = Field(pattern=_NAME)
    higher_is_better: bool


@artifact_type("research_brief")
class ResearchBrief(BaseModel):
    """What the researcher wants investigated; created by ``rtool new``."""

    topic: str
    questions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    domain_pack: str = "generic"
    seed_queries: list[str] = Field(default_factory=list)
    seed_papers: list[str] = Field(default_factory=list)  # DOIs or arXiv ids
    notes: str = ""
    # Limits the planner must meet; unlike free-text constraints, the plan checks enforce them.
    seeds: int | None = Field(default=None, ge=1)  # exactly this many seeds per condition
    max_minutes_per_run: float | None = Field(default=None, gt=0)  # one condition x one seed
    datasets: list[str] = Field(default_factory=list)  # only these datasets; empty: any
    conditions: list[BriefCondition] = Field(default_factory=list)  # the plan must have these; empty: any
    primary_metric: BriefMetric | None = None  # the plan's primary outcome; None: the planner chooses

    def requirements(self) -> list[str]:
        """Everything the researcher fixed, as lines for the planning and coding prompts."""
        lines = list(self.constraints)
        if self.seeds:
            lines.append(f"Use exactly {self.seeds} seeds per condition.")
        if self.max_minutes_per_run:
            lines.append(f"Each run (one condition, one seed) must finish within {self.max_minutes_per_run:g} minutes.")
        if self.datasets:
            lines.append(f"Use only these datasets: {', '.join(self.datasets)}.")
        if self.conditions:
            arms = "; ".join(
                f"`{c.name}`{' (the baseline)' if c.baseline else ''}{': ' + c.description if c.description else ''}"
                for c in self.conditions
            )
            lines.append(f"Include these conditions, with exactly these names: {arms}.")
        if self.primary_metric:
            direction = "higher is better" if self.primary_metric.higher_is_better else "lower is better"
            lines.append(f"The primary metric is `{self.primary_metric.name}` ({direction}).")
        return [*lines, self.notes] if self.notes else lines


class SelectedItem(BaseModel):
    """Written by a selection gate: the option the human (or auto mode) chose."""

    index: int
    item: dict[str, Any]
    edited: bool = False
    note: str = ""
