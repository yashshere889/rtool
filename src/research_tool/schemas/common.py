"""Artifacts shared by every pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.artifacts import artifact_type


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

    def requirements(self) -> list[str]:
        """Everything the researcher fixed, as lines for the planning and coding prompts."""
        lines = list(self.constraints)
        if self.seeds:
            lines.append(f"Use exactly {self.seeds} seeds per condition.")
        if self.max_minutes_per_run:
            lines.append(f"Each run (one condition, one seed) must finish within {self.max_minutes_per_run:g} minutes.")
        if self.datasets:
            lines.append(f"Use only these datasets: {', '.join(self.datasets)}.")
        return [*lines, self.notes] if self.notes else lines


class SelectedItem(BaseModel):
    """Written by a selection gate: the option the human (or auto mode) chose."""

    index: int
    item: dict[str, Any]
    edited: bool = False
    note: str = ""
