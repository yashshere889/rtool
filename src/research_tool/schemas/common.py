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


class SelectedItem(BaseModel):
    """Written by a selection gate: the option the human (or auto mode) chose."""

    index: int
    item: dict[str, Any]
    edited: bool = False
    note: str = ""
