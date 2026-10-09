"""Hypothesis agent outputs: drafted, checked and ranked hypotheses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..core.artifacts import artifact_type, register_artifact_type
from .common import SelectedItem

# Written by the hypothesis selection gate; registered here so pipelines can also supply it directly.
register_artifact_type("selected_hypothesis", SelectedItem)

ComputeClass = Literal["cpu_only", "small_gpu", "medium_gpu", "large_gpu"]
COMPUTE_HOURS = {"cpu_only": 0, "small_gpu": 8, "medium_gpu": 72, "large_gpu": 500}  # upper bound, GPU-hours


class EvidenceRef(BaseModel):
    paper_id: str = Field(description="Id of a paper from the list given")
    quote: str = Field(description="Sentence copied word for word from that paper")


class HypothesisDraft(BaseModel):
    title: str = Field(description="Short name, at most 10 words")
    statement: str = Field(description="Testable claim: if/when <intervention or condition>, <measurable outcome>")
    null_hypothesis: str
    rationale: str = Field(description="Why this follows from the evidence, 2-3 sentences")
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=4)
    independent_variables: list[str] = Field(min_length=1)
    dependent_variables: list[str] = Field(min_length=1)
    controls: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list, description="Datasets, populations or resources to use")
    methods: list[str] = Field(default_factory=list)
    success_criteria: str = Field(description="What result would support it and what would refute it")
    risks: list[str] = Field(default_factory=list)
    compute: ComputeClass = Field(
        description="cpu_only; small_gpu (<8 GPU-hours); medium_gpu (8-72 GPU-hours); large_gpu (>72 GPU-hours)"
    )


class HypothesisDrafts(BaseModel):
    hypotheses: list[HypothesisDraft] = Field(min_length=1, max_length=2)


class NoveltyJudgment(BaseModel):
    novel: bool = Field(description="False if a listed paper already tests essentially this hypothesis")
    closest_paper_id: str = Field(default="", description="Id of the most similar listed paper, if any")
    overlap: str = Field(description="One sentence on how it differs from (or repeats) the closest paper")
    score: int = Field(ge=0, le=10, description="10 = clearly new, 0 = already done")


class RubricScores(BaseModel):
    novelty: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    testability: int = Field(ge=1, le=5)
    significance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    comment: str = Field(description="One sentence: the main strength or weakness")


class CheckedEvidence(BaseModel):
    paper_id: str
    quote: str
    verified: bool
    support: float


class Hypothesis(BaseModel):
    """A hypothesis with everything the planner and the human need."""

    id: str
    title: str
    statement: str
    null_hypothesis: str
    rationale: str
    evidence: list[CheckedEvidence]
    independent_variables: list[str]
    dependent_variables: list[str]
    controls: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    success_criteria: str
    risks: list[str] = Field(default_factory=list)
    compute: ComputeClass
    source_opportunity: str = ""
    # the papers behind the gap it addresses, then the nearest papers found in the novelty check
    related_paper_ids: list[str] = Field(default_factory=list)
    novelty: NoveltyJudgment | None = None
    feasibility_notes: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    rubric_comments: list[str] = Field(default_factory=list)  # the judges' one-line verdicts, without repeats
    total_score: float = 0.0
    rejected_reason: str | None = None


@artifact_type("hypothesis_set")
class HypothesisSet(BaseModel):
    topic: str
    hypotheses: list[Hypothesis] = Field(description="The ranked shortlist offered at the selection gate")
    candidates: list[Hypothesis] = Field(default_factory=list, description="Everything drafted, with scores")
    markdown_path: str = ""
