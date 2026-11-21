"""What the study pipeline's stages exchange: the protocol and the uploaded data."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.artifacts import artifact_type
from ...schemas.plan import PlanCoreUnits
from ...schemas.report import Check

DATA_FILE = "data.csv"


class ProtocolDraft(PlanCoreUnits):
    """A study design a person can carry out: the experiment's design plus what the protocol needs."""

    expected_effect_size: float = Field(
        gt=0, le=5, description="Cohen's d of the smallest effect on the primary metric worth detecting"
    )
    materials: list[str] = Field(min_length=1, description="What the study needs: instruments, reagents, forms")
    blinding: str = Field(description="Who does not know which condition a unit is in (or why nobody can be blinded)")


@artifact_type("protocol")
class Protocol(BaseModel):
    unit: str
    conditions: list[str]
    baseline: str
    primary_metric: str
    effect_size: float  # Cohen's d the sample size is computed for
    alpha: float
    power: float
    paired: bool  # every unit is measured under every condition
    n_per_condition: int  # units per condition (paired: units in all)
    allocation_path: str  # run-relative CSV: which unit gets which condition (paired: in which order)
    materials: list[str]
    blinding: str
    data_format: str  # what the uploaded data.csv must look like
    checklist: list[Check] = Field(default_factory=list)
    markdown_path: str = ""


@artifact_type("study_data")
class StudyData(BaseModel):
    path: str  # run-relative copy of the uploaded data.csv
    sha256: str
    rows: int
    units: dict[str, int]  # condition -> units with data
