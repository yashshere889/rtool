"""Persistent run and stage state (one JSON file per stage, replaced atomically)."""

from __future__ import annotations

import secrets
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .fs import atomic_write_json, now_iso, read_json


class StageStatus(StrEnum):
    PENDING = "pending"  # not started, or ready to continue (resume=True)
    RUNNING = "running"  # an episode is executing it (or died while doing so)
    SUSPENDED = "suspended"  # waiting for Slurm jobs
    AWAITING_GATE = "awaiting_gate"  # waiting for a human decision
    DONE = "done"
    STALE = "stale"  # must re-run: inputs changed, change requested, or gate rejected
    FAILED = "failed"  # needs `rtool retry`


class GateStatus(StrEnum):
    OPEN = "open"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


def new_gate_nonce() -> str:
    return secrets.token_hex(6)


class GateState(BaseModel):
    gate_id: str
    in_stage: bool = False  # requested by the agent mid-run (ctx.gate) rather than after the stage
    nonce: str = ""  # new every time a gate opens: a decision can name the exact gate it was made at
    status: GateStatus = GateStatus.OPEN
    summary: dict[str, Any] = Field(default_factory=dict)
    opened_at: str = Field(default_factory=now_iso)
    resolved_at: str | None = None
    resolution: dict[str, Any] = Field(default_factory=dict)


class GateResolution(BaseModel):
    """What an agent receives back from ``ctx.gate(...)``."""

    approved: bool
    feedback: str = ""
    auto: bool = False
    data: dict[str, Any] = Field(default_factory=dict)


class ChangeRequest(BaseModel):
    """A downstream stage asking an upstream stage to redo its work."""

    from_stage: str
    target_stage: str
    target_artifact: str | None = None
    reason: str
    required_changes: list[str] = Field(default_factory=list)
    severity: Literal["minor", "major", "blocking"] = "major"
    created_at: str = Field(default_factory=now_iso)


class FeedbackKind(StrEnum):
    HUMAN_REJECTION = "human_rejection"
    CHANGE_REQUEST = "change_request"
    NOTE = "note"


class FeedbackItem(BaseModel):
    """Context handed to the next attempt of a stage (rejection reasons, change requests, notes)."""

    kind: FeedbackKind
    text: str
    from_stage: str | None = None
    required_changes: list[str] = Field(default_factory=list)
    severity: str = "major"
    gate_id: str | None = None
    created_at: str = Field(default_factory=now_iso)
    consumed_by_attempt: int | None = None


class StageState(BaseModel):
    stage_id: str
    status: StageStatus = StageStatus.PENDING
    attempt: int = 0
    resume: bool = False  # next execution continues the current attempt instead of starting fresh
    token: str = ""  # identifies the current attempt's commit (idempotent re-commit)
    commit_pending: bool = False
    input_versions: dict[str, int] = Field(default_factory=dict)
    output_versions: dict[str, int] = Field(default_factory=dict)
    agent_state: dict[str, Any] = Field(default_factory=dict)
    waiting_on: list[str] = Field(default_factory=list)
    job_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    gate: GateState | None = None
    feedback: list[FeedbackItem] = Field(default_factory=list)
    outbox: list[ChangeRequest] = Field(default_factory=list)
    revision_count: int = 0
    failures: int = 0
    last_error: str | None = None
    stale_reason: str | None = None
    usage: dict[str, float] = Field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None

    def pending_feedback(self) -> list[FeedbackItem]:
        return [f for f in self.feedback if f.consumed_by_attempt is None]


class RunState(BaseModel):
    run_id: str
    nonce: str  # makes commit tokens unique across runs that reuse an id
    last_outcome: str = "new"
    active_episode: str | None = None
    episodes: int = 0
    processed_actions: list[str] = Field(default_factory=list)
    usage: dict[str, float] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=now_iso)


class StateStore:
    def __init__(self, state_dir: Path) -> None:
        self.dir = Path(state_dir)

    @property
    def run_path(self) -> Path:
        return self.dir / "run.json"

    def stage_path(self, stage_id: str) -> Path:
        return self.dir / "stages" / f"{stage_id}.json"

    def load_run(self) -> RunState | None:
        if not self.run_path.exists():
            return None
        return RunState.model_validate(read_json(self.run_path))

    def save_run(self, state: RunState) -> None:
        state.updated_at = now_iso()
        atomic_write_json(self.run_path, state.model_dump(mode="json"))

    def load_stage(self, stage_id: str) -> StageState | None:
        path = self.stage_path(stage_id)
        if not path.exists():
            return None
        return StageState.model_validate(read_json(path))

    def save_stage(self, state: StageState) -> None:
        atomic_write_json(self.stage_path(state.stage_id), state.model_dump(mode="json"))

    def load_stages(self, stage_ids: list[str]) -> dict[str, StageState]:
        return {sid: self.load_stage(sid) or StageState(stage_id=sid) for sid in stage_ids}
