"""Slurm client interface (the real implementation arrives in M3)."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from .jobspec import JobSpec


class JobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def succeeded(self) -> bool:
        return self is JobState.COMPLETED


_TERMINAL = {
    JobState.COMPLETED,
    JobState.FAILED,
    JobState.CANCELLED,
    JobState.TIMEOUT,
    JobState.OUT_OF_MEMORY,
    JobState.NODE_FAIL,
    JobState.PREEMPTED,  # final only once the job has left the queue; a requeued job shows as PENDING
}


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    exit_code: int | None = None
    reason: str = ""
    name: str = ""


class SlurmClient(Protocol):
    def submit(self, spec: JobSpec) -> str: ...

    def status(self, job_id: str) -> JobStatus: ...

    def status_many(self, job_ids: Iterable[str]) -> dict[str, JobStatus]: ...

    def cancel(self, job_id: str) -> None: ...

    def estimate_start(self, spec: JobSpec) -> float | None:
        """When Slurm expects ``spec`` to start if submitted now (epoch seconds), or None if unknown."""
        ...
