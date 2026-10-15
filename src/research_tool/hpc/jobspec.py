"""Description of a batch job, independent of how it is submitted.

The real Slurm client (M3) renders this into an sbatch script following the
Barkla rules: ``#!/bin/bash -l``, one partition, ``--gres=gpu:<type>:<n>`` and
no CPU/memory requests for GPU jobs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

_TIME = re.compile(r"^(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$")


def parse_time_limit(value: str) -> int:
    """'D-HH:MM:SS' or 'HH:MM:SS' -> seconds."""
    match = _TIME.match(value.strip())
    if not match:
        raise ValueError(f"time limit {value!r} must look like HH:MM:SS or D-HH:MM:SS")
    days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    if minutes > 59 or seconds > 59:
        raise ValueError(f"time limit {value!r} has minutes/seconds above 59")
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def format_time_limit(seconds: float) -> str:
    """Seconds -> Slurm 'D-HH:MM:SS' / 'HH:MM:SS', rounded up to whole minutes."""
    seconds = int(-(-seconds // 60) * 60)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    return f"{days}-{hours:02d}:{minutes:02d}:00" if days else f"{hours:02d}:{minutes:02d}:00"


class GPURequest(BaseModel):
    type: str | None = None  # e.g. "l40s", "h100", "v100"
    count: int = Field(default=1, ge=1)

    def gres(self) -> str:
        return f"gpu:{self.type}:{self.count}" if self.type else f"gpu:{self.count}"


class JobSpec(BaseModel):
    name: str
    partition: str
    command: list[str] | str
    workdir: Path
    time_limit: str = "01:00:00"
    gpus: GPURequest | None = None
    ntasks: int | None = None  # CPU jobs only; GPU jobs get cores automatically on Barkla
    array: str | None = None
    dependency: list[str] = Field(default_factory=list)
    dependency_type: Literal["afterany", "afterok"] = "afterany"
    env: dict[str, str] = Field(default_factory=dict)
    modules: list[str] = Field(default_factory=list)
    requeue: bool = False
    exclude: list[str] = Field(default_factory=list)  # nodes Slurm must not use (e.g. one that could not start vLLM)
    signal: str | None = None  # e.g. "B:USR1@600"
    stdout: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("time_limit")
    @classmethod
    def _valid_time(cls, value: str) -> str:
        parse_time_limit(value)
        return value

    @property
    def time_limit_seconds(self) -> int:
        return parse_time_limit(self.time_limit)

    @property
    def gpu_hours_requested(self) -> float:
        gpus = self.gpus.count if self.gpus else 0
        return gpus * self.time_limit_seconds / 3600
