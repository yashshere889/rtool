"""Workspace jail, budgets, job specs, fake Slurm."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from research_tool.core.budgets import BudgetTracker
from research_tool.core.config import BudgetLimits
from research_tool.core.errors import BudgetExceeded, PathEscape
from research_tool.core.workspace import Workspace
from research_tool.hpc.fake_slurm import FakeSlurm
from research_tool.hpc.jobspec import GPURequest, JobSpec, parse_time_limit
from research_tool.hpc.slurm import JobState


def test_workspace_jail(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    ws.write_text("src/main.py", "print(1)\n")
    assert ws.read_text("src/main.py") == "print(1)\n"
    assert ws.listdir("src") == ["main.py"]
    with pytest.raises(PathEscape):
        ws.path("../outside.txt")
    os.symlink(tmp_path, ws.root / "link")
    with pytest.raises(PathEscape):
        ws.path("link/escape.txt")


def test_budget_tracker_limits_stage_and_run() -> None:
    tracker = BudgetTracker(BudgetLimits(max_llm_calls=2), BudgetLimits(max_prompt_tokens=100))
    tracker.charge(llm_calls=1, prompt_tokens=40)
    assert tracker.would_exceed(llm_calls=2) is not None
    assert tracker.stage_usage.llm_calls == 1  # would_exceed has no side effects
    with pytest.raises(BudgetExceeded, match="run budget exceeded: prompt_tokens"):
        tracker.charge(prompt_tokens=70)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("00:30:00", 1800), ("12:30:00", 45000), ("3-00:00:00", 259200), ("1-02:03:04", 93784)],
)
def test_parse_time_limit(text: str, seconds: int) -> None:
    assert parse_time_limit(text) == seconds


def test_jobspec_validation_and_gpu_hours(tmp_path: Path) -> None:
    spec = JobSpec(
        name="x",
        partition="gpu-h100",
        command="true",
        workdir=tmp_path,
        time_limit="02:00:00",
        gpus=GPURequest(type="h100", count=4),
    )
    assert spec.gpus is not None and spec.gpus.gres() == "gpu:h100:4"
    assert spec.gpu_hours_requested == pytest.approx(8.0)
    with pytest.raises(ValueError):
        JobSpec(name="x", partition="p", command="true", workdir=tmp_path, time_limit="90 minutes")


def test_fake_slurm_lifecycle_dependencies_and_persistence(tmp_path: Path) -> None:
    state = tmp_path / "slurm.json"
    slurm = FakeSlurm(state)
    a = slurm.submit(JobSpec(name="train", partition="p", command="true", workdir=tmp_path))
    b = slurm.submit(
        JobSpec(name="eval", partition="p", command="true", workdir=tmp_path, dependency=[a], dependency_type="afterok")
    )
    slurm.set_outcome("train", JobState.FAILED, exit_code=3)
    assert slurm.status(a).state is JobState.PENDING
    slurm.tick()
    assert FakeSlurm(state).status(a).state is JobState.RUNNING  # persisted for other processes
    slurm.finish_all()
    assert slurm.status(a).state is JobState.FAILED and slurm.status(a).exit_code == 3
    assert slurm.status(b).state is JobState.CANCELLED  # afterok never satisfied
    assert slurm.status("999").state is JobState.UNKNOWN


def test_fake_slurm_cancel(tmp_path: Path) -> None:
    slurm = FakeSlurm()
    job = slurm.submit(JobSpec(name="x", partition="p", command="true", workdir=tmp_path))
    slurm.cancel(job)
    assert slurm.status(job).state is JobState.CANCELLED


def test_event_log_cursor_tails_complete_lines(tmp_path: Path) -> None:
    from research_tool.core.events import EventLog

    log = EventLog(tmp_path / "events.jsonl", "r1")
    assert log.read_from(0) == ([], 0)
    for n in range(3):
        log.emit("tick", n=n)
    events, cursor = log.read_from(0, limit=2)
    assert [e["data"]["n"] for e in events] == [0, 1]
    with open(log.path, "ab") as f:
        f.write(b'{"kind": "half-writ')  # an append in progress
    events, cursor = log.read_from(cursor)
    assert [e["data"]["n"] for e in events] == [2]
    assert log.read_from(cursor) == ([], cursor)  # the torn line waits until it is complete
    with open(log.path, "ab") as f:
        f.write(b'ten"}\n')
    events, end = log.read_from(cursor)
    assert [e["kind"] for e in events] == ["half-written"] and end == log.path.stat().st_size
    assert log.read_from(end + 100)[0][0]["data"]["n"] == 0  # a cursor past the end starts over
