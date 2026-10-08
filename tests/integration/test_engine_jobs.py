"""Slurm suspend/resume, in-stage gates, and invalidation while waiting."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.engine import Engine, Outcome, load_status
from research_tool.core.run import RunHandle
from research_tool.core.services import Services
from research_tool.hpc.fake_slurm import FakeSlurm
from research_tool.hpc.slurm import JobState
from research_tool.llm.mock_backend import MockLLM
from research_tool.testing.harness import drive, make_run, make_services, queue
from research_tool.testing.toy_agents import ToyNumbers

JOB = {"name": "job", "stages": [{"id": "source", "agent": "toy_source"}, {"id": "analyze", "agent": "toy_analyzer"}]}
GPU_GATE = {
    "name": "gpu",
    "stages": [
        {"id": "source", "agent": "toy_source"},
        {"id": "analyze", "agent": "toy_analyzer", "config": {"gate": "gpu_runs"}},
    ],
}


def setup(runs_root: Path, spec: dict, mock_llm: MockLLM, **kw) -> tuple[RunHandle, FakeSlurm, Services]:
    run = make_run(runs_root, spec, **kw)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    return run, slurm, make_services(run, mock_llm, slurm)


def test_stage_waits_for_its_job_without_resubmitting(runs_root: Path, mock_llm: MockLLM) -> None:
    run, slurm, services = setup(runs_root, JOB, mock_llm)
    first = Engine(run, services).advance()
    assert first.outcome is Outcome.WAITING_JOB and first.waiting_jobs == ["1000"]
    slurm.tick()  # running, not finished yet
    assert Engine(run, services).advance().outcome is Outcome.WAITING_JOB
    slurm.finish_all()
    final = Engine(run, services).advance()
    assert final.outcome is Outcome.DONE and final.ran == ["analyze"]
    assert list(slurm.jobs) == ["1000"]  # submitted exactly once
    assert ArtifactStore(run.artifacts_dir).get("toy_stats").job_ids == ["1000"]
    _, stages, _ = load_status(run)
    assert stages["analyze"].job_results["1000"]["state"] == "COMPLETED"
    assert stages["analyze"].attempt == 1


def test_failed_job_fails_the_stage_and_retry_resubmits(runs_root: Path, mock_llm: MockLLM) -> None:
    run, slurm, services = setup(runs_root, JOB, mock_llm)
    slurm.set_outcome("toy-analysis", JobState.FAILED)
    Engine(run, services).advance()
    slurm.finish_all()
    failed = Engine(run, services).advance()
    assert failed.outcome is Outcome.NEEDS_HUMAN
    assert "ended FAILED" in failed.failed[0]["error"]
    slurm.set_outcome("toy-analysis", JobState.COMPLETED, exit_code=0)
    queue(run, ActionKind.RETRY, stage="analyze")
    assert drive(run, services).outcome is Outcome.DONE
    assert sorted(slurm.jobs) == ["1000", "1001"]


def test_in_stage_gate_pauses_before_spending_compute(runs_root: Path, mock_llm: MockLLM) -> None:
    run, slurm, services = setup(runs_root, GPU_GATE, mock_llm)
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.WAITING_GATE
    gate = result.open_gates[0]
    assert gate["gate"] == "gpu_runs" and gate["in_stage"] and gate["summary"]["partition"] == "gpu-l40s"
    assert slurm.jobs == {}  # nothing submitted before approval
    queue(run, ActionKind.APPROVE)
    assert Engine(run, services).advance().outcome is Outcome.WAITING_JOB
    slurm.finish_all()
    assert Engine(run, services).advance().outcome is Outcome.DONE


def test_rejecting_an_in_stage_gate_hands_back_to_the_agent(runs_root: Path, mock_llm: MockLLM) -> None:
    run, slurm, services = setup(runs_root, GPU_GATE, mock_llm)
    Engine(run, services).advance()
    queue(run, ActionKind.REJECT, payload={"feedback": "too expensive"})
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.NEEDS_HUMAN
    assert "too expensive" in result.failed[0]["error"]
    assert slurm.jobs == {}


def test_auto_mode_for_in_stage_gates(runs_root: Path, mock_llm: MockLLM) -> None:
    run, _, services = setup(runs_root, GPU_GATE, mock_llm, overrides={"gates": {"modes": {"gpu_runs": "auto"}}})
    assert drive(run, services).outcome is Outcome.DONE


def test_upstream_change_cancels_the_wait_and_restarts(runs_root: Path, mock_llm: MockLLM) -> None:
    run, slurm, services = setup(runs_root, JOB, mock_llm)
    Engine(run, services).advance()  # analyze is waiting on job 1000
    ArtifactStore(run.artifacts_dir).put(
        "toy_numbers", ToyNumbers(values=[10.0]), producer=Producer(stage="user", agent="manual edit")
    )
    result = Engine(run, services).advance()
    assert slurm.status("1000").state is JobState.CANCELLED
    assert result.outcome is Outcome.WAITING_JOB and result.waiting_jobs == ["1001"]
    slurm.finish_all()
    assert Engine(run, services).advance().outcome is Outcome.DONE
    assert ArtifactStore(run.artifacts_dir).get("toy_stats").mean == pytest.approx(10.0)
