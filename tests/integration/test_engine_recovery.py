"""Killing an episode at awkward moments loses no work."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from research_tool.core.artifacts import ArtifactStore
from research_tool.core.engine import Engine, Outcome, load_status
from research_tool.core.errors import LeaseHeld
from research_tool.core.events import EventLog
from research_tool.core.fs import FileLease
from research_tool.core.registry import registry
from research_tool.core.run import RunHandle
from research_tool.core.state import RunState, StageStatus, StateStore
from research_tool.llm.mock_backend import MockLLM
from research_tool.testing.harness import make_run, make_services


def kinds(run: RunHandle) -> list[str]:
    return [e["kind"] for e in EventLog(run.events_path, run.run_id)]


def test_killed_stage_resumes_from_its_checkpoint(runs_root: Path) -> None:
    run = make_run(runs_root, {"name": "x", "stages": [{"id": "crash", "agent": "test_crashy"}]})
    crashy = registry.get("test_crashy")
    before = crashy.runs  # type: ignore[attr-defined]
    with pytest.raises(SystemExit):
        Engine(run, make_services(run)).advance()
    _, stages, _ = load_status(run)
    assert stages["crash"].status is StageStatus.RUNNING  # what a dead episode leaves behind
    result = Engine(run, make_services(run)).advance()
    assert result.outcome is Outcome.DONE
    assert crashy.runs - before == 2  # type: ignore[attr-defined]
    _, stages, _ = load_status(run)
    assert stages["crash"].attempt == 1  # resumed, not restarted
    assert ArtifactStore(run.artifacts_dir).get("test_value").value == 42  # staged output survived
    assert "stage_interrupted" in kinds(run)


def test_lease_left_by_a_dead_episode_expires(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, {"name": "x", "stages": [{"id": "source", "agent": "toy_source"}]})
    assert FileLease(run.lease_path, ttl=1, owner="dead", clock=lambda: 0.0).acquire()
    assert Engine(run, make_services(run, mock_llm)).advance().outcome is Outcome.DONE


def test_a_stage_that_outlives_the_lease_ttl_keeps_the_run(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ttl = 0.3
    stages = {"name": "x", "stages": [{"id": "count", "agent": "test_counter"}]}
    run = make_run(runs_root, stages, overrides={"episode": {"lease_ttl_seconds": ttl}})
    real = Engine._run_stage
    second_episode: list[bool] = []

    def hours_long(self: Engine, sid: str) -> None:
        time.sleep(3 * ttl)  # without renewals during the stage, the lease would have expired twice over
        second_episode.append(FileLease(run.lease_path, ttl=ttl, owner="ep-second").acquire())
        real(self, sid)

    monkeypatch.setattr(Engine, "_run_stage", hours_long)
    assert Engine(run, make_services(run)).advance().outcome is Outcome.DONE
    assert second_episode == [False]
    assert not run.lease_path.exists()  # released, and no late renewal wrote it back


def test_an_episode_that_lost_the_run_leaves_its_state_alone(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = make_run(runs_root, {"name": "x", "stages": [{"id": "count", "agent": "test_counter"}]})
    real = Engine._run_stage

    def stalled_past_the_ttl(self: Engine, sid: str) -> None:
        real(self, sid)
        run.lease_path.unlink()  # meanwhile another episode took the run over
        assert FileLease(run.lease_path, owner="ep-other").acquire()
        StateStore(run.state_dir).save_run(RunState(run_id=run.run_id, nonce="other", active_episode="ep-other"))

    monkeypatch.setattr(Engine, "_run_stage", stalled_past_the_ttl)
    with pytest.raises(LeaseHeld):
        Engine(run, make_services(run)).advance()
    assert StateStore(run.state_dir).load_run().active_episode == "ep-other"  # type: ignore[union-attr]


def test_crash_during_commit_is_finished_without_rerunning(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = make_run(runs_root, {"name": "x", "stages": [{"id": "count", "agent": "test_counter"}]})
    counter = registry.get("test_counter")
    before = counter.runs  # type: ignore[attr-defined]

    def die(self: Engine, sid: str) -> None:
        raise SystemExit("died mid-commit")

    monkeypatch.setattr(Engine, "_finish_commit", die)
    with pytest.raises(SystemExit):
        Engine(run, make_services(run)).advance()
    monkeypatch.undo()
    _, stages, _ = load_status(run)
    assert stages["count"].commit_pending
    assert Engine(run, make_services(run)).advance().outcome is Outcome.DONE
    assert counter.runs - before == 1  # type: ignore[attr-defined]
    assert "commit_recovered" in kinds(run)
    assert ArtifactStore(run.artifacts_dir).versions("test_value") == [1]
