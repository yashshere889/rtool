"""Killing an episode at awkward moments loses no work."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.core.artifacts import ArtifactStore
from research_tool.core.engine import Engine, Outcome, load_status
from research_tool.core.events import EventLog
from research_tool.core.fs import FileLease
from research_tool.core.registry import registry
from research_tool.core.run import RunHandle
from research_tool.core.state import StageStatus
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
