"""Killing an episode at awkward moments loses no work."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from research_tool.agents.base import Agent, AgentContext, register_agent
from research_tool.core.artifacts import ArtifactStore
from research_tool.core.engine import AdvanceResult, Engine, Outcome, load_status
from research_tool.core.events import EventLog
from research_tool.core.fs import FileLease
from research_tool.core.registry import registry
from research_tool.core.run import RunHandle
from research_tool.core.state import StageStatus
from research_tool.llm.mock_backend import MockLLM
from research_tool.testing.harness import make_run, make_services

LONG = {"name": "x", "stages": [{"id": "long", "agent": "test_long_stage"}]}


@register_agent("test_long_stage")
class LongStage(Agent):
    """Calls ``work(ctx)`` (set by each test), then produces its output."""

    consumes = ("research_brief",)
    produces = ("test_value",)
    needs_llm = False

    @staticmethod
    def work(ctx: AgentContext) -> None:
        return None

    def run(self, ctx: AgentContext) -> None:
        self.work(ctx)
        ctx.outputs.put("test_value", {"value": 1})


def kinds(run: RunHandle) -> list[str]:
    return [e["kind"] for e in EventLog(run.events_path, run.run_id)]


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


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


def test_a_stage_that_outlives_the_lease_ttl_keeps_the_run(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = make_run(runs_root, LONG)
    now = [1_000.0]
    services = make_services(run)
    services.clock = lambda: now[0]
    ttl = run.config.episode.lease_ttl_seconds
    rivals: list[AdvanceResult] = []

    def hours_of_work(ctx: AgentContext) -> None:
        now[0] += 12 * ttl  # renewed only between stages, the lease would have expired long ago
        lease = FileLease(run.lease_path, ttl=ttl)
        wait_for(lambda: (info := lease.holder()) is not None and info.expires_at > now[0])  # the heartbeat
        rivals.append(Engine(run, services).advance())  # e.g. a second episode job starting now

    monkeypatch.setattr(LongStage, "work", staticmethod(hours_of_work))
    result = Engine(run, services, lease_heartbeat=0.01).advance()
    assert [r.outcome for r in rivals] == [Outcome.BUSY]
    assert result.outcome is Outcome.DONE and result.ran == ["long"]
    assert "stage_interrupted" not in kinds(run)  # nobody recovered (and re-ran) the running stage
    _, stages, _ = load_status(run)
    assert stages["long"].attempt == 1
    assert not run.lease_path.exists()


def test_an_episode_that_lost_its_lease_writes_nothing_more(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stalled for longer than the ttl and taken over, it stops at its next checkpoint."""
    run = make_run(runs_root, LONG)
    now = [1_000.0]
    services = make_services(run)
    services.clock = lambda: now[0]
    ttl = run.config.episode.lease_ttl_seconds
    at_takeover: dict[str, bytes] = {}
    went_on: list[bool] = []

    def stalls(ctx: AgentContext) -> None:
        now[0] += 2 * ttl  # no renewal for longer than the ttl: a hung filesystem, a stopped process
        assert FileLease(run.lease_path, ttl=ttl, owner="ep-rival", clock=lambda: now[0]).acquire()
        at_takeover.update(files(run.state_dir))
        ctx.state["progress"] = 1
        ctx.checkpoint()
        went_on.append(True)

    monkeypatch.setattr(LongStage, "work", staticmethod(stalls))
    result = Engine(run, services, lease_heartbeat=3600).advance()  # no beat during the test
    assert result.outcome is Outcome.BUSY and "ep-rival" in result.message
    assert went_on == []
    assert files(run.state_dir) == at_takeover  # no state written, the rival's lease neither released nor renewed
    assert ArtifactStore(run.artifacts_dir).head("test_value") is None
    assert "lease_lost" in kinds(run)
    _, stages, _ = load_status(run)
    assert stages["long"].status is StageStatus.RUNNING and stages["long"].agent_state == {}  # for the rival
