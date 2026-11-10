"""The run service shared by the CLI and the web UI: create, list, inspect, decide, stop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from research_tool.cli.main import app
from research_tool.core import actions
from research_tool.core.actions import Action, ActionInbox, ActionKind
from research_tool.core.engine import AdvanceResult, Engine, Outcome
from research_tool.core.errors import RToolError
from research_tool.core.events import EventLog
from research_tool.core.services import build_slurm
from research_tool.core.state import StageStatus
from research_tool.hpc.episode import EpisodeTracker, submit_episode
from research_tool.hpc.slurm import JobState
from research_tool.llm.mock_backend import MockLLM
from research_tool.service import (
    create_run,
    list_runs,
    load_app_config,
    open_run,
    queue_action,
    run_status,
    stop_run,
)
from research_tool.testing.harness import make_services


def new_run(runs_root: Path, project: str = "sleep", run_id: str = "r1", pipeline: str = "toy") -> Any:
    config = load_app_config(runs_root=runs_root)
    return create_run(config, project, "Sleep spindles and memory", pipeline=pipeline, run_id=run_id)


def test_create_list_and_inspect_runs(runs_root: Path, mock_llm: MockLLM) -> None:
    run = new_run(runs_root)
    new_run(runs_root, project="ocean", run_id="r2")
    broken = runs_root / "sleep" / "broken"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text("{not json")

    listed = {(s.project, s.run_id): s for s in list_runs(runs_root)}
    assert set(listed) == {("sleep", "r1"), ("ocean", "r2"), ("sleep", "broken")}
    assert listed[("sleep", "broken")].error  # listed with its error, not hidden
    first = listed[("sleep", "r1")]
    assert first.topic == "Sleep spindles and memory" and first.pipeline == "toy" and first.last_outcome == "new"
    assert [s.id for s in first.stages] == ["source", "analyze", "write"]

    Engine(run, make_services(run, mock_llm)).advance()
    status = run_status(open_run(runs_root, "sleep", "r1"))
    gate = status.open_gates[0]
    assert (gate.stage, gate.gate, gate.kind, gate.actions) == (
        "source",
        "numbers_review",
        "approve",
        ["approve", "reject"],
    )
    assert gate.nonce and gate.summary["artifacts"] == {"toy_numbers": 1}
    assert status.stages[0].status is StageStatus.AWAITING_GATE and status.last_outcome == "waiting_gate"


@pytest.mark.parametrize("project,run_id", [("..", "r1"), ("sleep", "../r1"), ("/etc", "r1"), ("sleep", "~")])
def test_open_run_refuses_paths(runs_root: Path, project: str, run_id: str) -> None:
    new_run(runs_root)
    with pytest.raises(RToolError):
        open_run(runs_root, project, run_id)


def test_queued_decisions_are_applied_by_the_next_episode(runs_root: Path, mock_llm: MockLLM) -> None:
    run = new_run(runs_root)
    services = make_services(run, mock_llm)
    Engine(run, services).advance()
    gate = run_status(run).open_gates[0]
    action = Action(
        kind=ActionKind.APPROVE, stage=gate.stage, gate=gate.gate, payload={"expect": {"gate_nonce": gate.nonce}}
    )
    result = queue_action(run, action)
    assert result is not None and not result.submitted  # fake Slurm: nothing to submit
    assert [a.id for a in run_status(run).queued_actions] == [action.id]
    Engine(run, services).advance()
    assert run_status(run).queued_actions == []
    assert ActionInbox(run.actions_dir).processed()[0]["outcome"]["status"] == "applied"


def test_status_reads_never_move_or_trip_over_actions(
    runs_root: Path, mock_llm: MockLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = new_run(runs_root)
    inbox = ActionInbox(run.actions_dir)
    applied = inbox.submit(Action(kind=ActionKind.NOTE, stage="source", payload={"text": "applied meanwhile"}))
    kept = Action(kind=ActionKind.NOTE, stage="source", payload={"text": "still queued"})
    inbox.submit(kept)
    newer = run.actions_dir / "00000000000000000000-0123456789ab.json"  # queued by a newer rtool than this one
    newer.write_text('{"kind": "a_kind_this_version_does_not_know"}', encoding="utf-8")
    read_json = actions.read_json

    def applied_after_the_listing(path: Path) -> Any:
        if path == applied:
            path.unlink()  # an episode applied it between the folder listing and this read
        return read_json(path)

    with monkeypatch.context() as patch:
        patch.setattr(actions, "read_json", applied_after_the_listing)
        assert [a.id for a in run_status(run).queued_actions] == [kept.id]
    assert newer.exists()  # left for the engine
    Engine(run, make_services(run, mock_llm)).advance()
    assert (inbox.invalid_dir / newer.name).exists()


def test_stop_cancels_the_episode_job(runs_root: Path) -> None:
    run = new_run(runs_root)
    slurm = build_slurm(run)
    job_id = submit_episode(run, slurm, needs_llm=False, reason="test")
    assert job_id is not None
    result = stop_run(run)
    assert result.cancelled == [job_id]
    assert build_slurm(run).status(job_id).state is JobState.CANCELLED
    assert EpisodeTracker(run, build_slurm(run)).current() is None
    assert "stop_requested" in [e["kind"] for e in EventLog(run.events_path, run.run_id)]


def test_list_and_stop_through_the_cli(runs_root: Path) -> None:
    run = new_run(runs_root)
    runner = CliRunner()
    out = runner.invoke(app, ["list", "--runs-root", str(runs_root)]).output
    assert "sleep/r1" in out and "Sleep spindles" in out and "source: pending" in out
    out = runner.invoke(app, ["stop", str(run.root)]).output
    assert "none were queued or running" in out


def test_an_episode_wakes_the_next_one_for_late_decisions(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A decision queued after the episode's last look at the inbox must not wait for the next human kick."""
    import research_tool.cli.main as cli

    run = new_run(runs_root)

    def episode_that_misses_a_decision(run_handle: Any, **_kwargs: Any) -> AdvanceResult:
        ActionInbox(run_handle.actions_dir).submit(Action(kind=ActionKind.APPROVE))  # arrives mid-episode
        return AdvanceResult(Outcome.WAITING_GATE, message="waiting")

    monkeypatch.setattr(cli, "run_episode", episode_that_misses_a_decision)
    result = CliRunner().invoke(cli.app, ["episode", str(run.root)])
    assert result.exit_code == 0, result.output
    record = EpisodeTracker(run, build_slurm(run)).current()
    assert record is not None and record["reason"] == "pending_actions"
