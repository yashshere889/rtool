"""Change requests: upstream re-runs, bounded rounds, the revision-limit gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore
from research_tool.core.engine import Engine, Outcome, load_status
from research_tool.core.events import EventLog
from research_tool.core.state import FeedbackKind
from research_tool.llm.mock_backend import MockLLM
from research_tool.testing.harness import make_run, make_services, queue

REVISE = {
    "name": "revise",
    "stages": [
        {"id": "source", "agent": "toy_source"},
        {"id": "analyze", "agent": "toy_analyzer", "config": {"use_job": False}},
        {"id": "write", "agent": "toy_writer", "config": {"require_variance": True}},
    ],
}


def test_change_request_reruns_upstream_then_requester(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, REVISE)
    result = Engine(run, make_services(run, mock_llm)).advance()
    assert result.outcome is Outcome.DONE
    assert result.ran == ["source", "analyze", "write", "analyze", "write"]
    store = ArtifactStore(run.artifacts_dir)
    assert store.get("toy_stats").variance == pytest.approx(1.25)
    assert store.get("toy_report").mentions_variance
    _, stages, _ = load_status(run)
    assert stages["source"].attempt == 1  # untouched
    assert stages["analyze"].revision_count == 1
    feedback = stages["analyze"].feedback[0]
    assert feedback.kind is FeedbackKind.CHANGE_REQUEST and feedback.from_stage == "write"
    assert feedback.consumed_by_attempt == 2


def test_revision_limit_opens_a_decision_gate(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, REVISE, overrides={"revisions": {"max_rounds": 0}})
    services = make_services(run, mock_llm)
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.WAITING_GATE
    gate = result.open_gates[0]
    assert (gate["gate"], gate["stage"]) == ("revision_limit", "write")
    assert gate["summary"]["requests"][0]["target_stage"] == "analyze"
    queue(run, ActionKind.APPROVE)  # accept the current outputs
    assert Engine(run, services).advance().outcome is Outcome.DONE
    assert not ArtifactStore(run.artifacts_dir).get("toy_report").mentions_variance


def test_rejecting_the_revision_limit_forces_one_more_round(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, REVISE, overrides={"revisions": {"max_rounds": 0}})
    services = make_services(run, mock_llm)
    Engine(run, services).advance()
    queue(run, ActionKind.REJECT, payload={"feedback": "we really need it"})
    assert Engine(run, services).advance().outcome is Outcome.DONE
    assert ArtifactStore(run.artifacts_dir).get("toy_report").mentions_variance
    _, stages, _ = load_status(run)
    assert "(human: we really need it)" in stages["analyze"].feedback[0].text


def test_revision_limit_in_auto_mode_accepts_outputs(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, REVISE, overrides={"revisions": {"max_rounds": 0}, "gates": {"default": "auto"}})
    assert Engine(run, make_services(run, mock_llm)).advance().outcome is Outcome.DONE


def test_requester_waits_while_a_revised_grandparent_is_paused(runs_root: Path, mock_llm: MockLLM) -> None:
    spec = {
        "name": "grandparent",
        "stages": [
            {"id": "source", "agent": "toy_source"},
            {"id": "analyze", "agent": "toy_analyzer", "config": {"use_job": False, "gate": "gpu_runs"}},
            {"id": "write", "agent": "toy_writer"},
            {"id": "critic", "agent": "test_report_critic"},
        ],
    }
    run = make_run(runs_root, spec)
    services = make_services(run, mock_llm)
    Engine(run, services).advance()  # analyze pauses at its in-stage gate
    queue(run, ActionKind.APPROVE)
    result = Engine(run, services).advance()
    # The critic asks 'analyze' (two levels up) for a revision. analyze re-runs and pauses at
    # its gate *before* committing anything, so 'write' still looks done — but the critic must
    # not run again until everything upstream of it has settled.
    assert result.ran == ["analyze", "write", "critic", "analyze"]
    assert result.outcome is Outcome.WAITING_GATE and result.open_gates[0]["gate"] == "gpu_runs"
    queue(run, ActionKind.APPROVE)
    result = Engine(run, services).advance()
    assert result.ran == ["analyze", "write", "critic"] and result.outcome is Outcome.DONE
    store = ArtifactStore(run.artifacts_dir)
    assert store.get("toy_stats").variance == pytest.approx(1.25)
    assert store.get("test_value").value == 2


def test_requests_to_stages_that_are_not_upstream_are_ignored(runs_root: Path) -> None:
    spec = {
        "name": "x",
        "stages": [
            {"id": "sibling", "agent": "toy_options"},
            {"id": "count", "agent": "test_counter"},
            {"id": "bad", "agent": "test_bad_requester"},
        ],
    }
    run = make_run(runs_root, spec)
    assert Engine(run, make_services(run)).advance().outcome is Outcome.DONE
    assert "change_request_ignored" in [e["kind"] for e in EventLog(run.events_path, run.run_id)]
