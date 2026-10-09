"""Linear runs, failures/retries, gates (approve, reject, select, auto) and the action inbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.core.actions import Action, ActionInbox, ActionKind
from research_tool.core.artifacts import ArtifactStore
from research_tool.core.engine import Engine, Outcome, load_status
from research_tool.core.events import EventLog
from research_tool.core.fs import FileLease
from research_tool.core.run import RunHandle
from research_tool.core.state import StageStatus
from research_tool.llm.mock_backend import MockLLM
from research_tool.testing.harness import drive, make_run, make_services, queue

LINEAR = {
    "name": "linear",
    "stages": [
        {"id": "source", "agent": "toy_source"},
        {"id": "analyze", "agent": "toy_analyzer", "config": {"use_job": False}},
        {"id": "write", "agent": "toy_writer"},
    ],
}
GATED = {
    "name": "gated",
    "gates": {"check": {"prompt": "look at the numbers"}},
    "stages": [
        {"id": "source", "agent": "toy_source", "gate": "check"},
        {"id": "analyze", "agent": "toy_analyzer", "config": {"use_job": False}},
    ],
}
SELECT = {
    "name": "select",
    "gates": {
        "pick": {
            "kind": "select",
            "options_artifact": "toy_options",
            "options_path": "options",
            "selection_artifact": "toy_choice",
        }
    },
    "stages": [{"id": "options", "agent": "toy_options", "gate": "pick"}, {"id": "choose", "agent": "toy_chooser"}],
}


def kinds(run: RunHandle) -> list[str]:
    return [e["kind"] for e in EventLog(run.events_path, run.run_id)]


def processed_outcomes(run: RunHandle) -> list[dict]:
    return [p["outcome"] for p in ActionInbox(run.actions_dir).processed()]


def test_linear_pipeline_runs_to_completion(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, LINEAR)
    services = make_services(run, mock_llm)
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.DONE
    assert result.ran == ["source", "analyze", "write"]
    store = ArtifactStore(run.artifacts_dir)
    assert store.get("toy_stats").mean == pytest.approx(2.5)
    envelope = store.envelope("toy_stats")
    assert envelope.inputs == {"toy_numbers": 1}
    assert envelope.producer.stage == "analyze" and envelope.producer.agent == "toy_analyzer"
    run_state, stages, _ = load_status(run)
    assert all(s.status is StageStatus.DONE for s in stages.values())
    assert run_state is not None and run_state.usage["llm_calls"] == 2
    assert kinds(run).count("stage_finished") == 3
    again = Engine(run, services).advance()  # nothing left to do
    assert again.outcome is Outcome.DONE and again.ran == []


def test_busy_when_another_episode_holds_the_run(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, LINEAR)
    with FileLease(run.lease_path, owner="someone-else"):
        assert Engine(run, make_services(run, mock_llm)).advance().outcome is Outcome.BUSY


def test_missing_outputs_fail_the_stage(runs_root: Path) -> None:
    run = make_run(runs_root, {"name": "x", "stages": [{"id": "lazy", "agent": "test_lazy"}]})
    result = Engine(run, make_services(run)).advance()
    assert result.outcome is Outcome.NEEDS_HUMAN
    assert "without producing ['test_value']" in result.failed[0]["error"]


def test_automatic_retries(runs_root: Path) -> None:
    spec = {"name": "x", "stages": [{"id": "flaky", "agent": "toy_flaky", "retries": 2, "config": {"fail_times": 2}}]}
    run = make_run(runs_root, spec)
    assert Engine(run, make_services(run)).advance().outcome is Outcome.DONE
    _, stages, _ = load_status(run)
    assert stages["flaky"].attempt == 3
    assert kinds(run).count("stage_retrying") == 2


def test_human_retry_of_failed_stage(runs_root: Path) -> None:
    run = make_run(runs_root, {"name": "x", "stages": [{"id": "flaky", "agent": "toy_flaky"}]})
    services = make_services(run)
    first = Engine(run, services).advance()
    assert first.outcome is Outcome.NEEDS_HUMAN
    assert "simulated failure on attempt 1" in first.failed[0]["error"]
    queue(run, ActionKind.RETRY)
    assert Engine(run, services).advance().outcome is Outcome.DONE


def test_gate_blocks_until_approved(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, GATED)
    services = make_services(run, mock_llm)
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.WAITING_GATE and result.ran == ["source"]
    gate = result.open_gates[0]
    assert gate["gate"] == "check" and gate["summary"]["artifacts"] == {"toy_numbers": 1}
    queue(run, ActionKind.APPROVE, payload={"note": "fine"})
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.DONE and result.ran == ["analyze"]


def test_rejection_reruns_stage_with_feedback(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, GATED)
    services = make_services(run, mock_llm)
    Engine(run, services).advance()
    queue(run, ActionKind.REJECT, payload={"feedback": "use more numbers"})
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.WAITING_GATE and result.ran == ["source"]
    store = ArtifactStore(run.artifacts_dir)
    assert store.get("toy_numbers").notes == ["use more numbers"]
    assert store.head("toy_numbers") == 2
    _, stages, _ = load_status(run)
    assert stages["source"].attempt == 2
    assert stages["source"].feedback[0].consumed_by_attempt == 2


def test_note_reaches_the_next_attempt(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, GATED)
    services = make_services(run, mock_llm)
    Engine(run, services).advance()
    queue(run, ActionKind.NOTE, stage="source", payload={"text": "prefer integers"})
    queue(run, ActionKind.REJECT, payload={"feedback": "again"})
    Engine(run, services).advance()
    assert ArtifactStore(run.artifacts_dir).get("toy_numbers").notes == ["prefer integers", "again"]


def test_auto_gates_run_unattended(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, "toy", overrides={"gates": {"default": "auto"}})
    result = drive(run, make_services(run, mock_llm))
    assert result.outcome is Outcome.DONE
    assert ArtifactStore(run.artifacts_dir).get("toy_report").mentions_variance is True


def test_bad_actions_are_rejected_without_stopping_the_run(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, GATED)
    queue(run, ActionKind.APPROVE)  # nothing is open yet
    result = Engine(run, make_services(run, mock_llm)).advance()
    assert result.outcome is Outcome.WAITING_GATE
    assert "action_rejected" in kinds(run)
    assert processed_outcomes(run)[0]["status"] == "rejected"


def test_duplicate_actions_apply_once(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, GATED)
    services = make_services(run, mock_llm)
    Engine(run, services).advance()
    inbox = ActionInbox(run.actions_dir)
    action = Action(kind=ActionKind.APPROVE)
    inbox.submit(action)
    inbox.submit(action)
    assert Engine(run, services).advance().outcome is Outcome.DONE
    assert [o["status"] for o in processed_outcomes(run)] == ["applied", "duplicate"]


def test_selection_gate(runs_root: Path) -> None:
    run = make_run(runs_root, SELECT, topic="T")
    services = make_services(run)
    result = Engine(run, services).advance()
    options = result.open_gates[0]["summary"]["options"]
    assert [o["index"] for o in options] == [0, 1, 2]
    assert options[1]["title"] == "T — option 1"
    queue(run, ActionKind.APPROVE)  # a selection gate needs a choice
    queue(run, ActionKind.SELECT, payload={"index": 7})  # out of range
    queue(run, ActionKind.SELECT, payload={"index": 1, "edit": {"detail": "tweaked"}})
    result = Engine(run, services).advance()
    assert result.outcome is Outcome.DONE
    assert [o["status"] for o in processed_outcomes(run)] == ["rejected", "rejected", "applied"]
    store = ArtifactStore(run.artifacts_dir)
    choice = store.get("toy_choice")
    assert choice.index == 1 and choice.edited and choice.item["detail"] == "tweaked"
    assert store.envelope("toy_choice").producer.stage == "options"
    assert store.get("toy_decision").chosen_title == "T — option 1"


def test_auto_selection_picks_the_first_option(runs_root: Path) -> None:
    run = make_run(runs_root, SELECT, topic="T", overrides={"gates": {"modes": {"pick": "auto"}}})
    assert Engine(run, make_services(run)).advance().outcome is Outcome.DONE
    assert ArtifactStore(run.artifacts_dir).get("toy_choice").index == 0


def test_several_open_gates_need_disambiguation(runs_root: Path, mock_llm: MockLLM) -> None:
    spec = {
        "name": "parallel",
        "gates": {"a": {}, "b": {}},
        "stages": [
            {"id": "source", "agent": "toy_source", "gate": "a"},
            {"id": "options", "agent": "toy_options", "gate": "b"},
        ],
    }
    run = make_run(runs_root, spec)
    services = make_services(run, mock_llm)
    assert len(Engine(run, services).advance().open_gates) == 2
    queue(run, ActionKind.APPROVE)
    queue(run, ActionKind.APPROVE, gate="b")
    result = Engine(run, services).advance()
    assert [g["gate"] for g in result.open_gates] == ["a"]
    assert "several gates are open" in processed_outcomes(run)[0]["error"]


def test_a_decision_names_the_gate_it_was_made_at(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, GATED)
    services = make_services(run, mock_llm)
    assert Engine(run, services).advance().outcome is Outcome.WAITING_GATE
    _, stages, _ = load_status(run)
    gate = stages["source"].gate
    assert gate is not None and len(gate.nonce) == 12
    seen = {"gate_nonce": gate.nonce, "artifacts": gate.summary["artifacts"]}

    queue(run, ActionKind.APPROVE, stage="source", gate="check", payload={"expect": {"gate_nonce": "0" * 12}})
    stale = {"gate_nonce": gate.nonce, "artifacts": {"toy_numbers": 99}}
    queue(run, ActionKind.APPROVE, stage="source", gate="check", payload={"expect": stale})
    assert Engine(run, services).advance().outcome is Outcome.WAITING_GATE  # both refused: the gate stays open
    errors = [o.get("error", "") for o in processed_outcomes(run)]
    assert "opened again" in errors[0] and "toy_numbers (v99 -> v1)" in errors[1]

    queue(run, ActionKind.APPROVE, stage="source", gate="check", payload={"expect": seen})
    assert Engine(run, services).advance().outcome is Outcome.DONE
