"""Editors are found by the artifact and the agent that owns it, and may change only their own stage's outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_tool.core.artifacts import ArtifactStore
from research_tool.core.config import load_config
from research_tool.core.editing import EDITORS, EditResult, run_editor
from research_tool.core.errors import ActionError
from research_tool.core.pipeline import find_pipeline, resolve_pipeline
from research_tool.testing.demo_run import build_demo_run


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    run = build_demo_run(tmp_path_factory.mktemp("runs"), "protocol_review", pipeline="study")
    return run, ArtifactStore(run.artifacts_dir), resolve_pipeline(find_pipeline("study"))


def edit(study, name: str, value: dict) -> EditResult:  # type: ignore[no-untyped-def]
    run, store, pipeline = study
    config = load_config(overrides={}, env={})
    return run_editor(pipeline, store, config, name, value, store.head(name), run.root)[2]


def test_the_same_artifact_goes_to_the_editor_of_its_own_agent(study) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = study
    plan = store.get("experiment_plan").model_dump(mode="json")
    result = edit(study, "experiment_plan", plan)  # produced by the protocol agent: the study design editor
    assert set(result.extra) == {"protocol"}
    assert ("experiment_plan", "planner") in EDITORS and ("experiment_plan", "protocol") in EDITORS
    with pytest.raises(ActionError, match="cannot be edited"):
        edit(study, "corpus", {})  # no editor is registered for the literature agent's corpus


def test_an_editor_cannot_change_another_stages_output(study, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    _, store, _ = study
    protocol = store.get("protocol")

    def meddle(value, ctx):  # type: ignore[no-untyped-def]
        return EditResult(protocol, extra={"corpus": store.get("corpus")})  # the literature stage's output

    monkeypatch.setitem(EDITORS, ("protocol", "protocol"), meddle)
    with pytest.raises(ActionError, match="cannot change corpus"):
        edit(study, "protocol", protocol.model_dump(mode="json"))

    def itself(value, ctx):  # type: ignore[no-untyped-def]
        return EditResult(protocol, extra={"protocol": protocol})

    monkeypatch.setitem(EDITORS, ("protocol", "protocol"), itself)
    with pytest.raises(ActionError, match="cannot change protocol"):
        edit(study, "protocol", protocol.model_dump(mode="json"))


def test_a_registered_editor_needs_its_agent_named() -> None:
    from research_tool.core.editing import artifact_editor

    with pytest.raises(TypeError):
        artifact_editor("x")  # type: ignore[call-arg]
    assert Path(__file__).exists()


def test_an_edit_that_changes_only_the_extra_artifact_is_stored(tmp_path: Path) -> None:
    """The protocol is unchanged but the plan's problems list was stale: the edit stores the plan alone."""
    from research_tool.core.actions import ActionKind
    from research_tool.core.artifacts import Producer
    from research_tool.service import run_status
    from research_tool.testing.demo_run import advance, demo_services
    from research_tool.testing.harness import queue

    run = build_demo_run(tmp_path, "protocol_review", pipeline="study")
    store = ArtifactStore(run.artifacts_dir)
    stale = store.get("experiment_plan")
    stale.issues = ["protocol checklist: blinding is stated"]  # as left behind by an edit made before editors did this
    store.put("experiment_plan", stale, producer=Producer(stage="protocol", agent="test"))
    assert store.head("experiment_plan") == 2 and store.head("protocol") == 1
    protocol = store.get("protocol").model_dump(mode="json")
    queue(
        run, ActionKind.EDIT, stage="protocol", payload={"artifact": "protocol", "base_version": 1, "value": protocol}
    )
    advance(run, demo_services(run))
    assert store.head("protocol") == 1  # nothing in it changed ...
    assert store.head("experiment_plan") == 3 and store.get("experiment_plan").issues == []  # ... the plan's list did
    gate = run_status(run).open_gates[0]
    assert gate.summary["artifacts"] == {"experiment_plan": 3, "protocol": 1}
