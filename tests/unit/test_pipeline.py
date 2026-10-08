from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from research_tool.agents.base import Agent, AgentContext, artifact_type
from research_tool.core.errors import PipelineError
from research_tool.core.pipeline import find_pipeline, parse_pipeline, resolve_pipeline
from research_tool.core.registry import AgentRegistry


def test_builtin_toy_pipeline_resolves_with_inferred_dependencies() -> None:
    resolved = resolve_pipeline(find_pipeline("toy"))
    assert resolved.order == ["source", "analyze", "write"]
    assert resolved.stage("analyze").deps == ("source",)
    assert resolved.stage("write").deps == ("analyze",)
    assert resolved.ancestors("write") == {"source", "analyze"}
    assert resolved.descendants("source") == {"analyze", "write"}


def _spec(stages: list[dict[str, Any]], gates: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": "t", "gates": gates or {}, "stages": stages}


def test_yaml_order_does_not_matter() -> None:
    spec = parse_pipeline(
        _spec(
            [
                {"id": "write", "agent": "toy_writer"},
                {"id": "analyze", "agent": "toy_analyzer"},
                {"id": "source", "agent": "toy_source"},
            ]
        )
    )
    assert resolve_pipeline(spec).order == ["source", "analyze", "write"]


def test_missing_producer_is_reported() -> None:
    spec = parse_pipeline(_spec([{"id": "write", "agent": "toy_writer"}]))
    with pytest.raises(PipelineError, match="consumes 'toy_stats' but no stage produces it"):
        resolve_pipeline(spec)


def test_unknown_agent_and_bad_config() -> None:
    with pytest.raises(PipelineError, match="unknown agent 'nope'"):
        resolve_pipeline(parse_pipeline(_spec([{"id": "a", "agent": "nope"}])))
    bad = parse_pipeline(_spec([{"id": "s", "agent": "toy_source", "config": {"count": "many"}}]))
    with pytest.raises(PipelineError, match="invalid config"):
        resolve_pipeline(bad)
    noconfig = parse_pipeline(_spec([{"id": "o", "agent": "toy_options", "config": {"x": 1}}]))
    with pytest.raises(PipelineError, match="takes no config"):
        resolve_pipeline(noconfig)


def test_spec_level_errors() -> None:
    with pytest.raises(PipelineError, match="duplicate stage ids"):
        parse_pipeline(_spec([{"id": "a", "agent": "toy_source"}, {"id": "a", "agent": "toy_source"}]))
    with pytest.raises(PipelineError, match="undeclared gate"):
        parse_pipeline(_spec([{"id": "a", "agent": "toy_source", "gate": "g"}]))
    with pytest.raises(PipelineError, match="select gates need"):
        parse_pipeline(_spec([{"id": "a", "agent": "toy_source", "gate": "g"}], {"g": {"kind": "select"}}))


def test_select_gate_adds_selection_artifact_and_dependency() -> None:
    spec = parse_pipeline(
        _spec(
            [{"id": "options", "agent": "toy_options", "gate": "pick"}, {"id": "choose", "agent": "toy_chooser"}],
            {
                "pick": {
                    "kind": "select",
                    "options_artifact": "toy_options",
                    "options_path": "options",
                    "selection_artifact": "toy_choice",
                }
            },
        )
    )
    resolved = resolve_pipeline(spec)
    assert "toy_choice" in resolved.stage("options").produces
    assert resolved.stage("choose").deps == ("options",)


def test_cycles_are_rejected() -> None:
    reg = AgentRegistry(builtin_modules=[])

    @artifact_type("cycle_a")
    class CA(BaseModel):
        x: int = 0

    @artifact_type("cycle_b")
    class CB(BaseModel):
        x: int = 0

    @reg.register("ping")
    class Ping(Agent):
        consumes = ("cycle_b",)
        produces = ("cycle_a",)

        def run(self, ctx: AgentContext) -> None: ...

    @reg.register("pong")
    class Pong(Agent):
        consumes = ("cycle_a",)
        produces = ("cycle_b",)

        def run(self, ctx: AgentContext) -> None: ...

    spec = parse_pipeline(_spec([{"id": "a", "agent": "ping"}, {"id": "b", "agent": "pong"}]))
    with pytest.raises(PipelineError, match="cycle"):
        resolve_pipeline(spec, reg)
