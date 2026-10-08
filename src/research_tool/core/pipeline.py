"""Pipeline specification (YAML) and its resolution into an executable DAG.

A pipeline lists stages; each stage names an agent. Dependencies are inferred
from data flow (a stage depends on whoever produces the artifacts it consumes)
plus any explicit ``needs``. Gates are declared once and attached to a stage;
they open after the stage commits its outputs.

    name: example
    gates:
      pick: {kind: select, options_artifact: hypothesis_set,
             options_path: hypotheses, selection_artifact: selected_hypothesis}
    stages:
      - {id: literature, agent: literature}
      - {id: hypotheses, agent: hypothesis, gate: pick}
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .agent import Agent
from .artifacts import ARTIFACT_TYPES, artifact_model, register_artifact_type
from .errors import ArtifactError, PipelineError, RegistryError
from .registry import AgentRegistry
from .registry import registry as default_registry

_ID = r"^[a-z][a-z0-9_]*$"


class GateKind(StrEnum):
    APPROVE = "approve"
    SELECT = "select"


class GateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: GateKind = GateKind.APPROVE
    prompt: str = ""
    # selection gates only
    options_artifact: str | None = None  # artifact (produced by the stage) holding the options
    options_path: str | None = None  # dotted path to the list inside that artifact's payload
    selection_artifact: str | None = None  # artifact written with the chosen option

    @model_validator(mode="after")
    def _check_select(self) -> GateSpec:
        if self.kind is GateKind.SELECT and not (
            self.options_artifact and self.options_path and self.selection_artifact
        ):
            raise ValueError("select gates need options_artifact, options_path and selection_artifact")
        return self


class StageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=_ID)
    agent: str
    needs: list[str] = Field(default_factory=list)
    gate: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    retries: int = Field(default=0, ge=0)


class PipelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    initial_artifacts: list[str] = Field(default_factory=lambda: ["research_brief"])
    gates: dict[str, GateSpec] = Field(default_factory=dict)
    stages: list[StageSpec]

    @model_validator(mode="after")
    def _check_references(self) -> PipelineSpec:
        ids = [s.id for s in self.stages]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate stage ids: {duplicates}")
        for stage in self.stages:
            if stage.gate and stage.gate not in self.gates:
                raise ValueError(f"stage {stage.id!r} uses undeclared gate {stage.gate!r}")
            unknown = [n for n in stage.needs if n not in ids]
            if unknown:
                raise ValueError(f"stage {stage.id!r} needs unknown stages {unknown}")
        attached = [s.gate for s in self.stages if s.gate]
        if len(attached) != len(set(attached)):
            raise ValueError("a gate can be attached to only one stage")
        return self


def parse_pipeline(data: dict[str, Any], source: str = "<pipeline>") -> PipelineSpec:
    try:
        return PipelineSpec.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise PipelineError(f"invalid pipeline {source}: {problems}") from None


def load_pipeline(path: Path) -> PipelineSpec:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PipelineError(f"pipeline file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise PipelineError(f"invalid YAML in {path}: {exc}") from None
    if not isinstance(data, dict):
        raise PipelineError(f"{path} must contain a mapping")
    return parse_pipeline(data, str(path))


def builtin_pipelines() -> list[str]:
    folder = resources.files("research_tool") / "pipelines"
    return sorted(p.name.removesuffix(".yaml") for p in folder.iterdir() if p.name.endswith(".yaml"))


def find_pipeline(name_or_path: str) -> PipelineSpec:
    """Load a pipeline by file path, or by name from the built-in ``pipelines/`` folder."""
    path = Path(name_or_path).expanduser()
    if path.suffix in {".yaml", ".yml"} or path.exists():
        return load_pipeline(path)
    resource = resources.files("research_tool") / "pipelines" / f"{name_or_path}.yaml"
    if not resource.is_file():
        raise PipelineError(f"no pipeline {name_or_path!r}; built-in pipelines: {builtin_pipelines()}")
    data = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return parse_pipeline(data, f"builtin:{name_or_path}")


def get_by_path(payload: Any, dotted: str) -> Any:
    node = payload
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(dotted)
    return node


@dataclass
class ResolvedStage:
    spec: StageSpec
    agent_cls: type[Agent]
    config: Any
    consumes: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    agent_produces: tuple[str, ...]  # what run() must stage
    produces: tuple[str, ...]  # agent outputs + selection artifact from its gate
    deps: tuple[str, ...]
    gate: GateSpec | None

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def all_inputs(self) -> tuple[str, ...]:
        return self.consumes + self.optional_inputs


class ResolvedPipeline:
    def __init__(self, spec: PipelineSpec, stages: dict[str, ResolvedStage], order: list[str]) -> None:
        self.spec = spec
        self.stages = stages
        self.order = order

    def stage(self, stage_id: str) -> ResolvedStage:
        try:
            return self.stages[stage_id]
        except KeyError:
            raise PipelineError(f"unknown stage {stage_id!r}") from None

    def ancestors(self, stage_id: str) -> set[str]:
        seen: set[str] = set()
        todo = list(self.stage(stage_id).deps)
        while todo:
            sid = todo.pop()
            if sid not in seen:
                seen.add(sid)
                todo.extend(self.stages[sid].deps)
        return seen

    def descendants(self, stage_id: str) -> set[str]:
        return {sid for sid in self.order if stage_id in self.ancestors(sid)}

    def producers_of(self, artifact: str) -> list[str]:
        return [sid for sid in self.order if artifact in self.stages[sid].produces]


def _ensure_selection_type(name: str) -> None:
    from ..schemas.common import SelectedItem

    existing = ARTIFACT_TYPES.get(name)
    if existing is None:
        register_artifact_type(name, SelectedItem)
    elif not issubclass(existing, SelectedItem):
        raise PipelineError(f"selection artifact {name!r} must use SelectedItem (or a subclass)")


def resolve_pipeline(spec: PipelineSpec, registry: AgentRegistry = default_registry) -> ResolvedPipeline:
    """Check the pipeline against the registered agents and compute a topological order."""
    import research_tool.schemas.common  # noqa: F401  (registers research_brief and friends)

    partial: dict[str, dict[str, Any]] = {}
    producers: dict[str, list[str]] = {}
    for stage in spec.stages:
        try:
            cls = registry.get(stage.agent)
        except RegistryError as exc:
            raise PipelineError(f"stage {stage.id!r}: {exc}") from None
        try:
            config = cls.validate_config(stage.config)
        except ValueError as exc:
            raise PipelineError(f"stage {stage.id!r}: {exc}") from None
        gate = spec.gates.get(stage.gate) if stage.gate else None
        produces = list(cls.produces)
        if gate is not None and gate.kind is GateKind.SELECT:
            if gate.options_artifact not in cls.produces:
                raise PipelineError(
                    f"gate {stage.gate!r}: options_artifact {gate.options_artifact!r} "
                    f"is not produced by stage {stage.id!r}"
                )
            _ensure_selection_type(gate.selection_artifact)  # type: ignore[arg-type]
            produces.append(gate.selection_artifact)  # type: ignore[arg-type]
        partial[stage.id] = {"spec": stage, "cls": cls, "config": config, "gate": gate, "produces": produces}
        for name in produces:
            producers.setdefault(name, []).append(stage.id)

    every_artifact = set(spec.initial_artifacts) | set(producers)
    for item in partial.values():
        every_artifact |= set(item["cls"].consumes) | set(item["cls"].optional_inputs)
    for name in sorted(every_artifact):
        try:
            artifact_model(name)
        except ArtifactError as exc:
            raise PipelineError(str(exc)) from None

    deps: dict[str, set[str]] = {}
    for sid, item in partial.items():
        cls = item["cls"]
        stage_deps = set(item["spec"].needs)
        for name in (*cls.consumes, *cls.optional_inputs):
            others = [p for p in producers.get(name, []) if p != sid]
            if not others and name in cls.consumes and name not in spec.initial_artifacts:
                raise PipelineError(f"stage {sid!r} consumes {name!r} but no stage produces it")
            stage_deps.update(others)
        deps[sid] = stage_deps

    # Kahn's algorithm, keeping YAML order among ready stages.
    order: list[str] = []
    remaining = [s.id for s in spec.stages]
    while remaining:
        ready = next((sid for sid in remaining if deps[sid] <= set(order)), None)
        if ready is None:
            raise PipelineError(f"dependency cycle among stages {remaining}")
        order.append(ready)
        remaining.remove(ready)

    stages = {
        sid: ResolvedStage(
            spec=item["spec"],
            agent_cls=item["cls"],
            config=item["config"],
            consumes=tuple(item["cls"].consumes),
            optional_inputs=tuple(item["cls"].optional_inputs),
            agent_produces=tuple(item["cls"].produces),
            produces=tuple(item["produces"]),
            deps=tuple(sorted(deps[sid], key=order.index)),
            gate=item["gate"],
        )
        for sid, item in partial.items()
    }
    return ResolvedPipeline(spec, stages, order)
