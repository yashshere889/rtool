"""Human edits to an agent's output.

A researcher may change some outputs by hand (e.g. the experiment plan). The
edit becomes a new version of the artifact, made by ``human:<user>``. Before it
is stored, the owning agent's *editor* normalises it: everything derived from
the edited fields is recomputed and checked, so a hand edit cannot leave the
artifact inconsistent, and files written from it (e.g. the Markdown write-up)
are produced again.

Agents register editors here (core never imports agents). The engine applies
edits as ``edit`` actions under the run lease; the web UI calls the same
editor to check a draft before it is sent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .artifacts import ArtifactEnvelope, ArtifactStore
from .config import AppConfig
from .errors import ActionError
from .pipeline import ResolvedPipeline
from .state import GateStatus, StageState, StageStatus


@dataclass
class EditContext:
    stage: str  # the stage that owns (produced) the artifact
    config: Any  # that stage's agent config
    app_config: AppConfig
    store: ArtifactStore  # to read the previous version's inputs; editors never write
    previous: ArtifactEnvelope
    run_root: Path  # to read files the inputs point to (run-relative paths); editors never write


@dataclass
class EditResult:
    value: BaseModel
    files: dict[str, str] = field(default_factory=dict)  # owner workspace-relative path -> text to (re)write
    meta: dict[str, Any] = field(default_factory=dict)


Editor = Callable[[dict[str, Any], EditContext], EditResult]
EDITORS: dict[str, Editor] = {}


def artifact_editor(name: str) -> Callable[[Editor], Editor]:
    """Register the function that turns a researcher's edit of ``name`` into a consistent new version."""

    def decorate(fn: Editor) -> Editor:
        EDITORS[name] = fn
        return fn

    return decorate


def describe(exc: ValidationError, limit: int = 4) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'value'}: {e['msg']}" for e in exc.errors()[:limit])


def owner_of(pipeline: ResolvedPipeline, name: str) -> str | None:
    return next((sid for sid in pipeline.order if name in pipeline.stage(sid).produces), None)


def run_editor(
    pipeline: ResolvedPipeline,
    store: ArtifactStore,
    app_config: AppConfig,
    name: str,
    value: Any,
    base_version: Any,
    run_root: Path,
) -> tuple[str, ArtifactEnvelope, EditResult]:
    """Check an edit of ``name`` made on ``base_version`` and normalise it. Raises ActionError if it is refused."""
    editor = EDITORS.get(name)
    owner = owner_of(pipeline, name)
    if editor is None or owner is None:
        raise ActionError(f"{name or 'this output'} cannot be edited by hand")
    head = store.head(name)
    if head is None:
        raise ActionError(f"there is no {name} to edit yet")
    if base_version != head:
        raise ActionError(f"{name} changed since you started editing (now version {head}); start again from it")
    if not isinstance(value, dict):
        raise ActionError("the edit must contain the whole edited value")
    previous = store.envelope(name, head)
    context = EditContext(owner, pipeline.stage(owner).config, app_config, store, previous, run_root)
    try:
        return owner, previous, EDITORS[name](value, context)
    except ValidationError as exc:
        raise ActionError(f"the edited {name} is not valid: {describe(exc)}") from None


def edit_refusal(stage: str, st: StageState) -> str | None:
    """Why the outputs of ``stage`` cannot be edited now (None if they can)."""
    in_stage_gate = st.gate is not None and st.gate.in_stage and st.gate.status is GateStatus.OPEN
    if st.status in (StageStatus.DONE, StageStatus.AWAITING_GATE) and not in_stage_gate:
        return None
    return (
        f"stage {stage!r} is {st.status.value}; its outputs can be edited once it has finished "
        "or while it waits for your approval"
    )
