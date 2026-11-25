"""Human edits to an agent's output.

A researcher may change some outputs by hand (e.g. the experiment plan). The
edit becomes a new version of the artifact, made by ``human:<user>``. Before it
is stored, the owning agent's *editor* normalises it: everything derived from
the edited fields is recomputed and checked, so a hand edit cannot leave the
artifact inconsistent, and files written from it (e.g. the Markdown write-up)
are produced again.

An editor is registered for an artifact *and the agent that owns it* (the same
artifact name can belong to different agents in different pipelines, and an
edit follows that agent's rules). An edit may also change other outputs of the
same stage that follow from it (``EditResult.extra``: a study's design edit
rewrites its protocol too); the engine stores them all as one edit.

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

from .actions import Action
from .artifacts import ArtifactEnvelope, ArtifactStore, Producer
from .config import AppConfig
from .errors import ActionError
from .fs import sha256_text
from .pipeline import ResolvedPipeline
from .state import GateStatus, StageState, StageStatus
from .workspace import Workspace

MAX_EDITED_FILE_BYTES = 1_000_000  # a file a researcher may rewrite (write_file)


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
    extra: dict[str, BaseModel] = field(default_factory=dict)  # other outputs of the same stage the edit changes


Editor = Callable[[dict[str, Any], EditContext], EditResult]
EDITORS: dict[tuple[str, str], Editor] = {}  # (artifact, the agent that owns it) -> editor


def artifact_editor(name: str, *, agent: str) -> Callable[[Editor], Editor]:
    """Register the function that turns a researcher's edit of ``name`` (made by ``agent``) into a new version."""

    def decorate(fn: Editor) -> Editor:
        EDITORS[(name, agent)] = fn
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
    owner = owner_of(pipeline, name)
    editor = EDITORS.get((name, pipeline.stage(owner).spec.agent)) if owner is not None else None
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
        result = editor(value, context)
    except ValidationError as exc:
        raise ActionError(f"the edited {name} is not valid: {describe(exc)}") from None
    for other in result.extra:  # an editor may change the stage's own outputs, never another stage's
        if other == name or other not in pipeline.stage(owner).produces or store.head(other) is None:
            raise ActionError(f"an edit of {name} cannot change {other}")
    return owner, previous, result


def edit_refusal(stage: str, st: StageState) -> str | None:
    """Why the outputs of ``stage`` cannot be edited now (None if they can)."""
    in_stage_gate = st.gate is not None and st.gate.in_stage and st.gate.status is GateStatus.OPEN
    if st.status in (StageStatus.DONE, StageStatus.AWAITING_GATE) and not in_stage_gate:
        return None
    return (
        f"stage {stage!r} is {st.status.value}; its outputs can be edited once it has finished "
        "or while it waits for your approval"
    )


class EditActions:
    """How the engine applies ``write_file`` and ``edit`` actions (mixed into ``Engine``, whose state it uses)."""

    def _write_file(self, action: Action) -> dict[str, Any]:
        """Rewrite a file the stage's agent lets researchers edit (e.g. the Coder's experiment.py).

        Allowed while the stage waits at its own checkpoint or after it failed. With ``retest`` the stage
        then continues: the checkpoint is answered with "test it again", a failed stage resumes.
        """
        sid = action.stage
        if sid is None or sid not in self.stages:
            raise ActionError("write_file needs a valid --stage")
        rs = self.pipeline.stage(sid)
        st = self.stages[sid]
        payload = action.payload
        path = str(payload.get("path") or "")
        if path not in rs.agent_cls.editable_files:
            allowed = ", ".join(rs.agent_cls.editable_files) or "no file"
            raise ActionError(f"in stage {sid!r} you can edit {allowed}, not {path!r}")
        at_gate = (
            st.status is StageStatus.AWAITING_GATE
            and st.gate is not None
            and st.gate.in_stage
            and st.gate.status is GateStatus.OPEN
        )
        if not (at_gate or st.status is StageStatus.FAILED):
            raise ActionError(
                f"{path} can be edited while stage {sid!r} waits for your decision or after it failed "
                f"(it is {st.status.value})"
            )
        content = payload.get("content")
        if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_EDITED_FILE_BYTES:
            raise ActionError(f"{path} must be text of at most {MAX_EDITED_FILE_BYTES // 1_000_000} MB")
        workspace = Workspace(self.run.work_dir(sid))
        target = workspace.path(path)
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        changed = content != current
        base = payload.get("base_sha")
        if changed and base is not None and sha256_text(current) != base:  # already holding it: nothing to refuse
            raise ActionError(f"{path} changed since you opened it; reload it and edit again")
        if changed:
            workspace.write_text(path, content)
            self._save(st)  # records this decision with the file (see StageState.last_action)
            self.events.emit("file_written", stage=sid, path=path, by=action.by, sha=sha256_text(content))
        outcome: dict[str, Any] = {"stage_id": sid, "path": path, "changed": changed}
        if payload.get("retest"):
            if at_gate:
                outcome.update(self._resolve_in_stage_gate(sid, False, "", {"retest": True}, action.by))
            else:  # failed: resume where it stopped (the agent re-tests what it finds)
                st.status = StageStatus.PENDING
                st.resume = True
                st.failures = 0
                st.last_error = None
                self._save(st)
                outcome["resumed"] = True
        return outcome

    def _edit(self, action: Action) -> dict[str, Any]:
        """Store a hand-edited output as a new version (normalised by the owning agent's editor).

        The owning stage keeps its state (an open gate stays open, now showing the edited version); later
        stages that used the old version become stale.
        """
        payload = action.payload
        name = str(payload.get("artifact") or "")
        value, base = payload.get("value"), payload.get("base_version")
        owner, previous, result = run_editor(self.pipeline, self.store, self.config, name, value, base, self.run.root)
        st = self.stages[owner]
        refusal = edit_refusal(owner, st)
        if refusal:
            raise ActionError(refusal)
        note = str(payload.get("note") or "").strip()
        rs = self.pipeline.stage(owner)
        stored: list[tuple[str, int]] = []  # the artifacts this edit made a new version of
        for artifact, edited in [(name, result.value), *result.extra.items()]:
            before = previous if artifact == name else self.store.envelope(artifact, self.store.head(artifact))
            meta = {**before.meta, **(result.meta if artifact == name else {})}
            meta |= {"edited_from": before.version, "edited_by": action.by}
            if note:
                meta["edit_note"] = note
            version = self.store.put(
                artifact,
                edited,
                producer=Producer(stage=owner, agent=f"human:{action.by}", attempt=st.attempt),  # no token: new version
                inputs=dict(before.inputs),
                meta=meta,
            )
            if version == before.version:
                continue
            st.output_versions[artifact] = version
            if artifact in rs.all_inputs:
                st.input_versions[artifact] = version
            if (
                st.gate is not None
                and st.gate.status is GateStatus.OPEN
                and artifact in st.gate.summary.get("artifacts", {})
            ):
                st.gate.summary["artifacts"][artifact] = version  # the checkpoint now asks about the edited version
            stored.append((artifact, version))
        if not stored:
            return {"stage_id": owner, "artifact": name, "version": previous.version, "unchanged": True}
        workspace = Workspace(self.run.work_dir(owner))
        for path, text in result.files.items():
            workspace.write_text(path, text)
        gate_artifacts = st.gate.summary.get("artifacts", {}) if st.gate is not None else {}
        if st.gate is not None and st.gate.status is GateStatus.OPEN and any(a in gate_artifacts for a, _ in stored):
            blocking = self._blocking(rs, st)  # what keeps the checkpoint from being approved, for the edited versions
            if blocking:
                st.gate.summary["blocking"] = blocking
            else:
                st.gate.summary.pop("blocking", None)
        self._save(st)
        for artifact, version in stored:
            self.events.emit("artifact_committed", stage=owner, artifact=artifact, version=version)
        before = {sid: self.stages[sid].status for sid in self.pipeline.order}
        self._propagate_staleness(owner)
        stale = [
            s
            for s in self.pipeline.order
            if self.stages[s].status is StageStatus.STALE and before[s] is not StageStatus.STALE
        ]
        first, version = next((a for a in stored if a[0] == name), stored[0])
        also = [a for a, _ in stored if a != first]
        self.events.emit(
            "artifact_edited", stage=owner, artifact=first, version=version, by=action.by, stale=stale, also=also
        )
        return {"stage_id": owner, "artifact": first, "version": version, "stale": stale, "also": also}
