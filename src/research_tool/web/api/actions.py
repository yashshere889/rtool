"""Decisions: queue one for the next episode, see what became of it, take it back while it is queued."""

from __future__ import annotations

import getpass
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ...core.actions import Action, ActionInbox, ActionKind
from ...core.artifacts import ArtifactStore
from ...core.engine import MAX_EDITED_FILE_BYTES, edit_option
from ...core.errors import RToolError
from ...core.fs import sha256_text
from ...core.pipeline import get_by_path, resolve_pipeline
from ...core.run import RunHandle
from ...core.state import StageStatus
from ...service import Affected, KickResult, OpenGate, RunStatus, check_edit, downstream_impact, run_status
from ..deps import get_run, runner_of
from .drafts import Draft, file_draft_key, read_draft, remove_draft, write_draft

router = APIRouter(tags=["actions"])

GATE_DECISIONS = {"approve", "reject", "select"}


class ActionRequest(BaseModel):
    id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$", description="Client-chosen id (idempotency)")
    kind: Literal["approve", "reject", "select", "retry", "note", "reopen", "edit", "write_file"]
    stage: str | None = None
    gate: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Queued(BaseModel):
    action: Action
    kick: KickResult


class ActionState(BaseModel):
    state: Literal["queued", "applied", "rejected", "duplicate", "invalid", "unknown"]
    error: str | None = None
    outcome: dict[str, Any] | None = None


def _user() -> str:
    try:
        return f"{getpass.getuser()} (web)"
    except Exception:  # noqa: BLE001
        return "web"


def check_choice_edit(run: RunHandle, gate: OpenGate, index: int, edit: Any) -> None:
    """An edited choice must still be a valid option (the engine checks again when it applies it)."""
    spec = run.spec.pipeline.gates.get(gate.gate)
    if spec is None or not spec.options_artifact or not spec.options_path:
        return
    version = (gate.summary.get("artifacts") or {}).get(spec.options_artifact)
    try:
        payload = ArtifactStore(run.artifacts_dir).envelope(spec.options_artifact, version).payload
        edit_option(spec, get_by_path(payload, spec.options_path)[index], edit)
    except RToolError as exc:
        raise HTTPException(422, str(exc)) from None


def check_write(run: RunHandle, status: RunStatus, stage_id: str | None, payload: dict[str, Any]) -> None:
    """A rewrite of a file the stage allows editing, at the right moment, of the version the page showed."""
    stage = next((s for s in status.stages if s.id == stage_id), None)
    if stage is None:
        raise HTTPException(422, "say which stage the file belongs to")
    path = str(payload.get("path") or "")
    if path not in resolve_pipeline(run.spec.pipeline).stage(stage.id).agent_cls.editable_files:
        raise HTTPException(422, f"{path or 'this file'} cannot be edited")
    at_gate = stage.gate is not None and stage.gate.in_stage
    if not (at_gate or stage.status is StageStatus.FAILED):
        raise HTTPException(409, "the file can be edited while the stage waits for your decision or after it failed")
    content = payload.get("content")
    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_EDITED_FILE_BYTES:
        raise HTTPException(422, "the file must be text of at most 1 MB")
    if any(a.kind is ActionKind.WRITE_FILE and a.stage == stage.id for a in status.queued_actions):
        raise HTTPException(409, "a change to this file is already on its way; undo it first to change it again")
    if payload.get("retest") and at_gate and any(a.stage == stage.id for a in status.queued_actions):
        raise HTTPException(409, "a decision for this checkpoint is already queued; undo it first")
    target = run.work_dir(stage.id) / path
    current = target.read_text(encoding="utf-8") if target.is_file() else ""
    if payload.get("base_sha") is not None and sha256_text(current) != payload["base_sha"]:
        raise HTTPException(409, "the file changed since you opened it; reload it (your text is kept) and save again")


def check(status: RunStatus, body: ActionRequest, run: RunHandle) -> dict[str, Any]:
    """Refuse decisions the engine would refuse, now rather than in the next episode. Returns the payload."""
    payload = dict(body.payload)
    queued = [a for a in status.queued_actions if a.kind.value in GATE_DECISIONS]
    if body.kind in GATE_DECISIONS:
        gate = next((g for g in status.open_gates if g.stage == body.stage and g.gate == body.gate), None)
        if gate is None:
            raise HTTPException(409, "this checkpoint is not open any more; reload the page")
        if body.kind not in gate.actions:
            raise HTTPException(422, f"checkpoint {gate.gate} takes {' or '.join(gate.actions)}, not {body.kind}")
        if any(a.stage == body.stage and a.gate == body.gate for a in queued):
            raise HTTPException(409, "a decision for this checkpoint is already queued; undo it first to change it")
        if any(a.kind is ActionKind.EDIT and a.stage == body.stage for a in status.queued_actions):
            raise HTTPException(409, "your edits to this stage are still on their way; decide once they are applied")
        if body.kind == "select":
            index = payload.get("index")
            options = gate.summary.get("options") or []
            if not isinstance(index, int) or not 0 <= index < len(options):
                raise HTTPException(422, f"choose an option between 0 and {len(options) - 1}")
            if payload.get("edit"):
                check_choice_edit(run, gate, index, payload["edit"])
        choices = [payload.get(k) for k in ("include", "exclude") if payload.get(k) is not None]
        if any(not isinstance(c, list) or len(c) > 1000 or not all(isinstance(i, str) for i in c) for c in choices):
            raise HTTPException(422, "include and exclude must be lists of paper ids")
        said = bool(str(payload.get("feedback") or "").strip()) or payload.get("retest") or any(choices)
        if body.kind == "reject" and not said:
            raise HTTPException(422, "say what should change: the stage re-runs with your feedback")
        # bind the decision to the gate as the user saw it (the server's view, if the page did not send its own)
        payload.setdefault("expect", {"gate_nonce": gate.nonce, "artifacts": gate.summary.get("artifacts", {})})
    elif body.kind == "retry":
        stage = next((s for s in status.stages if s.id == body.stage), None)
        if stage is None or stage.status is not StageStatus.FAILED:
            raise HTTPException(409, "only a failed stage can be retried")
    elif body.kind == "reopen":
        stage = next((s for s in status.stages if s.id == body.stage), None)
        if stage is None or not stage.checkpoint:
            raise HTTPException(422, "this stage has no checkpoint to reopen")
        if stage.status is not StageStatus.DONE:
            raise HTTPException(409, "only a finished stage can be reopened")
        if any(a.kind is ActionKind.REOPEN and a.stage == body.stage for a in status.queued_actions):
            raise HTTPException(409, "reopening this checkpoint is already on its way")
    elif body.kind == "edit":
        artifact = str(payload.get("artifact") or "")
        if any(a.kind is ActionKind.EDIT and a.payload.get("artifact") == artifact for a in status.queued_actions):
            raise HTTPException(409, "an edit of this output is already on its way; undo it first to change it")
        try:
            checked = check_edit(run, artifact, payload.get("base_version"), payload.get("value"))
        except RToolError as exc:
            raise HTTPException(422, str(exc)) from None
        if not checked.changed:
            raise HTTPException(422, "nothing was changed")
        body.stage = checked.stage
    elif body.kind == "write_file":
        check_write(run, status, body.stage, payload)
    elif body.kind == "note":
        if body.stage not in {s.id for s in status.stages}:
            raise HTTPException(422, "a note needs the stage it is for")
        if not str(payload.get("text") or "").strip():
            raise HTTPException(422, "the note is empty")
    return payload


@router.post("/runs/{project}/{run_id}/actions", response_model=Queued, status_code=202)
def submit(project: str, run_id: str, body: ActionRequest, request: Request) -> Queued:
    run = get_run(request, project, run_id)
    payload = check(run_status(run), body, run)
    fields: dict[str, Any] = {"id": body.id} if body.id else {}
    action = Action(kind=ActionKind(body.kind), stage=body.stage, gate=body.gate, payload=payload, by=_user(), **fields)
    state, _ = ActionInbox(run.actions_dir).lookup(action.id)
    if state != "unknown":
        raise HTTPException(409, "this decision was already sent")
    ActionInbox(run.actions_dir).submit(action)
    if action.kind is ActionKind.EDIT:
        remove_draft(run, str(payload["artifact"]))  # the queued action holds the edit now
    elif action.kind is ActionKind.WRITE_FILE:
        remove_draft(run, file_draft_key(str(payload["path"])))
    return Queued(action=action, kick=runner_of(request).wake(run, f"action:{body.kind}"))


@router.get("/runs/{project}/{run_id}/actions/{action_id}", response_model=ActionState)
def action_state(project: str, run_id: str, action_id: str, request: Request) -> ActionState:
    run = get_run(request, project, run_id)
    state, record = ActionInbox(run.actions_dir).lookup(action_id)
    if state == "processed" and record is not None:
        outcome = record.get("outcome") or {}
        status = outcome.get("status", "applied")
        return ActionState(state=status, error=outcome.get("error"), outcome=outcome)
    if state == "invalid":
        return ActionState(state="invalid", error="the action file could not be read")
    return ActionState(state=state)  # type: ignore[arg-type]


@router.delete("/runs/{project}/{run_id}/actions/{action_id}", status_code=204)
def undo(project: str, run_id: str, action_id: str, request: Request) -> Response:
    run = get_run(request, project, run_id)
    inbox = ActionInbox(run.actions_dir)
    _, record = inbox.lookup(action_id)
    if not inbox.withdraw(action_id):
        raise HTTPException(409, "too late to undo: the decision is no longer queued")
    edit = (record or {}).get("payload") or {}
    kind = (record or {}).get("kind")
    # back to editing: the withdrawn edit becomes the draft again
    if kind == "edit" and read_draft(run, str(edit.get("artifact"))) is None:
        draft = Draft(base_version=edit["base_version"], value=edit["value"], note=str(edit.get("note") or ""))
        write_draft(run, str(edit["artifact"]), draft)
    elif kind == "write_file" and read_draft(run, key := file_draft_key(str(edit.get("path")))) is None:
        value = {"content": edit.get("content", ""), "base_sha": edit.get("base_sha")}
        write_draft(run, key, Draft(base_version=0, value=value))
    return Response(status_code=204)


@router.get("/runs/{project}/{run_id}/stages/{stage}/impact", response_model=list[Affected])
def impact(project: str, run_id: str, stage: str, request: Request) -> list[Affected]:
    """What a change to this stage's outputs would do to the later stages (shown before reopening or editing)."""
    run = get_run(request, project, run_id)
    try:
        return downstream_impact(run, stage)
    except RToolError as exc:
        raise HTTPException(404, str(exc)) from None
