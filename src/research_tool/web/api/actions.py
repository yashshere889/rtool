"""Decisions: queue one for the next episode, see what became of it, take it back while it is queued."""

from __future__ import annotations

import getpass
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ...core.actions import Action, ActionInbox, ActionKind
from ...core.state import StageStatus
from ...service import KickResult, RunStatus, run_status
from ..deps import get_run, runner_of

router = APIRouter(tags=["actions"])

GATE_DECISIONS = {"approve", "reject", "select"}


class ActionRequest(BaseModel):
    id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$", description="Client-chosen id (idempotency)")
    kind: Literal["approve", "reject", "select", "retry", "note"]
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


def check(status: RunStatus, body: ActionRequest) -> dict[str, Any]:
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
        if body.kind == "select":
            index = payload.get("index")
            options = gate.summary.get("options") or []
            if not isinstance(index, int) or not 0 <= index < len(options):
                raise HTTPException(422, f"choose an option between 0 and {len(options) - 1}")
        if body.kind == "reject" and not str(payload.get("feedback") or "").strip():
            raise HTTPException(422, "say what should change: the stage re-runs with your feedback")
        # bind the decision to the gate as the user saw it (the server's view, if the page did not send its own)
        payload.setdefault("expect", {"gate_nonce": gate.nonce, "artifacts": gate.summary.get("artifacts", {})})
    elif body.kind == "retry":
        stage = next((s for s in status.stages if s.id == body.stage), None)
        if stage is None or stage.status is not StageStatus.FAILED:
            raise HTTPException(409, "only a failed stage can be retried")
    elif body.kind == "note":
        if body.stage not in {s.id for s in status.stages}:
            raise HTTPException(422, "a note needs the stage it is for")
        if not str(payload.get("text") or "").strip():
            raise HTTPException(422, "the note is empty")
    return payload


@router.post("/runs/{project}/{run_id}/actions", response_model=Queued, status_code=202)
def submit(project: str, run_id: str, body: ActionRequest, request: Request) -> Queued:
    run = get_run(request, project, run_id)
    payload = check(run_status(run), body)
    fields: dict[str, Any] = {"id": body.id} if body.id else {}
    action = Action(kind=ActionKind(body.kind), stage=body.stage, gate=body.gate, payload=payload, by=_user(), **fields)
    state, _ = ActionInbox(run.actions_dir).lookup(action.id)
    if state != "unknown":
        raise HTTPException(409, "this decision was already sent")
    ActionInbox(run.actions_dir).submit(action)
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
    if not ActionInbox(run.actions_dir).withdraw(action_id):
        raise HTTPException(409, "too late to undo: the decision is no longer queued")
    return Response(status_code=204)
