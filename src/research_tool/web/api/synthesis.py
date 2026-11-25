"""The literature review page: the review (any version) and live checks of an edited draft.

An edit is checked with the synthesis agent's own editor, the one the engine runs when it applies the edit, so the
page shows the sentence that would be refused and why. Nothing here calls the model or writes engine state: an edit
is sent as an ``edit`` action.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...agents.synthesis.agent import Synthesis
from ...core.artifacts import ArtifactStore
from ...core.editing import edit_refusal
from ...core.engine import load_status
from ...core.errors import ArtifactError, RToolError
from ...service import check_edit
from ..deps import get_run
from .plan import CheckRequest

router = APIRouter(tags=["synthesis"])

ARTIFACT = "synthesis"


class SynthesisView(BaseModel):
    stage: str
    version: int
    versions: list[int]
    head: int
    synthesis: Synthesis
    by: str | None  # who edited this version by hand (None: the agent wrote it)
    edited_from: int | None
    edit_note: str
    created_at: str
    editable: bool
    why_not_editable: str


class SynthesisCheck(BaseModel):
    ok: bool
    error: str
    synthesis: Synthesis | None
    changed: list[str]


@router.get("/runs/{project}/{run_id}/synthesis", response_model=SynthesisView)
def synthesis(project: str, run_id: str, request: Request, version: int | None = None) -> SynthesisView:
    run = get_run(request, project, run_id)
    store = ArtifactStore(run.artifacts_dir)
    try:
        envelope = store.envelope(ARTIFACT, version)
    except ArtifactError:
        raise HTTPException(404, "no review yet" if version is None else f"no version {version}") from None
    _, states, _ = load_status(run)
    sid = envelope.producer.stage
    head = store.head(ARTIFACT) or envelope.version
    refusal = edit_refusal(sid, states[sid]) if sid in states else "this review was not made by a stage of this run"
    if envelope.version != head:
        refusal = "this is an earlier version; edits start from the latest"
    agent = envelope.producer.agent
    return SynthesisView(
        stage=sid,
        version=envelope.version,
        versions=store.versions(ARTIFACT),
        head=head,
        synthesis=Synthesis.model_validate(envelope.payload),
        by=agent.removeprefix("human:") if agent.startswith("human:") else None,
        edited_from=envelope.meta.get("edited_from"),
        edit_note=str(envelope.meta.get("edit_note") or ""),
        created_at=envelope.created_at,
        editable=refusal is None,
        why_not_editable=refusal or "",
    )


@router.post("/runs/{project}/{run_id}/synthesis/check", response_model=SynthesisCheck)
def check(project: str, run_id: str, body: CheckRequest, request: Request) -> SynthesisCheck:
    """What an edited review would become, or why a sentence would be refused; nothing is stored."""
    run = get_run(request, project, run_id)
    try:
        checked = check_edit(run, ARTIFACT, body.base_version, body.value)
    except RToolError as exc:
        return SynthesisCheck(ok=False, error=str(exc), synthesis=None, changed=[])
    return SynthesisCheck(ok=True, error="", synthesis=Synthesis.model_validate(checked.value), changed=checked.changed)
