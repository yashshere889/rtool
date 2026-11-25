"""The study pipeline's pages: the protocol (read, edited, checked) and the data the researcher brings back.

The protocol is edited with the protocol agent's own editor, the one the engine runs when it applies an edit,
so a draft is checked exactly as it will be stored. The upload is checked against the protocol before it is
kept, so a file that does not fit never reaches the stage. Nothing here calls the model or writes engine state:
an edit is an ``edit`` action, and approving the data is the checkpoint's ordinary ``approve``.
"""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...agents.study.schemas import DATA_FILE, Protocol, StudyData
from ...agents.study.tasks import read_table
from ...core.artifacts import ArtifactStore
from ...core.editing import edit_refusal
from ...core.engine import load_status
from ...core.errors import ArtifactError, RToolError
from ...core.fs import atomic_write_text
from ...core.run import RunHandle
from ...core.state import GateStatus, StageStatus
from ...schemas.plan import ExperimentPlan
from ...service import check_edit
from ..deps import get_run
from ..security import MAX_BODY
from .plan import CheckRequest

router = APIRouter(tags=["study"])

PROTOCOL, PLAN, DATA = "protocol", "experiment_plan", "study_data"
PREVIEW_ROWS = 12


class ProtocolView(BaseModel):
    stage: str
    version: int
    versions: list[int]
    head: int
    protocol: Protocol
    plan: ExperimentPlan  # the design the protocol is computed from (read-only here)
    hypothesis_title: str
    by: str | None  # who edited this version by hand (None: written by the agent)
    edited_from: int | None
    edit_note: str
    created_at: str
    editable: bool
    why_not_editable: str
    allocation_rows: int
    allocation_preview: list[dict[str, str]]


class ProtocolCheck(BaseModel):
    ok: bool
    error: str
    protocol: Protocol | None
    changed: list[str]


class CollectView(BaseModel):
    stage: str
    status: StageStatus
    waiting: bool  # the data checkpoint is open
    can_upload: bool
    why_not: str
    columns: list[str]
    unit: str
    conditions: list[str]
    n_per_condition: int
    paired: bool
    data_format: str
    template: str  # a CSV with the unit and condition columns filled in from the allocation
    inbox_file: bool  # a data.csv is waiting for the stage to read
    received: StudyData | None  # what the stage accepted (and the analysis used)


class Received(BaseModel):
    rows: int
    units: dict[str, int]


def _allocation(run: RunHandle, protocol: Protocol) -> list[dict[str, str]]:
    path = run.root / protocol.allocation_path
    if not protocol.allocation_path or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _protocol_stage(run: RunHandle) -> str:
    sid = next((s.id for s in run.spec.pipeline.stages if s.agent == "protocol"), None)
    if sid is None:
        raise HTTPException(404, "this run has no protocol stage (use the study pipeline)")
    return sid


@router.get("/runs/{project}/{run_id}/protocol", response_model=ProtocolView)
def protocol(project: str, run_id: str, request: Request, version: int | None = None) -> ProtocolView:
    run = get_run(request, project, run_id)
    sid = _protocol_stage(run)
    store = ArtifactStore(run.artifacts_dir)
    try:
        envelope = store.envelope(PROTOCOL, version)
    except ArtifactError:
        raise HTTPException(404, "no protocol yet" if version is None else f"no version {version}") from None
    _, states, _ = load_status(run)
    head = store.head(PROTOCOL) or envelope.version
    refusal = edit_refusal(sid, states[sid])
    if envelope.version != head:
        refusal = "this is an earlier version; edits start from the latest"
    try:
        plan = ExperimentPlan.model_validate(store.envelope(PLAN).payload)
        chosen = store.envelope("selected_hypothesis").payload
        title = str((chosen.get("item") or {}).get("title") or "")
    except ArtifactError:
        raise HTTPException(404, "no protocol yet") from None
    value = Protocol.model_validate(envelope.payload)
    rows = _allocation(run, value)
    agent = envelope.producer.agent
    return ProtocolView(
        stage=sid,
        version=envelope.version,
        versions=store.versions(PROTOCOL),
        head=head,
        protocol=value,
        plan=plan,
        hypothesis_title=title,
        by=agent.removeprefix("human:") if agent.startswith("human:") else None,
        edited_from=envelope.meta.get("edited_from"),
        edit_note=str(envelope.meta.get("edit_note") or ""),
        created_at=envelope.created_at,
        editable=refusal is None,
        why_not_editable=refusal or "",
        allocation_rows=len(rows),
        allocation_preview=rows[:PREVIEW_ROWS],
    )


@router.post("/runs/{project}/{run_id}/protocol/check", response_model=ProtocolCheck)
def check(project: str, run_id: str, body: CheckRequest, request: Request) -> ProtocolCheck:
    """What an edited protocol would become (sample size, allocation, checklist), without storing anything."""
    run = get_run(request, project, run_id)
    try:
        checked = check_edit(run, PROTOCOL, body.base_version, body.value)
    except RToolError as exc:
        return ProtocolCheck(ok=False, error=str(exc), protocol=None, changed=[])
    return ProtocolCheck(ok=True, error="", protocol=Protocol.model_validate(checked.value), changed=checked.changed)


def _collect_stage(run: RunHandle) -> str:
    sid = next((s.id for s in run.spec.pipeline.stages if s.agent == "collect_data"), None)
    if sid is None:
        raise HTTPException(404, "this run has no data collection stage (use the study pipeline)")
    return sid


def _can_upload(run: RunHandle, sid: str) -> tuple[bool, str]:
    st = load_status(run)[1][sid]
    gate_open = st.gate is not None and st.gate.status is GateStatus.OPEN
    if gate_open or st.status in (StageStatus.FAILED, StageStatus.PENDING):
        return True, ""
    return (
        False,
        f"the data have been {'read' if st.status is StageStatus.DONE else 'asked for'} already ({st.status.value})",
    )


@router.get("/runs/{project}/{run_id}/collect", response_model=CollectView)
def collect(project: str, run_id: str, request: Request) -> CollectView:
    run = get_run(request, project, run_id)
    sid = _collect_stage(run)
    store = ArtifactStore(run.artifacts_dir)
    try:
        plan = ExperimentPlan.model_validate(store.envelope(PLAN).payload)
        value = Protocol.model_validate(store.envelope(PROTOCOL).payload)
    except ArtifactError:
        raise HTTPException(404, "the protocol is not written yet") from None
    st = load_status(run)[1][sid]
    metrics = [m.name for m in plan.metrics]
    columns = ["unit", "condition", *metrics]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(columns)
    for row in _allocation(run, value):
        if "condition" in row:
            writer.writerow([row["unit"], row["condition"], *[""] * len(metrics)])
        else:  # paired: every unit is measured under every condition, in its own order
            for condition in value.conditions:
                writer.writerow([row["unit"], condition, *[""] * len(metrics)])
    can, why = _can_upload(run, sid)
    received = None
    if store.head(DATA) is not None and st.status is StageStatus.DONE:
        received = StudyData.model_validate(store.envelope(DATA).payload)
    return CollectView(
        stage=sid,
        status=st.status,
        waiting=st.gate is not None and st.gate.status is GateStatus.OPEN,
        can_upload=can,
        why_not=why,
        columns=columns,
        unit=value.unit,
        conditions=value.conditions,
        n_per_condition=value.n_per_condition,
        paired=value.paired,
        data_format=value.data_format,
        template=out.getvalue(),
        inbox_file=(run.work_dir(sid) / "inbox" / DATA_FILE).is_file(),
        received=received,
    )


@router.put("/runs/{project}/{run_id}/collect/inbox", response_model=Received, status_code=201)
async def upload(project: str, run_id: str, request: Request) -> Received:
    """The study's data.csv as the request body. It is kept only if it follows the protocol."""
    run = get_run(request, project, run_id)
    sid = _collect_stage(run)
    can, why = _can_upload(run, sid)
    if not can:
        raise HTTPException(409, why)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY:  # the app's body cap, also checked here for clients that stream
            raise HTTPException(413, f"data files are limited to {MAX_BODY // (1024 * 1024)} MB")
    try:
        text = bytes(body).decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "this is not a text (UTF-8) CSV file") from None
    try:
        plan = ExperimentPlan.model_validate(ArtifactStore(run.artifacts_dir).envelope(PLAN).payload)
    except ArtifactError:
        raise HTTPException(409, "the protocol is not written yet") from None
    inbox = run.work_dir(sid) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    temp = inbox / f".{DATA_FILE}.upload"  # the stage ignores dot-files, so it never reads a half-written copy
    atomic_write_text(temp, text)
    rows, units, problems = read_table(temp, plan)
    if problems:
        temp.unlink(missing_ok=True)
        shown = "; ".join(problems[:8]) + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "")
        raise HTTPException(422, f"{DATA_FILE} does not match the protocol: {shown}")
    temp.replace(inbox / DATA_FILE)
    return Received(rows=rows, units=units)
