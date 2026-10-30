"""The Experiment plan page: the plan (any version), live checks of an edited draft, and the downloads.

Edits are checked here with the planner's own editor (the same code the engine
runs when it applies them), so the page shows exactly what would be stored:
recomputed runs and GPU-hours, the results schema, and the remaining issues.
Nothing here calls the model or writes engine state; an edit is sent as an
``edit`` action.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...core.artifacts import ArtifactStore
from ...core.editing import edit_refusal
from ...core.engine import load_status
from ...core.errors import ArtifactError, RToolError
from ...core.events import EventLog
from ...core.run import RunHandle
from ...core.state import StageStatus
from ...schemas.plan import ExperimentPlan, ResourceManifest
from ...service import check_edit
from ..deps import get_run

router = APIRouter(tags=["plan"])

PLAN = "experiment_plan"
MANIFEST = "resource_manifest"


class PartitionView(BaseModel):
    name: str
    gpu_type: str | None
    gpus_per_node: int
    gpu_memory_gb: float | None
    max_time: str
    preemptible: bool
    note: str
    allowed: bool  # experiments may run here


class ClusterView(BaseModel):
    partitions: list[PartitionView]  # empty: no cluster profile configured, nothing to check against
    budget_gpu_hours: float | None
    min_seeds: int
    max_dataset_gb: float


class PlanView(BaseModel):
    stage: str
    version: int
    versions: list[int]
    head: int
    plan: ExperimentPlan
    hypothesis_title: str
    by: str | None  # who edited this version by hand (None: written by the planner)
    edited_from: int | None
    edit_note: str
    created_at: str
    editable: bool  # this is the latest version and the stage allows edits now
    why_not_editable: str
    cluster: ClusterView


class CheckRequest(BaseModel):
    base_version: int
    value: dict[str, Any]


class PlanCheck(BaseModel):
    ok: bool
    error: str  # why the edit would be refused (when not ok)
    plan: ExperimentPlan | None  # the edit as it would be stored
    changed: list[str]


class DownloadItem(BaseModel):
    name: str
    kind: str
    source: str
    identifier: str
    status: Literal["waiting", "ready", "deferred", "manual", "failed"]
    bytes: int
    files: int
    path: str | None
    note: str


class DownloadsView(BaseModel):
    stage: str | None
    status: StageStatus | None
    started: bool
    last_error: str | None
    plan_version: int | None  # the plan version the downloads are for
    root: str
    items: list[DownloadItem]


def _cluster(run: RunHandle, config: Any) -> ClusterView:
    slurm = run.config.slurm
    allowed = set(slurm.experiment_partitions)
    return ClusterView(
        partitions=[
            PartitionView(
                name=name,
                gpu_type=spec.gpu_type,
                gpus_per_node=spec.gpus_per_node,
                gpu_memory_gb=spec.gpu_memory_gb,
                max_time=spec.max_time,
                preemptible=spec.preemptible,
                note=spec.note,
                allowed=name in allowed if allowed else bool(spec.gpus_per_node),
            )
            for name, spec in slurm.partitions.items()
        ],
        budget_gpu_hours=run.config.budgets.run.max_gpu_hours,
        min_seeds=getattr(config, "min_seeds", 1),
        max_dataset_gb=getattr(config, "max_dataset_gb", 0),
    )


@router.get("/runs/{project}/{run_id}/plan", response_model=PlanView)
def plan(project: str, run_id: str, request: Request, version: int | None = None) -> PlanView:
    """The experiment plan (latest, or ``version``) with what the editor needs to know about the cluster."""
    run = get_run(request, project, run_id)
    store = ArtifactStore(run.artifacts_dir)
    try:
        envelope = store.envelope(PLAN, version)
    except ArtifactError:
        raise HTTPException(404, "no experiment plan yet" if version is None else f"no version {version}") from None
    _, states, pipeline = load_status(run)
    sid = envelope.producer.stage
    head = store.head(PLAN) or envelope.version
    refusal = edit_refusal(sid, states[sid]) if sid in states else "this plan was not made by a stage of this run"
    if envelope.version != head:
        refusal = "this is an earlier version; edits start from the latest"
    try:
        chosen = store.envelope("selected_hypothesis", envelope.inputs.get("selected_hypothesis")).payload
        title = str((chosen.get("item") or {}).get("title") or "")
    except ArtifactError:
        title = ""
    agent = envelope.producer.agent
    return PlanView(
        stage=sid,
        version=envelope.version,
        versions=store.versions(PLAN),
        head=head,
        plan=ExperimentPlan.model_validate(envelope.payload),
        hypothesis_title=title,
        by=agent.removeprefix("human:") if agent.startswith("human:") else None,
        edited_from=envelope.meta.get("edited_from"),
        edit_note=str(envelope.meta.get("edit_note") or ""),
        created_at=envelope.created_at,
        editable=refusal is None,
        why_not_editable=refusal or "",
        cluster=_cluster(run, pipeline.stage(sid).config if sid in states else None),
    )


@router.post("/runs/{project}/{run_id}/plan/check", response_model=PlanCheck)
def check(project: str, run_id: str, body: CheckRequest, request: Request) -> PlanCheck:
    """What an edited plan would become (recomputed, checked), without storing anything."""
    run = get_run(request, project, run_id)
    try:
        checked = check_edit(run, PLAN, body.base_version, body.value)
    except RToolError as exc:
        return PlanCheck(ok=False, error=str(exc), plan=None, changed=[])
    return PlanCheck(ok=True, error="", plan=ExperimentPlan.model_validate(checked.value), changed=checked.changed)


@router.get("/runs/{project}/{run_id}/downloads", response_model=DownloadsView)
def downloads(project: str, run_id: str, request: Request) -> DownloadsView:
    """The datasets and models the approved plan needs, as the download step got (or is getting) them."""
    run = get_run(request, project, run_id)
    _, states, pipeline = load_status(run)
    sid = next((s for s in pipeline.order if MANIFEST in pipeline.stage(s).produces), None)
    st = states.get(sid) if sid else None
    store = ArtifactStore(run.artifacts_dir)
    if st is None or st.attempt == 0:
        status = st.status if st else None
        return DownloadsView(
            stage=sid, status=status, started=False, last_error=None, plan_version=None, root="", items=[]
        )
    plan_version = st.input_versions.get(PLAN)
    try:
        resources = ExperimentPlan.model_validate(store.envelope(PLAN, plan_version).payload).resources
    except ArtifactError:
        resources = []
    items = {
        r.name: DownloadItem(
            name=r.name,
            kind=r.kind,
            source=r.source,
            identifier=r.identifier,
            status="waiting",
            bytes=0,
            files=0,
            path=None,
            note="",
        )
        for r in resources
    }
    root = ""
    if st.status is StageStatus.DONE and store.head(MANIFEST) is not None:
        manifest = ResourceManifest.model_validate(store.envelope(MANIFEST, st.output_versions.get(MANIFEST)).payload)
        root = _shown(run, manifest.root)
        for r in manifest.resources:
            items[r.name] = DownloadItem(**r.model_dump() | {"path": _shown(run, r.path) if r.path else None})
    else:  # still downloading, or stopped: what this attempt has fetched so far
        for event in _attempt_events(run, sid):
            data = event.get("data") or {}
            if event.get("kind") == "resource_fetched" and data.get("name") in items:
                item = items[data["name"]]
                items[data["name"]] = item.model_copy(
                    update={"status": data.get("status", "ready"), "bytes": int(data.get("bytes") or 0)}
                )
    return DownloadsView(
        stage=sid,
        status=st.status,
        started=True,
        last_error=st.last_error,
        plan_version=plan_version,
        root=root,
        items=list(items.values()),
    )


def _shown(run: RunHandle, path: str) -> str:
    """Paths inside the run relative to it (also if the runs folder was moved); shared folders as they are."""
    try:
        return Path(path).relative_to(run.root).as_posix()
    except ValueError:
        pass
    marker = f"/{run.spec.project}/{run.run_id}/"
    return path.split(marker, 1)[1] if marker in path else path


def _attempt_events(run: RunHandle, stage: str) -> list[dict[str, Any]]:
    """The stage's events since its last fresh start."""
    events, _ = EventLog(run.events_path, run.run_id).read_from(0)
    mine = [e for e in events if e.get("stage") == stage]
    starts = [
        i for i, e in enumerate(mine) if e.get("kind") == "stage_started" and not (e.get("data") or {}).get("resumed")
    ]
    return mine[starts[-1] :] if starts else mine
