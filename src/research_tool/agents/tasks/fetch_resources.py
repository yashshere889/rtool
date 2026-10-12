"""fetch_resources: download the plan's datasets and models (on Barkla), quota-checked and resumable.

Resources the plan marks "manual" are not public: their folders start empty. If one is still empty,
the stage stops at the ``manual_resources`` gate so a person can put the files there, let the Coder
build it in code, or send the plan back for a public resource.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ...core.agent import Task
from ...core.context import AgentContext
from ...core.errors import StageFailed
from ...core.registry import register_agent
from ...hpc.storage import QuotaError, ensure_room
from ...schemas.plan import ExperimentPlan, FetchedResource, ResourceManifest
from ...tools.http import PoliteHttp
from ...tools.resources.fetch import Fetcher, tree_size

GATE = "manual_resources"


def provided(resource: FetchedResource) -> FetchedResource:
    """A manual resource as it is now: someone may have put files in its folder since it was created."""
    if resource.status != "manual" or not resource.path or not Path(resource.path).is_dir():
        return resource
    size, files = tree_size(Path(resource.path))
    note = f"provided by hand ({files} files)" if files else f"not provided: its folder {resource.path} is empty"
    return resource.model_copy(update={"bytes": size, "files": files, "note": note})


def gate_message(missing: list[FetchedResource]) -> str:
    where = "; ".join(f"{r.name}: {r.path}" for r in missing)
    return (
        f"{len(missing)} resource(s) in the plan are not public, so nothing was downloaded for them ({where}). "
        "Put their files in those folders and approve; or approve without them, and the Coder will try to "
        "build them in code (check that code at the GPU-runs checkpoint: the results rest on it). "
        "Request changes to have the plan revised instead, e.g. to use a public dataset or pip package."
    )


class FetchConfig(BaseModel):
    root: str | None = None  # default: paths.resources_root, else <run>/resources
    max_fraction: float = Field(default=0.9, gt=0, le=1)  # refuse downloads that would fill a quota past this


@register_agent("fetch_resources")
class FetchResources(Task):
    description = "Downloads the plan's datasets and models into a shared, quota-checked folder."
    consumes = ("experiment_plan",)
    produces = ("resource_manifest",)
    config_model = FetchConfig

    def run(self, ctx: AgentContext) -> None:
        cfg: FetchConfig = self.config
        plan: ExperimentPlan = ctx.inputs.get("experiment_plan")
        configured = cfg.root or (
            str(ctx.app_config.paths.resources_root) if ctx.app_config.paths.resources_root else None
        )
        root = Path(os.path.expandvars(os.path.expanduser(configured))) if configured else ctx.run_root / "resources"
        planned_gb = sum(r.size_gb or 0 for r in plan.resources if r.source == "huggingface")
        try:
            ensure_room(root, extra_bytes=int(planned_gb * 1e9), max_fraction=cfg.max_fraction)
        except QuotaError as exc:
            raise StageFailed(str(exc), needs_human=True) from None

        factory = ctx.tools.get("resource_fetcher")
        fetcher = factory(root) if factory else Fetcher(root, http=PoliteHttp(), hf_token=os.environ.get("HF_TOKEN"))
        done: dict[str, dict] = ctx.state.setdefault("fetched", {})
        results: list[FetchedResource] = []
        for resource in plan.resources:
            if resource.name in done and done[resource.name]["status"] != "failed":
                results.append(FetchedResource.model_validate(done[resource.name]))
                continue
            result = fetcher.fetch(resource)
            done[resource.name] = result.model_dump()
            ctx.checkpoint()  # a resumed stage never downloads the same thing twice
            ctx.emit("resource_fetched", name=resource.name, status=result.status, bytes=result.bytes)
            results.append(result)
        failed = [r for r in results if r.status == "failed"]
        if failed:
            details = "; ".join(f"{r.name}: {r.note}" for r in failed)
            raise StageFailed(
                f"could not fetch {details}. Fix it (e.g. HF_TOKEN for gated data), then `rtool retry <run> --resume`",
                needs_human=True,
            )
        results = [provided(r) for r in results]
        missing = [r for r in results if r.status == "manual" and not r.files]
        if missing:
            decision = ctx.gate(
                GATE,
                {
                    "message": gate_message(missing),
                    "missing": [{"name": r.name, "path": r.path} for r in missing],
                },
            )
            if not decision.approved:  # the plan is redone with this feedback, then this stage again
                ctx.request_change(
                    artifact="experiment_plan",
                    reason=f"the researcher at the {GATE} checkpoint: "
                    + (decision.feedback or "replace the manual resources with public ones"),
                    required_changes=[decision.feedback] if decision.feedback else [],
                )
            results = [provided(r) for r in results]  # files put there while the gate was open
        ctx.outputs.put("resource_manifest", ResourceManifest(resources=results, root=str(root)))
