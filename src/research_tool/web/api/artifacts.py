"""Artifacts (every version, with provenance) and files inside a run directory."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...core.artifacts import ARTIFACT_TYPES, ArtifactEnvelope, ArtifactStore, Producer, artifact_model
from ...core.errors import ArtifactError
from ..deps import get_run, jailed

router = APIRouter(tags=["artifacts"])

# What the browser may fetch from a run directory, and how it is served. Text is always text/plain (never
# rendered as a page); HTML paper copies and SVG figures get a sandbox so no script in them can run.
SERVED_TYPES = {
    ".pdf": "application/pdf",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    **{
        ext: "text/plain; charset=utf-8"
        for ext in (".md", ".txt", ".json", ".jsonl", ".csv", ".py", ".log", ".out", ".bib", ".diff", ".yaml", ".j2")
    },
}
SANDBOXED = {".html", ".htm", ".svg"}


class ArtifactInfo(BaseModel):
    name: str
    versions: list[int]
    head: int
    producer: Producer
    created_at: str


@router.get("/runs/{project}/{run_id}/artifacts", response_model=list[ArtifactInfo])
def artifacts(project: str, run_id: str, request: Request) -> list[ArtifactInfo]:
    store = ArtifactStore(get_run(request, project, run_id).artifacts_dir)
    out = []
    for name in store.names():
        versions = store.versions(name)
        if not versions:
            continue
        envelope = store.envelope(name)
        out.append(
            ArtifactInfo(
                name=name,
                versions=versions,
                head=versions[-1],
                producer=envelope.producer,
                created_at=envelope.created_at,
            )
        )
    return out


@router.get("/runs/{project}/{run_id}/artifacts/{name}", response_model=ArtifactEnvelope)
def artifact(project: str, run_id: str, name: str, request: Request, response: Response, version: int | None = None):
    """One version (the latest by default). Versions never change, so ETags make reloads cheap."""
    store = ArtifactStore(get_run(request, project, run_id).artifacts_dir)
    try:
        envelope = store.envelope(name, version)
    except ArtifactError as exc:
        raise HTTPException(404, str(exc)) from None
    etag = f'"{envelope.name}-v{envelope.version}-{envelope.content_hash[:12]}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"etag": etag})
    response.headers["etag"] = etag
    response.headers["cache-control"] = "no-cache" if version is None else "private, max-age=31536000, immutable"
    return envelope


@router.get("/artifact-types")
def artifact_types() -> dict[str, Any]:
    """JSON Schema of every artifact type, so pages for new agents can be generated."""
    from ...core.artifacts import _load_builtin_schemas

    _load_builtin_schemas()
    return {name: artifact_model(name).model_json_schema() for name in sorted(ARTIFACT_TYPES)}


@router.get("/runs/{project}/{run_id}/files/{path:path}")
def run_file(project: str, run_id: str, path: str, request: Request) -> FileResponse:
    """A file from the run directory (papers, figures, reports, logs) — only known kinds, never outside the run."""
    run = get_run(request, project, run_id)
    target = jailed(run, path)
    suffix = target.suffix.lower()
    media_type = SERVED_TYPES.get(suffix)
    if media_type is None:
        raise HTTPException(404, "this kind of file is not served")
    headers = {"cache-control": "no-cache"}
    if suffix in SANDBOXED:
        headers["content-security-policy"] = "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'"
    return FileResponse(target, media_type=media_type, headers=headers, content_disposition_type="inline")
