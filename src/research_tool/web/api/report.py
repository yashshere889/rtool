"""The Report and Review pages: the report as sections (templates and rendered), the placeholder catalogue,
the citable papers, live checks of an edit, the review with its checks and diff, and a download bundle.

Every number in a report comes from a placeholder rendered from the results;
these routes render with the same table the Writer used. Edits are sent as
``edit`` actions and checked here first with the agents' own editors.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ...core.artifacts import ArtifactEnvelope, ArtifactStore
from ...core.editing import edit_refusal
from ...core.engine import load_status
from ...core.errors import ArtifactError, RToolError
from ...core.run import RunHandle
from ...schemas.code import ExperimentResults
from ...schemas.literature import Corpus
from ...schemas.plan import ExperimentPlan
from ...schemas.report import Finding, Report, Review
from ...service import check_edit
from ...tools.report.render import Value, bib_keys, problems, render, value_table
from ..deps import get_run, run_relative

router = APIRouter(tags=["report"])

Which = Literal["report", "final_report"]
MAX_DIFF = 200_000


class PlaceholderInfo(BaseModel):
    key: str
    text: str  # what it renders to
    meaning: str


class PaperInfo(BaseModel):
    key: str
    paper_id: str
    title: str
    authors: str
    year: int | None
    label: str  # how the text cites it: "Morgan et al., 2019", "Novak & Hale, 2020"


class SectionView(BaseModel):
    id: str
    title: str
    origin: str  # rtool | llm | reviewer | human
    template: str  # with {{placeholders}} and [@citations]
    rendered: str  # numbers filled in; citations left as [@key] for the page to link
    editable: bool
    problems: list[str]


class ReportView(BaseModel):
    artifact: Which
    stage: str
    version: int
    versions: list[int]
    head: int
    by: str | None  # who edited this version by hand (None: an agent wrote it)
    edited_from: int | None
    edit_note: str
    created_at: str
    title: str
    title_origin: str
    sections: list[SectionView]
    findings: list[Finding]
    citations: list[str]
    references: list[PaperInfo]  # the cited papers, in order of first citation
    figures: list[str]  # run-relative
    markdown_path: str
    bibtex_path: str
    base_dir: str  # run-relative folder of the Markdown file (images in it are relative to it)
    placeholders: list[PlaceholderInfo]
    papers: list[PaperInfo]  # everything the report may cite
    editable: bool
    why_not_editable: str


class CheckRequest(BaseModel):
    artifact: Which
    base_version: int
    value: dict[str, Any]


class ReportCheck(BaseModel):
    ok: bool
    error: str
    title: str
    sections: list[SectionView]
    findings: list[Finding]
    changed: list[str]


class ReviewView(BaseModel):
    stage: str
    version: int
    review: Review
    final_version: int | None  # the final report this review produced
    final_edited: bool  # the final report was edited by hand afterwards (that edit was not reviewed)
    diff: str  # the Reviewer's edits as a unified diff
    min_score: float
    rounds_used: int  # revision rounds the report has had
    rounds_max: int
    report_stage: str | None


# ------------------------------------------------------------------ helpers
class _Material:
    """What a report version was written from: placeholder values and citable papers."""

    def __init__(self, store: ArtifactStore, envelope: ArtifactEnvelope) -> None:
        inputs = envelope.inputs

        def payload(name: str) -> dict[str, Any]:
            return store.envelope(name, inputs.get(name)).payload

        try:
            plan = ExperimentPlan.model_validate(payload("experiment_plan"))
            results = ExperimentResults.model_validate(payload("experiment_results"))
            self.corpus = Corpus.model_validate(payload("corpus"))
        except ArtifactError as exc:
            raise HTTPException(409, f"the report's inputs cannot be read: {exc}") from None
        self.values: dict[str, Value] = value_table(plan, results)
        self.keys = set(bib_keys(self.corpus))

    def papers(self) -> dict[str, PaperInfo]:
        out = {}
        for p in self.corpus.papers:
            if p.bib_key and p.bib_key in self.keys:
                authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
                out[p.bib_key] = PaperInfo(
                    key=p.bib_key,
                    paper_id=p.id,
                    title=p.title,
                    authors=authors,
                    year=p.year,
                    label=f"{_surnames(p.authors) or p.bib_key}, {p.year or 'n.d.'}",
                )
        return out

    def section_view(self, section: Any) -> SectionView:
        keys = self.keys if section.id != "abstract" else set()
        found = [] if section.origin == "rtool" else problems(section.template, self.values, keys, allowed_numbers=None)
        return SectionView(
            id=section.id,
            title=section.title,
            origin=section.origin,
            template=section.template,
            rendered=render(section.template, self.values),
            editable=section.origin != "rtool",
            problems=found,
        )


def _surnames(authors: list[str]) -> str:
    names = [a.split()[-1] for a in authors if a.split()]
    if len(names) > 2:
        return f"{names[0]} et al."
    return " & ".join(names)


def _envelope(store: ArtifactStore, which: str, version: int | None) -> ArtifactEnvelope:
    try:
        return store.envelope(which, version)
    except ArtifactError:
        missing = f"no {which.replace('_', ' ')} yet" if version is None else f"no version {version}"
        raise HTTPException(404, missing) from None


def _report_view(run: RunHandle, which: Which, version: int | None) -> ReportView:
    store = ArtifactStore(run.artifacts_dir)
    envelope = _envelope(store, which, version)
    report = Report.model_validate(envelope.payload)
    material = _Material(store, envelope)
    papers = material.papers()
    _, states, _ = load_status(run)
    sid = envelope.producer.stage
    head = store.head(which) or envelope.version
    refusal = edit_refusal(sid, states[sid]) if sid in states else "this report was not made by a stage of this run"
    if envelope.version != head:
        refusal = "this is an earlier version; edits start from the latest"
    agent = envelope.producer.agent
    return ReportView(
        artifact=which,
        stage=sid,
        version=envelope.version,
        versions=store.versions(which),
        head=head,
        by=agent.removeprefix("human:") if agent.startswith("human:") else None,
        edited_from=envelope.meta.get("edited_from"),
        edit_note=str(envelope.meta.get("edit_note") or ""),
        created_at=envelope.created_at,
        title=report.title,
        title_origin=report.title_origin,
        sections=[material.section_view(s) for s in report.sections],
        findings=report.findings,
        citations=report.citations,
        references=[papers[k] for k in report.citations if k in papers],
        figures=report.figures,
        markdown_path=report.markdown_path,
        bibtex_path=report.bibtex_path,
        base_dir=Path(report.markdown_path).parent.as_posix(),
        placeholders=[
            PlaceholderInfo(key=k, text=v.text, meaning=v.meaning) for k, v in sorted(material.values.items())
        ],
        papers=sorted(papers.values(), key=lambda p: p.key),
        editable=refusal is None,
        why_not_editable=refusal or "",
    )


# ------------------------------------------------------------------ routes
@router.get("/runs/{project}/{run_id}/report", response_model=ReportView)
def report(
    project: str, run_id: str, request: Request, which: Which = "report", version: int | None = None
) -> ReportView:
    """The report (``which=report``) or the final report after the review, as sections with what they render to."""
    return _report_view(get_run(request, project, run_id), which, version)


@router.post("/runs/{project}/{run_id}/report/check", response_model=ReportCheck)
def check(project: str, run_id: str, body: CheckRequest, request: Request) -> ReportCheck:
    """What an edited report would become (rendered, checked), without storing anything."""
    run = get_run(request, project, run_id)
    try:
        checked = check_edit(run, body.artifact, body.base_version, body.value)
    except RToolError as exc:
        return ReportCheck(ok=False, error=str(exc), title="", sections=[], findings=[], changed=[])
    store = ArtifactStore(run.artifacts_dir)
    material = _Material(store, _envelope(store, body.artifact, body.base_version))
    edited = Report.model_validate(checked.value)
    return ReportCheck(
        ok=True,
        error="",
        title=edited.title,
        sections=[material.section_view(s) for s in edited.sections],
        findings=edited.findings,
        changed=checked.changed,
    )


@router.get("/runs/{project}/{run_id}/review", response_model=ReviewView)
def review(project: str, run_id: str, request: Request, version: int | None = None) -> ReviewView:
    """The review: verdict, rubric scores, automated checks, issues, the Reviewer's edits, revision rounds."""
    run = get_run(request, project, run_id)
    store = ArtifactStore(run.artifacts_dir)
    envelope = _envelope(store, "review", version)
    result = Review.model_validate(envelope.payload)
    _, states, pipeline = load_status(run)
    sid = envelope.producer.stage
    st = states.get(sid)
    final_version = st.output_versions.get("final_report") if st else None
    head = store.head("final_report")
    edited_by = store.envelope("final_report").producer.agent if head is not None else ""
    final_edited = head != final_version and edited_by.startswith("human:")
    diff = ""
    if result.diff_path:
        path = run.root / result.diff_path
        if run_relative(run, path) is not None and path.is_file():
            diff = path.read_text(encoding="utf-8", errors="replace")[:MAX_DIFF]
    report_stage = next((s for s in pipeline.order if "report" in pipeline.stage(s).agent_produces), None)
    config = pipeline.stage(sid).config if sid in states else None
    return ReviewView(
        stage=sid,
        version=envelope.version,
        review=result,
        final_version=final_version,
        final_edited=final_edited,
        diff=diff if diff.strip() != "(no edits)" else "",
        min_score=float(getattr(config, "min_score", 3.0)),
        rounds_used=states[report_stage].revision_count if report_stage in states else 0,
        rounds_max=run.config.revisions.max_rounds,
        report_stage=report_stage,
    )


@router.get("/runs/{project}/{run_id}/report/bundle")
def bundle(project: str, run_id: str, request: Request) -> Response:
    """The final report (or the report) with its references, figures, review and results, as a zip file."""
    run = get_run(request, project, run_id)
    store = ArtifactStore(run.artifacts_dir)
    which = "final_report" if store.head("final_report") is not None else "report"
    report_ = Report.model_validate(_envelope(store, which, None).payload)
    files: dict[str, Path] = {
        Path(report_.markdown_path).name: run.root / report_.markdown_path,
        "refs.bib": run.root / report_.bibtex_path,
    }
    files |= {f"figures/{Path(f).name}": run.root / f for f in report_.figures}
    if store.head("review") is not None:
        files["review.md"] = run.root / Review.model_validate(store.envelope("review").payload).review_path
    if store.head("experiment_results") is not None:
        results = ExperimentResults.model_validate(store.envelope("experiment_results").payload)
        files["results/results.json"] = run.root / results.results_path
        if results.tables_path:
            files["results/results.md"] = run.root / results.tables_path
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in files.items():
            if run_relative(run, path) is not None and path.is_file():
                zf.write(path, f"{run_id}-report/{name}")
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"content-disposition": f'attachment; filename="{run_id}-report.zip"'},
    )
