"""The Hypotheses page: the shortlist side by side, with the papers, citations and references behind each.

Everything comes from the hypothesis set the agent produced (its evidence was
checked against the papers then) and the corpus it was drawn from. Nothing
here calls the model.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...core.artifacts import ArtifactStore
from ...core.engine import load_status
from ...core.errors import ArtifactError
from ...core.run import RunHandle
from ...core.state import GateState
from ...knowledge.store import KnowledgeStore
from ...schemas.hypothesis import COMPUTE_HOURS, Hypothesis, HypothesisSet
from ...tools.report.render import bibtex_subset
from ..deps import get_run
from .literature import PaperCard, _Literature

router = APIRouter(tags=["hypotheses"])

OPTIONS = "hypothesis_set"
Role = Literal["evidence", "closest", "related"]


class PaperUse(BaseModel):
    paper: PaperCard
    # evidence: quoted in its support; closest: the nearest earlier work (novelty check); related: behind the gap
    roles: list[Role]


class HypothesisCard(BaseModel):
    index: int | None  # its option number at the checkpoint (0-based); None for candidates that were not offered
    hypothesis: Hypothesis
    papers: list[PaperUse]
    bibtex: str  # every paper it uses, for a reference manager
    gpu_hours: int  # the most its compute class may need


class FieldChange(BaseModel):
    field: str
    before: Any
    after: Any


class Selection(BaseModel):
    version: int
    index: int
    title: str
    edited: bool
    changes: list[FieldChange]  # what the researcher changed before choosing it
    note: str
    from_version: int | None  # the hypothesis set it was chosen from
    by: str | None  # who chose it (None when it was chosen automatically)
    created_at: str


class HypothesesView(BaseModel):
    stage: str
    version: int
    versions: list[int]
    topic: str
    shortlist: list[HypothesisCard]
    others: list[HypothesisCard]  # drafted but not offered: rejected by a check, or ranked lower
    selection: Selection | None
    weights: dict[str, float]  # how the rubric scores were combined into the total
    budget_gpu_hours: float | None
    markdown: str  # run-relative path of the agent's own write-up


def _selection_artifact(run: RunHandle) -> str | None:
    for gate in run.spec.pipeline.gates.values():
        if gate.options_artifact == OPTIONS and gate.selection_artifact:
            return gate.selection_artifact
    return None


class _Papers:
    """Paper cards and BibTeX for the papers a hypothesis uses, from the corpus it was drawn from."""

    def __init__(self, lit: _Literature, kb: KnowledgeStore | None) -> None:
        self.lit, self.kb = lit, kb
        bib = lit.run.root / lit.corpus.bibtex_path if lit.corpus.bibtex_path else None
        self.bib = bib.read_text(encoding="utf-8") if bib is not None and bib.is_file() else ""
        self._cards: dict[str, PaperCard] = {}

    def card(self, paper_id: str) -> PaperCard:
        if paper_id not in self._cards:
            paper = self.lit.by_id[paper_id]
            self._cards[paper_id] = self.lit.card(paper, self.kb)
        return self._cards[paper_id]

    def uses(self, h: Hypothesis) -> list[PaperUse]:
        roles: dict[str, list[Role]] = {}

        def add(paper_id: str, role: Role) -> None:
            if paper_id in self.lit.by_id and role not in roles.setdefault(paper_id, []):
                roles[paper_id].append(role)

        for item in h.evidence:
            add(item.paper_id, "evidence")
        if h.novelty and h.novelty.closest_paper_id:
            add(h.novelty.closest_paper_id, "closest")
        for paper_id in h.related_paper_ids:
            add(paper_id, "related")
        return [PaperUse(paper=self.card(pid), roles=r) for pid, r in roles.items() if r]

    def view(self, h: Hypothesis, index: int | None) -> HypothesisCard:
        papers = self.uses(h)
        keys = [p.paper.bib_key for p in papers if p.paper.bib_key]
        return HypothesisCard(
            index=index,
            hypothesis=h,
            papers=papers,
            bibtex=bibtex_subset(self.bib, keys).strip() if self.bib and keys else "",
            gpu_hours=COMPUTE_HOURS[h.compute],
        )


def _selection(store: ArtifactStore, name: str | None, gate: GateState | None) -> Selection | None:
    if name is None or store.head(name) is None:
        return None
    envelope = store.envelope(name)
    chosen = envelope.payload
    index = int(chosen.get("index", 0))
    item: dict[str, Any] = chosen.get("item") or {}
    from_version = envelope.inputs.get(OPTIONS)
    try:
        offered = store.envelope(OPTIONS, from_version).payload["hypotheses"][index]
    except (ArtifactError, KeyError, IndexError):
        offered = {}
    changes = [
        FieldChange(field=key, before=offered.get(key), after=value)
        for key, value in item.items()
        if offered and offered.get(key) != value
    ]
    resolution = gate.resolution if gate is not None else {}
    return Selection(
        version=envelope.version,
        index=index,
        title=str(item.get("title", "")),
        edited=bool(chosen.get("edited")),
        changes=changes,
        note=str(chosen.get("note", "")),
        from_version=from_version,
        by=None if resolution.get("auto") else resolution.get("by"),
        created_at=envelope.created_at,
    )


@router.get("/runs/{project}/{run_id}/hypotheses", response_model=HypothesesView)
def hypotheses(project: str, run_id: str, request: Request, version: int | None = None) -> HypothesesView:
    """The shortlist offered at the checkpoint (latest set, or ``version``), the other drafts, and the choice."""
    run = get_run(request, project, run_id)
    store = ArtifactStore(run.artifacts_dir)
    try:
        envelope = store.envelope(OPTIONS, version)
    except ArtifactError:
        raise HTTPException(404, "no hypotheses yet" if version is None else f"no version {version}") from None
    hs = HypothesisSet.model_validate(envelope.payload)
    sid = envelope.producer.stage
    _, states, pipeline = load_status(run)
    config = pipeline.stage(sid).config if sid in states else None
    lit = _Literature(run, envelope.inputs.get("corpus"))
    offered = {h.id for h in hs.hypotheses}
    with lit.kb() as kb:
        papers = _Papers(lit, kb)
        shortlist = [papers.view(h, i) for i, h in enumerate(hs.hypotheses)]
        others = [papers.view(h, None) for h in hs.candidates if h.id not in offered]
    others.sort(key=lambda c: (c.hypothesis.rejected_reason is not None, -c.hypothesis.total_score))
    return HypothesesView(
        stage=sid,
        version=envelope.version,
        versions=store.versions(OPTIONS),
        topic=hs.topic,
        shortlist=shortlist,
        others=others,
        selection=_selection(store, _selection_artifact(run), states[sid].gate if sid in states else None),
        weights=dict(getattr(config, "weights", None) or {}),
        budget_gpu_hours=run.config.budgets.run.max_gpu_hours,
        markdown=hs.markdown_path,
    )
