"""The Literature page: the corpus, each paper's extracted facts, its text, full-text search, and adding papers.

Most per-paper facts (problem, approach, methods, the parsed text) live only in
the knowledge-base snapshot that comes with the corpus, so it is opened here
read-only, one connection per request. Nothing here calls the model.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ...core.artifacts import ArtifactStore
from ...core.errors import ArtifactError
from ...core.fs import atomic_write_json
from ...core.run import RunHandle
from ...knowledge.store import KnowledgeStore
from ...knowledge.verify import locate_quote
from ...schemas.literature import Claim, Corpus, KnowledgeMap, PaperRecord, QualityCheck, TopicScope
from ...tools.report.render import bibtex_subset
from ..deps import get_run

router = APIRouter(tags=["literature"])

MAX_UPLOAD_BYTES = 60 * 1024 * 1024
METHOD_SECTION = re.compile(r"method|approach|materials|experimental (design|setup)|procedure", re.I)
_SLUG = re.compile(r"[^A-Za-z0-9]+")

Read = Literal["full_text", "abstract", "none"]


class PaperLinks(BaseModel):
    doi: str | None = None
    arxiv: str | None = None
    openalex: str | None = None
    landing: str | None = None
    open_access: list[str] = Field(default_factory=list)


class PaperCard(BaseModel):
    id: str
    bib_key: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    abstract: str
    citation_count: int | None
    relevance: float | None
    relevance_reason: str
    status: str
    note: str
    file: str | None  # run-relative path of the copy, served by the files endpoint
    file_kind: Literal["pdf", "html"] | None
    read: Read  # how much of it the Literature agent read
    included: bool
    needs_upload: bool  # included, but no open-access copy was found
    found_by: list[str]
    topics: list[str]
    claims: dict[str, int]  # verified claims by kind: result, limitation, future_work
    links: PaperLinks
    summary: str  # the AI digest's summary ("" if there is none)
    override: Literal["include", "exclude"] | None  # kept or dropped by the researcher


class TopicView(BaseModel):
    label: str
    summary: str
    paper_ids: list[str]


class EntityView(BaseModel):
    kind: str
    name: str
    paper_ids: list[str]


class CoverageView(BaseModel):
    method: str
    dataset: str
    paper_ids: list[str]
    results: list[str]


class GapView(BaseModel):
    kind: str
    description: str
    paper_ids: list[str]


class OpenProblemView(BaseModel):
    statement: str
    why_open: str
    paper_ids: list[str]


class SubtopicView(BaseModel):
    name: str  # a sub-topic the search plan said the corpus should cover
    paper_ids: list[str]  # included papers that screening found to cover it


class LiteratureView(BaseModel):
    stage: str  # the stage that produced the corpus
    corpus_version: int
    map_version: int | None
    topic: str
    queries: list[str]
    stats: dict[str, int]
    papers: list[PaperCard]
    topics: list[TopicView]
    entities: list[EntityView]
    coverage: list[CoverageView]
    gaps: list[GapView]
    open_problems: list[OpenProblemView]
    timeline: dict[str, int]
    overview: str  # cites papers as [bib_key]; ``keys`` resolves them
    keys: dict[str, str]  # bib key -> paper id
    scope: TopicScope | None = None  # how the agent read the topic (None for corpora built before it existed)
    subtopics: list[SubtopicView] = Field(default_factory=list)
    quality: list[QualityCheck] = Field(default_factory=list)  # measured at the end of the stage


class DigestView(BaseModel):
    """The model's plain-language digest of a paper (AI-written: not checked against quotes)."""

    summary: str
    design: str
    sample: str
    procedure: str
    gaps: list[str]
    unverified_numbers: list[str]  # numbers in the digest that the paper text does not contain


class ClaimView(BaseModel):
    index: int  # position among this paper's claims (highlights refer to it)
    kind: str
    text: str
    quote: str
    verified: bool
    support: float
    method: str
    dataset: str
    metric: str
    value: str
    unquoted_numbers: list[str] = Field(default_factory=list)  # numbers the claim states that its quote does not


class PaperRef(BaseModel):
    id: str
    title: str
    year: int | None
    bib_key: str


class PaperDetail(BaseModel):
    paper: PaperCard
    problem: str
    approach: str
    methods: list[str]
    datasets: list[str]
    metrics: list[str]
    claims: list[ClaimView]
    methods_section: str  # the paper's own methods text (verbatim, may be cut)
    section_titles: list[str]
    gaps: list[GapView]
    open_problems: list[OpenProblemView]
    coverage: list[CoverageView]
    cites: list[PaperRef]  # papers in this corpus that this paper cites
    cited_by: list[PaperRef]  # papers in this corpus that cite this paper
    bibtex: str
    digest: DigestView | None


class TextSection(BaseModel):
    title: str
    text: str


class Highlight(BaseModel):
    claim: int
    section: int
    start: int
    end: int
    score: float


class PaperText(BaseModel):
    source: Read
    sections: list[TextSection]
    highlights: list[Highlight]
    focus: Highlight | None = None  # where the ``quote`` asked about was found (claim = -1)


class SearchHit(BaseModel):
    paper_id: str
    title: str
    bib_key: str
    section: str
    snippet: str


class Uploaded(BaseModel):
    file: str
    paper_id: str | None
    title: str


# ------------------------------------------------------------------ loading
class _Literature:
    def __init__(self, run: RunHandle, corpus_version: int | None = None) -> None:
        """The latest corpus, or ``corpus_version`` (e.g. the one a set of hypotheses was drawn from)."""
        store = ArtifactStore(run.artifacts_dir)
        try:
            corpus_env = store.envelope("corpus", corpus_version)
        except ArtifactError:
            raise HTTPException(404, "the literature stage has not produced a corpus yet") from None
        self.run = run
        self.corpus_version = corpus_env.version
        self.stage = corpus_env.producer.stage
        self.corpus = Corpus.model_validate(corpus_env.payload)
        try:
            map_env = store.envelope("knowledge_map")
            self.km: KnowledgeMap | None = KnowledgeMap.model_validate(map_env.payload)
            self.map_version: int | None = map_env.version
        except ArtifactError:
            self.km, self.map_version = None, None
        self.by_id = {p.id: p for p in self.corpus.papers}
        self.by_openalex = {p.openalex_id: p.id for p in self.corpus.papers if p.openalex_id}

    @contextlib.contextmanager
    def kb(self) -> Iterator[KnowledgeStore | None]:
        path = self.run.root / self.corpus.kb_path if self.corpus.kb_path else None
        if path is None or not path.is_file():
            yield None
            return
        store = KnowledgeStore(path, readonly=True)
        try:
            yield store
        finally:
            store.close()

    def claims_of(self, paper_id: str) -> list[Claim]:
        return [c for c in (self.km.claims if self.km else []) if c.paper_id == paper_id]

    def card(self, paper: PaperRecord, kb: KnowledgeStore | None) -> PaperCard:
        found = kb.extraction(paper.id) if kb is not None else None
        read: Read = "none" if found is None else "abstract" if found[1].get("abstract_only") else "full_text"
        digest = (found[0].get("digest") or {}) if found else {}
        included = paper.status not in ("candidate", "excluded")
        claims: dict[str, int] = {}
        for claim in self.claims_of(paper.id):
            if claim.verified:
                claims[claim.kind] = claims.get(claim.kind, 0) + 1
        suffix = Path(paper.file).suffix.lower() if paper.file else ""
        return PaperCard(
            id=paper.id,
            bib_key=paper.bib_key,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            venue=paper.venue,
            abstract=paper.abstract,
            citation_count=paper.citation_count,
            relevance=paper.relevance,
            relevance_reason=paper.relevance_reason,
            status=paper.status,
            note=paper.note,
            file=paper.file,
            file_kind="pdf" if suffix == ".pdf" else "html" if suffix in (".html", ".htm") else None,
            read=read,
            included=included,
            needs_upload=included and paper.status == "unavailable",
            found_by=paper.found_by,
            topics=[t.label for t in (self.km.topics if self.km else []) if paper.id in t.paper_ids],
            claims=claims,
            summary=str(digest.get("summary", "")),
            override=paper.override,
            links=PaperLinks(
                doi=f"https://doi.org/{paper.doi}" if paper.doi else None,
                arxiv=f"https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else None,
                openalex=f"https://openalex.org/{paper.openalex_id}" if paper.openalex_id else None,
                landing=paper.url,
                open_access=paper.oa_urls,
            ),
        )

    @staticmethod
    def how_read(kb: KnowledgeStore | None, paper: PaperRecord) -> Read:
        found = kb.extraction(paper.id) if kb is not None else None
        if found is None:
            return "none"
        return "abstract" if found[1].get("abstract_only") else "full_text"

    def gaps_for(self, paper_id: str | None = None) -> list[GapView]:
        out = []
        for gap in self.km.gaps if self.km else []:
            ids = [e for e in gap.evidence if e in self.by_id]
            if paper_id is None or paper_id in ids:
                out.append(GapView(kind=gap.kind, description=gap.description, paper_ids=ids))
        return out

    def problems_for(self, paper_id: str | None = None) -> list[OpenProblemView]:
        return [
            OpenProblemView(statement=p.statement, why_open=p.why_open, paper_ids=p.paper_ids)
            for p in (self.km.open_problems if self.km else [])
            if paper_id is None or paper_id in p.paper_ids
        ]

    def ref(self, paper_id: str) -> PaperRef:
        paper = self.by_id[paper_id]
        return PaperRef(id=paper.id, title=paper.title, year=paper.year, bib_key=paper.bib_key)


def _paper(lit: _Literature, paper_id: str) -> PaperRecord:
    paper = lit.by_id.get(paper_id)
    if paper is None:
        raise HTTPException(404, f"no paper {paper_id!r} in this corpus")
    return paper


# ------------------------------------------------------------------ routes
@router.get("/runs/{project}/{run_id}/literature", response_model=LiteratureView)
def literature(project: str, run_id: str, request: Request) -> LiteratureView:
    lit = _Literature(get_run(request, project, run_id))
    km = lit.km
    with lit.kb() as kb:
        cards = [lit.card(p, kb) for p in lit.corpus.papers]
    return LiteratureView(
        stage=lit.stage,
        corpus_version=lit.corpus_version,
        map_version=lit.map_version,
        topic=lit.corpus.topic,
        queries=lit.corpus.queries,
        stats=lit.corpus.stats,
        papers=cards,
        topics=[TopicView(label=t.label, summary=t.summary, paper_ids=t.paper_ids) for t in (km.topics if km else [])],
        entities=[EntityView(kind=e.kind, name=e.name, paper_ids=e.paper_ids) for e in (km.entities if km else [])],
        coverage=[
            CoverageView(method=c.method, dataset=c.dataset, paper_ids=c.paper_ids, results=c.results)
            for c in (km.coverage if km else [])
        ],
        gaps=lit.gaps_for(),
        open_problems=lit.problems_for(),
        timeline=km.timeline if km else {},
        overview=km.overview if km else "",
        keys={p.bib_key: p.id for p in lit.corpus.papers if p.bib_key},
        scope=lit.corpus.scope,
        subtopics=[
            SubtopicView(name=name, paper_ids=[p.id for p in lit.corpus.included() if name in p.subtopics])
            for name in lit.corpus.must_cover
        ],
        quality=km.quality if km else [],
    )


@router.get("/runs/{project}/{run_id}/literature/paper", response_model=PaperDetail)
def paper_detail(project: str, run_id: str, id: str, request: Request) -> PaperDetail:  # noqa: A002
    lit = _Literature(get_run(request, project, run_id))
    paper = _paper(lit, id)
    with lit.kb() as kb:
        card = lit.card(paper, kb)
        extraction = (kb.extraction(paper.id) if kb else None) or ({}, {})
        sections = kb.sections(paper.id) if kb else []
        edges = kb.citations() if kb else []
    data = extraction[0]
    methods_text = "\n\n".join(s.text for s in sections if METHOD_SECTION.search(s.title))
    cites = {lit.by_openalex[r] for r in paper.reference_ids if r in lit.by_openalex}
    cites |= {dst for src, dst in edges if src == paper.id and dst in lit.by_id}
    cited_by = {p.id for p in lit.corpus.papers if paper.openalex_id and paper.openalex_id in p.reference_ids}
    cited_by |= {src for src, dst in edges if dst == paper.id and src in lit.by_id}
    bibtex = ""
    bib_path = lit.run.root / lit.corpus.bibtex_path if lit.corpus.bibtex_path else None
    if paper.bib_key and bib_path is not None and bib_path.is_file():
        bibtex = bibtex_subset(bib_path.read_text(encoding="utf-8"), [paper.bib_key]).strip()
    digest = data.get("digest")
    return PaperDetail(
        paper=card,
        digest=DigestView.model_validate({"unverified_numbers": [], **digest}) if digest else None,
        problem=str(data.get("problem", "")),
        approach=str(data.get("approach", "")),
        methods=list(data.get("methods", [])),
        datasets=list(data.get("datasets", [])),
        metrics=list(data.get("metrics", [])),
        claims=[
            ClaimView(index=i, **c.model_dump(exclude={"paper_id"})) for i, c in enumerate(lit.claims_of(paper.id))
        ],
        methods_section=methods_text[:6000],
        section_titles=[s.title for s in sections],
        gaps=lit.gaps_for(paper.id),
        open_problems=lit.problems_for(paper.id),
        coverage=[
            CoverageView(method=c.method, dataset=c.dataset, paper_ids=c.paper_ids, results=c.results)
            for c in (lit.km.coverage if lit.km else [])
            if paper.id in c.paper_ids
        ],
        cites=[lit.ref(i) for i in sorted(cites - {paper.id})],
        cited_by=[lit.ref(i) for i in sorted(cited_by - {paper.id})],
        bibtex=bibtex,
    )


@router.get("/runs/{project}/{run_id}/literature/paper/text", response_model=PaperText)
def paper_text(
    project: str,
    run_id: str,
    id: str,  # noqa: A002
    request: Request,
    quote: str | None = Query(default=None, max_length=2000),
) -> PaperText:
    """The text the agent read (full text by section, or the abstract), with each claim's quote located in it.

    ``quote`` (e.g. a hypothesis's evidence) is located too, as ``focus``.
    """
    lit = _Literature(get_run(request, project, run_id))
    paper = _paper(lit, id)
    with lit.kb() as kb:
        sections = kb.sections(paper.id) if kb else []
        read = lit.how_read(kb, paper)
    if not sections:
        sections_out = [TextSection(title="Abstract", text=paper.abstract)] if paper.abstract else []
        source: Read = "abstract" if paper.abstract else "none"
    else:
        sections_out = [TextSection(title=s.title, text=s.text) for s in sections]
        source = "full_text" if read != "abstract" else "abstract"
    highlights = []
    for i, claim in enumerate(lit.claims_of(paper.id)):
        for s, section in enumerate(sections_out):
            found = locate_quote(claim.quote, section.text)
            if found is not None:
                start, end, score = found
                highlights.append(Highlight(claim=i, section=s, start=start, end=end, score=score))
                break
    focus = None
    for s, section in enumerate(sections_out if quote and quote.strip() else []):
        found = locate_quote(quote or "", section.text)
        if found is not None:
            focus = Highlight(claim=-1, section=s, start=found[0], end=found[1], score=found[2])
            break
    return PaperText(source=source, sections=sections_out, highlights=highlights, focus=focus)


@router.get("/runs/{project}/{run_id}/literature/search", response_model=list[SearchHit])
def search(project: str, run_id: str, q: str, request: Request, k: int = 20) -> list[SearchHit]:
    """Full-text search over every paper the agent read (BM25)."""
    lit = _Literature(get_run(request, project, run_id))
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]{2,}", q)]
    with lit.kb() as kb:
        hits = kb.search(q, k=max(1, min(k, 50))) if kb is not None and terms else []
    out = []
    for hit in hits:
        paper = lit.by_id.get(hit["paper_id"])
        if paper is None:
            continue
        text = " ".join(str(hit["text"]).split())
        lowered = text.lower()
        first = min((pos for t in terms if (pos := lowered.find(t)) >= 0), default=0)
        start = max(0, first - 120)
        snippet = ("…" if start else "") + text[start : start + 320] + ("…" if start + 320 < len(text) else "")
        out.append(
            SearchHit(
                paper_id=paper.id, title=paper.title, bib_key=paper.bib_key, section=hit["section"], snippet=snippet
            )
        )
    return out


@router.put("/runs/{project}/{run_id}/literature/inbox", response_model=Uploaded, status_code=201)
async def upload(
    project: str,
    run_id: str,
    request: Request,
    title: str,
    paper_id: str | None = None,
    doi: str | None = None,
) -> Uploaded:
    """A copy of a paper you have (e.g. paywalled, through your library), as the request body (PDF or HTML).

    It goes into the literature stage's inbox; the agent reads it the next time that stage runs.
    """
    run = get_run(request, project, run_id)
    stages = [s.id for s in run.spec.pipeline.stages if s.agent == "literature"]
    if not stages:
        raise HTTPException(409, "this run has no literature stage")
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    suffix = {"application/pdf": ".pdf", "text/html": ".html"}.get(content_type)
    if suffix is None:
        raise HTTPException(415, "send a PDF (application/pdf) or an HTML page (text/html)")
    title = title.strip()
    if not title:
        raise HTTPException(422, "the paper needs a title")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"papers are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if suffix == ".pdf" and not bytes(body[:1024]).lstrip().startswith(b"%PDF"):
        raise HTTPException(422, "this is not a PDF file")
    if (
        suffix == ".html"
        and b"<html" not in bytes(body[:4096]).lower()
        and b"<!doctype" not in bytes(body[:4096]).lower()
    ):
        raise HTTPException(422, "this is not an HTML page")
    inbox = run.work_dir(stages[0]) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    stem = _SLUG.sub("-", paper_id or title).strip("-")[:80] or "paper"
    target = inbox / f"{stem}{suffix}"
    sidecar = {k: v for k, v in {"title": title, "doi": doi, "paper_id": paper_id}.items() if v}
    atomic_write_json(target.with_suffix(".json"), sidecar)
    temp = inbox / f".{stem}{suffix}.upload"  # the agent skips dot-files, so it never sees a half-written copy
    temp.write_bytes(bytes(body))
    temp.replace(target)
    return Uploaded(file=str(target.relative_to(run.root)), paper_id=paper_id, title=title)
