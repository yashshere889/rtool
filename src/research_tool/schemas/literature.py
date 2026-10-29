"""Artifacts produced by the Literature agent, and the small schemas its LLM calls fill in.

The LLM schemas are deliberately narrow (one paper, one decision at a time)
because the backend is a 12B model: small, checkable steps beat one big ask.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.artifacts import artifact_type

PaperStatus = Literal["candidate", "excluded", "included", "downloaded", "unavailable", "parsed", "extracted", "failed"]


class PaperRecord(BaseModel):
    id: str  # stable key: doi:..., arxiv:..., openalex:W..., s2:..., title:<hash>
    bib_key: str = ""
    title: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    oa_urls: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    reference_ids: list[str] = Field(default_factory=list)  # OpenAlex W-ids, for snowballing
    sources: list[str] = Field(default_factory=list)  # which APIs returned it
    found_by: list[str] = Field(default_factory=list)  # query text, "refs:<id>", "cites:<id>", "manual"
    round: int = 0
    relevance: float | None = None
    relevance_reason: str = ""
    model_score: float | None = None  # the model's own 0-10 at screening (relevance is the checklist's)
    status: PaperStatus = "candidate"
    note: str = ""
    override: Literal["include", "exclude"] | None = None  # the researcher's call; screening never changes it
    subtopics: list[str] = Field(default_factory=list)  # the search plan's must-cover sub-topics it covers (screening)
    file: str | None = None  # relative to the run directory
    file_sha256: str | None = None


class TopicScope(BaseModel):
    """How the Literature agent read the research topic, before searching: shown to the researcher, and used to
    write the queries and to screen every paper (a paper must be about the key concepts, not just share words)."""

    interpretation: str = Field(
        description="Two or three sentences: what the research topic is about, in the field's own terms"
    )
    concepts: list[str] = Field(
        min_length=1,
        max_length=4,
        description="2 to 4 distinct key concepts a relevant paper must be about, each a short specific phrase",
    )
    out_of_scope: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Neighbouring areas that share words with the topic but are not what it is about",
    )
    other_readings: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Other plausible meanings of the topic (a short topic usually has some); empty only if it is clear",
    )


@artifact_type("corpus")
class Corpus(BaseModel):
    topic: str
    queries: list[str]
    papers: list[PaperRecord]
    stats: dict[str, int] = Field(default_factory=dict)
    needs_manual: list[str] = Field(default_factory=list)  # included but no open-access copy found
    kb_path: str  # immutable knowledge-base snapshot, relative to the run directory
    kb_sha256: str
    bibtex_path: str
    scope: TopicScope | None = None  # how the topic was read (None for corpora built before it existed)
    must_cover: list[str] = Field(default_factory=list)  # sub-topics the search plan said the corpus should cover

    def included(self) -> list[PaperRecord]:
        return [p for p in self.papers if p.status not in ("candidate", "excluded")]


# ------------------------------------------------------------ LLM outputs
class SearchPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=12, description="Short keyword queries for paper search")
    year_from: int | None = Field(default=None, description="Ignore papers before this year, if the topic is recent")
    must_cover: list[str] = Field(default_factory=list, description="Sub-topics the corpus should include")


class ConceptMatch(BaseModel):
    concept: str = Field(description="The key concept, as listed")
    match: Literal["yes", "partly", "no"] = Field(
        description="yes = the paper is about it; partly = touches it or only uses it as a tool; no = not about it"
    )


class RelevanceJudgment(BaseModel):
    """One paper, screened. The reason comes first (a small model that writes the score first just justifies
    it), and with a topic scope the score is computed from the concept checklist, not taken from the model."""

    reason: str = Field(description="One or two sentences: what the paper studies, then how that relates to the topic")
    concepts: list[ConceptMatch] = Field(
        default_factory=list, max_length=6, description="One entry per key concept listed, in the same order"
    )
    subtopics: list[int] = Field(
        default_factory=list, description="Numbers of the listed sub-topics the paper covers; empty if none"
    )
    out_of_scope: bool = Field(
        default=False, description="true if the paper is mainly about one of the areas listed as out of scope"
    )
    score: int = Field(ge=0, le=10, description="Overall: 0 = unrelated, 10 = central to the research topic")


class KeyResult(BaseModel):
    claim: str = Field(description="One finding, in your words, with no number that is not in the quote")
    quote: str = Field(description="Sentence copied verbatim from the paper that supports the claim")
    method: str = Field(default="", description="Short name of the method the result is about, if any")
    dataset: str = Field(default="", description="Short name of the dataset or population, if any")
    metric: str = Field(default="", description="Short name of what was measured, if any")
    value: str = Field(default="", description="The number reported, exactly as written, if any")


class EvidenceItem(BaseModel):
    statement: str
    quote: str = Field(description="Sentence copied verbatim from the paper")


class PaperExtraction(BaseModel):
    problem: str = Field(description="The research problem, one or two sentences")
    approach: str = Field(description="What the authors did, one or two sentences")
    methods: list[str] = Field(
        default_factory=list,
        description="Names of the methods, models, algorithms or tools used or proposed: short names, not steps",
    )
    datasets: list[str] = Field(
        default_factory=list,
        description="Names of the datasets, benchmarks, environments or study populations: short names",
    )
    metrics: list[str] = Field(
        default_factory=list, description="Names of the evaluation metrics or outcome measures: short names"
    )
    key_results: list[KeyResult] = Field(default_factory=list, max_length=8)
    limitations: list[EvidenceItem] = Field(default_factory=list, max_length=5)
    future_work: list[EvidenceItem] = Field(default_factory=list, max_length=5)


class PaperDigest(BaseModel):
    """A reader's digest of one paper (prose by the model: shown as AI-written, not checked against quotes)."""

    summary: str = Field(description="2-3 plain sentences: what was studied, how, and the main finding")
    design: str = Field(description="The study design in one sentence (e.g. randomised crossover, simulation)")
    sample: str = Field(description="Who or what was studied (participants, datasets, systems), or 'not stated'")
    procedure: str = Field(description="The main steps, in 2-4 short sentences")
    gaps: list[str] = Field(
        default_factory=list, max_length=3, description="Open questions or weaknesses the paper itself leaves"
    )


class TopicLabel(BaseModel):
    label: str = Field(description="2-6 word name for the group")
    summary: str = Field(description="One sentence on what these papers have in common")


class OpenProblem(BaseModel):
    statement: str
    paper_ids: list[str] = Field(description="Ids of the papers (from the list given) that raise it")
    why_open: str


class OpenProblems(BaseModel):
    problems: list[OpenProblem] = Field(max_length=8)


class Overview(BaseModel):
    text: str = Field(description="State of the art in 2-4 paragraphs; cite papers as [bib_key]")


# ------------------------------------------------------- knowledge map artifact
class Claim(BaseModel):
    paper_id: str
    kind: Literal["result", "limitation", "future_work"]
    text: str
    quote: str
    verified: bool
    support: float  # 0..1, how well the quote matches the paper text
    method: str = ""
    dataset: str = ""
    metric: str = ""
    value: str = ""
    # numbers the claim states that its quote does not (any of them missing from the whole paper unverifies it)
    unquoted_numbers: list[str] = Field(default_factory=list)


class Topic(BaseModel):
    label: str
    summary: str
    paper_ids: list[str]


class Entity(BaseModel):
    kind: Literal["method", "dataset", "metric"]
    name: str
    aliases: list[str] = Field(default_factory=list)
    paper_ids: list[str]


class CoverageCell(BaseModel):
    method: str
    dataset: str
    paper_ids: list[str]
    results: list[str] = Field(default_factory=list)  # "metric=value (paper)"


class Gap(BaseModel):
    # thin_topic and unresolved_limitation are no longer produced (kept so older maps still load)
    kind: Literal["uncovered_subtopic", "untested_combination", "thin_topic", "stale_topic", "unresolved_limitation"]
    description: str
    evidence: list[str] = Field(default_factory=list)


class QualityCheck(BaseModel):
    """One measured property of the literature stage's output, for the researcher at the review gate."""

    check: str  # short name, e.g. "full text"
    value: str  # e.g. "71 of 100 papers"
    ok: bool
    note: str = ""  # what it means when it is not ok


@artifact_type("knowledge_map")
class KnowledgeMap(BaseModel):
    topic: str
    overview: str
    topics: list[Topic] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    coverage: list[CoverageCell] = Field(default_factory=list)
    open_problems: list[OpenProblem] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    timeline: dict[str, int] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    markdown_path: str = ""  # human-readable map for the review gate (relative to the run directory)
    scope: TopicScope | None = None
    quality: list[QualityCheck] = Field(default_factory=list)
