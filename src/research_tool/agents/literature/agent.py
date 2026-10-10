"""Literature agent: plan → search → screen → snowball → download → parse → extract → embed → map.

Every phase is idempotent against the knowledge base (a SQLite file in the
stage workspace that survives across attempts), and progress is checkpointed,
so a killed episode or a re-run after the human rejects the literature gate
never repeats downloads or LLM reading it has already done. Only legal
open-access copies are downloaded; anything else is listed in
``corpus.needs_manual`` and can be dropped into ``inbox/`` by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...core.context import AgentContext
from ...core.errors import StageFailed
from ...core.registry import register_agent
from ...hpc.storage import QuotaError, ensure_room
from ...knowledge.store import KnowledgeStore
from ...knowledge.vectors import agglomerative, suggested_clusters
from ...llm.prompts import PromptLibrary
from ...schemas.common import ResearchBrief
from ...schemas.literature import (
    Corpus,
    KnowledgeMap,
    OpenProblems,
    Overview,
    PaperDigest,
    PaperExtraction,
    PaperRecord,
    RelevanceJudgment,
    SearchPlan,
    Topic,
    TopicLabel,
)
from ...tools.documents.convert import ConversionError, document_to_text
from ...tools.documents.sections import budgeted_text, chunk_sections, main_text, split_sections
from ...tools.http import HttpError
from ...tools.report.render import bare_numbers
from ...tools.scholarly.base import PaperHit, bib_key
from ...tools.scholarly.merge import PaperIndex
from . import mapping
from .toolkit import LiteratureToolkit, build_toolkit

SYSTEM = "You are a careful research assistant. Follow the instructions exactly and never invent facts or citations."
PHASES = ("plan", "search", "screen", "snowball", "download", "parse", "extract", "digest", "embed", "map")
INCLUDED = ("included", "downloaded", "unavailable", "parsed", "extracted", "failed")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class LiteratureConfig(BaseModel):
    sources: list[Literal["openalex", "semantic_scholar", "arxiv"]] = Field(
        default_factory=lambda: ["openalex", "semantic_scholar", "arxiv"]
    )
    max_queries: int = Field(default=8, ge=1, le=12)
    results_per_query: int = Field(default=25, ge=1, le=100)
    max_papers: int = Field(default=100, ge=1)  # focused corpus: ~30-100 papers
    min_papers: int = Field(default=30, ge=0)
    relevance_threshold: int = Field(default=6, ge=0, le=10)
    snowball_rounds: int = Field(default=2, ge=0)
    snowball_seeds: int = Field(default=10, ge=1)
    snowball_per_paper: int = Field(default=30, ge=1)
    year_from: int | None = None
    download: bool = True
    max_download_mb: int = Field(default=60, ge=1)
    extract_chars: int = Field(default=40_000, ge=2_000)  # ~10k tokens of paper text per extraction call
    digest: bool = True  # a short plain-language digest per paper (one small extra call each)
    digest_chars: int = Field(default=12_000, ge=1_000)
    min_text_chars: int = Field(default=1_500, ge=0)  # less than this after parsing = scanned PDF or bad download
    quote_threshold: float = Field(default=0.85, gt=0, le=1)
    overview_papers: int = Field(default=40, ge=1)
    local_kb_copy: bool = True  # work on a node-local copy of the knowledge base when available


def local_scratch() -> Path | None:
    """Barkla's per-node NVMe (``/tmp/users/$USER``) if present, else None."""
    explicit = os.environ.get("RTOOL_LOCAL_SCRATCH")
    if explicit:
        return Path(explicit) / "rtool-kb"
    candidate = Path("/tmp/users") / os.environ.get("USER", "")
    return candidate / "rtool-kb" if os.environ.get("USER") and candidate.is_dir() else None


def is_included(paper: PaperRecord) -> bool:
    return paper.status in INCLUDED


def paper_stem(paper_id: str) -> str:
    """File name (without extension) of a paper's copy: readable, and unique per id.

    Ids are full of dots (``doi:10.1038/...``, ``arxiv:2401.12345``), so never use
    ``Path.with_suffix`` on the result: it would cut the name at the first dot and
    let different papers overwrite each other.
    """
    digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:8]
    return f"{_SAFE.sub('_', paper_id)[:100]}-{digest}"


@register_agent("literature")
class LiteratureAgent(Agent):
    description = "Searches, screens and reads open-access papers; builds the corpus and knowledge map."
    consumes = ("research_brief",)
    produces = ("corpus", "knowledge_map")
    config_model = LiteratureConfig

    def run(self, ctx: AgentContext) -> None:
        config: LiteratureConfig = self.config
        cache = ctx.workspace.path("http_cache.sqlite")
        factory = ctx.tools.get("literature_toolkit")
        toolkit = factory(config, cache) if factory else build_toolkit(config, cache, ctx.app_config.knowledge)
        try:
            local = local_scratch() if config.local_kb_copy else None
            with KnowledgeStore.working_copy(ctx.workspace.path("kb.sqlite"), local) as kb:
                _LiteratureRun(ctx, config, toolkit, kb).execute()
        finally:
            toolkit.close()


class _LiteratureRun:
    def __init__(self, ctx: AgentContext, config: LiteratureConfig, toolkit: LiteratureToolkit, kb: KnowledgeStore):
        self.ctx = ctx
        self.cfg = config
        self.tools = toolkit
        self.kb = kb
        self.brief: ResearchBrief = ctx.inputs.get("research_brief")
        self.prompts = PromptLibrary.for_agent(__package__, "literature", pack=self.brief.domain_pack)
        # what the researcher wrote (choices made by clicking are applied below, not handed to the model)
        self.feedback = [f.text for f in ctx.feedback if not f.data.get("choices_only")]
        self.choices: dict[str, str] = {}  # paper id -> include | exclude, the latest decision wins
        for item in ctx.feedback:
            for choice in ("include", "exclude"):
                self.choices.update({str(pid): choice for pid in item.data.get(choice, []) or []})
        self.state = ctx.state
        self.index = PaperIndex(kb.papers())
        self.provenance: dict[str, Any] = self.state.setdefault("provenance", {})

    # ------------------------------------------------------------- plumbing
    def save(self) -> None:
        self.kb.upsert_papers(self.index.all())

    def checkpoint(self, sync: bool = False) -> None:
        self.save()
        self.ctx.checkpoint()
        if sync:
            self.kb.sync()

    def included(self) -> list[PaperRecord]:
        return [p for p in self.index.all() if is_included(p)]

    def render(self, name: str, **variables: Any) -> str:
        prompt = self.prompts.render(name, **variables)
        self.provenance[name] = prompt.template_hash
        return prompt.text

    def execute(self) -> None:
        self.record_choices()
        done = set(self.state.get("done", []))
        for phase in PHASES:
            if phase in done:
                continue
            getattr(self, f"phase_{phase}")()
            done.add(phase)
            self.state["done"] = sorted(done)
            self.ctx.emit("literature_phase_done", phase=phase, papers=len(self.index), included=len(self.included()))
            self.checkpoint(sync=True)

    # ----------------------------------------------------------------- plan
    def phase_plan(self) -> None:
        text = self.render("search_plan.j2", brief=self.brief, feedback=self.feedback, n_queries=self.cfg.max_queries)
        plan = self.ctx.llm.structured(SearchPlan, text, profile="reason", system=SYSTEM)
        queries = list(dict.fromkeys([*self.brief.seed_queries, *(q.strip() for q in plan.queries if q.strip())]))
        year_now = datetime.now(UTC).year
        year_from = self.cfg.year_from
        if year_from is None and plan.year_from and 1950 <= plan.year_from <= year_now - 2:
            year_from = plan.year_from  # the model's suggestion, if it isn't absurd
        self.state.update(
            queries=queries[: self.cfg.max_queries + len(self.brief.seed_queries)],
            year_from=year_from,
            must_cover=plan.must_cover,
        )
        self.ctx.emit("literature_plan", queries=self.state["queries"], year_from=year_from)

    # --------------------------------------------------------------- search
    def phase_search(self) -> None:
        searched = set(self.state.get("searched", []))
        for query in self.state["queries"]:
            for name, source in self.tools.sources.items():
                key = f"{name}|{query}"
                if key in searched:
                    continue
                try:
                    hits = source.search(query, self.cfg.results_per_query, year_from=self.state.get("year_from"))
                except HttpError as exc:
                    self.ctx.emit("literature_search_failed", source=name, query=query, error=str(exc))
                    hits = []
                for hit in hits:
                    self.index.add(hit, found_by=f"query:{query}")
                searched.add(key)
            self.state["searched"] = sorted(searched)
            self.checkpoint()
        self.ctx.emit("literature_search_done", candidates=len(self.index))

    # --------------------------------------------------------------- screen
    def phase_screen(self) -> None:
        self.screen(rescreen=bool(self.feedback) and not self.state.get("rescreened"))
        self.state["rescreened"] = True

    def record_choices(self) -> None:
        """Papers the researcher marked to keep or drop: remembered on the paper, so later screening respects them."""
        papers = {p.id: p for p in self.index.all()}
        applied = {pid: choice for pid, choice in self.choices.items() if pid in papers}
        for pid, choice in applied.items():
            papers[pid].override = choice  # type: ignore[assignment]
        if applied:
            self.apply_overrides()
            self.save()
            self.ctx.emit(
                "literature_choices_applied",
                kept=sum(c == "include" for c in applied.values()),
                dropped=sum(c == "exclude" for c in applied.values()),
            )

    def apply_overrides(self) -> None:
        for paper in self.index.all():
            if paper.override == "exclude" and paper.status != "excluded":
                paper.status, paper.note = "excluded", "dropped by you"
            elif paper.override == "include" and paper.status in ("candidate", "excluded"):
                paper.status, paper.note = "included", "kept by you"

    def screen(self, rescreen: bool = False) -> None:
        todo = [p for p in self.index.all() if p.relevance is None or (rescreen and p.status != "candidate")]
        if not todo:
            self.apply_overrides()
            return
        prompts = [self.render("relevance.j2", brief=self.brief, paper=p, feedback=self.feedback) for p in todo]
        results = self.ctx.llm.structured_many(RelevanceJudgment, prompts, profile="extract", system=SYSTEM)
        failures = 0
        for paper, result in zip(todo, results, strict=True):
            if isinstance(result, Exception):
                failures += 1
                paper.note = f"screening failed: {result}"[:200]
                continue
            paper.relevance, paper.relevance_reason = result.score, result.reason
            keep = result.score >= self.cfg.relevance_threshold
            if paper.status in ("candidate", "excluded", "included"):
                paper.status = "included" if keep else "excluded"
            elif not keep:
                paper.status = "excluded"  # already read, but no longer relevant
        self.apply_cap()
        self.apply_overrides()
        self.save()
        self.ctx.emit("literature_screened", screened=len(todo), failures=failures, included=len(self.included()))

    def apply_cap(self) -> None:
        ranked = sorted(self.included(), key=lambda p: (p.relevance or 0, p.citation_count or 0), reverse=True)
        kept = [p for p in ranked if p.override == "include"]  # the researcher's picks count first
        ranked = kept + [p for p in ranked if p.override != "include"]
        for paper in ranked[self.cfg.max_papers :]:
            paper.status = "excluded"
            paper.note = "below the corpus size cap"

    # ------------------------------------------------------------- snowball
    def neighbours(self, seed: PaperRecord) -> list[tuple[PaperHit, str]]:
        """References ("refs") and citing papers ("cites") from whichever source can provide them."""
        limit = self.cfg.snowball_per_paper
        found: list[tuple[PaperHit, str]] = []
        s2 = self.tools.sources.get("semantic_scholar")
        openalex = self.tools.sources.get("openalex")
        try:
            ref = s2.reference_for(seed) if s2 is not None else None
            if ref:
                found += [(h, "refs") for h in s2.references(ref, limit)]
                found += [(h, "cites") for h in s2.citations(ref, limit)]
            elif openalex is not None and seed.openalex_id:
                found += [(h, "refs") for h in openalex.by_ids(seed.reference_ids[:limit])]
                found += [(h, "cites") for h in openalex.citing(seed.openalex_id, limit)]
        except HttpError as exc:
            self.ctx.emit("literature_snowball_failed", paper=seed.id, error=str(exc))
        return found

    def phase_snowball(self) -> None:
        for round_number in range(1, self.cfg.snowball_rounds + 1):
            if len(self.included()) >= self.cfg.max_papers:
                break
            expanded = set(self.state.get("expanded", []))
            seeds = sorted(
                (p for p in self.included() if p.id not in expanded),
                key=lambda p: (p.relevance or 0, p.citation_count or 0),
                reverse=True,
            )[: self.cfg.snowball_seeds]
            if not seeds:
                break
            new, edges = 0, []
            for seed in seeds:
                for hit, direction in self.neighbours(seed):
                    record, created = self.index.add(hit, found_by=f"{direction}:{seed.id}", round_number=round_number)
                    new += int(created)
                    edges.append((seed.id, record.id) if direction == "refs" else (record.id, seed.id))
                expanded.add(seed.id)
                self.state["expanded"] = sorted(expanded)
                self.checkpoint()
            self.kb.add_citations(edges)
            self.ctx.emit("literature_snowball", round=round_number, seeds=len(seeds), new_candidates=new)
            if new == 0:
                break
            self.screen()
        if len(self.included()) < self.cfg.min_papers:
            self.ctx.emit("literature_small_corpus", included=len(self.included()), wanted=self.cfg.min_papers)

    # ------------------------------------------------------------- download
    def ingest_inbox(self, papers_dir: Path) -> None:
        """PDF/HTML files the researcher dropped into ``inbox/`` (optional sidecar ``<name>.json``)."""
        inbox = self.ctx.workspace.mkdir("inbox")
        for path in sorted(inbox.iterdir()):
            if path.suffix.lower() not in (".pdf", ".html", ".htm") or path.name.startswith("."):
                continue
            sidecar = path.with_suffix(".json")
            meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            record = self.index.records.get(meta.get("paper_id", ""))
            if record is None:
                hit = PaperHit(
                    source="manual",
                    source_id=path.stem,
                    title=meta.get("title") or path.stem,
                    doi=meta.get("doi"),
                    arxiv_id=meta.get("arxiv_id"),
                    year=meta.get("year"),
                )
                record, _ = self.index.add(hit, found_by="manual")
                record.relevance = record.relevance if record.relevance is not None else 10
            dest = papers_dir / f"{paper_stem(record.id)}{path.suffix.lower()}"
            shutil.move(str(path), dest)
            sidecar.unlink(missing_ok=True)
            record.file, record.status, record.note = self.ctx.relative(dest), "downloaded", "added by hand"
            self.ctx.emit("literature_manual_paper", paper=record.id)

    def candidate_urls(self, paper: PaperRecord) -> list[str]:
        urls = []
        if paper.arxiv_id:
            urls += [f"https://arxiv.org/html/{paper.arxiv_id}", f"https://arxiv.org/pdf/{paper.arxiv_id}"]
        urls += paper.oa_urls
        if paper.doi and self.tools.unpaywall is not None:
            try:
                urls += self.tools.unpaywall.oa_urls(paper.doi)
            except HttpError as exc:
                self.ctx.emit("literature_unpaywall_failed", paper=paper.id, error=str(exc))
        return list(dict.fromkeys(urls))

    def phase_download(self) -> None:
        papers_dir = self.ctx.workspace.mkdir("papers")
        self.ingest_inbox(papers_dir)
        todo = [p for p in self.included() if p.status == "included"]
        if not self.cfg.download or not todo:
            return
        try:
            ensure_room(papers_dir, extra_bytes=len(todo) * 5 * 1024 * 1024, extra_files=len(todo))
        except QuotaError as exc:
            raise StageFailed(f"not enough storage for {len(todo)} papers: {exc}", needs_human=True) from None
        max_bytes = self.cfg.max_download_mb * 1024 * 1024
        for n, paper in enumerate(todo, 1):
            stem = paper_stem(paper.id)
            for url in self.candidate_urls(paper):
                try:
                    result = self.tools.http.download(url, papers_dir / f"{stem}.part", max_bytes=max_bytes)
                except HttpError:
                    continue
                suffix = ".pdf" if result.content_type == "application/pdf" else ".html"
                final = papers_dir / f"{stem}{suffix}"
                result.path.replace(final)
                paper.file, paper.file_sha256, paper.status = self.ctx.relative(final), result.sha256, "downloaded"
                break
            else:
                paper.status, paper.note = "unavailable", "no open-access copy found"
            if n % 10 == 0:
                self.checkpoint()
        self.ctx.emit(
            "literature_downloaded",
            downloaded=sum(p.status == "downloaded" for p in todo),
            unavailable=sum(p.status == "unavailable" for p in todo),
        )

    # ---------------------------------------------------------------- parse
    def phase_parse(self) -> None:
        for paper in [p for p in self.included() if p.status == "downloaded" and p.file]:
            try:
                text = document_to_text(self.ctx.absolute(paper.file), pdf_converter=self.tools.pdf_converter)
            except (ConversionError, OSError) as exc:
                paper.status, paper.note = "failed", f"could not read file: {exc}"[:200]
                continue
            body, _references = main_text(split_sections(text))
            if sum(len(s.text) for s in body) < self.cfg.min_text_chars:
                paper.status, paper.note = "failed", "too little text extracted (scanned PDF?)"
                continue
            self.kb.set_sections(paper.id, body)
            self.kb.set_chunks(paper.id, chunk_sections(body))
            paper.status = "parsed"
        self.save()

    # -------------------------------------------------------------- extract
    def phase_extract(self) -> None:
        done = set(self.kb.extractions())
        todo: list[tuple[PaperRecord, str, bool]] = []
        for paper in self.included():
            if paper.id in done:
                continue
            if paper.status == "parsed":
                todo.append((paper, budgeted_text(self.kb.sections(paper.id), self.cfg.extract_chars), False))
            elif paper.status in ("unavailable", "failed", "included") and paper.abstract:
                todo.append((paper, paper.abstract, True))
        if not todo:
            return
        prompts = [
            self.render("extract.j2", topic=self.brief.topic, title=p.title, text=text, abstract_only=abstract_only)
            for p, text, abstract_only in todo
        ]
        results = self.ctx.llm.structured_many(PaperExtraction, prompts, profile="extract", system=SYSTEM)
        failures = 0
        for (paper, text, abstract_only), result in zip(todo, results, strict=True):
            if isinstance(result, Exception):
                failures += 1
                paper.note = f"extraction failed: {result}"[:200]
                continue
            meta = {"abstract_only": abstract_only, "chars": len(text), "prompt_hash": self.provenance["extract.j2"]}
            self.kb.put_extraction(paper.id, result.model_dump(), meta)
            if paper.status == "parsed":
                paper.status = "extracted"
        self.save()
        self.ctx.emit("literature_extracted", papers=len(todo) - failures, failures=failures)

    # --------------------------------------------------------------- digest
    def phase_digest(self) -> None:
        """A digest per paper (summary, design, sample, procedure, gaps) for the reader: model prose, so the UI
        marks it as AI-written, and numbers in it that the paper text does not contain are listed.

        Papers that already have one are skipped, so an older knowledge base is filled in on the next run.
        """
        if not self.cfg.digest:
            return
        todo: list[tuple[PaperRecord, dict[str, Any], dict[str, Any], str]] = []
        for paper in self.included():
            found = self.kb.extraction(paper.id)
            if found is None or "digest" in found[0]:
                continue
            sections = self.kb.sections(paper.id)
            text = budgeted_text(sections, self.cfg.digest_chars) if sections else paper.abstract
            if text:
                todo.append((paper, *found, text))
        if not todo:
            return
        prompts = [
            self.render(
                "digest.j2",
                topic=self.brief.topic,
                title=paper.title,
                text=text,
                facts=PaperExtraction.model_validate(data),
            )
            for paper, data, _meta, text in todo
        ]
        results = self.ctx.llm.structured_many(PaperDigest, prompts, profile="extract", system=SYSTEM)
        failures = 0
        for (paper, data, meta, text), result in zip(todo, results, strict=True):
            if isinstance(result, Exception):
                failures += 1
                continue
            prose = " ".join([result.summary, result.design, result.sample, result.procedure, *result.gaps])
            numbers = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}
            digest = {**result.model_dump(), "unverified_numbers": bare_numbers(prose, numbers)}
            meta = {**meta, "digest_hash": self.provenance["digest.j2"]}
            self.kb.put_extraction(paper.id, {**data, "digest": digest}, meta)
        self.ctx.emit("literature_digested", papers=len(todo) - failures, failures=failures)

    # ---------------------------------------------------------------- embed
    def phase_embed(self) -> None:
        extractions = self.kb.extractions()
        papers = self.included()
        texts = []
        for paper in papers:
            extra = extractions.get(paper.id, {})
            texts.append(" ".join([paper.title, paper.abstract, extra.get("problem", ""), extra.get("approach", "")]))
        if papers:
            vectors = self.tools.embedder.embed(texts)
            self.kb.put_vectors(
                "paper", self.tools.embedder.name, dict(zip((p.id for p in papers), vectors, strict=True))
            )

    # ------------------------------------------------------------------ map
    def phase_map(self) -> None:
        papers = {p.id: p for p in sorted(self.included(), key=lambda p: (p.year or 0, p.title))}
        taken: set[str] = set()
        for paper in papers.values():
            paper.bib_key = bib_key(paper.authors, paper.year, paper.title, taken)
        extractions = {
            pid: PaperExtraction.model_validate(data) for pid, data in self.kb.extractions().items() if pid in papers
        }
        texts = {pid: self.kb.full_text(pid) or papers[pid].abstract for pid in extractions}
        claims = mapping.build_claims(extractions, texts, self.cfg.quote_threshold)
        entities = mapping.build_entities(extractions)
        coverage = mapping.build_coverage(extractions, entities)
        topics = self.label_topics(papers)
        problems = self.open_problems(papers, claims)
        overview, unknown_citations = self.overview(papers, extractions, claims)
        gaps = mapping.find_gaps(entities, coverage, topics, papers)

        statuses = [p.status for p in self.index.all()]
        stats: dict[str, Any] = {
            "candidates": len(statuses),
            "included": len(papers),
            "excluded": statuses.count("excluded"),
            "full_text": sum(p.status == "extracted" for p in papers.values()),
            "abstract_only": sum(p.status != "extracted" for p in papers.values()),
            "unavailable": statuses.count("unavailable"),
            "claims": len(claims),
            "verified_claims": sum(c.verified for c in claims),
            "unknown_citations_removed": unknown_citations,
        }
        knowledge_map = KnowledgeMap(
            topic=self.brief.topic,
            overview=overview,
            topics=topics,
            entities=entities,
            claims=claims,
            coverage=coverage,
            open_problems=problems,
            gaps=gaps,
            timeline=mapping.timeline(papers.values()),
            stats=stats,
        )
        markdown = self.ctx.workspace.write_text("knowledge_map.md", mapping.render_markdown(knowledge_map, papers))
        knowledge_map.markdown_path = self.ctx.relative(markdown)
        bib = self.ctx.workspace.write_text("refs.bib", mapping.bibtex(papers.values()))
        self.save()
        snapshot = self.ctx.workspace.path(f"snapshots/kb-attempt{self.ctx.attempt}.sqlite")
        digest = self.kb.copy_to(snapshot)
        corpus = Corpus(
            topic=self.brief.topic,
            queries=list(self.state.get("queries", [])),
            papers=sorted(self.index.all(), key=lambda p: (not is_included(p), -(p.relevance or 0), p.title)),
            stats={k: v for k, v in stats.items() if isinstance(v, int)},
            needs_manual=[p.id for p in papers.values() if p.status == "unavailable"],
            kb_path=self.ctx.relative(snapshot),
            kb_sha256=digest,
            bibtex_path=self.ctx.relative(bib),
        )
        meta = {"review": knowledge_map.markdown_path, "prompts": dict(self.provenance)}
        self.ctx.outputs.put("corpus", corpus, meta=meta)
        self.ctx.outputs.put("knowledge_map", knowledge_map, meta=meta)

    def label_topics(self, papers: dict[str, PaperRecord]) -> list[Topic]:
        vectors = self.kb.vectors("paper", self.tools.embedder.name)
        ids = [pid for pid in papers if pid in vectors]
        if not ids:
            return []
        groups = [
            [ids[i] for i in group] for group in agglomerative([vectors[i] for i in ids], suggested_clusters(len(ids)))
        ]
        prompts = [self.render("topic_label.j2", papers=[papers[p] for p in group][:15]) for group in groups]
        labels = self.ctx.llm.structured_many(TopicLabel, prompts, profile="extract", system=SYSTEM)
        topics = []
        for n, (group, label) in enumerate(zip(groups, labels, strict=True), 1):
            if isinstance(label, Exception):
                label = TopicLabel(label=f"Topic {n}", summary="(labelling failed)")
            topics.append(Topic(label=label.label, summary=label.summary, paper_ids=group))
        return topics

    def open_problems(self, papers: dict[str, PaperRecord], claims: list[Any]) -> list[Any]:
        items = [
            {"paper_id": c.paper_id, "text": c.text}
            for c in claims
            if c.verified and c.kind in ("limitation", "future_work")
        ][:120]
        if not items:
            return []
        text = self.render("open_problems.j2", topic=self.brief.topic, items=items, max_problems=8)
        result = self.ctx.llm.structured(OpenProblems, text, profile="reason", system=SYSTEM)
        kept = []
        for problem in result.problems:
            problem.paper_ids = [p for p in problem.paper_ids if p in papers]  # drop ids the model made up
            if problem.paper_ids:
                kept.append(problem)
        return kept

    def overview(
        self, papers: dict[str, PaperRecord], extractions: dict[str, PaperExtraction], claims: list[Any]
    ) -> tuple[str, int]:
        ranked = sorted(papers.values(), key=lambda p: (p.relevance or 0, p.citation_count or 0), reverse=True)
        digest = []
        for paper in ranked[: self.cfg.overview_papers]:
            extraction = extractions.get(paper.id)
            results = [c.text for c in claims if c.paper_id == paper.id and c.kind == "result" and c.verified][:3]
            digest.append(
                {
                    "key": paper.bib_key,
                    "title": paper.title,
                    "year": paper.year,
                    "problem": extraction.problem if extraction else paper.abstract[:300],
                    "approach": extraction.approach if extraction else "",
                    "results": results,
                }
            )
        if not digest:
            return "", 0
        text = self.render("overview.j2", topic=self.brief.topic, papers=digest)
        result = self.ctx.llm.structured(Overview, text, profile="write", system=SYSTEM)
        return mapping.check_citations(result.text, {p.bib_key for p in papers.values()})
