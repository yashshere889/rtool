"""Hypothesis agent: mine opportunities → draft → check grounding → dedupe → novelty → score → pick 3.

Designed for a 12B model: each LLM call sees one opportunity (or one
hypothesis) with the exact evidence it may cite, and every claim it makes is
checked mechanically afterwards — quotes against paper text, paper ids against
the corpus, compute against the configured partitions.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...core.context import AgentContext
from ...core.errors import StageFailed
from ...core.registry import register_agent
from ...knowledge.store import KnowledgeStore
from ...knowledge.vectors import HashingEmbedder, dot
from ...knowledge.verify import normalize, quote_support
from ...llm.prompts import PromptLibrary
from ...schemas.common import ResearchBrief
from ...schemas.hypothesis import (
    COMPUTE_HOURS,
    CheckedEvidence,
    Hypothesis,
    HypothesisDraft,
    HypothesisDrafts,
    HypothesisSet,
    NoveltyJudgment,
    RubricScores,
)
from ...schemas.literature import Corpus, KnowledgeMap, PaperRecord
from ..common import SYSTEM, compute_profile, open_corpus_kb

CRITERIA = ("novelty", "feasibility", "testability", "significance", "clarity")


class HypothesisConfig(BaseModel):
    shortlist: int = Field(default=3, ge=1, le=6)
    max_opportunities: int = Field(default=8, ge=1)
    duplicate_similarity: float = Field(default=0.85, gt=0, le=1)
    min_verified_evidence: int = Field(default=1, ge=0)
    quote_threshold: float = Field(default=0.85, gt=0, le=1)
    judge_samples: int | None = Field(default=None, ge=1, le=7)  # None = the "judge" profile's samples
    diversity: float = Field(default=0.3, ge=0, le=1)  # 0 = pure score ranking, 1 = strongly prefer variety
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "novelty": 0.3,
            "feasibility": 0.25,
            "testability": 0.2,
            "significance": 0.15,
            "clarity": 0.1,
        }
    )


@dataclass
class Opportunity:
    id: str
    kind: str
    description: str
    evidence: list[dict[str, str]] = field(default_factory=list)  # {"paper_id", "text"} — verbatim sentences
    paper_ids: list[str] = field(default_factory=list)


def mine_opportunities(km: KnowledgeMap, papers: dict[str, PaperRecord], limit: int) -> list[Opportunity]:
    """Where the literature leaves room: open problems, untested combinations, recurring limitations."""
    verified = [c for c in km.claims if c.verified and c.paper_id in papers]

    def quotes_from(paper_ids: list[str], kinds: tuple[str, ...]) -> list[dict[str, str]]:
        picked = [c for c in verified if c.paper_id in paper_ids and c.kind in kinds]
        return [{"paper_id": c.paper_id, "text": c.quote} for c in picked[:4]]

    found: list[Opportunity] = []
    for problem in km.open_problems:
        evidence = quotes_from(problem.paper_ids, ("limitation", "future_work", "result"))
        found.append(
            Opportunity(
                f"problem-{len(found) + 1}",
                "open_problem",
                f"Open problem: {problem.statement} (still open because: {problem.why_open})",
                evidence,
                list(problem.paper_ids),
            )
        )
    for gap in km.gaps:
        if gap.kind != "untested_combination":
            continue
        ids = [p for p in gap.evidence if p in papers]
        found.append(
            Opportunity(
                f"gap-{len(found) + 1}",
                "untested_combination",
                f"Gap: {gap.description}",
                quotes_from(ids, ("result", "limitation")),
                ids,
            )
        )
    covered = {pid for op in found for pid in op.paper_ids}
    for claim in verified:
        if claim.kind == "limitation" and claim.paper_id not in covered:
            covered.add(claim.paper_id)
            found.append(
                Opportunity(
                    f"limitation-{len(found) + 1}",
                    "limitation",
                    f"A limitation the authors report: {claim.text}",
                    [{"paper_id": claim.paper_id, "text": claim.quote}],
                    [claim.paper_id],
                )
            )
    return [op for op in found if op.evidence][:limit]


def mmr(items: list[Hypothesis], vectors: dict[str, list[float]], k: int, diversity: float) -> list[Hypothesis]:
    """Maximal marginal relevance: high scores, but not near-copies of each other."""
    pool = sorted(items, key=lambda h: h.total_score, reverse=True)
    chosen: list[Hypothesis] = []
    while pool and len(chosen) < k:

        def value(h: Hypothesis) -> float:
            similarity = max((dot(vectors[h.id], vectors[c.id]) for c in chosen), default=0.0)
            return (1 - diversity) * h.total_score - diversity * similarity

        best = max(pool, key=value)
        chosen.append(best)
        pool.remove(best)
    return chosen


@register_agent("hypothesis")
class HypothesisAgent(Agent):
    description = "Drafts, checks and ranks hypotheses grounded in the knowledge map; offers the best few."
    consumes = ("research_brief", "corpus", "knowledge_map")
    produces = ("hypothesis_set",)
    config_model = HypothesisConfig

    def run(self, ctx: AgentContext) -> None:
        cfg: HypothesisConfig = self.config
        brief: ResearchBrief = ctx.inputs.get("research_brief")
        corpus: Corpus = ctx.inputs.get("corpus")
        km: KnowledgeMap = ctx.inputs.get("knowledge_map")
        prompts = PromptLibrary.for_agent(__package__, "hypothesis", pack=brief.domain_pack)
        papers = {p.id: p for p in corpus.included()}
        kb = open_corpus_kb(ctx, corpus)
        try:
            run = _HypothesisRun(ctx, cfg, brief, km, papers, kb, prompts)
            hypothesis_set = run.execute()
        finally:
            if kb is not None:
                kb.close()
        markdown = ctx.workspace.write_text("hypotheses.md", render_markdown(hypothesis_set, papers))
        hypothesis_set.markdown_path = ctx.relative(markdown)
        ctx.outputs.put(
            "hypothesis_set", hypothesis_set, meta={"review": hypothesis_set.markdown_path, "prompts": run.provenance}
        )


class _HypothesisRun:
    def __init__(
        self,
        ctx: AgentContext,
        cfg: HypothesisConfig,
        brief: ResearchBrief,
        km: KnowledgeMap,
        papers: dict[str, PaperRecord],
        kb: KnowledgeStore | None,
        prompts: PromptLibrary,
    ) -> None:
        self.ctx, self.cfg, self.brief, self.km, self.papers, self.kb, self.prompts = (
            ctx,
            cfg,
            brief,
            km,
            papers,
            kb,
            prompts,
        )
        self.compute = compute_profile(ctx.app_config)
        self.feedback = [f.text for f in ctx.feedback]
        self.provenance: dict[str, str] = {}
        self._texts: dict[str, str] = {}

    def render(self, name: str, **variables: Any) -> str:
        prompt = self.prompts.render(name, **variables)
        self.provenance[name] = prompt.template_hash
        return prompt.text

    def paper_text(self, paper_id: str) -> str:
        if paper_id not in self._texts:
            full = self.kb.full_text(paper_id) if self.kb is not None else ""
            paper = self.papers.get(paper_id)
            self._texts[paper_id] = normalize(full or (paper.abstract if paper else ""))
        return self._texts[paper_id]

    # ----------------------------------------------------------- pipeline
    def execute(self) -> HypothesisSet:
        opportunities = mine_opportunities(self.km, self.papers, self.cfg.max_opportunities)
        if not opportunities:
            raise StageFailed(
                "the knowledge map has no verified evidence to build hypotheses on; "
                "reject the literature gate with feedback to widen the corpus",
                needs_human=True,
            )
        candidates = self.draft(opportunities)
        self.check_grounding(candidates)
        self.dedupe(candidates)
        live = [h for h in candidates if h.rejected_reason is None]
        self.check_novelty(live)
        self.check_feasibility(live)
        self.score([h for h in live if h.rejected_reason is None])
        eligible = [h for h in candidates if h.rejected_reason is None]
        if not eligible:
            reasons = sorted({h.rejected_reason or "" for h in candidates})
            raise StageFailed(f"no hypothesis survived the checks ({'; '.join(reasons)[:400]})", needs_human=True)
        vectors = dict(
            zip((h.id for h in eligible), HashingEmbedder().embed([h.statement for h in eligible]), strict=True)
        )
        shortlist = mmr(eligible, vectors, self.cfg.shortlist, self.cfg.diversity)
        self.ctx.emit(
            "hypotheses_ranked", drafted=len(candidates), eligible=len(eligible), shortlist=[h.id for h in shortlist]
        )
        return HypothesisSet(topic=self.brief.topic, hypotheses=shortlist, candidates=candidates)

    def draft(self, opportunities: list[Opportunity]) -> list[Hypothesis]:
        related = sorted(self.papers.values(), key=lambda p: p.relevance or 0, reverse=True)[:15]
        texts = [
            self.render(
                "draft.j2",
                topic=self.brief.topic,
                questions=self.brief.questions,
                feedback=self.feedback,
                opportunity=op,
                papers=related,
                compute=self.compute,
            )
            for op in opportunities
        ]
        results = self.ctx.llm.structured_many(HypothesisDrafts, texts, profile="reason", system=SYSTEM)
        candidates: list[Hypothesis] = []
        for op, result in zip(opportunities, results, strict=True):
            if isinstance(result, Exception):
                self.ctx.emit("hypothesis_draft_failed", opportunity=op.id, error=str(result)[:300])
                continue
            for draft in result.hypotheses:
                candidates.append(self.to_hypothesis(f"h{len(candidates) + 1}", draft, op))
        return candidates

    def to_hypothesis(self, hid: str, draft: HypothesisDraft, opportunity: Opportunity) -> Hypothesis:
        data = draft.model_dump(exclude={"evidence"})
        evidence = [
            CheckedEvidence(paper_id=e.paper_id, quote=e.quote, verified=False, support=0.0) for e in draft.evidence
        ]
        related = [pid for pid in opportunity.paper_ids if pid in self.papers]
        return Hypothesis(
            id=hid, evidence=evidence, source_opportunity=opportunity.id, related_paper_ids=related, **data
        )

    def check_grounding(self, candidates: list[Hypothesis]) -> None:
        for h in candidates:
            for item in h.evidence:
                if item.paper_id in self.papers:
                    item.support = quote_support(item.quote, "", normalized_source=self.paper_text(item.paper_id))
                    item.verified = item.support >= self.cfg.quote_threshold
            if sum(e.verified for e in h.evidence) < self.cfg.min_verified_evidence:
                h.rejected_reason = "no evidence quote could be found in the cited papers"

    def dedupe(self, candidates: list[Hypothesis]) -> None:
        live = [h for h in candidates if h.rejected_reason is None]
        vectors = HashingEmbedder().embed([h.statement for h in live])
        for i, h in enumerate(live):
            for j in range(i):
                if live[j].rejected_reason is None and dot(vectors[i], vectors[j]) >= self.cfg.duplicate_similarity:
                    h.rejected_reason = f"near-duplicate of {live[j].id}"
                    break

    def neighbours(self, h: Hypothesis, k: int = 3) -> list[dict[str, Any]]:
        ranked: list[str] = []
        if self.kb is not None:
            for hit in self.kb.search(h.statement, k=20, paper_ids=list(self.papers)):
                if hit["paper_id"] not in ranked:
                    ranked.append(hit["paper_id"])
        if len(ranked) < k:
            ids = [pid for pid in self.papers if pid not in ranked]
            embedder = HashingEmbedder()
            query = embedder.embed([h.statement])[0]
            vectors = embedder.embed([f"{self.papers[p].title} {self.papers[p].abstract}" for p in ids])
            ranked += [ids[i] for i, _ in sorted(enumerate(vectors), key=lambda iv: -dot(query, iv[1]))]
        out = []
        for pid in ranked[:k]:
            paper = self.papers[pid]
            out.append({"id": pid, "title": paper.title, "year": paper.year, "summary": paper.abstract[:400]})
        return out

    def check_novelty(self, live: list[Hypothesis]) -> None:
        texts = []
        for h in live:
            neighbours = self.neighbours(h)
            h.related_paper_ids = list(dict.fromkeys([*h.related_paper_ids, *(n["id"] for n in neighbours)]))
            texts.append(self.render("novelty.j2", statement=h.statement, neighbours=neighbours))
        results = self.ctx.llm.structured_many(NoveltyJudgment, texts, profile="judge", system=SYSTEM)
        for h, result in zip(live, results, strict=True):
            if isinstance(result, Exception):
                h.feasibility_notes.append("novelty check failed; treated as uncertain")
                continue
            h.novelty = result
            if not result.novel and result.closest_paper_id in self.papers:
                h.rejected_reason = f"already tested by {result.closest_paper_id}"

    def check_feasibility(self, live: list[Hypothesis]) -> None:
        budget = self.ctx.app_config.budgets.run.max_gpu_hours
        partitions = self.ctx.app_config.slurm.experiment_partitions
        for h in live:
            hours = COMPUTE_HOURS[h.compute]
            if budget is not None and hours > budget:
                h.feasibility_notes.append(f"needs up to {hours} GPU-hours; the budget is {budget:g}")
            if h.compute != "cpu_only" and self.ctx.app_config.slurm.partitions and not partitions:
                h.feasibility_notes.append("no GPU partitions are allowed for experiments")
            if not h.datasets:
                h.feasibility_notes.append("no dataset named yet; the planner must find one")

    def score(self, live: list[Hypothesis]) -> None:
        samples = self.cfg.judge_samples or self.ctx.app_config.llm.profile("judge").samples
        jobs = []
        for h in live:
            novelty = h.novelty.overlap if h.novelty else "not checked"
            text = self.render("rubric.j2", topic=self.brief.topic, compute=self.compute, h=h, novelty=novelty)
            jobs += [(h, text)] * samples
        results = self.ctx.llm.structured_many(RubricScores, [t for _, t in jobs], profile="judge", system=SYSTEM)
        by_hypothesis: dict[str, list[RubricScores]] = {}
        for (h, _), result in zip(jobs, results, strict=True):
            if not isinstance(result, Exception):
                by_hypothesis.setdefault(h.id, []).append(result)
        weights = self.cfg.weights
        for h in live:
            scores = by_hypothesis.get(h.id)
            if not scores:
                h.rejected_reason = "could not be scored"
                continue
            h.scores = {c: float(statistics.median(getattr(s, c) for s in scores)) for c in CRITERIA}
            h.rubric_comments = list(dict.fromkeys(s.comment.strip() for s in scores if s.comment.strip()))
            if any("GPU-hours" in note for note in h.feasibility_notes):
                h.scores["feasibility"] = min(h.scores["feasibility"], 2.0)
            total = sum(weights.get(c, 0) * (h.scores[c] - 1) / 4 for c in CRITERIA) / max(sum(weights.values()), 1e-9)
            grounded = sum(e.verified for e in h.evidence) / max(1, len(h.evidence))
            h.total_score = round(total * (0.75 + 0.25 * grounded), 4)  # partly unverified evidence costs a little


def render_markdown(hypothesis_set: HypothesisSet, papers: dict[str, PaperRecord]) -> str:
    lines = [
        f"# Hypotheses: {hypothesis_set.topic}",
        "",
        "Choose one with `rtool select <run> <number>` (or reject with feedback).",
        "",
    ]
    for n, h in enumerate(hypothesis_set.hypotheses):
        lines += [
            f"## [{n}] {h.title}  (score {h.total_score:.2f})",
            "",
            f"**Hypothesis.** {h.statement}",
            "",
            f"**Null.** {h.null_hypothesis}",
            "",
            f"**Why.** {h.rationale}",
            "",
            f"**Variables.** independent: {', '.join(h.independent_variables)}; "
            f"dependent: {', '.join(h.dependent_variables)}; controls: {', '.join(h.controls) or '-'}",
            "",
            f"**Data / methods.** {', '.join(h.datasets) or '-'} / {', '.join(h.methods) or '-'}",
            "",
            f"**Success criteria.** {h.success_criteria}",
            "",
            f"**Compute.** {h.compute}",
            "",
        ]
        if h.novelty:
            lines += [f"**Novelty.** {h.novelty.overlap} (score {h.novelty.score}/10)", ""]
        lines += ["**Evidence.**"]
        for e in h.evidence:
            title = papers[e.paper_id].title if e.paper_id in papers else "unknown paper"
            mark = "verified" if e.verified else "NOT found in paper"
            lines.append(f"- “{e.quote}” — {title} ({mark})")
        notes = h.feasibility_notes + [f"risk: {r}" for r in h.risks]
        if notes:
            lines += ["", "**Notes.**", *[f"- {n}" for n in notes]]
        scores = ", ".join(f"{k} {v:g}" for k, v in h.scores.items())
        lines += ["", f"_Rubric: {scores}_", ""]
        if h.rubric_comments:
            lines += ["**Judges.**", *[f"- {c}" for c in h.rubric_comments], ""]
    rejected = [h for h in hypothesis_set.candidates if h.rejected_reason]
    if rejected:
        lines += ["## Considered and rejected", ""]
        lines += [f"- {h.title}: {h.rejected_reason}" for h in rejected]
    return "\n".join(lines) + "\n"
