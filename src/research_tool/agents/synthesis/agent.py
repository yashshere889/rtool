"""Synthesis: a literature review in which every sentence rests on verified claims from the corpus.

The literature-only pipeline (pipelines/synthesis.yaml) ends here instead of in an experiment. For each
topic of the knowledge map the model writes a paragraph sentence by sentence and names the claims (C1,
C2, ...) each sentence rests on. Code keeps a sentence only if every claim it names was offered and is
verified (its quote was found in the paper) and every number it states appears in those claims; the
rest are dropped, with the reason, and counted. The review cites the claims' papers and ends with the
quotes it rests on.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ...core.agent import Agent
from ...core.artifacts import artifact_type
from ...core.context import AgentContext
from ...core.fs import atomic_write_text
from ...core.registry import register_agent
from ...llm.prompts import PromptLibrary
from ...schemas.common import ResearchBrief
from ...schemas.literature import Claim, Corpus, KnowledgeMap, Topic
from ...schemas.report import Check, Report, ReportSection
from ...tools.report.render import bibtex_subset, reference_list
from ..common import SYSTEM

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class SentenceDraft(BaseModel):
    text: str = Field(description="One sentence of the review, saying only what the listed findings say")
    claims: list[str] = Field(description='Ids of the findings this sentence rests on, at least one, e.g. ["C2"]')


class ParagraphDraft(BaseModel):
    sentences: list[SentenceDraft] = Field(min_length=1, max_length=14)


class KeptSentence(BaseModel):
    text: str
    claims: list[str]  # claim ids
    papers: list[str]  # bib keys of the claims' papers


class DroppedSentence(BaseModel):
    text: str
    claims: list[str]
    reason: str


class SynthesisSection(BaseModel):
    title: str
    sentences: list[KeptSentence] = Field(default_factory=list)


class ClaimRef(BaseModel):
    paper_id: str
    bib_key: str
    text: str
    quote: str


@artifact_type("synthesis")
class Synthesis(BaseModel):
    """The review's sentences with the claims each rests on, for checking and scoring (the text is final_report)."""

    topic: str
    sections: list[SynthesisSection]
    dropped: list[DroppedSentence] = Field(default_factory=list)
    claims: dict[str, ClaimRef] = Field(default_factory=dict)  # every claim offered, by id
    checks: list[Check] = Field(default_factory=list)
    markdown_path: str = ""

    def kept(self) -> list[KeptSentence]:
        return [s for section in self.sections for s in section.sentences]


class SynthesisConfig(BaseModel):
    max_claims_per_topic: int = Field(default=12, ge=1, le=40)


def numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text)}


def judge(draft: SentenceDraft, offered: dict[str, Claim]) -> str:
    """'' if the sentence may stay, else why it goes."""
    if not draft.claims:
        return "cites no finding"
    unknown = [c for c in draft.claims if c not in offered]
    if unknown:
        return f"cites findings that were not offered: {', '.join(unknown)}"
    unverified = [c for c in draft.claims if not offered[c].verified]
    if unverified:
        return f"cites findings whose quotes were not found in the paper: {', '.join(unverified)}"
    sources = " ".join(f"{offered[c].text} {offered[c].quote}" for c in draft.claims)
    extra = sorted(numbers(draft.text) - numbers(sources))
    if extra:
        return f"states numbers its findings do not: {', '.join(extra)}"
    return ""


@register_agent("synthesis")
class SynthesisAgent(Agent):
    description = "Writes a literature review in which every sentence rests on verified claims, with their quotes."
    consumes = ("research_brief", "corpus", "knowledge_map")
    produces = ("synthesis", "final_report")
    config_model = SynthesisConfig

    def run(self, ctx: AgentContext) -> None:
        cfg: SynthesisConfig = self.config
        brief: ResearchBrief = ctx.inputs.get("research_brief")
        corpus: Corpus = ctx.inputs.get("corpus")
        km: KnowledgeMap = ctx.inputs.get("knowledge_map")
        prompts = PromptLibrary.for_agent(__package__, "synthesis", pack=brief.domain_pack)
        keys = {p.id: p.bib_key for p in corpus.papers if p.bib_key}
        verified = [c for c in km.claims if c.verified and c.kind == "result" and c.paper_id in keys]
        ids = {f"C{i}": c for i, c in enumerate(verified, 1)}
        topics = km.topics or [Topic(label="Findings", summary="", paper_ids=sorted(keys))]
        sections, dropped, provenance = [], [], {}
        for topic in topics:
            offered = {cid: c for cid, c in ids.items() if c.paper_id in topic.paper_ids}
            offered = dict(list(offered.items())[: cfg.max_claims_per_topic])
            if not offered:
                continue
            prompt = prompts.render("paragraph.j2", topic=topic, claims=list(offered.items()), brief=brief)
            provenance["paragraph.j2"] = prompt.template_hash
            draft = ctx.llm.structured(ParagraphDraft, prompt.text, profile="write", system=SYSTEM)
            section = SynthesisSection(title=topic.label)
            for sentence in draft.sentences:
                reason = judge(sentence, offered)
                if reason:
                    dropped.append(DroppedSentence(text=sentence.text, claims=sentence.claims, reason=reason))
                    continue
                papers = list(dict.fromkeys(keys[offered[c].paper_id] for c in sentence.claims))
                section.sentences.append(
                    KeptSentence(text=sentence.text.strip(), claims=sentence.claims, papers=papers)
                )
            if section.sentences:
                sections.append(section)
        refs = {
            cid: ClaimRef(paper_id=c.paper_id, bib_key=keys[c.paper_id], text=c.text, quote=c.quote)
            for cid, c in ids.items()
        }
        synthesis = Synthesis(topic=brief.topic, sections=sections, dropped=dropped, claims=refs)
        synthesis.checks = checks(synthesis, len(ids))
        self.write(ctx, brief, corpus, synthesis, provenance)

    def write(
        self, ctx: AgentContext, brief: ResearchBrief, corpus: Corpus, synthesis: Synthesis, provenance: dict[str, Any]
    ) -> None:
        cited = list(dict.fromkeys(k for s in synthesis.kept() for k in s.papers))
        sections = [
            ReportSection(
                id=f"topic_{i}",
                title=section.title,
                template=" ".join(
                    f"{s.text.rstrip('.')} [{'; '.join('@' + k for k in s.papers)}]." for s in section.sentences
                ),
                origin="llm",
            )
            for i, section in enumerate(synthesis.sections, 1)
        ]
        used = sorted({c for s in synthesis.kept() for c in s.claims}, key=lambda c: int(c[1:]))
        evidence = "\n".join(
            f'- **{c}** [@{synthesis.claims[c].bib_key}]: {synthesis.claims[c].text} — "{synthesis.claims[c].quote}"'
            for c in used
        )
        sections.append(ReportSection(id="evidence", title="Evidence", template=evidence, origin="llm"))
        title = f"Literature review: {brief.topic}"
        body = "\n\n".join(f"## {s.title}\n\n{s.template}" for s in sections)
        markdown = f"# {title}\n\n{body}\n\n## References\n\n{reference_list(corpus, cited)}\n"
        out = ctx.workspace.path("review")
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out / "review.md", markdown)
        source_bib = ctx.absolute(corpus.bibtex_path) if corpus.bibtex_path else None
        bibtex = source_bib.read_text(encoding="utf-8") if source_bib is not None and source_bib.exists() else ""
        atomic_write_text(out / "refs.bib", bibtex_subset(bibtex, cited))
        synthesis.markdown_path = ctx.relative(out / "review.md")
        ctx.outputs.put("synthesis", synthesis, meta={"prompts": provenance})
        ctx.outputs.put(
            "final_report",
            Report(
                title=title,
                sections=sections,
                markdown_path=synthesis.markdown_path,
                bibtex_path=ctx.relative(out / "refs.bib"),
                citations=cited,
            ),
            meta={"review": synthesis.markdown_path},
        )


def checks(synthesis: Synthesis, offered: int) -> list[Check]:
    """What the researcher sees at the final checkpoint: how much the model wrote that its sources did not say."""
    kept, dropped = len(synthesis.kept()), len(synthesis.dropped)
    used = {c for s in synthesis.kept() for c in s.claims}
    return [
        Check(name="sentences resting on verified findings", ok=kept > 0, detail=f"{kept} kept, {dropped} dropped"),
        Check(
            name="findings used",
            ok=bool(used),
            detail=f"{len(used)} of {offered} verified findings are cited",
        ),
        Check(
            name="unsupported sentences",
            ok=dropped <= kept,
            detail="; ".join(f"{d.reason}" for d in synthesis.dropped[:3]),
        ),
    ]
