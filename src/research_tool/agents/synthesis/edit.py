"""Hand edits of a literature review: the sentences, each tied to the verified findings it rests on.

A researcher may rewrite a sentence, change which findings it cites, add sentences and remove them. Every sentence is
judged by the same rule the agent applied (its findings were offered and verified, and every number it states is in
them); a sentence that fails is refused with the reason, not silently dropped. The report (text, Markdown and BibTeX)
is rebuilt from the edited review, so the two never diverge. The review's topics, the findings offered and the
sentences the model wrote that were dropped are the agent's and stay as they are.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ...core.editing import EditContext, EditResult, artifact_editor
from ...core.errors import ActionError, ArtifactError
from ...schemas.common import ResearchBrief
from ...schemas.literature import Corpus
from ...schemas.report import Report
from ...validity.note import statement_for
from .agent import KeptSentence, SentenceDraft, Synthesis, SynthesisSection, checks, judge, render_review


@artifact_editor("synthesis", agent="synthesis")
def edit_review(value: dict[str, Any], ctx: EditContext) -> EditResult:
    previous = Synthesis.model_validate(ctx.previous.payload)
    edited = Synthesis.model_validate({**previous.model_dump(mode="json"), "sections": value.get("sections")})
    if [s.title for s in edited.sections] != [s.title for s in previous.sections]:
        raise ActionError("the review's topics cannot be added, removed or renamed: edit the sentences under them")
    offered = {  # every finding the agent offered was verified
        cid: SimpleNamespace(verified=True, text=ref.text, quote=ref.quote) for cid, ref in previous.claims.items()
    }
    before = {s.title: [(x.text, x.claims) for x in s.sentences] for s in previous.sections}
    sections: list[SynthesisSection] = []
    refused: list[str] = []
    for section in edited.sections:
        if not section.sentences:
            raise ActionError(f"“{section.title}” has no sentences: a topic needs at least one (or leave it as it was)")
        kept: list[KeptSentence] = []
        for n, sentence in enumerate(section.sentences, 1):
            text, claims = sentence.text.strip(), [c.strip() for c in sentence.claims if c.strip()]
            if not text:
                raise ActionError(f"sentence {n} of “{section.title}” is empty: remove it instead")
            reason = judge(SentenceDraft(text=text, claims=claims), offered)  # type: ignore[arg-type]
            if reason:
                refused.append(f"sentence {n} of “{section.title}”: {reason}")
                continue
            papers = list(dict.fromkeys(previous.claims[c].bib_key for c in claims))
            kept.append(KeptSentence(text=text, claims=claims, papers=papers))
        sections.append(SynthesisSection(title=section.title, sentences=kept))
    if refused:
        raise ActionError("; ".join(refused[:4]) + (f"; and {len(refused) - 4} more" if len(refused) > 4 else ""))
    review = previous.model_copy(update={"sections": sections})
    review.checks = checks(review, len(previous.claims))

    try:
        corpus = Corpus.model_validate(ctx.store.envelope("corpus", ctx.previous.inputs.get("corpus")).payload)
        brief = ResearchBrief.model_validate(
            ctx.store.envelope("research_brief", ctx.previous.inputs.get("research_brief")).payload
        )
        report = Report.model_validate(ctx.store.envelope("final_report").payload)
    except ArtifactError:
        raise ActionError("the review's sources are missing, so it cannot be rebuilt") from None
    source_bib = ctx.run_root / corpus.bibtex_path if corpus.bibtex_path else None
    bib_text = source_bib.read_text(encoding="utf-8") if source_bib is not None and source_bib.exists() else ""
    kept_validity = next((s.template for s in report.sections if s.id == "validity"), None)
    validity = kept_validity or statement_for(ctx.app_config.paths.resolved_runs_root(), brief.domain_pack)
    human = frozenset(
        s.title for s in sections if [(x.text, x.claims) for x in s.sentences] != before[s.title]
    ) | frozenset(s.title for s in report.sections if s.origin == "human")
    _, report_sections, markdown, bibtex, cited = render_review(brief.topic, corpus, review, validity, bib_text, human)
    review.markdown_path = previous.markdown_path
    rebuilt = report.model_copy(update={"sections": report_sections, "citations": cited})
    return EditResult(
        review,
        files={"review/review.md": markdown, "review/refs.bib": bibtex},
        extra={"final_report": rebuilt},
    )


@artifact_editor("final_report", agent="synthesis")
def refuse_report_edit(value: dict[str, Any], ctx: EditContext) -> EditResult:
    raise ActionError(
        "a literature review is edited sentence by sentence (the synthesis); its report is rebuilt from it"
    )
