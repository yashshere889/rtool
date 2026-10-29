"""Hand edits of a report (or the final report): the researcher rewrites prose; rtool's fact sections stay.

The same rules as for the model apply, except the number rule: results may
only enter the text as placeholders and citations must be keys of papers in
the corpus, or the edit is refused; a number typed by hand is allowed but
listed as a finding (the Reviewer flags it too). Edited sections are marked
``origin: human``, so a later run of the Writer keeps them. The rendered
Markdown and the BibTeX of the cited papers are produced again.
"""

from __future__ import annotations

from typing import Any

from ...core.editing import EditContext, EditResult
from ...core.errors import ActionError, ArtifactError
from ...schemas.code import ExperimentResults
from ...schemas.literature import Corpus
from ...schemas.plan import ExperimentPlan
from ...schemas.report import Finding, Report
from .render import (
    Value,
    assemble,
    bare_numbers,
    bib_keys,
    bibtex_subset,
    citation_keys,
    placeholders,
    problems,
    reference_list,
    value_table,
)

HAND_CHECKED = ("citation", "number", "placeholder")  # findings about a section's text, redone for edited sections


def report_material(ctx: EditContext) -> tuple[dict[str, Value], set[str], Corpus, str]:
    """Placeholder values, citable keys, the corpus and its BibTeX, as the report's version was made from them."""
    inputs = ctx.previous.inputs

    def payload(name: str) -> dict[str, Any]:
        return ctx.store.envelope(name, inputs.get(name)).payload

    try:
        plan = ExperimentPlan.model_validate(payload("experiment_plan"))
        results = ExperimentResults.model_validate(payload("experiment_results"))
        corpus = Corpus.model_validate(payload("corpus"))
    except ArtifactError as exc:
        raise ActionError(f"the report's inputs cannot be read: {exc}") from None
    bib = ctx.run_root / corpus.bibtex_path if corpus.bibtex_path else None
    bibtex = bib.read_text(encoding="utf-8") if bib is not None and bib.is_file() else ""
    return value_table(plan, results), set(bib_keys(corpus)), corpus, bibtex


def _in_workspace(ctx: EditContext, run_relative: str) -> str:
    prefix = f"work/{ctx.stage}/"
    if not run_relative.startswith(prefix):
        raise ActionError(f"{run_relative} is not in the {ctx.stage} stage's folder")
    return run_relative.removeprefix(prefix)


def edit_report(value: dict[str, Any], ctx: EditContext) -> EditResult:
    previous = Report.model_validate(ctx.previous.payload)
    edited = Report.model_validate({**previous.model_dump(mode="json"), **value})
    if [s.id for s in edited.sections] != [s.id for s in previous.sections]:
        raise ActionError("sections cannot be added, removed or reordered; edit their text")
    values, keys, corpus, bibtex = report_material(ctx)

    sections, changed, typed = [], set(), []
    for new, old in zip(edited.sections, previous.sections, strict=True):
        if new.template.strip() == old.template.strip() and new.title.strip() == old.title:
            sections.append(old)
            continue
        if old.origin == "rtool":
            raise ActionError(f"“{old.title}” is written by rtool from the results and cannot be edited")
        found = problems(new.template, values, keys if old.id != "abstract" else set(), allowed_numbers=None)
        if found:
            raise ActionError(f"{old.title}: " + "; ".join(found[:3]))
        typed += [(old, snippet) for snippet in bare_numbers(new.template)]
        changed.add(old.id)
        update = {"template": new.template.strip(), "title": new.title.strip() or old.title, "origin": "human"}
        sections.append(old.model_copy(update=update))

    findings = [f for f in previous.findings if not (f.section in changed and f.kind in HAND_CHECKED)]
    findings += [
        Finding(
            kind="number",
            section=section.id,
            severity="minor",
            detail=f"number typed by hand instead of a placeholder: {snippet}",
        )
        for section, snippet in typed
    ]
    title = edited.title.strip() or previous.title
    cited = citation_keys("\n".join(s.template for s in sections))
    markdown = assemble(title, sections, values, reference_list(corpus, cited))
    report = previous.model_copy(
        update={
            "title": title,
            "title_origin": "human" if title != previous.title else previous.title_origin,
            "sections": sections,
            "citations": cited,
            "findings": findings,
            "numbers": sum(len(set(values) & set(placeholders(s.template))) for s in sections),
        }
    )
    files = {
        _in_workspace(ctx, previous.markdown_path): markdown,
        _in_workspace(ctx, previous.bibtex_path): bibtex_subset(bibtex, cited),
    }
    return EditResult(report, files=files)
