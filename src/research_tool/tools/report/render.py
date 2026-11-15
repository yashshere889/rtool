"""Report templates -> Markdown: result placeholders, citations, the number rule, references.

Templates are Markdown in which results appear only as placeholders
(``{{mean:acc:new}}``, ``{{p:acc:new}}``, ...) and prior work only as pandoc
citations (``[@key]``, ``[@a; @b]``). The number rule: any other number in a
section must come from the material that section was written from (e.g. a
verified quote from a paper), so nothing in the report is made up.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ...schemas.code import ExperimentResults
from ...schemas.literature import Corpus
from ...schemas.plan import ExperimentPlan
from ...schemas.report import ReportSection
from .numbers import fmt, fmt_ci

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_]+(?::[A-Za-z0-9_.\-]+)*)\s*\}\}")
CITATION = re.compile(r"\[\s*(@[^\]]+?)\s*\]")
_KEY = re.compile(r"@([A-Za-z0-9_:.\-]+)")
_CODE = re.compile(r"`[^`\n]*`")
_LINK_TARGET = re.compile(r"\]\([^)]*\)|https?://\S+")
_NUMBER = re.compile(r"(?<![\w.@/])(?<![A-Za-z]-)(\d+(?:\.\d+)?)")
# "95% CI {{ci:...}}" and "95% CI [{{ci:...}}]" render the label twice: the value already carries it
_DOUBLE_CI = re.compile(r"95\s?%\s?(?:CI|confidence interval)[:,]?\s*(\[\s*)?(95% CI \[[^\[\]]*\])(?(1)\s*\])", re.I)
# "p = {{p:...}}" renders "p = p = 0.008": the value already says "p ="
_DOUBLE_P = re.compile(r"\bp\s*(?:=|<)\s*(?=p (?:=|<) )")
# values that carry their noun: "{{pts:...}} percentage points" and "{{conditions}} conditions" say it twice
_DOUBLE_NOUN = re.compile(r"\b(percentage points|conditions|seeds)(?:\s+\1\b)+")
UNRESOLVED = "[?]"  # what a number nobody could check becomes when it cannot simply be left out
_UNRESOLVED_ASIDE = re.compile(r"\s*\([^()]*\[\?\][^()]*\)")


@dataclass(frozen=True)
class Value:
    text: str  # what the placeholder renders to
    meaning: str  # shown to the model so it picks the right one


def _p_text(p: float | None) -> str:
    if p is None:
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def movement(difference: float) -> str:
    """The word for a change, so a fall is written as a fall and not as "a reduction of -7.5%"."""
    return "rise" if difference > 0 else "fall" if difference < 0 else "change"


def value_table(plan: ExperimentPlan, results: ExperimentResults) -> dict[str, Value]:
    """Every placeholder a report may use, with its rendered text."""
    values: dict[str, Value] = {}
    for s in results.summaries:
        key = f"{s.metric}:{s.condition}"
        values[f"mean:{key}"] = Value(fmt(s.mean), f"mean {s.metric} of {s.condition} ({s.n} runs)")
        values[f"ci:{key}"] = Value(f"95% CI {fmt_ci(s.ci_low, s.ci_high)}", "95% CI of that mean")
        values[f"std:{key}"] = Value(fmt(s.std), f"standard deviation of {s.metric} for {s.condition}")
        values[f"n:{key}"] = Value(str(s.n), f"successful runs of {s.condition}")
    for c in results.comparisons:
        key = f"{c.metric}:{c.condition}"
        what = f"{c.condition} minus {c.baseline} on {c.metric}"
        values[f"diff:{key}"] = Value(fmt(c.difference), f"difference in means, {what}")
        values[f"diff_ci:{key}"] = Value(f"95% CI {fmt_ci(c.ci_low, c.ci_high)}", f"95% CI of the difference, {what}")
        values[f"p:{key}"] = Value(_p_text(c.p_value), f"{c.test} p-value, {what}")
        if c.p_adjusted is not None:
            values[f"p_adj:{key}"] = Value(_p_text(c.p_adjusted), f"corrected p-value, {what}")
        values[f"d:{key}"] = Value(fmt(c.effect_size), f"effect size (Cohen's d), {what}")
        # sizes without a sign, after a word for the direction: signed ones read as "-5.3 points lower"
        moved, base = movement(c.difference), results.summary(c.baseline, c.metric)
        if base is not None and base.mean:
            values[f"rel:{key}"] = Value(
                f"{abs(c.difference) / abs(base.mean):.1%}",
                f"size of the relative {moved}, {what}, as a share of the baseline mean (no sign: say it is a {moved})",
            )
        mean = results.summary(c.condition, c.metric)
        if base is not None and mean is not None and all(0 <= m <= 1 for m in (base.mean, mean.mean)):
            values[f"pts:{key}"] = Value(
                f"{abs(c.difference) * 100:.1f} percentage points",
                f"size of the absolute {moved} in percentage points, {what} (not the relative {moved}; no sign)",
            )
    ok = sum(1 for r in results.runs if r.status == "ok")
    values["runs_ok"] = Value(str(ok), "experiment runs that produced results")
    values["runs_total"] = Value(str(len(results.runs)), "experiment runs planned")
    # counts carry their noun, so a bare placeholder still reads "across 2 conditions", not "across 2."
    n = len(plan.seeds)
    values["seeds"] = Value(f"{n} seed{'s' * (n != 1)}", "seeds (independent repetitions) per condition, with the word")
    n = len(plan.run_conditions())
    values["conditions"] = Value(f"{n} condition{'s' * (n != 1)}", "conditions, ablations included, with the word")
    return values


# ------------------------------------------------------------ rendering
def placeholders(template: str) -> list[str]:
    return [m.group(1) for m in PLACEHOLDER.finditer(template)]


def render(template: str, values: dict[str, Value]) -> str:
    text = PLACEHOLDER.sub(lambda m: values[m.group(1)].text if m.group(1) in values else m.group(0), template)
    return _DOUBLE_NOUN.sub(r"\1", _DOUBLE_P.sub("", _DOUBLE_CI.sub(r"\2", text)))


def plan_numbers(plan: ExperimentPlan) -> str:
    """Facts from the plan that prose may repeat as typed numbers: agent counts in condition names and
    descriptions, the levels of the independent variables, seeds, alpha."""
    parts = [plan.design]
    for c in plan.run_conditions():
        parts += [c.name.replace("_", " "), c.description]
    for v in plan.independent_variables:
        parts += [" ".join(value.strip("./ ") for value in v.values)]
    parts += [f"{len(plan.seeds)} seeds", *(f"alpha {t.alpha:g}" for t in plan.statistical_tests)]
    return "\n".join(parts)


def unresolved_count(text: str) -> int:
    return text.count(UNRESOLVED)


def citation_keys(text: str) -> list[str]:
    keys: list[str] = []
    for group in CITATION.findall(text):
        for key in _KEY.findall(group):
            if key not in keys:
                keys.append(key)
    return keys


def _floats(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(text)}


def bare_numbers(template: str, allowed: Iterable[float] = ()) -> list[str]:
    """Numbers typed into the text instead of a placeholder (years, '95%' and ``allowed`` excepted)."""
    allowed_set = set(allowed)
    text = PLACEHOLDER.sub(" ", template)
    text = CITATION.sub(" ", text)
    text = _CODE.sub(" ", text)
    text = _LINK_TARGET.sub(" ", text)
    found = []
    for match in _NUMBER.finditer(text):
        number = match.group(1)
        value = float(number)
        if (number.isdigit() and len(number) == 4 and 1900 <= value <= 2099) or value == 95:
            continue
        if any(abs(value - a) <= 1e-9 * max(1.0, abs(a)) for a in allowed_set):
            continue
        start, end = max(0, match.start() - 30), min(len(text), match.end() + 30)
        found.append("…" + " ".join(text[start:end].split()) + "…")
    return found


def problems(
    template: str, values: dict[str, Value], keys: set[str], *, allowed_numbers: str | None = None
) -> list[str]:
    """Everything that stops a section template from being rendered honestly.

    ``allowed_numbers`` is the material the section was written from: numbers that appear there
    may be repeated (e.g. figures from a cited paper). ``None`` disables the number rule.
    """
    found = [f"unknown placeholder {{{{{p}}}}}" for p in placeholders(template) if p not in values]
    found += [f"unknown citation key @{k} (cite only the keys given)" for k in citation_keys(template) if k not in keys]
    if allowed_numbers is not None:
        found += [
            f"number typed instead of a placeholder: {snippet}"
            for snippet in bare_numbers(template, _floats(allowed_numbers))
        ]
    return found


def strip_problems(template: str, values: dict[str, Value], keys: set[str], allowed_numbers: str | None) -> str:
    """Last resort after failed repairs: drop unknown placeholders/citations and typed numbers."""
    text = PLACEHOLDER.sub(lambda m: m.group(0) if m.group(1) in values else UNRESOLVED, template)

    def keep_known(match: re.Match[str]) -> str:
        known = [k for k in _KEY.findall(match.group(2)) if k in keys]
        return f"{match.group(1)}[{'; '.join('@' + k for k in known)}]" if known else ""  # no stray space left

    text = re.sub(r"(\s*)" + CITATION.pattern, keep_known, text)
    if allowed_numbers is not None:
        allowed = _floats(allowed_numbers)
        protected = [m.span() for p in (PLACEHOLDER, CITATION, _CODE, _LINK_TARGET) for m in p.finditer(text)]

        def scrub(match: re.Match[str]) -> str:
            value = float(match.group(1))
            inside = any(a <= match.start() < b for a, b in protected)
            keep = inside or value == 95 or (match.group(1).isdigit() and 1900 <= value <= 2099)
            keep = keep or any(abs(value - a) <= 1e-9 * max(1.0, abs(a)) for a in allowed)
            return match.group(0) if keep else UNRESOLVED

        text = _NUMBER.sub(scrub, text)
    return _UNRESOLVED_ASIDE.sub("", text)  # "(p < [?])" goes; a [?] in the middle of a sentence stays visible


# --------------------------------------------------------------- references
def bib_keys(corpus: Corpus) -> dict[str, str]:
    """bib key -> paper id, for every paper in the corpus that has one."""
    return {p.bib_key: p.id for p in corpus.included() if p.bib_key}


def to_pandoc_citations(text: str, keys: set[str]) -> str:
    """'[smith2020]' (the knowledge map's style) -> '[@smith2020]' for known keys."""
    return re.sub(r"\[([A-Za-z0-9_:.\-]+)\]", lambda m: f"[@{m.group(1)}]" if m.group(1) in keys else m.group(0), text)


def reference_list(corpus: Corpus, keys: list[str]) -> str:
    papers = {p.bib_key: p for p in corpus.papers if p.bib_key}
    lines = []
    for key in keys:
        p = papers.get(key)
        if p is None:
            continue
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "") if p.authors else "Anon."
        where = f" *{p.venue}*." if p.venue else ""
        link = f" https://doi.org/{p.doi}" if p.doi else (f" https://arxiv.org/abs/{p.arxiv_id}" if p.arxiv_id else "")
        lines.append(f"- **[{key}]** {authors} ({p.year or 'n.d.'}). {p.title}.{where}{link}")
    return "\n".join(lines)


def bibtex_subset(bibtex: str, keys: list[str]) -> str:
    """The BibTeX entries for ``keys`` (in that order) out of a whole bibliography."""
    entries: dict[str, str] = {}
    starts = [m for m in re.finditer(r"^@\w+\s*\{\s*([^,\s]+)\s*,", bibtex, re.M)]
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(bibtex)
        entries[match.group(1)] = bibtex[match.start() : end].strip()
    return "\n\n".join(entries[k] for k in keys if k in entries) + "\n"


def assemble(title: str, sections: list[ReportSection], values: dict[str, Value], references: str) -> str:
    """The rendered Markdown document."""
    lines = [f"# {title}", ""]
    numbered = 0
    for section in sections:
        if section.id == "abstract":
            lines += ["## Abstract", ""]
        else:
            numbered += 1
            lines += [f"## {numbered}. {section.title}", ""]
        lines += [render(section.template, values).strip(), ""]
    if references:
        lines += ["## References", "", references, ""]
    return "\n".join(lines)
