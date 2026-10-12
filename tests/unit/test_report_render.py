"""Report templates: placeholders, citations, the number rule, references."""

from __future__ import annotations

from research_tool.agents.coder.results import aggregate
from research_tool.schemas.code import ExperimentResults, RunRecord
from research_tool.schemas.literature import Corpus, PaperRecord
from research_tool.schemas.report import ReportSection
from research_tool.testing.samples import sample_plan as plan
from research_tool.testing.samples import sample_records as records
from research_tool.tools.report.render import (
    Value,
    assemble,
    bare_numbers,
    bibtex_subset,
    citation_keys,
    plan_numbers,
    problems,
    reference_list,
    render,
    strip_problems,
    to_pandoc_citations,
    unresolved_count,
    value_table,
)

VALUES = {"mean:acc:new": Value("0.718", "mean acc of new"), "p:acc:new": Value("p = 0.010", "p")}
KEYS = {"smith2020", "lee2021"}


def results() -> ExperimentResults:
    p = plan()
    summaries, comparisons = aggregate(p, records())
    return ExperimentResults(
        runs=records(), summaries=summaries, comparisons=comparisons, complete=True, results_path="r.json"
    )


def test_value_table_covers_summaries_and_comparisons() -> None:
    values = value_table(plan(), results())
    assert values["mean:acc:new"].text == "0.713" and values["n:acc:new"].text == "3"
    assert values["ci:acc:new"].text.startswith("95% CI [")
    assert values["p:acc:new"].text == "p = 0.010" and values["runs_total"].text == "6"
    assert "diff:loss:new" in values and "mean:acc:base" in values


def test_render_and_citations() -> None:
    text = "New reached {{mean:acc:new}} ({{p:acc:new}}), unlike [@smith2020; @lee2021] and [@smith2020]."
    assert render(text, VALUES) == "New reached 0.718 (p = 0.010), unlike [@smith2020; @lee2021] and [@smith2020]."
    assert citation_keys(text) == ["smith2020", "lee2021"]
    assert to_pandoc_citations("as shown [smith2020] and [unknown]", KEYS) == "as shown [@smith2020] and [unknown]"


def test_the_number_rule() -> None:
    clean = "In 2023 the gain was {{mean:acc:new}} with a 95% CI; see `run_2.py` and [@smith2020]."
    assert bare_numbers(clean) == []  # years, 95%, code and citations are fine
    typed = "The new method reached 0.72 accuracy, a gain of 7 points."
    assert len(bare_numbers(typed)) == 2
    assert bare_numbers(typed, allowed=[0.72, 7.0]) == []  # both were in the material
    assert bare_numbers("ResNet50 and GPT-4 are models") == []
    found = problems("We got 0.72 {{mean:acc:old}} [@nobody]", VALUES, KEYS, allowed_numbers="")
    assert any("unknown placeholder" in p for p in found) and any("@nobody" in p for p in found)
    assert any("number typed" in p for p in found)
    assert problems("We got 0.72", VALUES, KEYS, allowed_numbers=None) == []  # rule off


def test_strip_problems_is_the_last_resort() -> None:
    text = "Accuracy was 0.72 {{mean:acc:old}} as shown [@nobody; @smith2020] and here [@nobody]."
    stripped = strip_problems(text, VALUES, KEYS, allowed_numbers="")
    assert stripped == "Accuracy was [?] [?] as shown [@smith2020] and here."
    assert unresolved_count(stripped) == 2
    # an aside that cannot be checked is left out rather than shown as "(p < [?])"
    aside = strip_problems("All comparisons were significant (p < 0.001), as expected.", VALUES, KEYS, "")
    assert aside == "All comparisons were significant, as expected."


def test_the_ci_label_is_not_doubled() -> None:
    """Seen on Barkla: '(95% CI 95% CI [1, 1])' and '95% CI [95% CI [-0.8, -0.8]]'."""
    values = {"ci:a:b": Value("95% CI [1, 1]", "ci"), "diff_ci:a:b": Value("95% CI [-0.8, -0.8]", "ci")}
    assert render("mean 1 (95% CI {{ci:a:b}})", values) == "mean 1 (95% CI [1, 1])"
    assert render("(-0.8; 95% CI [{{diff_ci:a:b}}])", values) == "(-0.8; 95% CI [-0.8, -0.8])"
    assert render("95 % CI: {{ci:a:b}}", values) == "95% CI [1, 1]"
    assert render("a mean of 1 ({{ci:a:b}})", values) == "a mean of 1 (95% CI [1, 1])"


def test_plan_facts_may_be_typed() -> None:
    p = plan()
    p.conditions[1].description = "Treatment condition with 10 agents"
    p.independent_variables[0].values = ["./2", "3", "10"]
    material = plan_numbers(p)
    assert bare_numbers("Experiments across 2 to 10 agents, N=10, three seeds.", allowed=[2.0, 10.0]) == []
    assert "10" in material and "2" in material.split() and "3 seeds" in material and "alpha 0.05" in material
    assert problems("From 2 to 10 agents at alpha 0.05.", VALUES, KEYS, allowed_numbers=material) == []


def test_references_and_bibtex() -> None:
    corpus = Corpus(
        topic="t",
        queries=[],
        papers=[
            PaperRecord(
                id="doi:1",
                bib_key="smith2020",
                title="Spindles",
                authors=["A Smith", "B Jones"],
                year=2020,
                venue="Sleep",
                doi="10.1/x",
                status="extracted",
            ),
        ],
        kb_path="kb.sqlite",
        kb_sha256="",
        bibtex_path="refs.bib",
    )
    assert reference_list(corpus, ["smith2020"]) == (
        "- **[smith2020]** A Smith, B Jones (2020). Spindles. *Sleep*. https://doi.org/10.1/x"
    )
    bib = "@article{smith2020,\n  title={Spindles}\n}\n\n@misc{lee2021,\n  title={Other}\n}\n"
    assert bibtex_subset(bib, ["lee2021"]) == "@misc{lee2021,\n  title={Other}\n}\n"
    doc = assemble(
        "Title",
        [
            ReportSection(id="abstract", title="Abstract", template="A {{mean:acc:new}}."),
            ReportSection(id="results", title="Results", template="R."),
        ],
        VALUES,
        "- ref",
    )
    assert doc.startswith("# Title\n\n## Abstract\n\nA 0.718.") and "## 1. Results" in doc and doc.endswith("- ref\n")


def test_partial_runs_still_render() -> None:
    runs = records()
    runs[0] = RunRecord(condition="base", seed=0, status="failed")
    summaries, comparisons = aggregate(plan(), runs)
    values = value_table(
        plan(),
        ExperimentResults(
            runs=runs, summaries=summaries, comparisons=comparisons, complete=False, results_path="r.json"
        ),
    )
    assert values["runs_ok"].text == "5" and values["n:acc:base"].text == "2"
