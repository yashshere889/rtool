"""The Literature agent's checks, one at a time: screening scores, copies of papers, claims, gaps, quality."""

from __future__ import annotations

from research_tool.agents.literature import quality
from research_tool.agents.literature.agent import ABSENCE, concept_score, copy_problem, judged_reason, rank_papers
from research_tool.agents.literature.mapping import (
    build_claims,
    build_entities,
    find_gaps,
    numbers_in,
    subtopic_papers,
    untested_combinations,
)
from research_tool.knowledge.vectors import absorb_small
from research_tool.schemas.literature import (
    Claim,
    CoverageCell,
    Entity,
    PaperExtraction,
    PaperRecord,
    RelevanceJudgment,
    Topic,
    TopicScope,
)
from research_tool.tools.documents.sections import Section

SCOPE = TopicScope(
    interpretation="LLM-based agents that carry out parts of scientific research.",
    concepts=["LLM-based agents", "automating scientific research", "multi-agent pipelines"],
    out_of_scope=["multi-agent reinforcement learning for robot control"],
)


def judgment(*matches: str, score: int = 9, out_of_scope: bool = False, names: list[str] | None = None):  # noqa: ANN201
    names = names or SCOPE.concepts
    return RelevanceJudgment(
        reason="A paper.",
        concepts=[{"concept": n, "match": m} for n, m in zip(names, matches, strict=False)],
        out_of_scope=out_of_scope,
        score=score,
    )


# --------------------------------------------------------------- screening
def test_the_checklist_decides_not_the_models_own_score() -> None:
    # how the first real run went wrong: a MARL paper shares "multi-agent" and the model gave it a 9
    marl = judgment("no", "no", "partly", score=9)
    assert concept_score(marl, SCOPE)[0] == 1.7
    core = judgment("yes", "yes", "yes", score=6)
    assert concept_score(core, SCOPE)[0] == 10.0
    background = judgment("yes", "no", "yes")
    assert concept_score(background, SCOPE)[0] == 6.7  # passes the default threshold of 6 as background


def test_concepts_match_by_name_then_by_position_and_missing_ones_count_as_no() -> None:
    reworded = judgment("yes", "yes", "yes", names=["LLM agents", "research automation", "pipelines"])
    assert concept_score(reworded, SCOPE)[0] == 10.0  # by position
    reordered = judgment("no", "yes", names=["multi-agent pipelines", "LLM-based agents"])
    score, matches = concept_score(reordered, SCOPE)
    assert matches == [
        ("LLM-based agents", "yes"),
        ("automating scientific research", "no"),
        ("multi-agent pipelines", "no"),
    ]
    assert score == 3.3


def test_out_of_scope_caps_the_score_and_no_scope_keeps_the_models_score() -> None:
    assert concept_score(judgment("yes", "yes", "yes", out_of_scope=True), SCOPE)[0] == 3.0
    assert concept_score(judgment("yes", score=7), None) == (7.0, [])
    assert concept_score(RelevanceJudgment(reason="x", score=4), SCOPE) == (4.0, [])  # no checklist given


def test_the_reason_says_which_concepts_matched() -> None:
    result = judgment("yes", "partly", "no", out_of_scope=True)
    _, matches = concept_score(result, SCOPE)
    text = judged_reason(result, matches)
    assert text.startswith("A paper. [about: LLM-based agents; partly about: automating scientific research;")
    assert text.endswith("not about: multi-agent pipelines (out of scope)]")


# ------------------------------------------------------------------ copies
def paper(title: str = "Agents that write research papers end to end", abstract: str = "") -> PaperRecord:
    return PaperRecord(id="doi:10.1/x", title=title, abstract=abstract)


def body(*texts: str, title: str = "front matter") -> list[Section]:
    return [Section(title=title, text=texts[0]), *(Section(title=f"s{i}", text=t) for i, t in enumerate(texts[1:]))]


def test_a_landing_page_with_the_abstract_is_not_the_paper() -> None:
    abstract = "We built agents. " * 80  # ~1,400 characters, like the Oxford repository page of 2026-10-12
    record = paper(abstract=abstract)
    landing = body(f"{record.title} - Research Archive", abstract, "Cite. Email. BibTeX. Terms of use.")
    problem = copy_problem(record, landing, min_chars=1500, ratio=4.0, title=True)
    assert problem is not None and "characters" in problem
    full = body(f"{record.title}\nAlice, Bob", abstract, "Methods. " * 900)
    assert copy_problem(record, full, min_chars=1500, ratio=4.0, title=True) is None


def test_a_different_document_is_caught_by_its_title() -> None:
    other = body("Crop yield prediction from satellite images\n" + "text " * 1000)
    problem = copy_problem(paper(), other, min_chars=1500, ratio=4.0, title=True)
    assert problem is not None and "title" in problem
    # a file the researcher added by hand is trusted: only the scanned-PDF check applies
    assert copy_problem(paper(), other, min_chars=1500, ratio=0.0, title=False) is None
    assert "characters" in (copy_problem(paper(), body("x" * 100), min_chars=1500, ratio=0.0, title=False) or "")


def test_absence_claims_are_recognised() -> None:
    for gap in ("The paper does not state how studies were selected.", "It is unclear whether results generalise.",
                "The authors fail to report variance.", "Sample size is not reported."):  # fmt: skip
        assert ABSENCE.search(gap), gap
    assert not ABSENCE.search("The sample was small and drawn from a single site.")


# ------------------------------------------------------------------ claims
def test_numbers_in_skips_years_and_names() -> None:
    assert numbers_in("GPT-4 and Claude 3.7 reached 87.4% in 2024 on 1,000 tasks (A5, v2)") == ["3.7", "87.4", "1000"]


def test_claims_with_numbers_found_nowhere_in_the_paper_are_not_verified() -> None:
    text = "Our method reached 87.4% accuracy. It used 64 adults. The baseline reached 80.1%."
    extraction = PaperExtraction(
        problem="p",
        approach="a",
        key_results=[
            {"claim": "87.4% accuracy", "quote": "Our method reached 87.4% accuracy."},
            {"claim": "87.4% accuracy on 64 adults", "quote": "Our method reached 87.4% accuracy."},
            {"claim": "a 50% gain over the baseline", "quote": "Our method reached 87.4% accuracy."},
        ],
    )
    exact, elsewhere, invented = build_claims({"p1": extraction}, {"p1": text})
    assert exact.verified and exact.unquoted_numbers == []
    assert elsewhere.verified and elsewhere.unquoted_numbers == ["64"]  # in the paper, just not in this quote
    assert not invented.verified and invented.support == 1.0 and invented.unquoted_numbers == ["50"]


def test_descriptions_are_not_entities() -> None:
    extraction = PaperExtraction(
        problem="p",
        approach="a",
        methods=["PPO", "Applying temporal clustering with HDBSCAN and dynamic time warping to find patterns"],
    )
    assert [e.name for e in build_entities({"p1": extraction})] == ["PPO"]


# -------------------------------------------------------------------- gaps
def test_gaps_rest_on_evidence_not_on_outlier_clusters() -> None:
    papers = {f"p{i}": PaperRecord(id=f"p{i}", title=f"T{i}", year=2010) for i in range(6)}
    papers["p0"].subtopics = ["idea generation"]
    subtopics = subtopic_papers(["idea generation", "experiment execution", "paper writing"], papers.values())
    assert subtopics == {"idea generation": ["p0"], "experiment execution": [], "paper writing": []}
    topics = [
        Topic(label="Big", summary="", paper_ids=["p0", "p1", "p2", "p3"]),
        Topic(label="Outliers", summary="", paper_ids=["p4", "p5"]),
    ]
    gaps = find_gaps([], [], topics, papers, subtopics=subtopics, now_year=2026)
    assert [g.kind for g in gaps] == ["uncovered_subtopic"] * 3 + ["stale_topic"]
    assert gaps[3].description == "No paper on 'Big' since 2010"  # the two-paper group is no "thin topic"
    assert gaps[0].evidence == ["p0"] and gaps[0].description.startswith("Only one paper")


def test_a_method_missing_from_its_benchmark_family_is_a_gap_and_an_unrelated_pair_is_not() -> None:
    """2026-10-14: the plain rule reported pairs like "GPT-4o has not been evaluated on UK Biobank"."""

    def cell(method: str, dataset: str, *ids: str) -> CoverageCell:
        return CoverageCell(method=method, dataset=dataset, paper_ids=list(ids))

    coverage = [
        cell("GPT-4o", "GSM8K", "p1", "p2"),
        cell("Llama 3", "GSM8K", "p3", "p4"),
        cell("Llama 3", "MATH", "p3", "p4"),  # two papers test MATH alongside GSM8K
        cell("Llama 3", "UK Biobank", "p5"),
        cell("Qwen", "UK Biobank", "p6"),
    ]
    entities = [
        Entity(kind="method", name="GPT-4o", paper_ids=["p1", "p2"]),
        Entity(kind="method", name="Llama 3", paper_ids=["p3", "p4", "p5"]),
        Entity(kind="dataset", name="GSM8K", paper_ids=["p1", "p2", "p3", "p4"]),
        Entity(kind="dataset", name="MATH", paper_ids=["p3", "p4"]),
        Entity(kind="dataset", name="UK Biobank", paper_ids=["p5", "p6"]),
    ]
    gaps = untested_combinations(entities, coverage)
    assert [g.description for g in gaps] == [
        "GPT-4o was evaluated on GSM8K but not on MATH, which 2 papers in this corpus test alongside GSM8K"
    ]
    assert gaps[0].evidence == ["p3", "p4"]


def test_ties_go_to_papers_linked_to_the_others_then_to_the_models_score() -> None:
    def paper(pid: str, cites: int, model: float = 9, refs: list[str] | None = None) -> PaperRecord:
        return PaperRecord(
            id=pid, title=pid, relevance=8.8, model_score=model, citation_count=cites, openalex_id=f"W{pid}",
            reference_ids=refs or [],
        )  # fmt: skip

    old_neighbour = paper("marl", cites=500)  # well cited, linked to nothing here
    preprint = paper("preprint", cites=0)
    linked_by_openalex = paper("survey", cites=3, refs=["Wseed"])
    seed = PaperRecord(id="seed", title="seed", relevance=10, openalex_id="Wseed")
    quiet = paper("quiet", cites=0, model=6)
    order = rank_papers([old_neighbour, quiet, preprint, linked_by_openalex, seed], [("seed", "preprint")])
    # one link each, same model score: citation count decides between them; the unlinked paper comes after
    assert [p.id for p in order] == ["seed", "survey", "preprint", "marl", "quiet"]


def test_small_clusters_join_their_nearest_large_one() -> None:
    vectors = [[1.0, 0.0], [0.99, 0.1], [0.98, 0.2], [0.0, 1.0], [0.1, 0.99], [0.2, 0.98], [0.9, 0.4]]
    groups = [[0, 1, 2], [3, 4, 5], [6]]
    assert absorb_small(groups, vectors) == [[0, 1, 2, 6], [3, 4, 5]]
    assert absorb_small([[0], [1]], vectors) == [[0], [1]]  # nothing large to join: left as it is


# ----------------------------------------------------------------- quality
def run_checks(relevances: list[float], **extra):  # noqa: ANN003, ANN201
    candidates = [PaperRecord(id=f"p{i}", title=f"T{i}", relevance=r) for i, r in enumerate(relevances)]
    included = {p.id: p.model_copy(update={"status": "extracted"}) for p in candidates if (p.relevance or 0) >= 6}
    named = PaperExtraction(problem="p", approach="a", methods=["PPO"])
    return {
        c.check: c
        for c in quality.checks(
            candidates=candidates,
            papers=included,
            extractions={pid: named for pid in included},
            claims=extra.get("claims", []),
            subtopics=extra.get("subtopics", {}),
            threshold=6,
            snowball_rounds=extra.get("rounds", 2),
        )
    }


def test_lenient_screening_is_flagged() -> None:
    lenient = run_checks([6, 7, 8, 8, 9, 10] * 5)  # the first real run: nothing below 6
    assert not lenient["screening"].ok and "lowest score 6" in lenient["screening"].value
    strict = run_checks([0, 1, 2, 3, 7, 9] * 5)
    assert strict["screening"].ok and strict["full text"].ok and strict["papers with named methods"].ok
    assert not strict["papers found through citations"].ok  # snowballing was on but found nothing
    assert run_checks([0, 9] * 10, rounds=0)["papers found through citations"].ok


def test_claims_and_subtopics_are_checked() -> None:
    claims = [Claim(paper_id="p1", kind="result", text="x", quote="q", verified=False, support=1.0)] * 2
    claims += [Claim(paper_id="p1", kind="result", text="x", quote="q", verified=True, support=1.0)] * 8
    checks = run_checks([0, 9] * 10, claims=claims, subtopics={"a": ["p1", "p3"], "b": []})
    assert not checks["results with numbers not in the paper"].ok
    assert checks["quotes found in the papers"].ok
    assert not checks["sub-topics covered"].ok and checks["sub-topics covered"].note.startswith(
        "Hardly any papers on: b"
    )
