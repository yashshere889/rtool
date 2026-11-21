"""BioModels (files, download, SBML check) and the sysbio pack's conservation check."""

from __future__ import annotations

from pathlib import Path

from research_tool.packs import load_pack, pack_result_issues
from research_tool.packs.sysbio.checks import conservation_issues
from research_tool.schemas.code import RunRecord
from research_tool.schemas.plan import ResourceChoice, ResourceSpec
from research_tool.testing.samples import SBML, biomodels_mock, sample_plan
from research_tool.tools.resources.checks import HubChecker
from research_tool.tools.resources.fetch import Fetcher
from research_tool.tools.resources.models import sbml_problem


def spec(identifier: str = "BIOMD0000000999") -> ResourceSpec:
    return ResourceSpec(kind="model", name="m", source="biomodels", identifier=identifier, purpose="")


def test_biomodels_models_are_checked_and_their_sbml_fetched(tmp_path: Path) -> None:
    http = biomodels_mock({"BIOMD0000000999_url.xml": SBML.encode()})
    found = HubChecker(http).check(ResourceChoice(**spec().model_dump(include=set(ResourceChoice.model_fields))))
    assert found.exists and found.note == "main files: BIOMD0000000999_url.xml"
    assert (
        "not a BioModels id"
        in HubChecker(http)
        .check(ResourceChoice(**spec("BIOMD12").model_dump(include=set(ResourceChoice.model_fields))))
        .note
    )  # noqa: E501
    fetched = Fetcher(tmp_path, http=http).fetch(spec())
    assert fetched.status == "ready" and fetched.files == 1  # the main file only, not the additional ones
    assert (tmp_path / "biomodels" / "BIOMD0000000999" / "BIOMD0000000999_url.xml").read_text() == SBML


def test_a_model_file_that_is_not_sbml_stops_the_fetch(tmp_path: Path) -> None:
    broken = Fetcher(tmp_path / "a", http=biomodels_mock({"m.xml": b"<html><body>Not found</body></html>"})).fetch(
        spec()
    )
    assert broken.status == "failed" and "is XML but not SBML (its root is <html>)" in broken.note
    garbled = Fetcher(tmp_path / "b", http=biomodels_mock({"m.xml": b"<sbml><model"})).fetch(spec())
    assert garbled.status == "failed" and "is not readable XML" in garbled.note
    empty = tmp_path / "empty.xml"
    empty.write_text('<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"/>')
    assert sbml_problem(empty) == "empty.xml is SBML without a <model>"


def records(errors: list[float]) -> list[RunRecord]:
    return [
        RunRecord(condition="new", seed=i, status="ok", metrics={"acc": 1.0, "mass_balance_error": e})
        for i, e in enumerate(errors)
    ]


def test_a_simulation_that_loses_mass_is_a_design_problem() -> None:
    plan = sample_plan()
    assert conservation_issues(plan, records([0.0, 1e-9, -5e-7])) == []
    (issue,) = conservation_issues(plan, records([0.0, 0.02]))
    assert issue.startswith("`mass_balance_error` reaches 0.02 (above 1e-06) in new/seed 1")
    assert conservation_issues(plan, [RunRecord(condition="a", seed=0, status="ok", metrics={"acc": 9.0})]) == []
    sysbio = load_pack("sysbio")
    assert sysbio.result_checks and pack_result_issues(sysbio, plan, records([0.5])) != []
    assert pack_result_issues(load_pack("generic"), plan, records([0.5])) == []  # other packs do not check it


def test_an_error_that_never_happens_is_no_sign_of_a_fake_experiment() -> None:
    from research_tool.agents.coder.sanity import results_warnings  # noqa: PLC0415

    plan = sample_plan()  # acc (higher is better) and loss (lower is better)

    def runs(loss: float) -> list[RunRecord]:
        return [
            RunRecord(condition=c, seed=s, status="ok", metrics={"acc": 0.5 + s / 10, "loss": loss})
            for c in ("base", "new")
            for s in range(3)
        ]

    assert results_warnings(plan, runs(0.0)) == []  # e.g. a conserving simulation's mass balance error
    (warning,) = results_warnings(plan, runs(0.3))  # a fixed non-zero value still looks computed
    assert "every seed gave exactly the same `loss`" in warning
