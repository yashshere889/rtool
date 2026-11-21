"""The GPU-heavy packs' parts: their own images, UniProt / AlphaFold / PDB, and their result checks."""

from __future__ import annotations

from pathlib import Path

from research_tool.core.config import ExperimentsConfig, PackImage
from research_tool.packs import load_pack
from research_tool.packs.md.checks import energy_drift_issues
from research_tool.packs.protein.checks import leakage_issues
from research_tool.schemas.code import RunRecord
from research_tool.schemas.plan import ResourceChoice, ResourceSpec
from research_tool.testing.samples import FASTA, sample_plan, structures_mock
from research_tool.tools.resources.checks import HubChecker
from research_tool.tools.resources.fetch import Fetcher


def choice(source: str, identifier: str) -> ResourceChoice:
    return ResourceChoice(kind="dataset", name="d", source=source, identifier=identifier, purpose="")  # type: ignore[arg-type]


def test_a_pack_can_have_its_own_image() -> None:
    config = ExperimentsConfig(
        container="$NEMOTRON_SIF",
        packages=["torch"],
        pack_images={"protein": PackImage(container="$PROTEIN_SIF", packages=["torch", "esm", "biopython"])},
    )
    assert config.for_pack("protein").container == "$PROTEIN_SIF"
    assert config.for_pack("protein").packages == ["torch", "esm", "biopython"]
    assert config.for_pack("ml_dl") is config  # no image of its own: the default one


def test_uniprot_alphafold_and_pdb_are_checked_and_fetched(tmp_path: Path) -> None:
    checker = HubChecker(structures_mock())
    assert checker.check(choice("uniprot", "P69905")).note.startswith("sp|P69905|HBA_HUMAN Hemoglobin")
    assert not checker.check(choice("uniprot", "Q99999")).exists
    assert "not a UniProt accession" in checker.check(choice("uniprot", "hemoglobin")).note
    assert checker.check(choice("alphafold", "P69905")).note == "AlphaFold model AF-P69905-F1, mean pLDDT 97.2"
    assert checker.check(choice("pdb", "4HHB")).exists and "not a PDB id" in checker.check(choice("pdb", "HHB4")).note
    fetcher = Fetcher(tmp_path, http=structures_mock())
    for source, identifier, path in (
        ("uniprot", "p69905", "uniprot/P69905.fasta"),
        ("alphafold", "P69905", "alphafold/AF-P69905-F1-model_v4.cif"),
        ("pdb", "4HHB", "pdb/4hhb.cif"),
    ):
        fetched = fetcher.fetch(ResourceSpec(**choice(source, identifier).model_dump()))
        assert fetched.status == "ready" and fetched.path == str(tmp_path / path), fetched.note
    assert (tmp_path / "uniprot" / "P69905.fasta").read_text() == FASTA


def records(name: str, values: list[float]) -> list[RunRecord]:
    return [
        RunRecord(condition="new", seed=i, status="ok", metrics={"acc": 0.9, name: v}) for i, v in enumerate(values)
    ]


def test_leaked_test_sequences_and_energy_drift_are_design_problems() -> None:
    plan = sample_plan()
    assert leakage_issues(plan, records("test_sequences_in_train", [0, 0])) == []
    (leak,) = leakage_issues(plan, records("test_sequences_in_train", [0, 3]))
    assert leak.startswith("`test_sequences_in_train` is 3: test sequences also appear in the training data")
    assert energy_drift_issues(plan, records("energy_drift", [0.001, -0.004])) == []
    (drift,) = energy_drift_issues(plan, records("energy_drift", [0.001, 0.05]))
    assert drift.startswith("`energy_drift` reaches 0.05 (above 0.01)")
    assert load_pack("protein").result_checks and load_pack("md").result_checks
