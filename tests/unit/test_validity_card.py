"""The validity card, the code version it names, and the cases shipped in benchmarks/."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research_tool.validity.card import build_card, render_markdown, wilson
from research_tool.validity.cases import load_cases
from research_tool.validity.provenance import code_hash, code_version
from research_tool.validity.runs import generate_data
from research_tool.validity.score import RunScore

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def test_wilson_intervals() -> None:
    low, high = wilson(0, 10)
    assert low == 0 and high == pytest.approx(0.2775, abs=1e-4)  # 0 of 10 still allows about 28 %
    assert wilson(10, 10)[1] == 1 and wilson(10, 10)[0] == pytest.approx(0.7225, abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)  # nothing measured: nothing known


def score(run_id: str, case: str, kind: str, level: int, **fields: object) -> RunScore:
    max_level = 1 if kind == "open" else 3
    return RunScore(
        run_id=run_id, case=case, kind=kind, variant=str(fields.pop("variant", "main")), repeat=1, commit="c",
        finished=bool(fields.pop("finished", True)), max_level=max_level, level=level, **fields,  # type: ignore[arg-type]
    )  # fmt: skip


def test_the_card_counts_confirmed_cases_and_reports_the_rest_apart() -> None:
    scores = [
        score("p1", "planted-a", "planted", 3, variant="clear"),
        score("p2", "planted-a", "planted", 2, variant="noisy", category="wrong_analysis"),
        score("n1", "null-a", "null", 3),
        score("n2", "null-a", "null", 2, category="wrong_analysis", false_claim=True),
        score("o1", "open-a", "open", 1),
        score("u1", "unconfirmed", "replication", 3, confirmed=False),
        score("f1", "planted-a", "planted", 0, category="stage_failed:code"),
        score("r1", "planted-a", "planted", -1, finished=False),
    ]
    card = build_card("generic", [(None, s) for s in scores], provenance={"code": "abc", "model": "m"})
    assert card.runs == 7 and card.unfinished == ["r1"]
    assert {kind: (r.k, r.n) for kind, r in card.headline.items()} == {
        "null": (1, 2),
        "open": (1, 1),
        "planted": (1, 3),
    }
    assert (card.unconfirmed["unconfirmed"].k, card.unconfirmed["unconfirmed"].n) == (1, 1)  # not in the headline
    assert (card.false_claims.k, card.false_claims.n) == (1, 2)
    assert card.levels["planted"] == {"L0": 1, "L2": 1, "L3": 1}
    assert {name: (r.k, r.n) for name, r in card.variants["planted-a"].items()} == {
        "clear": (1, 1), "main": (0, 1), "noisy": (0, 1),
    }  # fmt: skip
    assert card.categories == {"stage_failed:code": ["f1"], "wrong_analysis": ["p2", "n2"]}
    text = render_markdown(card)
    assert "# Validity card: generic" in text and "**planted**: 1 / 3 (33 %" in text
    assert "**False claims** on null cases (the report says supported): 1 / 2" in text
    assert (
        "## Not in the headline: expected outcome not confirmed yet" in text and "Still running, not scored: r1" in text
    )
    assert "What this card does not measure" in text and "model `m`" in text


def test_the_code_version_is_a_hash_and_git_says_whether_it_is_committed(tmp_path: Path) -> None:
    package = tmp_path / "repo" / "pkg"
    package.mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"compiled")
    before = code_hash(package)
    (package / "__pycache__" / "b.pyc").write_bytes(b"more")
    assert code_hash(package) == before  # compiled files do not change the version
    assert code_version(package).commit == ""  # no git here (as on Barkla): the hash alone names the code
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    (tmp_path / "repo" / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")  # as in rtool's repo
    subprocess.run([*git, "init", "-q"], cwd=tmp_path / "repo", check=True)
    subprocess.run([*git, "add", ".gitignore", "pkg/a.py"], cwd=tmp_path / "repo", check=True)
    subprocess.run([*git, "commit", "-qm", "c"], cwd=tmp_path / "repo", check=True)
    clean = code_version(package)
    assert len(clean.commit) == 40 and not clean.dirty and clean.code == before and clean.label == before[:12]
    (package / "a.py").write_text("x = 2\n", encoding="utf-8")
    dirty = code_version(package)
    assert dirty.dirty and dirty.code != before


@pytest.mark.parametrize("pack", sorted(p.name for p in BENCHMARKS.iterdir() if p.is_dir()))
def test_the_shipped_cases_load_and_their_data_can_be_made(pack: str, tmp_path: Path) -> None:
    cases = load_cases(BENCHMARKS, pack)
    assert cases and all(c.confirmed or c.kind in ("planted", "replication") for c in cases)
    for case in cases:
        for variant in case.variant_names():
            case.expectation(variant), case.research_brief(variant)  # every variant is complete
            for name, entry in generate_data(case, variant, tmp_path).items():
                assert any(Path(entry["path"]).iterdir()), f"{case.id}/{variant}: {name} is empty"
