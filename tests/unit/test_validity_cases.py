"""Validity cases: what case.yaml may say, and what a case turns into."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from research_tool.core.config import deep_merge
from research_tool.validity.cases import MIN_SEEDS, CaseError, load_case, load_cases


def planted(**changes: Any) -> dict[str, Any]:
    case = {
        "id": "two-servers",
        "kind": "planted",
        "pack": "generic",
        "brief": {
            "topic": "Do two servers shorten the time in an M/M/c queue?",
            "seeds": 8,
            "constraints": ["CPU only"],
            "conditions": [
                {"name": "one_server", "baseline": True, "description": "M/M/1, mu = 1"},
                {"name": "two_servers", "description": "M/M/2, mu = 1 each"},
            ],
            "primary_metric": {"name": "time_in_system", "higher_is_better": False},
        },
        "hypothesis": {
            "statement": "Two servers give a shorter mean time in system than one at lambda = 0.8.",
            "null_hypothesis": "The mean time in system does not differ.",
            "success_criteria": "two_servers has a lower mean time_in_system than one_server, paired t-test p < 0.05.",
            "independent_variables": ["servers"],
            "dependent_variables": ["time_in_system"],
        },
        "expect": {"significant": True, "direction": "better"},
    }
    return deep_merge(case, changes)


def write(root: Path, case: dict[str, Any], folder: str | None = None) -> Path:
    path = root / case["pack"] / (folder or case["id"])
    path.mkdir(parents=True, exist_ok=True)
    (path / "case.yaml").write_text(yaml.safe_dump(case), encoding="utf-8")
    return path


def test_a_planted_case_becomes_a_brief_a_hypothesis_and_an_expectation(tmp_path: Path) -> None:
    case = load_case(write(tmp_path, planted()))
    assert case.start == "plan" and case.baseline == "one_server" and case.variant_names() == ["main"]
    brief = case.research_brief()
    assert brief.domain_pack == "generic" and brief.seeds == 8
    assert brief.constraints[0] == "CPU only"
    assert "`one_server` (the baseline): M/M/1, mu = 1; `two_servers`: M/M/2, mu = 1 each" in brief.constraints[1]
    assert brief.constraints[2] == "The primary metric is `time_in_system` (lower is better)."
    hypothesis = case.hypothesis.hypothesis(case.id)  # type: ignore[union-attr]
    assert hypothesis.id == "two-servers" and hypothesis.title == hypothesis.statement and hypothesis.evidence == []
    expect = case.expectation()
    # the only non-baseline condition is the one compared; a planted case expects "supported" unless it says
    assert expect.condition == "two_servers" and expect.outcome == ["supported"]


def test_a_null_case_expects_no_effect_and_no_claim(tmp_path: Path) -> None:
    case = load_case(
        write(tmp_path, planted(id="rng-null", kind="null", expect={"significant": False, "direction": None}))
    )
    assert case.expectation().outcome == ["refuted", "inconclusive"]


def test_variants_change_the_expectation(tmp_path: Path) -> None:
    data = planted(
        expect={"difference": {"low": -5, "high": -2}},
        variants=[{"name": "busy"}, {"name": "light", "expect": {"difference": {"low": -1, "high": 0}}}],
    )
    case = load_case(write(tmp_path, data))
    assert case.variant_names() == ["busy", "light"]
    assert case.expectation("busy").difference.high == -2  # type: ignore[union-attr]
    assert case.expectation("light").difference.high == 0  # type: ignore[union-attr]
    assert case.expectation("light").significant is True  # the rest comes from the case
    with pytest.raises(CaseError, match="no variant 'heavy'"):
        case.expectation("heavy")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"brief": {"seeds": MIN_SEEDS - 1}}, "at least 6 seeds"),
        ({"expect": {"significant": False}}, "planted case expects a significant effect"),
        ({"kind": "null"}, "null case expects no effect"),
        ({"kind": "replication"}, "needs expect.value"),
        ({"kind": "open"}, "open case fixes nothing"),
        ({"hypothesis": None}, "needs a hypothesis"),
        ({"brief": {"primary_metric": None}}, "primary_metric is required"),
        ({"brief": {"seeds": None}}, "brief.seeds is required"),
        ({"expect": {"condition": "one_server"}}, "not a non-baseline condition"),
        ({"expect": {"difference": {"low": 2, "high": 1}}}, "is above high"),
        ({"brief": {"conditions": [{"name": "Two Servers", "baseline": True}]}}, "should match pattern"),
        ({"data": [{"name": "queue_log", "generator": "generate.py"}]}, "brief.datasets must list"),
        ({"variants": [{"name": "a"}, {"name": "a"}]}, "variant names must be unique"),
        ({"variants": [{"name": "zero", "expect": {"significant": False}}]}, "(variant 'zero')"),
        ({"expct": {"significant": True}}, "Extra inputs are not permitted"),  # a typo fails
    ],
)
def test_case_mistakes_are_caught(tmp_path: Path, changes: dict[str, Any], message: str) -> None:
    with pytest.raises(CaseError, match=f"case.yaml.*{message.replace('(', '[(]').replace(')', '[)]')}"):
        load_case(write(tmp_path, planted(**changes)))


def test_two_baselines_are_one_too_many(tmp_path: Path) -> None:
    data = planted()
    data["brief"]["conditions"][1]["baseline"] = True
    with pytest.raises(CaseError, match="exactly one of brief.conditions must be the baseline"):
        load_case(write(tmp_path, data))


def test_a_case_lives_in_its_own_folder_and_brings_its_generator(tmp_path: Path) -> None:
    with pytest.raises(CaseError, match="must be its folder's name"):
        load_case(write(tmp_path, planted(), folder="elsewhere"))
    with pytest.raises(CaseError, match="unknown domain pack 'chemistry'"):
        load_case(write(tmp_path, planted(pack="chemistry")))
    data = planted(brief={"datasets": ["queue_log"]}, data=[{"name": "queue_log", "generator": "../generate.py"}])
    folder = write(tmp_path, data)
    (folder.parent / "generate.py").write_text("print('outside the case')\n", encoding="utf-8")
    with pytest.raises(CaseError, match="no generator '../generate.py' in the case folder"):
        load_case(folder)
    (folder / "generate.py").write_text("print('inside')\n", encoding="utf-8")
    data["data"][0]["generator"] = "generate.py"
    (folder / "case.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_case(folder).data[0].name == "queue_log"


def test_load_cases_reads_one_pack_in_folder_order(tmp_path: Path) -> None:
    write(tmp_path, planted(id="b-case"))
    write(tmp_path, planted(id="a-case"))
    write(tmp_path, planted(id="other", pack="ml_dl"))
    assert [c.id for c in load_cases(tmp_path, "generic")] == ["a-case", "b-case"]
    misplaced = planted(id="misplaced", pack="ml_dl")
    folder = tmp_path / "generic" / "misplaced"
    folder.mkdir()
    (folder / "case.yaml").write_text(yaml.safe_dump(misplaced), encoding="utf-8")
    with pytest.raises(CaseError, match="must use the 'generic' pack"):
        load_cases(tmp_path, "generic")
