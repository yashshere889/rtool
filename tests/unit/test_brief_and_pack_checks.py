"""What the researcher fixes in the brief, tests that cannot be significant, pack checks and local data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_tool.agents.planner.checks import (
    brief_issues,
    design_issues,
    follow_brief,
    plan_issues,
    rank_test_issues,
    source_issues,
)
from research_tool.core.config import LocalDataset, load_config
from research_tool.core.errors import ConfigError
from research_tool.packs import Pack, _plan_check, load_pack
from research_tool.packs.ml_dl.checks import held_out_issues
from research_tool.schemas.common import BriefCondition, BriefMetric, ResearchBrief
from research_tool.schemas.plan import PlanCore, ResourceChoice, ResourceSpec, StatisticalTest
from research_tool.testing.samples import sample_plan
from research_tool.tools.http import PoliteHttp
from research_tool.tools.resources.checks import HubChecker
from research_tool.tools.resources.fetch import Fetcher

PINNED = ResearchBrief(
    topic="t",
    seeds=6,
    conditions=[BriefCondition(name="base", baseline=True), BriefCondition(name="new", description="the new one")],
    primary_metric=BriefMetric(name="loss", higher_is_better=False),
)


def core(conditions: list[dict[str, Any]], test: str = "paired t-test", n_seeds: int = 3) -> PlanCore:
    return PlanCore(
        objective="o",
        design="d",
        independent_variables=[{"name": "x", "description": "x"}],
        dependent_variables=[{"name": "y", "description": "y"}],
        conditions=conditions,
        metrics=[
            {"name": "acc", "definition": "a", "higher_is_better": True, "primary": True},
            {"name": "loss", "definition": "l", "higher_is_better": True},
        ],
        statistical_tests=[{"test": test, "comparison": "new vs base"}],
        n_seeds=n_seeds,
        procedure=["1", "2", "3"],
        expected_results="r",
    )


def test_what_the_brief_fixes_is_set_not_asked_again() -> None:
    planned = core([{"name": "base", "description": "b"}, {"name": "new", "description": "n", "is_baseline": True}])
    followed = follow_brief(planned, PINNED)
    assert followed.n_seeds == 6
    assert [(c.name, c.is_baseline) for c in followed.conditions] == [("base", True), ("new", False)]
    assert [(m.name, m.primary, m.higher_is_better) for m in followed.metrics] == [
        ("acc", False, True),
        ("loss", True, False),
    ]
    assert brief_issues(PINNED, conditions=followed.conditions, metrics=followed.metrics) == []


def test_what_the_plan_left_out_is_an_issue() -> None:
    planned = follow_brief(
        core([{"name": "control", "description": "c", "is_baseline": True}, {"name": "other", "description": "o"}]),
        PINNED,
    )
    # the fixed baseline is missing, so the model's own baseline stays and the issue names what to add
    assert planned.conditions[0].is_baseline
    issues = brief_issues(PINNED, conditions=planned.conditions, metrics=planned.metrics)
    assert issues[0].startswith("the researcher fixed a condition named 'base'")
    assert "the plan has control, other" in issues[0]
    assert "'new' (the new one)" in issues[1]
    # a hand edit can still break what follow_brief would have set
    edited = sample_plan(primary="acc")
    issues = brief_issues(PINNED, conditions=edited.conditions, metrics=edited.metrics)
    assert any("fixed the primary metric 'loss' (lower is better)" in i for i in issues)
    edited.metrics = [
        m.model_copy(update={"primary": m.name == "loss", "higher_is_better": True}) for m in edited.metrics
    ]
    assert brief_issues(PINNED, metrics=edited.metrics) == [
        "the primary metric loss must have lower is better, as the researcher set"
    ]


@pytest.mark.parametrize(
    ("test", "seeds", "needed"),
    [("Wilcoxon signed-rank", 5, 6), ("Mann-Whitney U", 3, 4), ("Wilcoxon signed-rank", 6, None), ("t-test", 2, None)],
)
def test_a_rank_test_that_cannot_be_significant_is_an_issue(test: str, seeds: int, needed: int | None) -> None:
    issues = rank_test_issues([StatisticalTest(test=test, comparison="new vs base")], seeds)
    assert issues == [] if needed is None else f"use at least {needed} seeds" in issues[0]
    arms = [{"name": "base", "description": "b", "is_baseline": True}, {"name": "new", "description": "n"}]
    planned = core(arms, test=test, n_seeds=seeds)
    assert any("whatever the results" in i for i in design_issues(planned, min_seeds=1)) == (needed is not None)


def test_a_pack_may_limit_the_sources_its_plans_use() -> None:
    resources = [
        ResourceSpec(kind="dataset", name="d", source="local", identifier="d", purpose=""),
        ResourceSpec(kind="dataset", name="mnist", source="torchvision", identifier="MNIST", purpose=""),
    ]
    assert source_issues(resources, Pack(name="any", description="", smoke="")) == []
    limited = Pack(name="omics", description="", smoke="", resource_sources=["local", "url"])
    assert source_issues(resources, limited) == [
        "mnist comes from 'torchvision', which the omics pack does not use: choose one of local, url"
    ]


def test_the_ml_pack_wants_held_out_data() -> None:
    def with_splits(*splits: str | None) -> Any:
        plan = sample_plan()
        plan.resources = [
            ResourceSpec(kind="dataset", name=f"d{i}", source="torchvision", identifier="MNIST", split=s, purpose="")
            for i, s in enumerate(splits)
        ]
        return plan

    assert held_out_issues(with_splits("train"), None) and held_out_issues(with_splits("train", "TRAIN"), None)
    assert held_out_issues(with_splits("train", "test"), None) == []
    assert held_out_issues(with_splits(None), None) == []  # no split named: it may be made in code
    assert load_pack("ml_dl").plan_checks == ["research_tool.packs.ml_dl.checks:held_out_issues"]
    # hand-edited plans get the pack's checks too
    config = load_config(env={})
    brief = ResearchBrief(topic="t", domain_pack="ml_dl")
    assert any("held-out" in i for i in plan_issues(with_splits("train"), config, 1, 100, brief))
    assert not any("held-out" in i for i in plan_issues(with_splits("train"), config, 1, 100, None))  # generic


def test_a_misspelt_pack_check_fails_when_the_pack_is_loaded() -> None:
    with pytest.raises(ConfigError, match="cannot be loaded"):
        _plan_check("x", "research_tool.packs.ml_dl.checks:no_such_check")
    with pytest.raises(ConfigError, match="cannot be loaded"):
        _plan_check("x", "research_tool.no_such_module:check")


def test_a_local_dataset_is_checked_and_copied_under_its_content(tmp_path: Path) -> None:
    data = tmp_path / "provided" / "xor"
    data.mkdir(parents=True)
    (data / "train.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    local = {"xor": LocalDataset(path=str(data), description="XOR"), "gone": LocalDataset(path=str(tmp_path / "no"))}
    checker = HubChecker(PoliteHttp(), local=local)

    def check(identifier: str) -> Any:
        return checker.check(
            ResourceChoice(kind="dataset", name="d", source="local", identifier=identifier, purpose="")
        )

    assert check("xor").exists and check("xor").note == "provided locally (1 files)"
    assert not check("gone").exists and "does not exist" in check("gone").note
    assert not check("other").exists and "the researcher provided: gone, xor" in check("other").note

    fetcher = Fetcher(tmp_path / "resources", provided=local)
    spec = ResourceSpec(kind="dataset", name="d", source="local", identifier="xor", purpose="")
    first = fetcher.fetch(spec)
    assert first.status == "ready" and first.files == 1 and first.note.startswith("provided locally, sha256 ")
    assert Path(first.path).parent == tmp_path / "resources" / "local"  # type: ignore[arg-type]
    assert (Path(first.path) / "train.csv").read_text(encoding="utf-8") == "x,y\n1,0\n"  # type: ignore[arg-type]
    assert fetcher.fetch(spec).path == first.path  # the same content: the same copy
    (data / "train.csv").write_text("x,y\n1,1\n", encoding="utf-8")
    assert fetcher.fetch(spec).path != first.path  # changed data never overwrites what a run read
    missing = fetcher.fetch(spec.model_copy(update={"identifier": "gone"}))
    assert missing.status == "failed" and "no local dataset 'gone' on disk" in missing.note
