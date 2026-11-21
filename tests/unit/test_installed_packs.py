"""Packs from installed Python packages, and what a deployment offers: which packs, and how many GPU hours."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from research_tool.cli.main import app
from research_tool.core.config import apply_overrides, deep_merge, load_config
from research_tool.core.errors import ConfigError, RToolError
from research_tool.core.run import RunHandle
from research_tool.llm.prompts import PromptLibrary
from research_tool.packs import available_packs, load_pack, pack_plan_issues, pack_sources
from research_tool.service.runs import create_run
from research_tool.testing.demo_run import CLUSTER, demo_case_data
from research_tool.testing.samples import sample_plan
from research_tool.validity.cases import Case
from research_tool.validity.runs import create_case_run

CHECKS = """
def issues(plan, brief):
    return [] if plan.metrics else ["no metrics"]
"""


def install(site: Path, name: str, module: str, *, checks: bool = False) -> None:
    """A pack as an installed package ships it: the package's folder, and an entry point naming it."""
    package = site / module
    (package / "templates").mkdir(parents=True)
    (package / "prompts" / "planner").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    lines = [f"name: {name}", f'title: "Field {name}"', "description: A field nobody built in.", "hints: [x]"]
    if checks:
        (package / "checks.py").write_text(CHECKS)
        lines.append(f"plan_checks: ['{module}.checks:issues']")
    (package / "pack.yaml").write_text("\n".join(lines) + "\n")
    (package / "templates" / "experiment.py.j2").write_text("# the field's own skeleton\n")
    (package / "prompts" / "planner" / "plan_core.j2").write_text("the field's own planner prompt\n")
    dist = site / f"{module}-0.3.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {module}\nVersion: 0.3\n")
    (dist / "entry_points.txt").write_text(f"[research_tool.packs]\n{name} = {module}\n")


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    site = tmp_path / "site"
    install(site, "fieldx", "rtool_pack_fieldx", checks=True)
    install(site, "omics", "rtool_pack_other_omics")  # tries to take a built-in pack's name
    monkeypatch.syspath_prepend(str(site))
    yield site
    for name in [m for m in sys.modules if m.startswith("rtool_pack_")]:  # the next test installs its own copy
        del sys.modules[name]


def test_an_installed_pack_is_used_like_a_built_in_one(installed: Path) -> None:
    pack = load_pack("fieldx")
    assert pack.origin == "rtool_pack_fieldx 0.3" and pack.title == "Field fieldx"
    assert pack.template == "# the field's own skeleton\n" and pack.hints == ["x"]
    assert pack_plan_issues(pack, sample_plan(), None) == []  # its own check, found by module path
    prompts = PromptLibrary.for_agent("research_tool.agents.planner", "planner", pack="fieldx")
    assert prompts.render("plan_core.j2").text.strip() == "the field's own planner prompt"
    assert "fieldx" in available_packs()


def test_an_installed_pack_cannot_take_a_built_in_name(installed: Path) -> None:
    assert load_pack("omics").origin == "built-in"
    (other,) = [s for s in pack_sources() if s.origin.startswith("rtool_pack_other_omics")]
    assert other.shadowed and available_packs().count("omics") == 1


def test_a_broken_installed_pack_is_reported_not_a_crash(installed: Path) -> None:
    (installed / "rtool_pack_fieldx" / "pack.yaml").write_text("title: Field x: a colon breaks this\n")
    with pytest.raises(ConfigError, match=r"pack 'fieldx' \(rtool_pack_fieldx 0.3\): pack.yaml is not valid YAML"):
        load_pack("fieldx")
    listed = CliRunner().invoke(app, ["packs"])
    assert listed.exit_code == 0 and "fieldx     broken" in listed.output


def test_a_pack_name_is_one_path_segment() -> None:
    for name in ("../generic", "generic/../omics", "", ".hidden"):
        with pytest.raises(ConfigError, match="unknown domain pack"):
            load_pack(name)


def site_config(root: Path, **packs: Any) -> Any:
    overrides = {"paths": {"runs_root": str(root)}, "packs": packs, "budgets": {"run": {"max_gpu_hours": 10}}}
    return load_config(overrides=overrides, env={})


def test_a_deployment_offers_only_its_packs(tmp_path: Path, installed: Path) -> None:
    config = site_config(tmp_path / "runs", allowed=["generic", "fieldx"])
    with pytest.raises(RToolError, match="the 'md' pack is not offered here; this deployment offers: generic, fieldx"):
        create_run(config, "p", "Folding times of a small protein", pack="md")
    assert create_run(config, "p", "A question in field x", pack="fieldx").spec.pack == "fieldx"
    cluster = load_config(
        overrides=deep_merge(
            CLUSTER, {"paths": {"runs_root": str(tmp_path / "cases")}, "packs": {"allowed": ["omics"]}}
        ),
        env={},
    )
    with pytest.raises(RToolError, match="the 'generic' pack is not offered here"):
        create_case_run(cluster, Case.model_validate(demo_case_data()), commit="0" * 12)


def test_a_pack_gpu_cap_only_lowers_the_budget_and_holds_over_config_files(tmp_path: Path) -> None:
    config = site_config(tmp_path / "runs", max_gpu_hours={"md": 2, "omics": 50})
    md = create_run(config, "p", "Folding times of a small protein", pack="md")
    omics = create_run(config, "p", "Donor-level expression", pack="omics")
    assert md.config.budgets.run.max_gpu_hours == 2 and omics.config.budgets.run.max_gpu_hours == 10
    reopened = RunHandle(md.root)  # an episode layers the site's config files over the run's snapshot
    reopened.config = apply_overrides(reopened.config, {"budgets": {"run": {"max_gpu_hours": 100}}})
    assert reopened.config.budgets.run.max_gpu_hours == 2
    unlimited = load_config(
        overrides={"paths": {"runs_root": str(tmp_path / "u")}, "packs": {"max_gpu_hours": {"md": 3}}}
    )
    assert create_run(unlimited, "p", "Folding", pack="md").config.budgets.run.max_gpu_hours == 3  # no budget: the cap


def test_a_pack_can_name_its_pipeline(tmp_path: Path) -> None:
    config = site_config(tmp_path / "runs")
    assert create_run(config, "p", "Do statins lower LDL?", pack="medicine").spec.pipeline.name == "synthesis"
    assert create_run(config, "p", "Does caffeine speed reactions?", pack="study").spec.pipeline.name == "study"
    named = create_run(config, "p", "Statins and LDL", pack="medicine", pipeline="research")
    assert (
        named.spec.pipeline.name == "research"
        and create_run(config, "p", "x", pack="omics").spec.pipeline.name == "research"
    )


def test_refusals_hold_for_installed_packs(tmp_path: Path, installed: Path) -> None:
    with pytest.raises(RToolError, match="rtool refuses this brief"):
        create_run(site_config(tmp_path / "runs"), "p", "Design a novel toxin protein", pack="fieldx")


def test_rtool_packs_lists_where_each_pack_comes_from(tmp_path: Path, installed: Path) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("packs:\n  allowed: [generic, fieldx]\n  max_gpu_hours: {fieldx: 4}\n")
    result = CliRunner().invoke(app, ["packs", "--config", str(site)])
    assert result.exit_code == 0, result.output
    rows = {tuple(line.split()[:2]): line for line in result.output.splitlines() if line.strip()}
    assert "rtool_pack_fieldx 0.3" in rows["fieldx", "offered"] and "max 4 GPU-h" in rows["fieldx", "offered"]
    assert "built-in" in rows["md", "not"] and "pipeline synthesis" in rows["medicine", "not"]
    assert "rtool_pack_other_omics 0.3" in rows["omics", "shadowed"]
