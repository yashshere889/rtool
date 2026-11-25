"""Every resource failure seen on Barkla (docs/reports/planner-resources-2026-11-24.md) is still caught.

No LLM is involved. Not automated (needs Barkla): replay the plan stage of the real runs with each planner model
(`rtool replay <run> --stage plan --tag <name> -c <overlay>`) and compare the issues per plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_tool.agents.planner.checks import blocking_issues, brief_issues, resource_issues
from research_tool.schemas.common import ResearchBrief
from research_tool.schemas.plan import ResourceSpec

CASES = json.loads((Path(__file__).with_name("resource_cases.json")).read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["case"] for c in CASES])
def test_a_historic_resource_failure_is_flagged(case: dict) -> None:
    resources = [ResourceSpec(purpose="", **r) for r in case["resources"]]
    brief = ResearchBrief(**case["brief"])
    found = resource_issues(resources, 200) + brief_issues(brief, resources=resources)
    for wanted in case["expect"]:
        assert any(wanted in issue for issue in found), (wanted, found)


def test_the_blocking_list_covers_a_missing_manual_or_gated_resource() -> None:
    risky = [ResourceSpec(purpose="", **r) for c in CASES for r in c["resources"] if r["source"] != "torchvision"]
    assert blocking_issues(risky)
