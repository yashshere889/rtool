"""The Experiment plan page's API on the demo runs: the plan, live checks, drafts, edits and the downloads."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from research_tool.service import open_run, run_status  # noqa: E402
from research_tool.testing.web import WRITE, StubRunner, web_client  # noqa: E402
from research_tool.web.runner import LocalRunner  # noqa: E402

AT_GATE = "/api/runs/sleep-demo/3-plan"
LATER = "/api/runs/sleep-demo/4-gpu-runs"


def edited(plan: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(plan)
    value["seeds"] = [0, 1, 2, 3, 4]
    value["compute"]["hours_per_run"] = 1.0
    value["ablations"] = [{"name": "Single Click", "description": "one click per up-state", "is_baseline": False}]
    return value


def test_the_plan_and_the_cluster_it_must_fit(runs_root: Path) -> None:
    view = web_client(runs_root).get(f"{AT_GATE}/plan").json()
    assert view["stage"] == "plan" and view["version"] == view["head"] == 1 and view["by"] is None
    assert view["editable"] and view["hypothesis_title"].startswith("Phase-locked clicks")
    cluster = view["cluster"]
    assert cluster["budget_gpu_hours"] == 40 and cluster["min_seeds"] == 3
    assert {p["name"] for p in cluster["partitions"] if p["allowed"]} == {"gpu-l40s", "gpu-h100"}
    plan = view["plan"]
    assert plan["compute"]["partition"] == "gpu-l40s" and plan["compute"]["runs"] == 12 and plan["issues"] == []
    assert web_client(runs_root).get(f"{AT_GATE}/plan", params={"version": 7}).status_code == 404


def test_an_edit_is_checked_as_it_would_be_stored(runs_root: Path) -> None:
    c = web_client(runs_root)
    plan = c.get(f"{AT_GATE}/plan").json()["plan"]
    checked = c.post(f"{AT_GATE}/plan/check", json={"base_version": 1, "value": edited(plan)}, headers=WRITE).json()
    assert checked["ok"] and set(checked["changed"]) == {"ablations", "seeds", "compute"}
    new = checked["plan"]
    assert [a["name"] for a in new["ablations"]] == ["single_click"]
    compute = new["compute"]
    assert (compute["runs"], compute["gpu_hours_total"], compute["time_limit"]) == (20, 20.0, "01:18:00")
    assert new["results_schema"]["properties"]["runs"]["minItems"] == 20

    greedy = edited(plan)
    greedy["compute"]["hours_per_run"] = 10.0
    issues = c.post(f"{AT_GATE}/plan/check", json={"base_version": 1, "value": greedy}, headers=WRITE).json()
    assert any("budget is 40" in i for i in issues["plan"]["issues"])
    broken = dict(plan, metrics=[])
    refused = c.post(f"{AT_GATE}/plan/check", json={"base_version": 1, "value": broken}, headers=WRITE).json()
    assert not refused["ok"] and "at least one metric" in refused["error"]
    stale = c.post(f"{AT_GATE}/plan/check", json={"base_version": 0, "value": plan}, headers=WRITE).json()
    assert "changed since you started editing" in stale["error"]


def test_drafts_edits_and_undo(runs_root: Path) -> None:
    c = web_client(runs_root, StubRunner())
    plan = c.get(f"{AT_GATE}/plan").json()["plan"]
    value = edited(plan)
    drafts = f"{AT_GATE}/drafts/experiment_plan"
    assert c.get(drafts).status_code == 404
    assert c.put(drafts, json={"base_version": 1, "value": value}).status_code == 403  # writes need the header
    saved = c.put(drafts, json={"base_version": 1, "value": value, "note": "more seeds"}, headers=WRITE).json()
    assert saved["saved_at"] and c.get(drafts).json()["value"]["seeds"] == [0, 1, 2, 3, 4]
    assert c.get(f"{AT_GATE}/drafts/..%2Fevil").status_code in (404, 422)

    actions = f"{AT_GATE}/actions"
    edit = {"kind": "edit", "payload": {"artifact": "experiment_plan", "base_version": 1, "value": value}}
    unchanged = {"kind": "edit", "payload": {"artifact": "experiment_plan", "base_version": 1, "value": plan}}
    assert c.post(actions, json=unchanged, headers=WRITE).status_code == 422
    sent = c.post(actions, json=edit, headers=WRITE)
    assert sent.status_code == 202 and sent.json()["action"]["stage"] == "plan"
    assert c.get(drafts).status_code == 404  # the queued edit holds it now
    assert c.post(actions, json=edit, headers=WRITE).status_code == 409
    approve = {"kind": "approve", "stage": "plan", "gate": "experiment_plan"}
    assert "still on their way" in c.post(actions, json=approve, headers=WRITE).json()["detail"]

    assert c.delete(f"{actions}/{sent.json()['action']['id']}", headers=WRITE).status_code == 204
    assert c.get(drafts).json()["value"] == value  # back to editing


def test_an_applied_edit_becomes_the_plan_at_the_checkpoint(runs_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(runs_root, runner)
    plan = c.get(f"{AT_GATE}/plan").json()["plan"]
    body = {"kind": "edit", "payload": {"artifact": "experiment_plan", "base_version": 1, "value": edited(plan)}}
    assert c.post(f"{AT_GATE}/actions", json=body, headers=WRITE).status_code == 202
    run = open_run(runs_root, "sleep-demo", "3-plan")
    runner.join(run, timeout=120)
    view = c.get(f"{AT_GATE}/plan").json()
    assert view["version"] == 2 and view["versions"] == [1, 2] and view["by"] and view["edited_from"] == 1
    assert view["plan"]["compute"]["runs"] == 20
    gate = run_status(run).open_gates[0]
    assert gate.gate == "experiment_plan" and gate.summary["artifacts"] == {"experiment_plan": 2}
    old = c.get(f"{AT_GATE}/plan", params={"version": 1}).json()
    assert not old["editable"] and "earlier version" in old["why_not_editable"]

    approve = {"kind": "approve", "stage": "plan", "gate": "experiment_plan"}  # bound to version 2 by the server
    assert c.post(f"{AT_GATE}/actions", json=approve, headers=WRITE).status_code == 202
    runner.join(run, timeout=180)
    status = run_status(run)
    assert [g.gate for g in status.open_gates] == ["gpu_runs"]  # the code was written and tested for the edited plan
    assert c.get(f"{AT_GATE}/downloads").json()["plan_version"] == 2


def test_downloads(runs_root: Path) -> None:
    c = web_client(runs_root)
    assert c.get(f"{AT_GATE}/downloads").json()["started"] is False
    view = c.get(f"{LATER}/downloads").json()
    assert view["status"] == "done" and view["plan_version"] == 1 and view["root"] == "resources"
    items = {i["name"]: i for i in view["items"]}
    assert items["Sleep-EDF"]["status"] == "ready" and items["Sleep-EDF"]["path"].startswith("resources/")
    assert items["numpy"]["status"] == "deferred"
    impact = {a["stage"]: a["effect"] for a in c.get(f"{LATER}/stages/plan/impact").json()}
    assert impact["fetch"] == "redo" and impact["code"] == "redo"  # editing a finished plan redoes these
