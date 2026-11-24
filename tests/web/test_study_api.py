"""The study pages' API on the demo runs: the protocol (read, checked, edited) and the data upload."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research_tool.service import open_run, run_status  # noqa: E402
from research_tool.testing.web import WRITE, StubRunner, web_client  # noqa: E402
from research_tool.web.runner import LocalRunner  # noqa: E402

AT_PROTOCOL = "/api/runs/sleep-study-demo/3-protocol"
AT_DATA = "/api/runs/sleep-study-demo/4-data"


def filled(template: str) -> str:
    """The template with a value in every metric cell."""
    rows = list(csv.reader(io.StringIO(template)))
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(rows[0])
    for i, row in enumerate(rows[1:]):
        writer.writerow([*row[:2], *[str(3 + i % 5)] * (len(row) - 2)])
    return out.getvalue()


def test_the_protocol_and_its_allocation(pipelines_root: Path) -> None:
    view = web_client(pipelines_root).get(f"{AT_PROTOCOL}/protocol").json()
    protocol = view["protocol"]
    assert view["stage"] == "protocol" and view["version"] == view["head"] == 1 and view["editable"]
    assert (protocol["n_per_condition"], protocol["baseline"], protocol["unit"]) == (26, "sham", "participant")
    assert view["allocation_rows"] == 52 and len(view["allocation_preview"]) == 12
    assert view["plan"]["replicates"] == "units" and view["hypothesis_title"].startswith("Phase-locked clicks")
    assert all(c["ok"] for c in protocol["checklist"])
    assert web_client(pipelines_root).get(f"{AT_PROTOCOL}/protocol", params={"version": 7}).status_code == 404


def test_an_edit_is_checked_as_it_would_be_stored(pipelines_root: Path) -> None:
    c = web_client(pipelines_root)
    protocol = c.get(f"{AT_PROTOCOL}/protocol").json()["protocol"]
    checked = c.post(
        f"{AT_PROTOCOL}/protocol/check",
        json={"base_version": 1, "value": {**protocol, "effect_size": 0.5, "blinding": ""}},
        headers=WRITE,
    ).json()
    assert checked["ok"] and checked["protocol"]["n_per_condition"] == 64
    assert set(checked["changed"]) == {"effect_size", "blinding"}  # what you changed, not what follows from it
    assert [c["name"] for c in checked["protocol"]["checklist"] if not c["ok"]] == ["blinding is stated"]
    for bad in ({"effect_size": 9}, {"effect_size": 0}):
        refused = c.post(
            f"{AT_PROTOCOL}/protocol/check", json={"base_version": 1, "value": {**protocol, **bad}}, headers=WRITE
        ).json()
        assert not refused["ok"] and "effect size" in refused["error"]
    stale = c.post(f"{AT_PROTOCOL}/protocol/check", json={"base_version": 0, "value": protocol}, headers=WRITE).json()
    assert "changed since you started editing" in stale["error"]


def test_an_applied_edit_rewrites_the_allocation_and_the_document(pipelines_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(pipelines_root, runner)
    protocol = c.get(f"{AT_PROTOCOL}/protocol").json()["protocol"]
    body = {
        "kind": "edit",
        "payload": {"artifact": "protocol", "base_version": 1, "value": {**protocol, "effect_size": 0.5}},
    }
    assert c.post(f"{AT_PROTOCOL}/actions", json=body, headers=WRITE).status_code == 202
    run = open_run(pipelines_root, "sleep-study-demo", "3-protocol")
    runner.join(run, timeout=120)
    view = c.get(f"{AT_PROTOCOL}/protocol").json()
    assert view["version"] == 2 and view["by"] and view["edited_from"] == 1
    assert view["protocol"]["n_per_condition"] == 64 and view["allocation_rows"] == 128
    document = (run.root / view["protocol"]["markdown_path"]).read_text()
    assert "64 units per condition" in document and "d = 0.5" in document
    gate = run_status(run).open_gates[0]
    assert gate.gate == "protocol_review" and gate.summary["artifacts"]["protocol"] == 2


def test_the_data_checkpoint_takes_a_file_that_follows_the_protocol(pipelines_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(pipelines_root, runner)
    view = c.get(f"{AT_DATA}/collect").json()
    assert view["waiting"] and view["can_upload"] and view["received"] is None and not view["inbox_file"]
    assert view["columns"] == ["unit", "condition", "recall_gain"] and view["n_per_condition"] == 26
    assert len(view["template"].splitlines()) == 53  # a header and a row for each of the 52 units

    put = {"headers": {**WRITE, "content-type": "text/csv"}}
    good = filled(view["template"])
    broken = good.replace(",phase_locked,", ",coffee,", 1)
    refused = c.put(f"{AT_DATA}/collect/inbox", content=broken, **put)
    assert refused.status_code == 422 and "condition 'coffee' is not one of" in refused.json()["detail"]
    assert not c.get(f"{AT_DATA}/collect").json()["inbox_file"]  # nothing was kept
    assert c.put(f"{AT_DATA}/collect/inbox", content=good).status_code == 403  # writes need the header

    sent = c.put(f"{AT_DATA}/collect/inbox", content=good, **put)
    assert sent.status_code == 201 and sent.json() == {"rows": 52, "units": {"phase_locked": 26, "sham": 26}}
    assert c.get(f"{AT_DATA}/collect").json()["inbox_file"]

    approve = {"kind": "approve", "stage": "collect", "gate": "data_upload"}
    assert c.post(f"{AT_DATA}/actions", json=approve, headers=WRITE).status_code == 202
    run = open_run(pipelines_root, "sleep-study-demo", "4-data")
    runner.join(run, timeout=180)
    assert [g.gate for g in run_status(run).open_gates] == ["final_report"]
    after = c.get(f"{AT_DATA}/collect").json()
    assert after["received"]["rows"] == 52 and not after["can_upload"] and "read already" in after["why_not"]
    assert c.put(f"{AT_DATA}/collect/inbox", content=good, **put).status_code == 409


def test_a_run_without_the_study_stages_has_no_study_pages(runs_root: Path) -> None:
    c = web_client(runs_root, StubRunner())
    assert c.get("/api/runs/sleep-demo/1-literature/protocol").status_code == 404
    assert c.get("/api/runs/sleep-demo/1-literature/collect").status_code == 404


def test_a_design_edit_rewrites_the_protocol_and_the_checkpoint_can_still_be_approved(pipelines_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(pipelines_root, runner)
    plan = c.get(f"{AT_PROTOCOL}/plan").json()["plan"]
    design = {
        **plan,
        "conditions": [
            *plan["conditions"],
            {"name": "Fixed Interval", "description": "untimed clicks", "is_baseline": False},
        ],
    }
    checked = c.post(f"{AT_PROTOCOL}/plan/check", json={"base_version": 1, "value": design}, headers=WRITE).json()
    assert checked["ok"] and [x["name"] for x in checked["plan"]["conditions"]][-1] == "fixed_interval"
    assert checked["plan"]["issues"] == []  # no resource checks: the table is uploaded later
    refused = c.post(
        f"{AT_PROTOCOL}/plan/check",
        json={"base_version": 1, "value": {**plan, "conditions": plan["conditions"][:1]}},
        headers=WRITE,
    ).json()
    assert not refused["ok"] and "at least two conditions" in refused["error"]

    body = {"kind": "edit", "payload": {"artifact": "experiment_plan", "base_version": 1, "value": design}}
    assert c.post(f"{AT_PROTOCOL}/actions", json=body, headers=WRITE).status_code == 202
    run = open_run(pipelines_root, "sleep-study-demo", "3-protocol")
    runner.join(run, timeout=120)
    view = c.get(f"{AT_PROTOCOL}/protocol").json()
    assert view["version"] == 2 and view["by"] and view["edited_from"] == 1  # the protocol followed the plan
    protocol = view["protocol"]
    assert protocol["conditions"] == ["sham", "phase_locked", "fixed_interval"] and protocol["n_per_condition"] > 26
    assert view["allocation_rows"] == protocol["n_per_condition"] * 3
    assert c.get(f"{AT_PROTOCOL}/plan").json()["version"] == 2
    gate = run_status(run).open_gates[0]
    assert gate.gate == "protocol_review" and gate.summary["artifacts"] == {"experiment_plan": 2, "protocol": 2}

    approve = {"kind": "approve", "stage": "protocol", "gate": "protocol_review"}  # bound to both new versions
    assert c.post(f"{AT_PROTOCOL}/actions", json=approve, headers=WRITE).status_code == 202
    runner.join(run, timeout=180)
    assert [g.gate for g in run_status(run).open_gates] == ["data_upload"]
    columns = c.get(f"{AT_PROTOCOL}/collect").json()
    assert columns["conditions"] == ["sham", "phase_locked", "fixed_interval"]


def test_a_protocol_edit_that_fails_a_check_puts_it_in_the_plans_problems(pipelines_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(pipelines_root, runner)
    protocol = c.get(f"{AT_PROTOCOL}/protocol").json()["protocol"]
    body = {
        "kind": "edit",
        "payload": {"artifact": "protocol", "base_version": 1, "value": {**protocol, "blinding": ""}},
    }
    assert c.post(f"{AT_PROTOCOL}/actions", json=body, headers=WRITE).status_code == 202
    run = open_run(pipelines_root, "sleep-study-demo", "3-protocol")
    runner.join(run, timeout=120)
    plan = c.get(f"{AT_PROTOCOL}/plan").json()
    assert plan["version"] == 2 and any(
        i.startswith("protocol checklist: blinding is stated") for i in plan["plan"]["issues"]
    )
    fixed = {
        "kind": "edit",
        "payload": {"artifact": "protocol", "base_version": 2, "value": {**protocol, "blinding": "double blind"}},
    }
    assert c.post(f"{AT_PROTOCOL}/actions", json=fixed, headers=WRITE).status_code == 202
    runner.join(run, timeout=120)
    assert not any(i.startswith("protocol checklist") for i in c.get(f"{AT_PROTOCOL}/plan").json()["plan"]["issues"])
