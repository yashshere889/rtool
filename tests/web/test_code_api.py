"""The Code & runs page's API on the demo runs: files, smoke tests, editing experiment.py, runs and results."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research_tool.core.events import EventLog  # noqa: E402
from research_tool.service import open_run  # noqa: E402
from research_tool.testing.web import WRITE, StubRunner, web_client  # noqa: E402
from research_tool.web.runner import LocalRunner  # noqa: E402

AT_GATE = "/api/runs/sleep-demo/4-gpu-runs"
DONE = "/api/runs/sleep-demo/6-done"


def test_the_code_at_the_gpu_checkpoint(runs_root: Path) -> None:
    c = web_client(runs_root)
    view = c.get(f"{AT_GATE}/code").json()
    assert view["stage"] == "code" and view["status"] == "awaiting_gate" and view["phase"] == "approve"
    files = {f["path"]: f for f in view["files"]}
    assert files["experiment.py"]["role"] == "experiment" and files["experiment.py"]["editable"]
    assert files["run.py"]["role"] == "harness" and not files["run.py"]["editable"] and ".rtool.json" not in files
    assert view["editable"] and view["experiment_sha"] == view["generated_sha"] == view["tested_sha"]
    assert {s["condition"] for s in view["smoke"]} == set(view["conditions"]) and all(s["ok"] for s in view["smoke"])
    assert view["baseline"] == "sham" and len(view["runs"]) == len(view["conditions"]) * len(view["seeds"])
    assert {r["status"] for r in view["runs"]} == {"waiting"}
    assert any(log["label"].startswith("Smoke test") for log in view["logs"])

    source = c.get(f"{AT_GATE}/code/file", params={"path": "experiment.py"}).json()
    assert source["content"].startswith('"""Closed-loop') and source["sha"] == view["experiment_sha"]
    assert c.get(f"{AT_GATE}/code/generated").json()["content"] == source["content"]
    assert c.get(f"{AT_GATE}/code/file", params={"path": "../../run.json"}).status_code == 404
    assert c.get(f"{AT_GATE}/code/file", params={"path": ".venv/x"}).status_code == 404
    smoke_log = c.get(f"{AT_GATE}/code/log", params={"name": view["logs"][0]["name"]}).json()
    assert smoke_log["name"].startswith("logs/") and not smoke_log["truncated"]
    assert c.get(f"{AT_GATE}/code/log", params={"name": "code/experiment.py"}).status_code == 404

    archive = c.get(f"{AT_GATE}/code/archive")
    assert archive.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
    assert "4-gpu-runs-code/experiment.py" in names and not any(".rtool.json" in n for n in names)


def test_the_agent_checks_code_before_it_is_sent(runs_root: Path) -> None:
    c = web_client(runs_root)
    bad = c.post(f"{AT_GATE}/code/check", json={"content": "def run(condition, seed:\n"}, headers=WRITE).json()
    assert bad["problems"] and bad["problems"][0].startswith("SyntaxError")
    contract = c.post(f"{AT_GATE}/code/check", json={"content": "def run(x):\n    return {}\n"}, headers=WRITE).json()
    assert "must accept" in contract["problems"][0]
    good = 'def run(condition, seed, resources, smoke, workdir):\n    return {"acc": len(condition)}\n'
    assert c.post(f"{AT_GATE}/code/check", json={"content": good}, headers=WRITE).json()["problems"] == []


def test_editing_experiment_py_and_testing_it_again(runs_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(runs_root, runner)
    view = c.get(f"{AT_GATE}/code").json()
    source = c.get(f"{AT_GATE}/code/file", params={"path": "experiment.py"}).json()["content"]
    edited = source.replace("nights = 3 if smoke else 40", "nights = 3 if smoke else 60")
    assert edited != source
    url = f"{AT_GATE}/actions"
    write = {"path": "code/experiment.py", "content": edited, "base_sha": view["experiment_sha"], "retest": True}
    body = {"kind": "write_file", "stage": "code", "payload": write}
    assert c.post(url, json={**body, "payload": {**write, "path": "code/run.py"}}, headers=WRITE).status_code == 422
    stale = c.post(url, json={**body, "payload": {**write, "base_sha": "0" * 64}}, headers=WRITE)
    assert stale.status_code == 409 and "changed since you opened it" in stale.json()["detail"]
    assert c.post(url, json=body, headers=WRITE).status_code == 202
    run = open_run(runs_root, "sleep-demo", "4-gpu-runs")
    runner.join(run, timeout=120)

    after = c.get(f"{AT_GATE}/code").json()
    assert after["status"] == "awaiting_gate" and after["experiment_sha"] != after["generated_sha"]
    assert after["tested_sha"] == after["experiment_sha"]  # the checkpoint now asks about the tested edit
    assert any(f["signature"] == "human_edit" for f in after["fixes"])
    assert "file_written" in [e["kind"] for e in EventLog(run.events_path, run.run_id)]
    approve_untested = {"kind": "reject", "stage": "code", "gate": "gpu_runs", "payload": {"retest": True}}
    assert c.post(url, json=approve_untested, headers=WRITE).status_code == 202  # "test again" needs no comment


def test_runs_and_results_of_a_finished_run(runs_root: Path) -> None:
    c = web_client(runs_root)
    view = c.get(f"{DONE}/code").json()
    assert view["phase"] == "finish" and not view["editable"] and "finished" in view["why_not_editable"]
    assert view["jobs"] and view["jobs"][0]["state"] == "COMPLETED" and view["jobs"][0]["partition"] == "gpu-l40s"
    assert {r["status"] for r in view["runs"]} == {"ok"} and all(r["log"] for r in view["runs"])
    assert set(view["runs"][0]["metrics"]) == set(view["metrics"])
    job_log = c.get(f"{DONE}/code/log", params={"name": view["runs"][0]["log"]}).json()
    assert job_log["name"].startswith(".rtool/logs/")

    results = c.get(f"{DONE}/results").json()
    assert results["baseline"] == "sham" and results["results"]["complete"]
    assert {m["name"] for m in results["metrics"]} == set(view["metrics"])
    comparison = next(x for x in results["results"]["comparisons"] if x["condition"] == "phase_locked")
    assert comparison["baseline"] == "sham" and comparison["significant"]
    assert results["results"]["figures"] and results["results"]["figures"][0].endswith(".svg")
    assert c.get(f"{AT_GATE}/results").status_code == 404


def test_an_undone_rewrite_becomes_a_draft_again(runs_root: Path) -> None:
    c = web_client(runs_root, StubRunner())
    view = c.get(f"{AT_GATE}/code").json()
    drafts = f"{AT_GATE}/drafts/file_code_experiment_py"
    draft = {"base_version": 0, "value": {"content": "# mine\n", "base_sha": view["experiment_sha"]}}
    assert c.put(drafts, json=draft, headers=WRITE).status_code == 200
    write = {"path": "code/experiment.py", "content": "# mine\n", "base_sha": view["experiment_sha"]}
    sent = c.post(f"{AT_GATE}/actions", json={"kind": "write_file", "stage": "code", "payload": write}, headers=WRITE)
    assert sent.status_code == 202 and c.get(drafts).status_code == 404
    again = c.post(f"{AT_GATE}/actions", json={"kind": "write_file", "stage": "code", "payload": write}, headers=WRITE)
    assert again.status_code == 409
    assert c.delete(f"{AT_GATE}/actions/{sent.json()['action']['id']}", headers=WRITE).status_code == 204
    assert c.get(drafts).json()["value"] == {"content": "# mine\n", "base_sha": view["experiment_sha"]}


def test_links_planted_in_the_code_folder_are_not_followed(runs_root: Path, tmp_path: Path) -> None:
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")
    code = open_run(runs_root, "sleep-demo", "4-gpu-runs").work_dir("code") / "code"
    (code / "notes.txt").symlink_to(secret)  # e.g. written by experiment code
    c = web_client(runs_root)
    assert "notes.txt" not in {f["path"] for f in c.get(f"{AT_GATE}/code").json()["files"]}
    assert c.get(f"{AT_GATE}/code/file", params={"path": "notes.txt"}).status_code == 404
    archive = zipfile.ZipFile(io.BytesIO(c.get(f"{AT_GATE}/code/archive").content))
    assert not any(n.endswith("notes.txt") for n in archive.namelist())
    assert all(b"PRIVATE KEY" not in archive.read(n) for n in archive.namelist())
