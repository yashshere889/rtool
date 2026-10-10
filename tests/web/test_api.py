"""The web API against demo runs: who may call it, what it shows, and decisions going through the inbox."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from research_tool.core.actions import ActionInbox  # noqa: E402
from research_tool.service import open_run, run_status  # noqa: E402
from research_tool.testing.web import BASE, TOKEN, WRITE, StubRunner  # noqa: E402
from research_tool.testing.web import web_client as client  # noqa: E402
from research_tool.web.runner import LocalRunner  # noqa: E402


# ---------------------------------------------------------------- security
def test_only_the_token_holder_gets_in(runs_root: Path) -> None:
    c = client(runs_root, signed_in=False)
    assert c.get("/api/health").json()["ok"] is True  # public
    assert c.get("/api/runs").status_code == 401
    assert c.post("/api/session", json={"token": TOKEN}).status_code == 403  # no X-Requested-With
    assert c.post("/api/session", json={"token": "wrong-token-123456"}, headers=WRITE).status_code == 401
    response = c.post("/api/session", json={"token": TOKEN}, headers=WRITE)
    assert response.status_code == 204 and "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=strict" in response.headers["set-cookie"].lower()
    assert c.get("/api/runs").status_code == 200  # the cookie works
    bearer = TestClient(c.app, base_url=BASE)
    assert bearer.get("/api/runs", headers={"authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_other_sites_and_hosts_are_refused(runs_root: Path) -> None:
    c = client(runs_root)
    assert c.get("/api/runs", headers={"host": "evil.example:8765"}).status_code == 400  # DNS rebinding
    assert c.get("/api/runs", headers={"host": "localhost:5173"}).status_code == 200  # the Vite dev proxy
    action = {"kind": "approve", "stage": "literature", "gate": "literature_review"}
    url = "/api/runs/sleep-demo/1-literature/actions"
    assert c.post(url, json=action).status_code == 403  # a plain cross-site form/fetch cannot add the header
    assert c.post(url, json=action, headers={**WRITE, "origin": "https://evil.example"}).status_code == 403
    headers = c.get("/api/runs").headers
    assert headers["x-content-type-options"] == "nosniff" and headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs/sleep-demo/..%2F..%2Fetc",
        "/api/runs/../../etc/passwd",
        "/api/runs/sleep-demo/1-literature/files/../../../../etc/passwd",
        "/api/runs/sleep-demo/1-literature/files/%2Fetc%2Fpasswd",
        "/api/runs/sleep-demo/1-literature/files/work/literature/kb.sqlite",  # not a kind of file that is served
        "/api/runs/nope/1-literature",
    ],
)
def test_nothing_outside_the_runs_is_served(runs_root: Path, path: str) -> None:
    response = client(runs_root).get(path)
    assert "root:" not in response.text
    # 404, or (when the client itself normalises "/api/runs/../../etc" to "/etc/...") the app shell, never the file
    assert response.status_code == 404 or "Research Tool" in response.text


def test_the_app_shell_never_serves_files_outside_its_folder(runs_root: Path) -> None:
    c = client(runs_root, signed_in=False)
    for path in ("/..%2F..%2F..%2F..%2Fetc%2Fpasswd", "/assets/..%2F..%2F..%2Fpyproject.toml"):
        response = c.get(path)
        assert "root:" not in response.text and "[project]" not in response.text and "Research Tool" in response.text


# ------------------------------------------------------------------- reads
def test_runs_and_a_run_at_a_checkpoint(runs_root: Path) -> None:
    c = client(runs_root)
    listed = {r["run_id"]: r for r in c.get("/api/runs").json()}
    assert set(listed) == {"1-literature", "2-hypotheses", "3-plan", "4-gpu-runs"}
    assert listed["1-literature"]["topic"].startswith("Closed-loop auditory stimulation")
    run = c.get("/api/runs/sleep-demo/1-literature").json()
    gate = run["open_gates"][0]
    assert (gate["stage"], gate["gate"], gate["actions"]) == ("literature", "literature_review", ["approve", "reject"])
    assert gate["summary"]["review_files"] == ["work/literature/knowledge_map.md"]  # run-relative, never absolute
    assert "path" not in run and str(runs_root) not in json.dumps(run)
    assert [s["status"] for s in run["stages"]][:2] == ["awaiting_gate", "pending"]


def test_artifacts_with_versions_and_etags(runs_root: Path) -> None:
    c = client(runs_root)
    names = {a["name"]: a for a in c.get("/api/runs/sleep-demo/1-literature/artifacts").json()}
    assert {"research_brief", "corpus", "knowledge_map"} <= set(names)
    assert names["corpus"]["producer"]["agent"] == "literature"
    first = c.get("/api/runs/sleep-demo/1-literature/artifacts/corpus")
    corpus = first.json()["payload"]
    assert corpus["stats"]["full_text"] == 9
    again = c.get(
        "/api/runs/sleep-demo/1-literature/artifacts/corpus", headers={"if-none-match": first.headers["etag"]}
    )
    assert again.status_code == 304
    assert c.get("/api/runs/sleep-demo/1-literature/artifacts/report").status_code == 404
    schemas = c.get("/api/artifact-types").json()
    assert "experiment_plan" in schemas and schemas["experiment_plan"]["title"] == "ExperimentPlan"


def test_papers_are_served_as_files_with_ranges(runs_root: Path) -> None:
    c = client(runs_root)
    corpus = c.get("/api/runs/sleep-demo/1-literature/artifacts/corpus").json()["payload"]
    pdf = next(p["file"] for p in corpus["papers"] if (p["file"] or "").endswith(".pdf"))
    whole = c.get(f"/api/runs/sleep-demo/1-literature/files/{pdf}")
    assert whole.status_code == 200 and whole.headers["content-type"] == "application/pdf"
    assert whole.content.startswith(b"%PDF") and "attachment" not in whole.headers.get("content-disposition", "")
    part = c.get(f"/api/runs/sleep-demo/1-literature/files/{pdf}", headers={"range": "bytes=0-7"})
    assert part.status_code == 206 and part.content == b"%PDF-1.4"
    html = next(p["file"] for p in corpus["papers"] if (p["file"] or "").endswith(".html"))
    page = c.get(f"/api/runs/sleep-demo/1-literature/files/{html}")
    assert page.status_code == 200 and "sandbox" in page.headers["content-security-policy"]
    md = c.get("/api/runs/sleep-demo/1-literature/files/work/literature/knowledge_map.md")
    assert md.headers["content-type"].startswith("text/plain")  # text is never rendered as a page


def test_events_page_and_stream(runs_root: Path) -> None:
    c = client(runs_root)
    page = c.get("/api/runs/sleep-demo/1-literature/events", params={"limit": 5}).json()
    assert len(page["events"]) <= 5 and page["cursor"] > 0
    assert all(e["kind"] != "llm_call" for e in page["events"])
    rest = c.get("/api/runs/sleep-demo/1-literature/events", params={"after": page["cursor"]}).json()
    assert "gate_opened" in [e["kind"] for e in rest["events"]]
    with c.stream("GET", "/api/runs/sleep-demo/1-literature/events/stream", params={"after": 0, "timeout": 1}) as s:
        text = "".join(s.iter_text())
    assert "event: ready" in text and "event: events" in text and '"literature_phase_done"' in text


# ---------------------------------------------------------------- decisions
def test_a_decision_is_queued_bound_to_the_gate_and_can_be_undone(runs_root: Path) -> None:
    runner = StubRunner()
    c = client(runs_root, runner)
    url = "/api/runs/sleep-demo/1-literature/actions"
    assert c.post(url, json={"kind": "reject", "stage": "literature", "gate": "literature_review"}, headers=WRITE)
    missing = c.post(url, json={"kind": "reject", "stage": "literature", "gate": "literature_review"}, headers=WRITE)
    assert missing.status_code == 422 and "what should change" in missing.json()["detail"]
    wrong = c.post(url, json={"kind": "approve", "stage": "hypotheses", "gate": "hypothesis_selection"}, headers=WRITE)
    assert wrong.status_code == 409
    queued = c.post(url, json={"kind": "approve", "stage": "literature", "gate": "literature_review"}, headers=WRITE)
    assert queued.status_code == 202 and runner.woken == ["action:approve"]
    action = queued.json()["action"]
    assert action["payload"]["expect"]["gate_nonce"] and action["by"].endswith("(web)")
    again = c.post(url, json={"kind": "approve", "stage": "literature", "gate": "literature_review"}, headers=WRITE)
    assert again.status_code == 409  # one decision per checkpoint at a time
    assert c.get(f"{url}/{action['id']}").json()["state"] == "queued"
    assert c.delete(f"{url}/{action['id']}", headers=WRITE).status_code == 204
    assert c.get(f"{url}/{action['id']}").json()["state"] == "unknown"
    assert c.delete(f"{url}/{action['id']}", headers=WRITE).status_code == 409
    run = open_run(runs_root, "sleep-demo", "1-literature")
    assert ActionInbox(run.actions_dir).pending() == []


def test_selecting_a_hypothesis_is_checked(runs_root: Path) -> None:
    c = client(runs_root)
    url = "/api/runs/sleep-demo/2-hypotheses/actions"
    body = {"kind": "select", "stage": "hypotheses", "gate": "hypothesis_selection", "payload": {"index": 7}}
    assert c.post(url, json=body, headers=WRITE).status_code == 422
    body["payload"] = {"index": 1}
    assert c.post(url, json=body, headers=WRITE).status_code == 202


def test_an_approval_advances_the_run_locally(runs_root: Path) -> None:
    runner = LocalRunner()
    c = client(runs_root, runner)
    url = "/api/runs/sleep-demo/1-literature/actions"
    queued = c.post(url, json={"kind": "approve", "stage": "literature", "gate": "literature_review"}, headers=WRITE)
    assert queued.status_code == 202 and queued.json()["kick"]["submitted"]
    run = open_run(runs_root, "sleep-demo", "1-literature")
    runner.join(run, timeout=120)
    assert c.get(f"{url}/{queued.json()['action']['id']}").json()["state"] == "applied"
    status = run_status(run)
    assert status.open_gates[0].gate == "hypothesis_selection" and status.stages[0].status.value == "done"


def test_starting_new_research(runs_root: Path) -> None:
    runner = StubRunner()
    c = client(runs_root, runner)
    body = {"topic": "Does sound during deep sleep help older adults remember?", "review": "gpu_and_final"}
    created = c.post("/api/runs", json=body, headers=WRITE)
    assert created.status_code == 201, created.text
    run = created.json()["run"]
    assert run["project"] == "sound-deep-sleep-help" and run["topic"] == body["topic"] and runner.woken == ["new run"]
    handle = open_run(runs_root, run["project"], run["run_id"])
    assert handle.config.gates.mode("literature_review").value == "auto"
    assert handle.config.gates.mode("gpu_runs").value == "ask"
    assert c.post("/api/runs", json={"topic": "x"}, headers=WRITE).status_code == 422


def test_the_app_shell_and_unknown_api_paths(runs_root: Path) -> None:
    c = client(runs_root, signed_in=False)
    for path in ("/", "/runs/sleep-demo/1-literature/literature"):
        page = c.get(path)
        assert page.status_code == 200 and "Research Tool" in page.text
        assert "script-src 'self'" in page.headers["content-security-policy"]
    assert c.get("/api/nope", headers={"authorization": f"Bearer {TOKEN}"}).status_code == 404
