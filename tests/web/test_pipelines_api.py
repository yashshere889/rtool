"""A demo server runs all three pipelines: the study and synthesis demos, and new runs of each."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research_tool.service import open_run  # noqa: E402
from research_tool.testing.web import WRITE, StubRunner  # noqa: E402
from research_tool.testing.web import web_client as client  # noqa: E402

STUDY = ["literature", "hypotheses", "protocol", "collect", "analyse", "report", "review"]
SYNTHESIS = ["literature", "synthesis"]


def stages(c, project: str, run_id: str) -> list[str]:  # type: ignore[no-untyped-def]
    return [s["id"] for s in c.get(f"/api/runs/{project}/{run_id}").json()["stages"]]


def test_the_study_demo_stops_at_each_of_its_checkpoints(pipelines_root: Path) -> None:
    c = client(pipelines_root)
    for run_id, gate in (
        ("3-protocol", "protocol_review"),
        ("4-data", "data_upload"),
        ("5-final-report", "final_report"),
    ):
        run = c.get(f"/api/runs/sleep-study-demo/{run_id}").json()
        assert [g["gate"] for g in run["open_gates"]] == [gate], run_id
        assert [s["id"] for s in run["stages"]] == STUDY
    done = c.get("/api/runs/sleep-study-demo/6-done").json()
    assert done["open_gates"] == [] and {s["status"] for s in done["stages"]} == {"done"}
    names = {a["name"] for a in c.get("/api/runs/sleep-study-demo/6-done/artifacts").json()}
    assert {"protocol", "study_data", "experiment_results", "final_report", "review"} <= names


def test_the_synthesis_demo_stops_at_each_of_its_checkpoints(pipelines_root: Path) -> None:
    c = client(pipelines_root)
    for run_id, gate in (("1-literature", "literature_review"), ("2-review", "final_report")):
        run = c.get(f"/api/runs/sleep-review-demo/{run_id}").json()
        assert [g["gate"] for g in run["open_gates"]] == [gate] and [s["id"] for s in run["stages"]] == SYNTHESIS
    names = {a["name"] for a in c.get("/api/runs/sleep-review-demo/3-done/artifacts").json()}
    assert {"synthesis", "final_report"} <= names


def test_a_demo_server_offers_the_packs_it_can_simulate(runs_root: Path) -> None:
    packs = {p["name"]: p["pipeline"] for p in client(runs_root).get("/api/settings").json()["packs"]}
    assert packs == {"generic": "research", "study": "study", "medicine": "synthesis"}


@pytest.mark.parametrize(("pack", "expected"), [("study", STUDY), ("medicine", SYNTHESIS)])
def test_new_research_in_a_demo_server_uses_the_packs_pipeline(runs_root: Path, pack: str, expected: list[str]) -> None:
    runner = StubRunner()
    c = client(runs_root, runner)
    created = c.post(
        "/api/runs", json={"topic": "Does sound during deep sleep help memory?", "pack": pack}, headers=WRITE
    )
    assert created.status_code == 201, created.text
    run = created.json()["run"]
    assert [s["id"] for s in run["stages"]] == expected and runner.woken == ["new run"]
    assert open_run(runs_root, run["project"], run["run_id"]).spec.pack == pack  # the brief's pack, not "generic"


def test_a_demo_server_refuses_a_pack_it_cannot_simulate(runs_root: Path) -> None:
    c = client(runs_root)
    refused = c.post(
        "/api/runs", json={"topic": "Does sound during deep sleep help memory?", "pack": "omics"}, headers=WRITE
    )
    assert refused.status_code == 422 and "offers only" in refused.json()["detail"]


REVIEW = "/api/runs/sleep-review-demo/2-review"


def test_a_review_is_edited_sentence_by_sentence_and_its_report_follows(pipelines_root: Path) -> None:
    from research_tool.core.artifacts import ArtifactStore
    from research_tool.service import run_status
    from research_tool.web.runner import LocalRunner

    runner = LocalRunner()
    c = client(pipelines_root, runner)
    view = c.get(f"{REVIEW}/synthesis").json()
    assert view["editable"] and view["version"] == view["head"] == 1 and view["stage"] == "synthesis"
    review = view["synthesis"]
    first = review["sections"][0]["sentences"][0]
    claims = list(review["claims"])

    def with_sentence(sentence: dict) -> dict:  # the first sentence of the first section replaced
        sections = [dict(s) for s in review["sections"]]
        sections[0] = {**sections[0], "sentences": [sentence, *sections[0]["sentences"][1:]]}
        return {**review, "sections": sections}

    def check(value: dict) -> dict:
        return c.post(f"{REVIEW}/synthesis/check", json={"base_version": 1, "value": value}, headers=WRITE).json()

    shorter = {"text": first["text"].rstrip(".") + ", as the study reports", "claims": first["claims"], "papers": []}
    assert check(with_sentence(shorter))["ok"]
    invented = check(with_sentence({**shorter, "text": "It helped 97 % of people", "claims": first["claims"]}))
    assert not invented["ok"] and "states numbers its findings do not: 97" in invented["error"]
    unknown = check(with_sentence({**shorter, "claims": ["C999"]}))
    assert not unknown["ok"] and "were not offered: C999" in unknown["error"]
    assert "cites no finding" in check(with_sentence({**shorter, "claims": []}))["error"]
    renamed = {**review, "sections": [{**review["sections"][0], "title": "Other"}, *review["sections"][1:]]}
    assert "cannot be added, removed or renamed" in check(renamed)["error"]
    assert claims  # the findings offered are the review's, not editable

    body = {"kind": "edit", "payload": {"artifact": "synthesis", "base_version": 1, "value": with_sentence(shorter)}}
    assert c.post(f"{REVIEW}/actions", json=body, headers=WRITE).status_code == 202
    run = open_run(pipelines_root, "sleep-review-demo", "2-review")
    runner.join(run, timeout=120)
    after = c.get(f"{REVIEW}/synthesis").json()
    assert (
        after["version"] == 2
        and after["by"]
        and after["synthesis"]["sections"][0]["sentences"][0]["text"] == shorter["text"]
    )
    store = ArtifactStore(run.artifacts_dir)
    report = store.get("final_report")
    assert store.head("final_report") == 2 and report.sections[0].origin == "human"  # the report was rebuilt with it
    assert shorter["text"].rstrip(".") in (run.root / report.markdown_path).read_text()
    assert next(s for s in report.sections if s.id == "validity").origin == "rtool"  # kept as it was, locked
    assert run_status(run).open_gates[0].summary["artifacts"] == {"synthesis": 2, "final_report": 2}
    direct = {
        "kind": "edit",
        "payload": {"artifact": "final_report", "base_version": 2, "value": report.model_dump(mode="json")},
    }
    assert "edited sentence by sentence" in c.post(f"{REVIEW}/actions", json=direct, headers=WRITE).json()["detail"]
