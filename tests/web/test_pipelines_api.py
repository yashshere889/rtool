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
