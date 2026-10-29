"""The Hypotheses page's API on the demo run: the shortlist with its papers, edited choices, changing the choice."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research_tool.core.artifacts import ArtifactStore  # noqa: E402
from research_tool.service import open_run, run_status  # noqa: E402
from research_tool.testing.web import WRITE, web_client  # noqa: E402
from research_tool.web.runner import LocalRunner  # noqa: E402

CHOOSING = "/api/runs/sleep-demo/2-hypotheses"
CHOSEN = "/api/runs/sleep-demo/3-plan"
GATE = {"stage": "hypotheses", "gate": "hypothesis_selection"}


def test_three_hypotheses_with_their_papers(runs_root: Path) -> None:
    view = web_client(runs_root).get(f"{CHOOSING}/hypotheses").json()
    assert view["stage"] == "hypotheses" and view["versions"] == [1] and view["selection"] is None
    assert [c["index"] for c in view["shortlist"]] == [0, 1, 2]
    assert sum(view["weights"].values()) == pytest.approx(1.0) and view["markdown"].endswith("hypotheses.md")
    for card in view["shortlist"]:
        h = card["hypothesis"]
        assert h["evidence"] and h["rubric_comments"] and h["novelty"]["closest_paper_id"]
        roles = {p["paper"]["id"]: p["roles"] for p in card["papers"]}
        assert {e["paper_id"] for e in h["evidence"]} <= {pid for pid, r in roles.items() if "evidence" in r}
        assert "closest" in roles[h["novelty"]["closest_paper_id"]]
        assert all(f"{{{p['paper']['bib_key']}," in card["bibtex"] for p in card["papers"])
        assert card["gpu_hours"] == {"cpu_only": 0, "small_gpu": 8}[h["compute"]]
    assert view["others"] and all(c["index"] is None for c in view["others"])
    assert web_client(runs_root).get(f"{CHOOSING}/hypotheses", params={"version": 9}).status_code == 404


def test_an_edited_choice_is_checked_then_kept(runs_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(runs_root, runner)
    url = f"{CHOOSING}/actions"

    def select(edit: dict) -> object:
        return c.post(url, json={"kind": "select", **GATE, "payload": {"index": 1, "edit": edit}}, headers=WRITE)

    bad = select({"compute": "huge_gpu"})
    assert bad.status_code == 422 and "compute" in bad.json()["detail"]
    assert select({"colour": "red"}).status_code == 422
    statement = "If the detection threshold adapts to each older adult, slow-oscillation responses match the young"
    assert select({"statement": statement, "datasets": ["MASS", "Sleep-EDF"]}).status_code == 202
    run = open_run(runs_root, "sleep-demo", "2-hypotheses")
    runner.join(run, timeout=120)
    selection = c.get(f"{CHOOSING}/hypotheses").json()["selection"]
    assert selection["index"] == 1 and selection["edited"] and selection["from_version"] == 1
    assert {ch["field"]: ch["after"] for ch in selection["changes"]} == {
        "statement": statement,
        "datasets": ["MASS", "Sleep-EDF"],
    }
    assert ArtifactStore(run.artifacts_dir).get("selected_hypothesis").item["statement"] == statement


def test_changing_my_choice(runs_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(runs_root, runner)
    impact = {a["stage"]: a["effect"] for a in c.get(f"{CHOSEN}/stages/hypotheses/impact").json()}
    assert impact == {"plan": "redo", "fetch": "not_started", "code": "not_started", "report": "not_started"} | {
        "review": "not_started"
    }
    assert c.get(f"{CHOSEN}/stages/nope/impact").status_code == 404

    url = f"{CHOSEN}/actions"
    assert c.post(url, json={"kind": "reopen", "stage": "plan"}, headers=WRITE).status_code == 409  # not finished
    assert c.post(url, json={"kind": "reopen", "stage": "fetch"}, headers=WRITE).status_code == 422  # no checkpoint
    assert c.post(url, json={"kind": "reopen", "stage": "hypotheses"}, headers=WRITE).status_code == 202
    run = open_run(runs_root, "sleep-demo", "3-plan")
    runner.join(run, timeout=120)
    gate = next(g for g in run_status(run).open_gates if g.stage == "hypotheses")
    assert gate.actions == ["select", "approve", "reject"] and gate.summary["current"]["index"] == 0

    kept = c.post(url, json={"kind": "approve", **GATE}, headers=WRITE)  # keep the current choice
    assert kept.status_code == 202
    runner.join(run, timeout=120)
    status = run_status(run)
    assert [g.stage for g in status.open_gates] == ["plan"]  # the plan was not touched
    assert ArtifactStore(run.artifacts_dir).head("experiment_plan") == 1


def test_a_quote_is_found_in_the_paper(runs_root: Path) -> None:
    c = web_client(runs_root)
    card = c.get(f"{CHOOSING}/hypotheses").json()["shortlist"][0]
    evidence = card["hypothesis"]["evidence"][0]
    url = "/api/runs/sleep-demo/2-hypotheses/literature/paper/text"
    text = c.get(url, params={"id": evidence["paper_id"], "quote": evidence["quote"]}).json()
    focus = text["focus"]
    span = text["sections"][focus["section"]]["text"][focus["start"] : focus["end"]]
    assert focus["claim"] == -1 and " ".join(span.split()) == " ".join(evidence["quote"].split())
    assert c.get(url, params={"id": evidence["paper_id"], "quote": "nothing like this at all"}).json()["focus"] is None
