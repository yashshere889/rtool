"""The Report and Review pages' API on the demo runs: sections, checks of an edit, the review, the bundle."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from research_tool.service import open_run, run_status  # noqa: E402
from research_tool.testing.web import WRITE, web_client  # noqa: E402
from research_tool.web.runner import LocalRunner  # noqa: E402

AT_GATE = "/api/runs/sleep-demo/5-final-report"


def with_section(view: dict[str, Any], section: str, template: str) -> dict[str, Any]:
    sections = [dict(s, template=template) if s["id"] == section else s for s in view["sections"]]
    fields = ("id", "title", "template", "origin")
    return {"title": view["title"], "sections": [{k: s[k] for k in fields} for s in sections]}


def test_the_report_as_sections(runs_root: Path) -> None:
    c = web_client(runs_root)
    view = c.get(f"{AT_GATE}/report").json()
    assert view["artifact"] == "report" and view["stage"] == "report" and view["editable"]
    origins = {s["id"]: (s["origin"], s["editable"]) for s in view["sections"]}
    assert origins["methods"] == ("rtool", False) and origins["discussion"] == ("llm", True)
    results = next(s for s in view["sections"] if s["id"] == "results")
    assert "{{" in results["template"] and "{{" not in results["rendered"] and not results["problems"]
    assert view["placeholders"] and all(p["meaning"] for p in view["placeholders"])
    assert view["references"] and {r["key"] for r in view["references"]} <= {p["key"] for p in view["papers"]}
    assert view["base_dir"] == str(Path(view["markdown_path"]).parent)
    final = c.get(f"{AT_GATE}/report", params={"which": "final_report"}).json()
    assert final["stage"] == "review" and final["editable"]  # while the final checkpoint is open
    assert c.get("/api/runs/sleep-demo/4-gpu-runs/report").status_code == 404


def test_an_edit_is_checked_before_it_is_sent(runs_root: Path) -> None:
    c = web_client(runs_root)
    view = c.get(f"{AT_GATE}/report").json()
    url = f"{AT_GATE}/report/check"

    def check(value: dict[str, Any]) -> dict[str, Any]:
        body = {"artifact": "report", "base_version": view["version"], "value": value}
        return c.post(url, json=body, headers=WRITE).json()

    bad = check(with_section(view, "discussion", "We saw {{mean:nothing:here}}."))
    assert not bad["ok"] and "unknown placeholder" in bad["error"]
    locked = check(with_section(view, "methods", "My own methods."))
    assert not locked["ok"] and "written by rtool" in locked["error"]
    key = view["papers"][0]["key"]
    metric = next(p["key"] for p in view["placeholders"] if p["key"].startswith("mean:"))
    good = check(with_section(view, "discussion", f"In short, {{{{{metric}}}}} [@{key}]. Limitations. Few seeds."))
    assert good["ok"] and good["changed"] == ["sections"]
    discussion = next(s for s in good["sections"] if s["id"] == "discussion")
    assert discussion["origin"] == "human" and "{{" not in discussion["rendered"]
    assert f"[@{key}]" in discussion["rendered"]


def test_an_applied_edit_is_reviewed_again(runs_root: Path) -> None:
    runner = LocalRunner()
    c = web_client(runs_root, runner)
    view = c.get(f"{AT_GATE}/report").json()
    value = with_section(view, "conclusion", "We conclude that timing matters.")
    body = {"kind": "edit", "payload": {"artifact": "report", "base_version": view["version"], "value": value}}
    assert c.post(f"{AT_GATE}/actions", json=body, headers=WRITE).status_code == 202
    run = open_run(runs_root, "sleep-demo", "5-final-report")
    runner.join(run, timeout=180)
    edited = c.get(f"{AT_GATE}/report").json()
    assert edited["version"] == view["version"] + 1 and edited["by"] and edited["edited_from"] == view["version"]
    status = run_status(run)
    assert [g.gate for g in status.open_gates] == ["final_report"]  # reviewed again, asking again
    assert next(s for s in status.stages if s.id == "review").attempt == 2
    final = c.get(f"{AT_GATE}/report", params={"which": "final_report"}).json()
    conclusion = next(s for s in final["sections"] if s["id"] == "conclusion")
    assert conclusion["template"] == "We conclude that timing matters." and conclusion["origin"] == "human"


def test_the_review_and_the_bundle(runs_root: Path) -> None:
    c = web_client(runs_root)
    view = c.get(f"{AT_GATE}/review").json()
    review = view["review"]
    assert review["verdict"] in ("accept", "needs_human") and len(review["checks"]) == 8
    assert set(review["scores"]) == {"soundness", "evidence", "clarity", "reproducibility", "limitations"}
    assert view["final_version"] == 1 and not view["final_edited"] and view["rounds_max"] >= 1
    assert view["report_stage"] == "report" and view["min_score"] == 3.0
    archive = c.get(f"{AT_GATE}/report/bundle")
    names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
    assert "5-final-report-report/final_report.md" in names and "5-final-report-report/refs.bib" in names
    assert any(n.endswith(".svg") for n in names) and "5-final-report-report/review.md" in names
    assert "5-final-report-report/results/results.json" in names
