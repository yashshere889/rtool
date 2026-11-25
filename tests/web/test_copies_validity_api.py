"""Validity notes and cards, replaying a stage in a copy, and archiving a run."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research_tool.core.fs import atomic_write_json  # noqa: E402
from research_tool.service import list_runs, open_run  # noqa: E402
from research_tool.testing.web import WRITE, StubRunner  # noqa: E402
from research_tool.testing.web import web_client as client  # noqa: E402
from research_tool.validity.card import build_card  # noqa: E402
from research_tool.validity.score import RunScore  # noqa: E402
from research_tool.web.api.copies import wait_for_archive  # noqa: E402

RUN = "/api/runs/sleep-demo/3-plan"


def write_card(runs_root: Path, pack: str, *, passed: bool, created: str) -> None:
    score = RunScore(
        run_id="r1", case="c", variant="main", kind="null", pack=pack, finished=True, level=3 if passed else 1,
        max_level=3, confirmed=True, repeat=1, commit="abc123",
    )  # fmt: skip
    card = build_card(pack, [(None, score)], provenance={"model": "Qwen/Qwen3.6-35B-A3B", "code": "abc123"})
    card.created_at = created
    folder = runs_root / ".validity-cards"
    folder.mkdir(exist_ok=True)
    atomic_write_json(folder / f"{pack}-{created[:4]}.json", card.model_dump(mode="json"))


def test_a_field_without_a_card_claims_no_level(runs_root: Path) -> None:
    note = client(runs_root).get(f"{RUN}/validity").json()
    assert note["pack"] == "generic" and not note["measured"] and note["runs"] == 0
    assert "no level is claimed" in note["text"]
    assert client(runs_root).get("/api/validity").json() == []


def test_the_newest_card_of_the_field_is_what_the_report_may_say(runs_root: Path) -> None:
    write_card(runs_root, "generic", passed=False, created="2026-11-01T00:00:00+00:00")
    write_card(runs_root, "generic", passed=True, created="2026-11-20T00:00:00+00:00")
    write_card(runs_root, "omics", passed=True, created="2026-11-02T00:00:00+00:00")
    (runs_root / ".validity-cards" / "broken.json").write_text("{not json")
    c = client(runs_root)
    note = c.get(f"{RUN}/validity").json()
    assert note["measured"] and note["runs"] == 1 and note["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert note["top_level"] == [{"kind": "null", "k": 1, "n": 1, "low": pytest.approx(0.2065, abs=1e-3), "high": 1.0}]
    assert "never claimed as trustworthy" in note["text"] and "false claims on null cases 0/1" in note["text"]
    cards = {card["pack"]: card for card in c.get("/api/validity").json()}
    assert set(cards) == {"generic", "omics"} and cards["generic"]["created_at"].startswith("2026-11-20")
    assert [r.project for r in list_runs(runs_root)] and ".validity-cards" not in {
        r.project for r in list_runs(runs_root)
    }


def test_a_stage_is_redone_in_a_copy_and_the_original_stays(runs_root: Path) -> None:
    runner = StubRunner()
    c = client(runs_root, runner)
    before = (runs_root / "sleep-demo" / "3-plan" / "run.json").read_bytes()
    made = c.post(f"{RUN}/replay", json={"stage": "plan", "tag": "again"}, headers=WRITE)
    assert made.status_code == 201, made.text
    assert made.json()["run_id"] == "3-plan-plan-again" and runner.woken == ["replay"]
    copy = open_run(runs_root, "sleep-demo", "3-plan-plan-again")
    assert copy.run_id == "3-plan-plan-again"
    stages = {s["id"]: s for s in c.get("/api/runs/sleep-demo/3-plan-plan-again").json()["stages"]}
    assert stages["plan"]["status"] == "stale" and stages["literature"]["status"] == "done"
    assert (runs_root / "sleep-demo" / "3-plan" / "run.json").read_bytes() == before
    again = c.post(f"{RUN}/replay", json={"stage": "plan", "tag": "again"}, headers=WRITE)
    assert again.status_code == 409 and "already exists" in again.json()["detail"]
    assert c.post(f"{RUN}/replay", json={"stage": "nope"}, headers=WRITE).status_code == 409
    assert c.post(f"{RUN}/replay", json={"stage": "plan", "tag": "../x"}, headers=WRITE).status_code == 422
    assert c.post(f"{RUN}/replay", json={"stage": "plan"}).status_code == 403  # writes need the header
    assert c.post("/api/runs/sleep-demo/1-literature/replay", json={"stage": "code"}, headers=WRITE).status_code == 409
    # a stage whose earlier stages have not finished has no complete inputs: refused


def test_archiving_needs_a_folder_and_packs_the_run_in_the_background(runs_root: Path, tmp_path: Path) -> None:
    c = client(runs_root)
    url = f"{RUN}/archive"
    view = c.get(url).json()
    assert not view["can_archive"] and "archive_root" in view["why_not"] and view["state"] == "none"
    assert view["run_bytes"] > 0 and view["default_max_file_mb"] == 200
    assert c.post(url, json={}, headers=WRITE).status_code == 409

    run = open_run(runs_root, "sleep-demo", "3-plan")
    spec = json.loads((run.root / "run.json").read_text())
    spec["config"]["paths"]["archive_root"] = str(tmp_path / "archive")
    (run.root / "run.json").write_text(json.dumps(spec))
    assert c.get(url).json()["can_archive"]
    assert c.post(url, json={"max_file_mb": 0}, headers=WRITE).status_code == 422
    assert c.post(url, json={"max_file_mb": 500}, headers=WRITE).status_code == 422  # lower the cap, never lift it
    assert c.post(url, json={}).status_code == 403
    started = c.post(url, json={"max_file_mb": 0.05}, headers=WRITE)
    assert started.status_code == 202 and started.json()["state"] in ("running", "done")
    wait_for_archive(run.root)
    done = c.get(url).json()
    assert done["state"] == "done" and done["files"] > 10 and done["path"].endswith("sleep-demo-3-plan.tar.gz")
    assert done["skipped"] and all(s["bytes"] > 50_000 for s in done["skipped"])  # files above the cap were left out
    with tarfile.open(done["path"]) as tar:
        names = tar.getnames()
    assert any(n.endswith("/MANIFEST.json") for n in names) and any(n.endswith("/run.json") for n in names)
    assert shutil.which("tar") is None or Path(done["path"]).stat().st_size > 0
