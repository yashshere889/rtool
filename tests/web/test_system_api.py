"""What this deployment can do: fields, pipelines, agents and the machine's checks."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from research_tool.testing.web import web_client as client  # noqa: E402


def test_fields_say_which_packs_are_offered_and_what_they_run(runs_root: Path, tmp_path: Path) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("packs:\n  allowed: [generic, study]\n  max_gpu_hours: {generic: 12}\n", encoding="utf-8")
    fields = {f["name"]: f for f in client(runs_root, mode="job", config_files=[site]).get("/api/fields").json()}
    assert fields["generic"]["state"] == "offered" and fields["generic"]["max_gpu_hours"] == 12
    assert fields["study"]["pipeline"] == "study" and fields["study"]["origin"] == "built-in"
    assert fields["omics"]["state"] == "not_offered" and fields["medicine"]["pipeline"] == "synthesis"
    choices = {
        p["name"]: p for p in client(runs_root, mode="job", config_files=[site]).get("/api/settings").json()["packs"]
    }
    assert (
        set(choices) == {"generic", "study"}
        and choices["generic"]["max_gpu_hours"] == 12
        and choices["study"]["description"]
    )


def test_a_demo_server_offers_only_what_it_can_simulate(runs_root: Path) -> None:
    fields = {f["name"]: f["state"] for f in client(runs_root).get("/api/fields").json()}
    assert fields["generic"] == fields["study"] == fields["medicine"] == "offered" and fields["omics"] == "not_offered"


def test_pipelines_and_agents(runs_root: Path) -> None:
    c = client(runs_root)
    pipelines = {p["name"]: p for p in c.get("/api/pipelines").json()}
    assert {"research", "study", "synthesis"} <= set(pipelines)
    study = pipelines["study"]["stages"]
    assert [s["id"] for s in study][:3] == ["literature", "hypotheses", "protocol"]
    assert next(s for s in study if s["id"] == "protocol")["gate"] == "protocol_review"
    assert next(s for s in study if s["id"] == "protocol")["produces"] == ["experiment_plan", "protocol"]
    agents = {a["name"]: a for a in c.get("/api/agents").json()}
    assert agents["protocol"]["uses_llm"] and not agents["collect_data"]["uses_llm"]
    assert "study_data" in agents["collect_data"]["produces"]


def test_the_machine_checks(runs_root: Path) -> None:
    checks = {c["name"]: c for c in client(runs_root).get("/api/doctor").json()}  # no network by default
    assert checks["rtool"]["status"] == "ok" and checks["sqlite FTS5"]["status"] in ("ok", "fail")
    assert not any(name in checks for name in ("OpenAlex", "Crossref"))  # the paper APIs were not contacted
