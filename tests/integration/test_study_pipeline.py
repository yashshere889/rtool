"""A study a person carries out: the protocol, the uploaded data (checked), its analysis and the report.

The demo model answers, except that the protocol stage gets the trial design written here.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from research_tool.cli.main import app
from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.config import deep_merge, load_config
from research_tool.core.engine import Outcome
from research_tool.core.pipeline import find_pipeline, pipeline_from
from research_tool.core.run import RunHandle
from research_tool.core.state import StageStatus, StateStore
from research_tool.testing.demo_run import CLUSTER, advance, demo_services
from research_tool.testing.harness import queue
from research_tool.validity.cases import load_case
from research_tool.validity.runs import create_case_run, generate_data, seeded_artifacts
from research_tool.validity.score import score_run

CASES = Path(__file__).resolve().parents[2] / "benchmarks" / "study"
DESIGN = {
    "objective": "Compare reaction time after caffeine and after placebo",
    "design": "A parallel, two-arm, randomised, double-blind trial",
    "independent_variables": [
        {"name": "drink", "description": "placebo or caffeine", "values": ["placebo", "caffeine"]}
    ],
    "dependent_variables": [{"name": "reaction_time_ms", "description": "mean simple reaction time"}],
    "controls": ["identical drinks", "the same test at the same time of day"],
    "conditions": [
        {"name": "placebo", "description": "decaffeinated drink", "is_baseline": True},
        {"name": "caffeine", "description": "drink with 200 mg caffeine"},
    ],
    "ablations": [],
    "metrics": [
        {
            "name": "reaction_time_ms",
            "definition": "mean simple reaction time",
            "higher_is_better": False,
            "primary": True,
        }
    ],
    "statistical_tests": [{"test": "Welch t-test", "comparison": "caffeine vs placebo"}],
    "n_seeds": 1,
    "procedure": [
        "Recruit adults",
        "Randomise by the allocation table",
        "Measure 30 minutes later",
        "Record one row each",
    ],
    "expected_results": "Shorter reaction times after caffeine",
    "risks": ["habitual coffee drinkers may respond less"],
    "unit_of_analysis": "participant",
    "expected_effect_size": 0.6,
    "materials": ["decaffeinated drink", "caffeine 200 mg", "reaction-time software"],
    "blinding": "participants and the person measuring do not know which drink was given",
}


def with_design(services: Any) -> Any:
    services.llm.backend.rules.insert(0, (lambda r: (r.json_schema or {}).get("title") == "ProtocolDraft", DESIGN))
    return services


def config(root: Path, **overrides: Any) -> Any:
    return load_config(
        overrides=deep_merge(deep_merge(CLUSTER, {"paths": {"runs_root": str(root)}}), overrides), env={}
    )


@pytest.fixture(scope="module")
def trial(tmp_path_factory: pytest.TempPathFactory) -> RunHandle:
    run = create_case_run(
        config(tmp_path_factory.mktemp("study") / "runs"),
        load_case(CASES / "caffeine-reaction-time"),
        commit="0" * 12,
        variant="trial_1",
    )
    result = advance(run, with_design(demo_services(run)), max_episodes=60)
    assert result.outcome is Outcome.DONE, f"{result.outcome.value}: {result.message}"
    return run


def test_the_protocol_computes_the_sample_size_and_the_allocation(trial: RunHandle) -> None:
    store = ArtifactStore(trial.artifacts_dir)
    protocol, plan = store.get("protocol"), store.get("experiment_plan")
    assert (protocol.n_per_condition, protocol.paired, protocol.unit) == (45, False, "participant")  # d = 0.6
    assert all(c.ok for c in protocol.checklist) and plan.issues == []
    with (trial.root / protocol.allocation_path).open() as handle:
        allocation = list(csv.DictReader(handle))
    assert len(allocation) == 90 and {r["condition"] for r in allocation[:2]} == {
        "placebo",
        "caffeine",
    }  # blocks of two
    assert "Upload it with `rtool add-data <run> data.csv`" in (trial.root / protocol.markdown_path).read_text()


def test_the_uploaded_trial_is_analysed_across_participants(trial: RunHandle) -> None:
    store = ArtifactStore(trial.artifacts_dir)
    data, results = store.get("study_data"), store.get("experiment_results")
    assert data.rows == 90 and data.units == {"placebo": 45, "caffeine": 45}
    primary = results.comparison("caffeine", "reaction_time_ms")
    assert primary.test == "Welch t-test" and primary.n == 45 and primary.significant and primary.better
    assert abs(primary.difference - -39.8) < 0.1  # the reference's difference for this trial
    assert (trial.root / store.get("code_bundle").root / "experiment.py").is_file()  # rtool's script, audited
    score = score_run(trial)
    assert (score.level, score.category) == (3, ""), [c for c in score.checks if not c.ok]


def test_the_run_waits_for_the_data_and_refuses_a_broken_file(tmp_path: Path) -> None:
    case = load_case(CASES / "caffeine-reaction-time")
    gates = {
        "gates": {
            "default": "ask",
            "modes": {"protocol_review": "auto", "final_report": "auto", "revision_limit": "auto"},
        }
    }
    cfg = config(tmp_path / "runs", **gates)
    spec = pipeline_from(find_pipeline("study"), "protocol")
    run = RunHandle.create(cfg.paths.resolved_runs_root(), "study", spec, cfg, run_id="trial")
    store = ArtifactStore(run.artifacts_dir)
    for name, value in seeded_artifacts(case).items():
        store.put(name, value, producer=Producer(stage="user", agent="test"))
    services = with_design(demo_services(run))
    waiting = advance(run, services)
    assert waiting.outcome is Outcome.WAITING_GATE and [g["gate"] for g in waiting.open_gates] == ["data_upload"]

    good = Path(generate_data(case, "trial_1", tmp_path / "data")["study_data"]["path"]) / "data.csv"
    broken = tmp_path / "broken.csv"
    broken.write_text(good.read_text().replace(",caffeine,", ",coffee,", 1).replace(",placebo,", ",placebo,abc", 1))
    runner = CliRunner()
    result = runner.invoke(app, ["add-data", str(run.root), str(broken)])
    assert result.exit_code == 0 and "queued approve" in result.output, result.output
    advance(run, services)
    collect = StateStore(run.state_dir).load_stage("collect")
    assert collect is not None and collect.status is StageStatus.FAILED
    assert "condition 'coffee' is not one of placebo, caffeine" in (collect.last_error or "")
    assert "is 'abc" in (collect.last_error or "") or "not a number" in (collect.last_error or "")

    assert runner.invoke(app, ["add-data", str(run.root), str(good)]).exit_code == 0  # no gate open now: just copied
    queue(run, ActionKind.RETRY, stage="collect")
    done = advance(run, services, max_episodes=60)
    assert done.outcome is Outcome.DONE, f"{done.outcome.value}: {done.message}"
    assert ArtifactStore(run.artifacts_dir).get("study_data").units == {"placebo": 45, "caffeine": 45}
