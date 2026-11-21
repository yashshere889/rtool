"""Validity runs: seeded runs that need nobody, driven to the end by the demo model, then scored."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.config import AppConfig, GateMode, deep_merge, load_config
from research_tool.core.engine import Outcome
from research_tool.core.errors import RToolError
from research_tool.core.fs import atomic_write_json, read_json
from research_tool.core.run import RunHandle
from research_tool.core.state import GateState, StageStatus, StateStore
from research_tool.llm.types import ChatRequest
from research_tool.schemas.hypothesis import Hypothesis
from research_tool.testing import demo_data
from research_tool.testing.demo_run import CLUSTER, advance, demo_case_data, demo_services
from research_tool.testing.harness import make_run
from research_tool.validity.cases import Case, CaseError, load_case
from research_tool.validity.runs import DATA_FOLDER, MARKER, NO_KNOWLEDGE_BASE, OVERLAY, create_case_run, generate_data
from research_tool.validity.score import RunScore, score_run

COMMIT = "0123456789abcdef"


def demo_case(**changes: Any) -> Case:
    """A planted case the demo model answers: the demo plan's own conditions, metric and simulated effect."""
    return Case.model_validate(demo_case_data(**changes))


def demo_config(runs_root: Path) -> AppConfig:
    return load_config(overrides=deep_merge(CLUSTER, {"paths": {"runs_root": str(runs_root)}}), env={})


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> RunHandle:
    """One planted-case run, driven by the demo model from the plan to the reviewed report."""
    runs_root = tmp_path_factory.mktemp("validity") / "runs"
    # the demo plan asks for a GPU per run, so this run gets GPU-hours (Stage 0's real cases get none)
    run = create_case_run(demo_config(runs_root), demo_case(), commit=COMMIT, budgets={"max_gpu_hours": 40})
    result = advance(run, demo_services(run), max_episodes=60)
    assert result.outcome is Outcome.DONE, f"{result.outcome.value}: {result.message}"
    return run


def copy_of(run: RunHandle, tmp_path: Path) -> RunHandle:
    target = tmp_path / "copy" / run.run_id
    shutil.copytree(run.root, target)
    return RunHandle(target)


def remark(run: RunHandle, case: Case) -> None:
    """Make ``run`` a run of another case (the scorer reads the case from the run's marker)."""
    marker = read_json(run.root / MARKER)
    atomic_write_json(run.root / MARKER, {**marker, "case": case.model_dump(mode="json")})


# ---------------------------------------------------------------- creation
def test_a_case_run_is_seeded_and_needs_nobody(runs_root: Path) -> None:
    run = create_case_run(demo_config(runs_root), demo_case(), commit=COMMIT, variant="main", repeat=2)
    assert run.spec.project == "validity-generic" and run.run_id == "phase-locked-recall-main-r2-0123456"
    assert run.spec.pipeline.name == "research-from-plan"
    store = ArtifactStore(run.artifacts_dir)
    assert set(store.heads()) == {"research_brief", "selected_hypothesis", "corpus", "knowledge_map"}
    assert store.envelope("research_brief").producer.agent == "rtool validity"
    brief = store.get("research_brief")
    assert brief.seeds == 6 and [c.name for c in brief.conditions] == ["sham", "phase_locked"]
    assert any("`sham` (the baseline)" in line for line in brief.requirements())
    chosen = Hypothesis.model_validate(store.get("selected_hypothesis").item)
    assert chosen.id == "phase-locked-recall" and chosen.statement == demo_data.HYPOTHESES[0]["statement"]
    corpus = store.get("corpus")
    assert corpus.papers == [] and corpus.kb_path == NO_KNOWLEDGE_BASE and not (run.root / NO_KNOWLEDGE_BASE).exists()
    # every gate the run can open decides itself, and the GPU budget is zero unless the caller sets one
    for gate in ("experiment_plan", "manual_resources", "gpu_runs", "final_report", "revision_limit", "other"):
        assert run.config.gates.mode(gate) is GateMode.AUTO
    assert run.config.budgets.run.max_gpu_hours == 0
    overlay = run.root / OVERLAY
    assert run.config.slurm.episode.config_files[-1] == str(overlay)  # episodes apply it last
    assert yaml.safe_load(overlay.read_text(encoding="utf-8"))["gates"]["modes"]["gpu_runs"] == "auto"
    marker = read_json(run.root / MARKER)
    assert (marker["variant"], marker["repeat"], marker["commit"]) == ("main", 2, COMMIT)
    assert Case.model_validate(marker["case"]) == demo_case()


def test_an_open_case_runs_the_whole_pipeline(runs_root: Path) -> None:
    case = Case.model_validate({"id": "open-topic", "kind": "open", "pack": "ml_dl", "brief": {"topic": "anything"}})
    run = create_case_run(demo_config(runs_root), case, commit=COMMIT)
    assert run.spec.project == "validity-ml_dl" and run.spec.pipeline.stages[0].id == "literature"
    assert set(ArtifactStore(run.artifacts_dir).heads()) == {"research_brief"}


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: demo_case(start_at="literature"), "where the pipeline still chooses the hypothesis"),
        # its generators are found next to case.yaml, so a case with data must come from load_case
        (lambda: demo_case(brief={"datasets": ["d"]}, data=[{"name": "d", "generator": "g.py"}]), "load it with"),
    ],
)
def test_case_runs_refuse_what_they_cannot_do(runs_root: Path, make: Callable[[], Case], message: str) -> None:
    with pytest.raises(CaseError, match=message):
        create_case_run(demo_config(runs_root), make(), commit=COMMIT)


def test_a_case_run_needs_a_gpu_budget_and_a_known_variant(runs_root: Path) -> None:
    with pytest.raises(CaseError, match="needs a GPU-hour budget"):
        create_case_run(demo_config(runs_root), demo_case(), commit=COMMIT, budgets={"max_gpu_hours": None})
    with pytest.raises(CaseError, match="no variant 'busy'"):
        create_case_run(demo_config(runs_root), demo_case(), commit=COMMIT, variant="busy")


# ----------------------------------------------------------------- scoring
def test_a_planted_case_run_reaches_the_right_answer(finished: RunHandle) -> None:
    score = score_run(finished)
    assert score.finished and score.level == 3 and score.category == "" and not score.false_claim
    assert all(c.ok for c in score.checks) and {c.level for c in score.checks} == {0, 1, 2, 3}
    comparison = score.facts["comparison"]
    assert comparison["significant"] and comparison["better"] and comparison["difference"] > 0
    assert score.facts["outcome"] == "supported" and score.facts["primary_metric"] == "recall_gain"
    assert score.facts["seeds"] == 6 and "phase_locked" in score.facts["conditions"]
    assert score.audit and set(score.audit) <= {"ok", "problem", "unconfirmed", "unanswered"}


def test_the_same_results_on_a_null_case_are_a_false_claim(finished: RunHandle, tmp_path: Path) -> None:
    run = copy_of(finished, tmp_path)
    remark(run, demo_case(kind="null", expect={"significant": False, "direction": None}))
    score = score_run(run)
    # the analysis found the (planted) effect, so it disagrees with a null truth; the report claims it too
    assert (score.level, score.category, score.false_claim) == (2, "wrong_analysis", True)
    failed = [c.name for c in score.checks if not c.ok]
    assert failed == ["not significant", "the report concludes refuted or inconclusive"]


def _fail_stage(run: RunHandle) -> None:
    store = StateStore(run.state_dir)
    stage = store.load_stage("code")
    assert stage is not None
    stage.status, stage.last_error = StageStatus.FAILED, "smoke test failed: ImportError\nTraceback ..."
    store.save_stage(stage)


def _human_action(run: RunHandle) -> None:
    store = StateStore(run.state_dir)
    state = store.load_run()
    assert state is not None
    state.processed_actions.append("act-1")
    store.save_run(state)


def _put(name: str, **changes: Any) -> Callable[[RunHandle], None]:
    def change(run: RunHandle) -> None:
        store = ArtifactStore(run.artifacts_dir)
        value = store.get(name).model_copy(update=changes)
        store.put(name, value, producer=Producer(stage="test", agent="test"))

    return change


@pytest.mark.parametrize(
    ("change", "level", "category"),
    [
        (_fail_stage, -1, "stage_failed:code"),
        (_human_action, -1, "human_intervened"),
        (_put("experiment_results", complete=False), 0, "incomplete_runs"),
        (_put("experiment_results", warnings=["every seed gave the same value"]), 0, "fake_experiment"),
        (_put("experiment_results", code_fingerprint="other code"), 0, "stale_results"),
        (_put("experiment_plan", issues=["sleep-edf could not be found"]), 1, "plan_issue"),
        (_put("final_report", hypothesis_outcome="refuted"), 2, "misread_results"),
    ],
)
def test_each_failure_stops_at_its_level(
    finished: RunHandle, tmp_path: Path, change: Callable[[RunHandle], None], level: int, category: str
) -> None:
    run = copy_of(finished, tmp_path)
    change(run)
    score = score_run(run)
    assert (score.level, score.category) == (level, category)
    assert [c.level for c in score.checks if not c.ok] == [level + 1]  # scoring stopped at that level


def test_a_condition_the_plan_left_out_is_missing(finished: RunHandle, tmp_path: Path) -> None:
    run = copy_of(finished, tmp_path)
    case = demo_case()
    case.brief.conditions[1].name = "closed_loop"
    remark(run, case.model_copy(update={"expect": case.expect.model_copy(update={"condition": "closed_loop"})}))
    score = score_run(run)
    assert (score.level, score.category) == (0, "missing_condition")


@pytest.mark.parametrize(("offset", "level", "category"), [(0.0, 3, ""), (1.0, 2, "wrong_analysis")])
def test_a_replication_case_checks_the_known_value(
    finished: RunHandle, tmp_path: Path, offset: float, level: int, category: str
) -> None:
    mean = ArtifactStore(finished.artifacts_dir).get("experiment_results").summary("phase_locked", "recall_gain").mean
    run = copy_of(finished, tmp_path)
    band = {"condition": "phase_locked", "low": mean + offset - 0.1, "high": mean + offset + 0.1}
    remark(run, demo_case(kind="replication", expect={"significant": None, "direction": None, "value": band}))
    score = score_run(run)
    assert (score.level, score.category) == (level, category)
    assert score.facts["mean"] == mean
    # a replication case expects no conclusion unless it names one, so the report is not checked
    assert [c.name for c in score.checks if c.level == 3] == [f"the mean of phase_locked is between {band['low']:g} "
                                                              f"and {band['high']:g}"]  # fmt: skip
    assert "phase_locked was summarised on every seed" in [c.name for c in score.checks if c.level == 2]


def test_a_difference_outside_its_band_is_a_wrong_analysis(finished: RunHandle, tmp_path: Path) -> None:
    run = copy_of(finished, tmp_path)
    remark(run, demo_case(expect={"difference": {"low": 0.0, "high": 1.0}}))  # the demo's effect is about 2
    score = score_run(run)
    assert (score.level, score.category) == (2, "wrong_analysis")
    assert [c.name for c in score.checks if not c.ok] == ["the difference is between 0 and 1"]


def test_an_open_case_is_scored_for_running_and_consistent_code_only(finished: RunHandle, tmp_path: Path) -> None:
    run = copy_of(finished, tmp_path)
    remark(run, Case.model_validate({"id": "open-topic", "kind": "open", "pack": "generic", "brief": {"topic": "t"}}))
    score = score_run(run)
    assert (score.max_level, score.level, score.category) == (1, 1, "")
    assert {c.level for c in score.checks} == {0, 1} and not score.false_claim


def test_a_run_waiting_at_a_checkpoint_stalled(finished: RunHandle, tmp_path: Path) -> None:
    run = copy_of(finished, tmp_path)
    store = StateStore(run.state_dir)
    stage = store.load_stage("review")
    assert stage is not None
    stage.status, stage.gate = StageStatus.AWAITING_GATE, GateState(gate_id="final_report")
    store.save_stage(stage)
    score = score_run(run)
    assert score.finished and (score.level, score.category) == (-1, "stalled_at_gate:final_report")


def test_a_run_still_going_is_not_scored(finished: RunHandle, tmp_path: Path) -> None:
    run = copy_of(finished, tmp_path)
    store = StateStore(run.state_dir)
    stage = store.load_stage("review")
    assert stage is not None
    stage.status = StageStatus.PENDING
    store.save_stage(stage)
    assert score_run(run) == RunScore(
        run_id=run.run_id, case="phase-locked-recall", kind="planted", variant="main", repeat=1, commit=COMMIT,
        finished=False, confirmed=False,
    )  # fmt: skip


def test_only_validity_runs_are_scored(runs_root: Path) -> None:
    with pytest.raises(RToolError, match="is not a validity run"):
        score_run(make_run(runs_root, "toy"))


# ------------------------------------------------------------ case data
# The demo experiment reads recordings.csv (night, n3_minutes) from its "Sleep-EDF" resource.
RECORDINGS = """import csv, json, random, sys
from pathlib import Path

out, args = Path(sys.argv[1]), json.loads(sys.argv[2])
if args.get("fail"):
    sys.exit("generator says: no such cohort")
rng = random.Random(args["seed"])
with open(out / "recordings.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["night", "n3_minutes"])
    for night in range(args["nights"]):
        writer.writerow([night + 1, round(rng.uniform(70, 100), 1)])
"""


def case_with_data(root: Path, **changes: Any) -> Case:
    """The demo case, written to a case folder with a generator for its "Sleep-EDF" recordings."""
    recordings = {
        "name": "Sleep-EDF",
        "generator": "generate.py",
        "args": {"seed": 1, "nights": 5},
        "description": "minutes of N3 sleep per night",
    }
    variants = [{"name": "short"}, {"name": "long", "data_args": {"nights": 9}}]
    data = demo_case_data(
        **{"brief": {"datasets": ["Sleep-EDF"]}, "data": [recordings], "variants": variants, **changes}
    )
    folder = root / "cases" / "generic" / data["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "generate.py").write_text(RECORDINGS, encoding="utf-8")
    (folder / "case.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_case(folder)


def test_case_data_is_generated_once_for_each_content(tmp_path: Path) -> None:
    case, root = case_with_data(tmp_path), tmp_path / "data"
    short = generate_data(case, "short", root)
    assert short == {"Sleep-EDF": {"path": short["Sleep-EDF"]["path"], "description": "minutes of N3 sleep per night"}}
    folder = Path(short["Sleep-EDF"]["path"])
    assert folder.parent == root and folder.name.startswith("Sleep-EDF-")
    assert len((folder / "recordings.csv").read_text(encoding="utf-8").splitlines()) == 6  # header + 5 nights
    assert generate_data(case, "short", root) == short  # the same arguments make the same data, kept once
    long = generate_data(case, "long", root)["Sleep-EDF"]["path"]
    assert long != str(folder) and len((Path(long) / "recordings.csv").read_text().splitlines()) == 10
    assert sorted(p.name for p in root.iterdir()) == sorted([folder.name, Path(long).name])  # no partial folders
    failing = case_with_data(
        tmp_path / "other", data=[{"name": "Sleep-EDF", "generator": "generate.py", "args": {"fail": True}}]
    )
    with pytest.raises(CaseError, match="failed \\(exit 1\\): generator says: no such cohort"):
        generate_data(failing, "short", root)


def test_a_case_run_with_its_own_data_uses_it_as_a_local_resource(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run = create_case_run(
        demo_config(runs_root), case_with_data(tmp_path), commit=COMMIT, variant="long", budgets={"max_gpu_hours": 40}
    )
    provided = run.config.experiments.local_data["Sleep-EDF"]
    assert Path(provided.path).parent == runs_root / DATA_FOLDER
    overlay = yaml.safe_load((run.root / OVERLAY).read_text(encoding="utf-8"))
    assert overlay["experiments"]["local_data"]["Sleep-EDF"]["path"] == provided.path  # episodes keep it
    assert read_json(run.root / MARKER)["data"] == {"Sleep-EDF": provided.path}

    services = demo_services(run)
    local = {
        "resources": [
            {"kind": "dataset", "name": "Sleep-EDF", "source": "local", "identifier": "Sleep-EDF", "purpose": "nights"},
            {"kind": "code", "name": "numpy", "source": "pip", "identifier": "numpy", "purpose": "maths"},
        ]
    }

    def asks_for_resources(request: ChatRequest) -> bool:
        return (request.json_schema or {}).get("title") == "ResourcePlan"

    services.llm.backend.rules.insert(0, (asks_for_resources, local))  # the planner picks the provided data
    result = advance(run, services, max_episodes=60)
    assert result.outcome is Outcome.DONE, f"{result.outcome.value}: {result.message}"
    store = ArtifactStore(run.artifacts_dir)
    assert [r.source for r in store.get("experiment_plan").resources] == ["local", "pip"]
    fetched = next(r for r in store.get("resource_manifest").resources if r.name == "Sleep-EDF")
    assert fetched.status == "ready" and Path(fetched.path).parent.name == "local"  # type: ignore[arg-type]
    assert (Path(fetched.path) / "recordings.csv").read_text() == (Path(provided.path) / "recordings.csv").read_text()  # type: ignore[arg-type]
    score = score_run(run)
    assert (score.level, score.category) == (3, ""), [c for c in score.checks if not c.ok]
