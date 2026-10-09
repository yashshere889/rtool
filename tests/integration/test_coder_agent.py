"""The Coder end to end with a mock LLM: write, smoke-test, repair, gate, run on (fake) Slurm, analyse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_tool.agents.planner.checks import results_schema
from research_tool.core.actions import ActionKind
from research_tool.core.artifacts import ArtifactStore, Producer
from research_tool.core.engine import Engine, Outcome, load_status
from research_tool.core.events import EventLog
from research_tool.core.run import RunHandle
from research_tool.core.services import Services
from research_tool.core.state import StageStatus
from research_tool.hpc.episode import llm_needed, run_episode
from research_tool.hpc.fake_slurm import FakeSlurm
from research_tool.hpc.slurm import JobState
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.types import ChatRequest
from research_tool.schemas.plan import (
    ComputeRequest,
    Condition,
    ExperimentPlan,
    Metric,
    ResourceManifest,
    StatisticalTest,
    Variable,
)
from research_tool.testing.harness import make_run, queue

GOOD = """# requirements:
import random

BATCH_SIZE = 16


def run(condition, seed, resources, smoke, workdir):
    rng = random.Random(seed)
    level = 0.65 if condition == "base" else 0.72
    acc = level + rng.uniform(-0.01, 0.01)
    return {"acc": acc, "loss": 1 - acc}
"""
CRASHES_IN_SMOKE = GOOD.replace("    rng = random", "    if smoke:\n        raise KeyError('label')\n    rng = random")
OOM_IN_FULL_RUNS = GOOD.replace(
    "    rng = random",
    "    if not smoke and BATCH_SIZE > 8:\n        raise MemoryError('batch too big')\n    rng = random",
)


def pipeline(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": "code-only",
        "initial_artifacts": ["research_brief", "experiment_plan", "resource_manifest"],
        "stages": [{"id": "code", "agent": "coder", "config": config or {}}],
    }


def setup(runs_root: Path, tmp_path: Path, config: dict[str, Any] | None = None) -> RunHandle:
    run = make_run(runs_root, pipeline(config))
    conditions = [
        Condition(name="base", description="baseline", is_baseline=True),
        Condition(name="new", description="n"),
    ]
    metrics = [Metric(name="acc", definition="accuracy"), Metric(name="loss", definition="l", higher_is_better=False)]
    plan = ExperimentPlan(
        hypothesis_id="h1",
        hypothesis="The new method is more accurate",
        null_hypothesis="No difference",
        objective="o",
        design="two conditions, three seeds",
        independent_variables=[Variable(name="method", description="m")],
        dependent_variables=[Variable(name="acc", description="a")],
        conditions=conditions,
        metrics=metrics,
        statistical_tests=[StatisticalTest(test="paired t-test", comparison="new vs base")],
        seeds=[0, 1, 2],
        procedure=["load", "train", "evaluate"],
        resources=[],
        compute=ComputeRequest(
            partition="gpu-l40s",
            gpu_type="l40s",
            gpus_per_run=1,
            time_limit="01:00:00",
            runs=6,
            hours_per_run=0.5,
            gpu_hours_total=3,
        ),  # fmt: skip
        expected_results="new is better",
        results_schema=results_schema(conditions, metrics, [0, 1, 2]),
    )
    store = ArtifactStore(run.artifacts_dir)
    store.put("experiment_plan", plan, producer=Producer(stage="user", agent="test"))
    store.put(
        "resource_manifest",
        ResourceManifest(resources=[], root=str(tmp_path)),
        producer=Producer(stage="user", agent="test"),
    )
    return run


def coder_llm(first: str, *, fix: str = GOOD) -> MockLLM:
    def implement(request: ChatRequest) -> str:
        return f"Here is the file.\n```python\n{first}```\n"

    return (
        MockLLM()
        .add(lambda r: "Write the complete Python file `experiment.py`" in r.messages[-1].content, implement)
        .add(lambda r: "fails. Fix it" in r.messages[-1].content, f"```python\n{fix}```")
        .add(lambda r: "Revise the experiment code below" in r.messages[-1].content, f"```python\n{GOOD}```")
    )


def services(run: RunHandle, llm: MockLLM) -> Services:
    return Services(
        llm=LLMClient(llm, run.config.llm), slurm=FakeSlurm(run.jobs_dir / "fake_slurm.json", run_commands=True)
    )


def advance(run: RunHandle, svc: Services, expected: Outcome) -> Any:
    result = Engine(run, svc).advance()
    assert result.outcome is expected, (result.outcome, result.message, result.failed)
    return result


def test_write_repair_approve_run_and_analyse(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path)
    llm = coder_llm(CRASHES_IN_SMOKE)
    svc = services(run, llm)

    gate = advance(run, svc, Outcome.WAITING_GATE).open_gates[0]
    assert gate["gate"] == "gpu_runs" and gate["in_stage"]
    summary = gate["summary"]
    assert "6 jobs" in summary["message"] and "base ok" in summary["smoke"]
    assert any(path.endswith("experiment.py") for path in summary["review_files"])
    fix_prompt = next(r for r in llm.calls if "fails. Fix it" in r.messages[-1].content).messages[-1].content
    assert "KeyError: 'label'" in fix_prompt  # the LLM saw the real traceback

    queue(run, ActionKind.APPROVE)
    waiting = advance(run, svc, Outcome.WAITING_JOB)
    job = svc.slurm.jobs[waiting.waiting_jobs[0]]["spec"]  # type: ignore[union-attr]
    assert job["array"] == "0-5%4" and job["gpus"]["type"] == "l40s" and job["partition"] == "gpu-l40s"
    svc.slurm.finish_all()  # type: ignore[union-attr]  (runs the six array tasks for real)
    advance(run, svc, Outcome.DONE)

    store = ArtifactStore(run.artifacts_dir)
    results, bundle = store.get("experiment_results"), store.get("code_bundle")
    assert results.complete and len(results.runs) == 6
    acc = results.comparison("new", "acc")
    assert acc is not None and acc.significant and acc.better and acc.test == "paired t-test"
    assert results.summary("base", "acc").n == 3  # type: ignore[union-attr]
    data = json.loads((run.root / results.results_path).read_text())
    assert len(data["runs"]) == 6 and all(set(r["metrics"]) == {"acc", "loss"} for r in data["runs"])
    assert all((run.root / f).read_text().startswith("<svg") for f in results.figures)
    assert "| new vs base |" in (run.root / results.tables_path).read_text()
    assert [f.signature for f in bundle.fixes] == ["traceback"]
    assert {f.path: f.origin for f in bundle.files}["experiment.py"] == "llm"
    assert all(s.ok for s in bundle.smoke) and bundle.fingerprint == results.code_fingerprint
    kinds = [e["kind"] for e in EventLog(run.events_path, run.run_id)]
    assert "experiment_submitted" in kinds and "code_fix" in kinds


def test_out_of_memory_in_full_runs_halves_the_batch_and_asks_again(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path)
    svc = services(run, coder_llm(OOM_IN_FULL_RUNS))
    advance(run, svc, Outcome.WAITING_GATE)
    queue(run, ActionKind.APPROVE)
    advance(run, svc, Outcome.WAITING_JOB)
    svc.slurm.finish_all()  # type: ignore[union-attr]  (every run raises MemoryError)
    again = advance(run, svc, Outcome.WAITING_GATE)  # the code changed, so the GPU spend is approved again
    assert again.open_gates[0]["gate"] == "gpu_runs"
    code = (run.work_dir("code") / "code" / "experiment.py").read_text()
    assert "BATCH_SIZE = 8" in code  # the playbook, not the LLM
    queue(run, ActionKind.APPROVE)
    advance(run, svc, Outcome.WAITING_JOB)
    svc.slurm.finish_all()  # type: ignore[union-attr]
    advance(run, svc, Outcome.DONE)
    store = ArtifactStore(run.artifacts_dir)
    assert store.get("experiment_results").complete and len(store.get("experiment_results").jobs) == 2
    fixes = store.get("code_bundle").fixes
    assert fixes[-1].signature == "cpu_oom" and "BATCH_SIZE to 8" in fixes[-1].action


def test_rejected_gate_revises_the_code(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path)
    llm = coder_llm(GOOD)
    svc = services(run, llm)
    advance(run, svc, Outcome.WAITING_GATE)
    queue(run, ActionKind.REJECT, payload={"feedback": "log the loss curve too"})
    advance(run, svc, Outcome.WAITING_GATE)  # revised, smoke-tested, and asking again
    revise = [r for r in llm.calls if "Revise the experiment code below" in r.messages[-1].content]
    assert revise and "log the loss curve too" in revise[0].messages[-1].content


def test_edits_at_the_gpu_gate_are_tested_before_any_gpu_time(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path)
    llm = coder_llm(GOOD)
    svc = services(run, llm)
    advance(run, svc, Outcome.WAITING_GATE)
    path = run.work_dir("code") / "code" / "experiment.py"
    path.write_text(path.read_text() + "\n# tuned by hand\n")  # the researcher edits the code at the gate...
    queue(run, ActionKind.APPROVE)  # ...and approves (the CLI shows experiment.py among the files to read)
    calls = len(llm.calls)
    again = advance(run, svc, Outcome.WAITING_GATE)  # nothing submitted: the edit is smoke-tested first
    assert again.open_gates[0]["gate"] == "gpu_runs" and len(llm.calls) == calls  # no model involved
    queue(run, ActionKind.APPROVE)
    advance(run, svc, Outcome.WAITING_JOB)  # the approval of the tested edit stands
    svc.slurm.finish_all()  # type: ignore[union-attr]
    advance(run, svc, Outcome.DONE)
    bundle = ArtifactStore(run.artifacts_dir).get("code_bundle")
    assert {f.path: f.origin for f in bundle.files}["experiment.py"] == "human"
    assert "# tuned by hand" in path.read_text() and "human_edit" in [f.signature for f in bundle.fixes]


def test_reject_with_retest_tests_again_without_the_model(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path)
    llm = coder_llm(GOOD)
    svc = services(run, llm)
    advance(run, svc, Outcome.WAITING_GATE)
    calls = len(llm.calls)
    queue(run, ActionKind.REJECT, payload={"feedback": "", "retest": True})
    assert advance(run, svc, Outcome.WAITING_GATE).open_gates[0]["gate"] == "gpu_runs"
    assert len(llm.calls) == calls  # re-tested, not revised


def test_unfixable_code_asks_a_human_then_resumes(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path, config={"max_fix_rounds": 1})
    svc = services(run, coder_llm(CRASHES_IN_SMOKE, fix=CRASHES_IN_SMOKE))
    failed = advance(run, svc, Outcome.NEEDS_HUMAN)
    assert "retry" in failed.failed[0]["error"] and "by hand" in failed.failed[0]["error"]
    (run.work_dir("code") / "code" / "experiment.py").write_text(GOOD)  # the researcher fixes it
    queue(run, ActionKind.RETRY, stage="code", payload={"resume": True})
    gate = advance(run, svc, Outcome.WAITING_GATE).open_gates[0]
    assert gate["gate"] == "gpu_runs"
    queue(run, ActionKind.APPROVE)
    advance(run, svc, Outcome.WAITING_JOB)
    svc.slurm.finish_all()  # type: ignore[union-attr]
    advance(run, svc, Outcome.DONE)
    bundle = ArtifactStore(run.artifacts_dir).get("code_bundle")
    assert {f.path: f.origin for f in bundle.files}["experiment.py"] == "human"
    assert "human_edit" in [f.signature for f in bundle.fixes]


CRASHES_IN_FULL_RUNS = GOOD.replace(
    "    rng = random", "    if not smoke:\n        raise ValueError('label column missing')\n    rng = random"
)


def test_bookkeeping_runs_on_cpu_nodes(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path)
    llm = coder_llm(GOOD)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json", run_commands=True)
    gpu = run_episode(run, slurm=slurm, llm_factory=lambda: LLMClient(llm, run.config.llm), has_gpu=True)
    assert gpu.outcome is Outcome.WAITING_GATE
    queue(run, ActionKind.APPROVE)
    assert not llm_needed(run)  # approving only submits jobs: a CPU episode will do
    calls = len(llm.calls)
    submitted = run_episode(run, slurm=slurm, has_gpu=False)
    assert submitted.outcome is Outcome.WAITING_JOB, submitted.message
    assert not llm_needed(run, (StageStatus.SUSPENDED,))  # collecting needs no GPU either
    slurm.finish_all()
    done = run_episode(run, slurm=slurm, has_gpu=False)
    assert done.outcome is Outcome.DONE and len(llm.calls) == calls  # no model calls after the approval
    assert ArtifactStore(run.artifacts_dir).get("experiment_results").complete


def test_a_fix_that_needs_the_model_moves_to_a_gpu_episode(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path)
    llm = coder_llm(CRASHES_IN_FULL_RUNS)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json", run_commands=True)
    factory = lambda: LLMClient(llm, run.config.llm)  # noqa: E731
    run_episode(run, slurm=slurm, llm_factory=factory, has_gpu=True)
    queue(run, ActionKind.APPROVE)
    run_episode(run, slurm=slurm, has_gpu=False)
    slurm.finish_all()  # every full run raises ValueError
    handed = run_episode(run, slurm=slurm, has_gpu=False)
    assert handed.outcome is Outcome.NEEDS_LLM and handed.deferred == ["code"]
    state = load_status(run)[1]["code"]
    assert state.agent_state["phase"] == "collect" and state.agent_state.get("rerun_rounds", 0) == 0  # nothing lost
    again = run_episode(run, slurm=slurm, llm_factory=factory, has_gpu=True)
    assert again.outcome is Outcome.WAITING_GATE  # fixed by the model, smoke-tested, asking for approval again
    assert any(
        "fails. Fix it" in r.messages[-1].content and "label column missing" in r.messages[-1].content
        for r in llm.calls
    )


def test_runs_that_never_ran_are_resubmitted_unchanged(runs_root: Path, tmp_path: Path) -> None:
    run = setup(runs_root, tmp_path)
    llm = coder_llm(GOOD)
    svc = services(run, llm)
    advance(run, svc, Outcome.WAITING_GATE)
    queue(run, ActionKind.APPROVE)
    advance(run, svc, Outcome.WAITING_JOB)
    slurm: FakeSlurm = svc.slurm  # type: ignore[assignment]
    slurm.set_outcome("rtool-exp-*", JobState.NODE_FAIL)  # the node died: nothing ran, no logs
    slurm.finish_all()
    calls = len(llm.calls)
    again = advance(run, svc, Outcome.WAITING_JOB)  # resubmitted straight away, no model, no new approval
    assert len(llm.calls) == calls and len(again.waiting_jobs) == 1
    slurm.set_outcome("rtool-exp-*", JobState.COMPLETED, 0)
    slurm.finish_all()
    advance(run, svc, Outcome.DONE)
    bundle = ArtifactStore(run.artifacts_dir).get("code_bundle")
    assert (
        bundle.fixes[-1].signature == "not_run" and ArtifactStore(run.artifacts_dir).get("experiment_results").complete
    )
