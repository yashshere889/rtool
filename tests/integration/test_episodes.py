"""Episode chaining: CPU episodes hand over to GPU ones, waits chain via dependencies, no duplicates."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from research_tool.core.config import load_config
from research_tool.core.engine import Outcome
from research_tool.core.events import EventLog
from research_tool.core.run import RunHandle
from research_tool.hpc.episode import (
    BAD_NODE_HOURS,
    EpisodeTracker,
    LazyLLM,
    bad_nodes,
    choose_gpu,
    episode_job_spec,
    gpu_available,
    llm_needed,
    note_bad_node,
    run_episode,
    schedule_next,
    submit_episode,
)
from research_tool.hpc.fake_slurm import FakeSlurm
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.server import VLLMServerError
from research_tool.testing.harness import make_run

JOB = {"name": "job", "stages": [{"id": "source", "agent": "toy_source"}, {"id": "analyze", "agent": "toy_analyzer"}]}


class Factory:
    """Counts how often an episode actually needed the LLM."""

    def __init__(self, run: RunHandle, llm: MockLLM) -> None:
        self.run, self.llm, self.calls = run, llm, 0

    def __call__(self) -> LLMClient:
        self.calls += 1
        return LLMClient(self.llm, self.run.config.llm)


def test_episode_job_spec(runs_root: Path) -> None:
    run = make_run(
        runs_root,
        JOB,
        overrides={"slurm": {"episode": {"activate": "/venv/bin/activate", "config_files": ["~/site.yaml"]}}},
    )
    gpu = episode_job_spec(run, needs_llm=True, dependency=["77"])
    assert gpu.partition == "gpu-l40s" and gpu.gpus is not None and gpu.gpus.gres() == "gpu:l40s:1"
    assert gpu.ntasks is None and gpu.dependency == ["77"] and gpu.signal == "B:USR1@600"
    assert gpu.command.startswith("unset PYTHONPATH PYTHONHOME && export PYTHONNOUSERSITE=1 && ")  # type: ignore[union-attr]
    assert "&& source /venv/bin/activate && exec rtool episode " in gpu.command
    assert "--config " + str(Path("~/site.yaml").expanduser()) in gpu.command
    cpu = episode_job_spec(run, needs_llm=False)
    assert cpu.partition == "nodes" and cpu.gpus is None and cpu.ntasks == 1


CHOICES = {
    "slurm": {
        "episode": {
            "gpu_choices": [
                {"partition": "gpu-l40s", "gpu_type": "l40s"},
                {"partition": "gpu-h100", "gpu_type": "h100"},
                {"partition": "gpu-l40s-low", "gpu_type": "l40s"},
            ],
            "prefer_within_minutes": 30,
        }
    }
}


def test_llm_episodes_take_the_gpu_that_starts_soonest(runs_root: Path) -> None:
    run = make_run(runs_root, JOB, overrides=CHOICES)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    now = time.time()
    assert choose_gpu(run, slurm)[0].partition == "gpu-l40s"  # no estimates: first choice
    slurm.set_estimate("gpu-l40s", "l40s", now + 2 * 86400)
    slurm.set_estimate("gpu-h100", "h100", now + 12 * 3600)
    slurm.set_estimate("gpu-l40s-low", "l40s", now)
    job = submit_episode(run, slurm, needs_llm=True, reason="test")
    spec = slurm.jobs[job]["spec"]  # type: ignore[index]
    assert spec["partition"] == "gpu-l40s-low" and spec["gpus"]["type"] == "l40s" and spec["requeue"]
    submitted = [e for e in EventLog(run.events_path, run.run_id) if e["kind"] == "episode_submitted"][-1]["data"]
    assert submitted["partition"] == "gpu-l40s-low" and submitted["expected_start"]
    # an earlier choice is kept when it starts nearly as soon
    slurm.set_estimate("gpu-l40s", "l40s", now + 20 * 60)
    assert choose_gpu(run, slurm)[0].partition == "gpu-l40s"


def test_gpu_episode_waiting_on_jobs_starts_on_a_cpu_node(runs_root: Path) -> None:
    run = make_run(runs_root, JOB, overrides=CHOICES)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    job = submit_episode(run, slurm, needs_llm=True, reason="waiting_job", dependency=["5"])
    spec = slurm.jobs[job]["spec"]  # type: ignore[index]
    assert spec["partition"] == "nodes" and spec["gpus"] is None and spec["dependency"] == ["5"]


def test_cpu_episode_hands_over_to_a_gpu_episode(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, JOB)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    factory = Factory(run, mock_llm)
    result = run_episode(run, slurm=slurm, llm_factory=factory, has_gpu=False)
    assert result.outcome is Outcome.NEEDS_LLM and result.deferred == ["source"]
    assert factory.calls == 0  # never started an LLM on a CPU node
    gpu_job = schedule_next(run, slurm, result, self_job_id=None)
    assert gpu_job is not None and slurm.jobs[gpu_job]["spec"]["partition"] == "gpu-l40s"
    # a second scheduler sees the queued episode and does not duplicate it
    assert submit_episode(run, slurm, needs_llm=True, reason="again") is None
    assert "episode_submitted" in [e["kind"] for e in EventLog(run.events_path, run.run_id)]


def test_gpu_episode_runs_then_waits_on_its_job_via_dependency(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, JOB)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    factory = Factory(run, mock_llm)
    result = run_episode(run, slurm=slurm, llm_factory=factory, has_gpu=True)
    assert result.outcome is Outcome.WAITING_JOB and factory.calls == 1
    experiment = result.waiting_jobs[0]
    follow_up = schedule_next(run, slurm, result, self_job_id=None)
    spec = slurm.jobs[follow_up]["spec"]  # type: ignore[index]
    assert spec["dependency"] == [experiment] and spec["dependency_type"] == "afterany"
    assert spec["partition"] == "nodes"  # the resuming stage (toy_analyzer) does not need the LLM
    slurm.finish_all()
    factory2 = Factory(run, mock_llm)
    final = run_episode(run, slurm=slurm, llm_factory=factory2, has_gpu=False)
    assert final.outcome is Outcome.DONE and factory2.calls == 0
    assert schedule_next(run, slurm, final, self_job_id=None) is None


def test_running_episode_ignores_itself_when_scheduling(runs_root: Path) -> None:
    run = make_run(runs_root, JOB)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    me = submit_episode(run, slurm, needs_llm=False, reason="first")
    slurm.tick()  # now RUNNING: this is "the current episode"
    tracker = EpisodeTracker(run, slurm)
    assert tracker.current() is not None
    assert tracker.current(ignore=me) is None
    assert submit_episode(run, slurm, needs_llm=True, reason="next", self_job_id=me) is not None


def test_gpu_episode_with_a_mock_llm_starts_no_server(runs_root: Path) -> None:
    run = make_run(runs_root, JOB)  # llm.backend defaults to mock
    lazy = LazyLLM(run.config, run.jobs_dir / "llm")
    lazy.bind(budget=None, on_call=None, stage="source")
    assert lazy.started and lazy._handle is None  # no vLLM process was launched


def test_llm_needed_and_gpu_detection(runs_root: Path) -> None:
    run = make_run(runs_root, JOB)
    assert llm_needed(run)  # toy_source (pending) needs the LLM
    assert not gpu_available({})
    assert gpu_available({"CUDA_VISIBLE_DEVICES": "0"}) and gpu_available({"SLURM_JOB_GPUS": "1"})
    assert not gpu_available({"CUDA_VISIBLE_DEVICES": "-1"})


def test_barkla_site_config_loads(monkeypatch) -> None:  # noqa: ANN001
    path = Path(__file__).resolve().parents[2] / "configs" / "barkla.yaml"
    cfg = load_config(path, env={"USER": "someone"})
    assert str(cfg.paths.runs_root) == "/mnt/scratch/users/someone/rtool/projects"
    assert cfg.llm.server.container == "$NEMOTRON_SIF"  # expanded later, when the server starts
    assert cfg.slurm.experiment_partitions == ["gpu-l40s", "gpu-h100", "gpu-v100"]
    assert cfg.slurm.partitions["gpu-h100"].gpus_per_node == 4
    assert cfg.knowledge.embedder == "fastembed" and cfg.paths.resources_root is not None
    choices = cfg.slurm.episode.gpu_choices
    assert choices[0].partition == "gpu-l40s" and len(choices) >= 4
    for choice in choices:  # every choice must pass the job validator
        partition = cfg.slurm.partitions[choice.partition]
        assert partition.allows_gpu_type(choice.gpu_type) and (partition.gpu_memory_gb or 0) >= 40  # type: ignore[arg-type]
    assert cfg.slurm.partitions["gpu-a-lowsmall"].preemptible


def test_llm_needed_looks_only_at_what_can_run_next(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, "toy")  # source (LLM, gate) -> analyze (no LLM) -> write (LLM)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    result = run_episode(run, slurm=slurm, llm_factory=Factory(run, mock_llm), has_gpu=True)
    assert result.open_gates[0]["gate"] == "numbers_review"
    # approving lets `analyze` run next; `write` (which needs the LLM) is further away
    assert not llm_needed(run)


NO_SHM = "vLLM exited with code 1 during startup (OSError: [Errno 28] No space left on device); last log lines:\n..."


def broken_server() -> LLMClient:
    raise VLLMServerError(NO_SHM)


def test_a_node_that_cannot_start_vllm_is_avoided_and_the_run_goes_on(
    runs_root: Path, mock_llm: MockLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-10-15: vLLM died twice on gpu45 (/dev/shm full); the episode crashed and nothing was resubmitted."""
    run = make_run(runs_root, JOB)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    monkeypatch.setenv("SLURMD_NODENAME", "gpu45")
    result = run_episode(run, slurm=slurm, llm_factory=broken_server, has_gpu=True)
    assert result.outcome is Outcome.NEEDS_LLM
    assert result.message.startswith("vLLM exited with code 1 during startup (OSError: [Errno 28] No space left")
    assert result.message.endswith("on gpu45; the next GPU episode avoids that node")
    job = schedule_next(run, slurm, result, self_job_id=None)
    assert job is not None and slurm.jobs[job]["spec"]["exclude"] == ["gpu45"]
    failed = [e["data"] for e in EventLog(run.events_path, run.run_id) if e["kind"] == "llm_server_failed"]
    assert failed and failed[0]["node"] == "gpu45" and not failed[0]["giving_up"]
    # the next episode, on a working node, picks the interrupted stage up again
    result = run_episode(run, slurm=slurm, llm_factory=Factory(run, mock_llm), has_gpu=True)
    assert result.outcome is Outcome.WAITING_JOB and "source" in result.ran


def test_repeated_server_failures_stop_for_a_human_and_are_forgotten_after_a_day(
    runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(runs_root, JOB)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    outcomes = []
    for node in ("gpu41", "gpu42", "gpu43"):
        monkeypatch.setenv("SLURMD_NODENAME", node)
        outcomes.append(run_episode(run, slurm=slurm, llm_factory=broken_server, has_gpu=True))
    assert [r.outcome for r in outcomes] == [Outcome.NEEDS_LLM, Outcome.NEEDS_LLM, Outcome.NEEDS_HUMAN]
    assert "failed to start 3 times in the last day (nodes gpu41, gpu42, gpu43)" in outcomes[-1].message
    assert schedule_next(run, slurm, outcomes[-1], self_job_id=None) is None
    later = time.time() + (BAD_NODE_HOURS + 1) * 3600
    assert bad_nodes(run, now=later) == {}
    assert sorted(note_bad_node(run, "gpu44", "x", now=later)) == ["gpu44"]
