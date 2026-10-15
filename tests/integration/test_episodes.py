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
    EpisodeTracker,
    LazyLLM,
    choose_gpu,
    episode_job_spec,
    gpu_available,
    llm_needed,
    run_episode,
    schedule_next,
    submit_episode,
)
from research_tool.hpc.fake_slurm import FakeSlurm
from research_tool.hpc.jobspec import JobSpec
from research_tool.hpc.slurm import JobState
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
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
    assert gpu.singleton and cpu.singleton and gpu.name == cpu.name == f"rtool-ep-{run.run_id}"


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
    assert spec["singleton"]  # but priced without it: it would make every choice wait for the running episode
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


def test_racing_kicks_never_run_two_episodes_at_once(runs_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two kicks that both look before either records its job submit two episodes; Slurm runs them in turn."""
    run = make_run(runs_root, JOB)
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    experiment = slurm.submit(JobSpec(name="experiment", partition="nodes", command="true", workdir=run.root))
    monkeypatch.setattr(EpisodeTracker, "current", lambda self, ignore=None: None)  # neither saw the other's job
    first = submit_episode(run, slurm, needs_llm=False, reason="ui")
    second = submit_episode(run, slurm, needs_llm=True, reason="cli")
    after_experiment = submit_episode(run, slurm, needs_llm=False, reason="waiting_job", dependency=[experiment])
    assert first and second and after_experiment

    def running() -> list[str]:
        return sorted(job_id for job_id, job in slurm.jobs.items() if job["state"] == JobState.RUNNING.value)

    slurm.tick()
    assert running() == sorted([experiment, first])  # jobs of other names are not held back
    slurm.tick()  # the first episode ends, then the next one in line starts
    assert running() == [second]  # its own dependency is met, but an episode is running
    slurm.tick()
    assert running() == [after_experiment]
    slurm.tick()
    assert running() == []
