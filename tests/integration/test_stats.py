"""`rtool stats`: where a run's wall time went, rebuilt from its event log."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from research_tool.cli.main import app
from research_tool.core.actions import ActionKind
from research_tool.core.engine import Outcome
from research_tool.core.events import EventLog
from research_tool.core.run import RunHandle
from research_tool.core.stats import (
    RunStats,
    compute_stats,
    iso,
    run_stats,
    split_wall_time,
    stats_for_run_dir,
    union_seconds,
)
from research_tool.hpc import episode as episode_module
from research_tool.hpc.episode import LazyLLM, run_episode, schedule_next, server_failed, submit_episode
from research_tool.hpc.fake_slurm import FakeSlurm
from research_tool.llm import server as server_module
from research_tool.llm.client import LLMClient
from research_tool.llm.mock_backend import MockLLM
from research_tool.llm.server import ServerHandle, VLLMServerError, node_facts
from research_tool.testing.demo_run import build_demo_run
from research_tool.testing.harness import make_run, queue

T0 = datetime(2026, 10, 10, 19, 14, 15, tzinfo=UTC).timestamp()


def ev(t: float, kind: str, episode: str | None = None, stage: str | None = None, **data: Any) -> dict[str, Any]:
    event: dict[str, Any] = {"ts": iso(T0 + t), "run": "r1", "episode": episode, "kind": kind}
    if stage:
        event["stage"] = stage
    if data:
        event["data"] = data
    return event


def call(t: float, episode: str, latency: float, finish: str = "stop") -> dict[str, Any]:
    return ev(
        t, "llm_call", episode, "literature", profile="extract", prompt_tokens=100, completion_tokens=10,
        latency_s=latency, finish_reason=finish, structured=True,
    )  # fmt: skip


# A CPU episode hands over to a GPU one, which is killed mid-stage and requeued; a person approves
# the gate; the next stage waits on an experiment job; a time-limit continuation's vLLM fails to start.
EVENTS = [
    ev(0, "episode_submitted", job="101", needs_llm=False, reason="rtool run", dependency=[], partition="nodes"),
    ev(30, "episode_started", "ep-101", number=1),
    ev(
        40,
        "episode_submitted",
        job="102",
        needs_llm=True,
        reason="needs_llm",
        dependency=[],
        partition="gpu-l40s",
        gpu="gpu:l40s:1",
    ),  # fmt: skip
    ev(41, "episode_finished", "ep-101", outcome="needs_llm", ran=[]),
    ev(641, "episode_started", "ep-102", number=2),
    ev(642, "stage_started", "ep-102", "literature", agent="literature", attempt=1, resumed=False),
    ev(
        822,
        "llm_server_ready",
        "ep-102",
        "literature",
        seconds=180.0,
        model="nemotron",
        node="gpu45",
        gpu="NVIDIA L40S, 46068 MiB",
        devices="0",
        partition="gpu-l40s",
        job="102",
    ),  # fmt: skip
    call(830, "ep-102", 8),
    call(835, "ep-102", 10),  # overlaps the call before it
    ev(842, "literature_phase_done", "ep-102", "literature", phase="plan", papers=0, included=0),
    call(900, "ep-102", 20, finish="length"),
    ev(1000, "literature_phase_done", "ep-102", "literature", phase="search", papers=40, included=0),
    call(1100, "ep-102", 5),  # ... and then the node died
    ev(1400, "episode_started", "ep-102", number=3),  # Slurm requeued the job
    ev(1400, "stage_interrupted", "ep-102", "literature", attempt=1),
    ev(1401, "stage_started", "ep-102", "literature", agent="literature", attempt=1, resumed=True),
    ev(1461, "llm_server_ready", "ep-102", "literature", seconds=60.0, host="gpu46", gpu="NVIDIA L40S, 46068 MiB"),
    ev(1500, "literature_phase_done", "ep-102", "literature", phase="screen", papers=40, included=12),
    ev(1510, "gate_opened", "ep-102", "literature", gate="literature_review", in_stage=False),
    ev(1511, "episode_finished", "ep-102", outcome="waiting_gate", ran=["literature"]),
    ev(
        3511, "episode_submitted", job="103", needs_llm=False, reason="action:approve", dependency=[], partition="nodes"
    ),  # fmt: skip
    ev(3690, "episode_started", "ep-103", number=4),
    ev(3700, "gate_resolved", "ep-103", "literature", gate="literature_review", approved=True, auto=False),
    ev(3700, "stage_finished", "ep-103", "literature", attempt=1),
    ev(3700, "action_applied", "ep-103", action="approve", by="yash", outcome={"status": "applied"}),
    ev(3701, "stage_started", "ep-103", "code", agent="coder", attempt=1, resumed=False),
    ev(3720, "stage_suspended", "ep-103", "code", jobs=["900"], reason="experiments"),
    ev(
        3721,
        "episode_submitted",
        job="104",
        needs_llm=False,
        reason="waiting_job",
        dependency=["900"],
        partition="nodes",
    ),  # fmt: skip
    ev(3722, "episode_finished", "ep-103", outcome="waiting_job", ran=["code"]),
    ev(7322, "episode_started", "ep-104", number=5),
    ev(7323, "stage_started", "ep-104", "code", agent="coder", attempt=1, resumed=True),
    ev(7400, "stage_finished", "ep-104", "code", attempt=1),
    ev(7400, "gate_auto_resolved", "ep-104", "code", gate="final"),
    ev(
        7401,
        "episode_submitted",
        job="105",
        needs_llm=True,
        reason="time_limit",
        dependency=["104"],
        partition="gpu-l40s",
        gpu="gpu:l40s:1",
    ),  # fmt: skip
    ev(7402, "episode_finished", "ep-104", outcome="continue", ran=["code"]),
    ev(7502, "episode_started", "ep-105", number=6),
    ev(7503, "stage_started", "ep-105", "report", agent="writer", attempt=1, resumed=False),
    ev(7533, "llm_server_failed", "ep-105", "report", seconds=30.0, node="gpu47", error="vLLM exited with code 1"),
    ev(7590, "episode_submitted", job="106", needs_llm=True, reason="rtool run", dependency=[], partition="gpu-l40s"),
    ev(7600, "episode_not_started", job="106", error="no secrets file"),
]
DECIDED = [
    {
        "kind": "approve",
        "stage": None,
        "gate": None,
        "by": "yash",
        "created_at": iso(T0 + 3511),
        "applied_at": iso(T0 + 3700),
        "outcome": {"status": "applied"},
    }
]


def test_union_and_wall_time_split() -> None:
    assert union_seconds([(0, 10), (5, 12), (20, 25), (24, 24)]) == 17
    # the earlier layer wins where layers overlap; the last item is what no layer covers
    assert split_wall_time([[(0, 10)], [(5, 20)]], 0, 30) == [10, 10, 10]


def test_where_the_wall_time_went() -> None:
    stats = compute_stats(EVENTS, DECIDED, stage_order=["literature", "code", "report"])
    wall = stats.wall
    assert (wall.total_s, wall.running_s, wall.person_s, wall.jobs_s, wall.queue_s, wall.idle_s) == (
        7600, 724, 2000, 3600, 1209, 67,
    )  # fmt: skip
    assert (wall.gpu_episodes, wall.gpu_run_s, wall.gpu_cold_start_s, wall.gpu_llm_busy_s) == (3, 601, 270, 38)
    assert stats.notes == [] and stats.auto_gates == 1 and stats.run == "r1"


def test_episodes() -> None:
    episodes = {(e.job, e.attempt): e for e in compute_stats(EVENTS, DECIDED).episodes}
    first = episodes["101", 1]
    assert (first.kind, first.partition, first.wait_s, first.queue_s, first.run_s, first.outcome) == (
        "cpu", "nodes", 30, 30, 11, "needs_llm",
    )  # fmt: skip
    gpu = episodes["102", 1]
    assert (gpu.kind, gpu.wait_s, gpu.queue_s, gpu.run_s, gpu.outcome) == ("gpu", 601, 601, 459, "unfinished")
    assert (gpu.node, gpu.gpu, gpu.server_start_s, gpu.server_ready) == ("gpu45", "NVIDIA L40S, 46068 MiB", 180, True)
    assert gpu.startup() == (180, False)
    assert (gpu.llm_calls, gpu.truncated, gpu.llm_busy_s, gpu.busy_fraction) == (4, 1, 38, 0.083)
    assert (gpu.prompt_tokens, gpu.completion_tokens, gpu.ran) == (400, 40, ["literature"])
    requeued = episodes["102", 2]
    assert (requeued.reason, requeued.wait_s, requeued.queue_s, requeued.server_start_s) == ("requeued", 300, 300, 60)
    assert requeued.outcome == "waiting_gate" and requeued.node == "gpu46"  # read from the older field name
    after_jobs = episodes["104", 1]
    assert (after_jobs.wait_s, after_jobs.queue_s, after_jobs.after_jobs) == (3601, None, ["900"])
    continuation = episodes["105", 1]  # queued behind its own predecessor: counted from when that ended
    assert (continuation.wait_s, continuation.queue_s, continuation.kind) == (101, 100, "gpu")
    failed = episodes["106", 1]
    assert (failed.outcome, failed.error, failed.started, failed.wait_s) == ("not_started", "no secrets file", None, 10)
    assert [e.job for e in compute_stats(EVENTS, DECIDED).episodes] == ["101", "102", "102", "103", "104", "105", "106"]


def test_gates_split_the_person_from_the_episode_that_applied_it() -> None:
    (gate,) = compute_stats(EVENTS, DECIDED).gates
    assert (gate.stage, gate.gate, gate.result, gate.by) == ("literature", "literature_review", "approved", "yash")
    assert (gate.wait_s, gate.human_s, gate.apply_s) == (2190, 2001, 189)
    # without the processed actions the episode the decision kicked off stands in for it, with a note
    fallback = compute_stats(EVENTS)
    assert (fallback.gates[0].human_s, fallback.gates[0].apply_s) == (2001, 189)
    assert "decision times are unknown" in fallback.notes[0]
    # with neither, the whole wait counts as the person's
    no_kick = [e for e in EVENTS if (e.get("data") or {}).get("reason") != "action:approve"]
    gate = compute_stats(no_kick).gates[0]
    assert (gate.human_s, gate.wait_s) == (None, 2190)


def test_stages_phases_and_vllm_starts() -> None:
    stats = compute_stats(EVENTS, DECIDED, stage_order=["literature", "code", "report"])
    stages = {s.stage: s for s in stats.stages}
    assert [s.stage for s in stats.stages] == ["literature", "code", "report"]
    lit = stages["literature"]
    assert (lit.turns, lit.active_s, lit.attempts, lit.last) == (2, 567, 1, "finished")
    assert (lit.llm_calls, lit.truncated, lit.llm_busy_s, lit.server_start_s) == (4, 1, 38, 240)
    assert (stages["code"].turns, stages["code"].active_s, stages["code"].last) == (2, 96, "finished")
    assert (stages["report"].active_s, stages["report"].last, stages["report"].server_start_s) == (30, "unfinished", 30)
    # a phase interrupted by the node failure counts its time in both turns
    assert [(p.phase, p.seconds) for p in stats.phases] == [("plan", 200), ("search", 158), ("screen", 199)]
    assert stats.phases[2].included == 12
    assert [(s.episode, s.seconds, s.ok) for s in stats.servers] == [
        ("ep-102", 180, True), ("ep-102", 60, True), ("ep-105", 30, False),
    ]  # fmt: skip
    assert stats.servers[2].error == "vLLM exited with code 1"
    assert (stats.llm.calls, stats.llm.truncated, stats.llm.latency_s, stats.llm.busy_s) == (4, 1, 43, 38)
    assert stats.llm.by_profile == {"extract": 4}


def test_logs_without_vllm_start_events_bound_it_by_the_first_llm_call() -> None:
    old = [e for e in EVENTS if e["kind"] not in ("llm_server_ready", "llm_server_failed")]
    stats = compute_stats(old, DECIDED)
    gpu = next(e for e in stats.episodes if e.job == "102")
    assert (gpu.server_start_s, gpu.first_llm_s) == (None, 181)  # first call started at 822, the episode at 641
    assert gpu.startup() == (181, True)
    wall = stats.wall
    assert (wall.gpu_startup_s, wall.gpu_startup_bounded, wall.gpu_cold_start_s) == (181, 1, 0)
    assert any(note.startswith("1 GPU episode(s) logged no llm_server_ready") for note in stats.notes)


# The code stage fails and the episode stops for a person; a week later they retry.
FAILED = [
    ev(0, "episode_submitted", job="201", needs_llm=False, reason="rtool run", dependency=[], partition="nodes"),
    ev(10, "episode_started", "ep-201", number=1),
    ev(11, "stage_started", "ep-201", "code", agent="coder", attempt=1, resumed=False),
    ev(70, "stage_failed", "ep-201", "code", attempt=1, error="smoke test failed", traceback=None),
    ev(71, "episode_finished", "ep-201", outcome="needs_human", ran=["code"]),
    ev(
        604871, "episode_submitted", job="202", needs_llm=False, reason="action:retry", dependency=[], partition="nodes"
    ),  # fmt: skip
    ev(604931, "episode_started", "ep-202", number=2),
    ev(604931, "action_applied", "ep-202", action="retry", by="yash", outcome={"status": "applied"}),
    ev(604932, "stage_started", "ep-202", "code", agent="coder", attempt=1, resumed=True),
    ev(604990, "stage_finished", "ep-202", "code", attempt=1),
    ev(604991, "episode_finished", "ep-202", outcome="done", ran=["code"]),
]


def test_a_failed_stage_waiting_for_a_person_is_its_own_part_of_the_wall_time() -> None:
    wall = compute_stats(FAILED).wall
    assert (wall.total_s, wall.running_s, wall.queue_s, wall.failed_s, wall.idle_s) == (604991, 121, 70, 604800, 0)
    # the person's queued action ends the wait even when no episode was submitted for it
    retried = [{"kind": "retry", "created_at": iso(T0 + 604850), "applied_at": iso(T0 + 604931)}]
    no_kick = [e for e in FAILED if e["kind"] != "episode_submitted" or e["data"]["job"] != "202"]
    wall = compute_stats(no_kick, retried).wall
    assert (wall.running_s, wall.queue_s, wall.failed_s, wall.idle_s) == (121, 10, 604779, 81)


# The bad-node branch's llm_server_failed: no episode, stage or seconds; node, excluded, giving_up.
FOREIGN = [
    ev(0, "episode_submitted", job="301", needs_llm=True, reason="needs_llm", dependency=[], partition="gpu-l40s"),
    ev(100, "episode_started", "ep-301", number=1),
    ev(101, "stage_started", "ep-301", "literature", agent="literature", attempt=1, resumed=False),
    call(190, "ep-301", 10),  # its first call started 80 s into the episode
    ev(400, "episode_finished", "ep-301", outcome="continue", ran=["literature"]),
    ev(401, "episode_submitted", job="302", needs_llm=True, reason="continue", dependency=[], partition="gpu-l40s"),
    ev(500, "episode_started", "ep-302", number=2),
    ev(501, "stage_started", "ep-302", "literature", agent="literature", attempt=1, resumed=True),
    ev(620, "llm_server_failed", node="gpu45", error="vLLM exited with code 1", excluded=["gpu45"], giving_up=True),
    ev(3620, "episode_submitted", job="303", needs_llm=True, reason="rtool run", dependency=[], partition="gpu-l40s"),
    ev(3700, "episode_started", "ep-303", number=3),
    ev(3701, "episode_finished", "ep-303", outcome="done", ran=[]),
]


def test_a_vllm_failure_logged_by_other_code(runs_root: Path) -> None:
    stats = compute_stats(FOREIGN)
    (failed,) = stats.servers
    assert (failed.episode, failed.stage, failed.node, failed.seconds, failed.ok) == (
        "ep-302", "literature", "gpu45", None, False,
    )  # fmt: skip
    episodes = {e.job: e for e in stats.episodes}
    # the episode that logged nothing about its vLLM is bounded by its first LLM call ...
    assert episodes["301"].startup() == (80, True)
    # ... and the one whose start failed shows that, with its run time up to the failure
    broken = episodes["302"]
    assert (broken.server_failures, broken.server_ready, broken.startup(), broken.run_s) == (
        1,
        False,
        (None, False),
        120,
    )
    wall = stats.wall
    assert (wall.gpu_startup_s, wall.gpu_startup_bounded) == (80, 1)
    assert any(note.startswith("1 GPU episode(s) logged no llm_server_ready") for note in stats.notes)
    # giving up stops the run for a person, until the next episode starts
    assert (wall.running_s, wall.queue_s, wall.failed_s, wall.idle_s) == (421, 279, 3000, 1)

    log = runs_root / "events.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("".join(json.dumps(e) + "\n" for e in FOREIGN), encoding="utf-8")
    code, out = invoke("stats", log)
    assert code == 0 and "<=1m20s" in out and "failed" in out and "a failed stage or run waited for a person" in out


def test_a_run_whose_run_json_this_rtool_cannot_load(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, "toy")
    drive_as_slurm_would(run, mock_llm)
    spec = json.loads((run.root / "run.json").read_text(encoding="utf-8"))
    spec["config"]["llm"]["from_a_newer_rtool"] = True  # a key this rtool does not know
    (run.root / "run.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError):
        RunHandle(run.root)
    stats = stats_for_run_dir(run.root)
    assert stats.run == run.run_id and [s.stage for s in stats.stages] == ["source", "analyze", "write"]
    assert stats.episodes and all(g.decided for g in stats.gates)  # the processed actions were read too
    assert stats.notes[-1] == (
        "run.json did not load (config.llm.from_a_newer_rtool: Extra inputs are not permitted): "
        "read events.jsonl and actions/ only"
    )
    code, out = invoke("stats", run.root)
    assert code == 0 and "run.json did not load" in out


def test_an_empty_log() -> None:
    stats = compute_stats([])
    assert stats.episodes == [] and stats.notes == ["the event log is empty"]


# ----------------------------------------------------------- vLLM start-up events
class FakeServer:
    fail = False

    def __init__(self, llm: Any, log_dir: Path) -> None:
        pass

    def start(self) -> ServerHandle:
        if self.fail:
            raise VLLMServerError("vLLM exited with code 1 during startup")
        return ServerHandle(url="http://127.0.0.1:9", api_key="k", pid=0, log_path=Path("vllm.log"), process=None)  # type: ignore[arg-type]

    @staticmethod
    def stop(handle: ServerHandle, grace_s: float = 30.0) -> None:
        pass


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch, mock_llm: MockLLM) -> type[FakeServer]:
    monkeypatch.setattr(episode_module, "VLLMServer", FakeServer)
    monkeypatch.setattr(episode_module, "build_llm", lambda config, endpoint=None: LLMClient(mock_llm, config.llm))
    monkeypatch.setattr(server_module, "gpu_names", lambda: "NVIDIA L40S, 46068 MiB")
    for name, value in {"SLURMD_NODENAME": "gpu45", "SLURM_JOB_PARTITION": "gpu-l40s", "SLURM_JOB_ID": "7"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    FakeServer.fail = False
    return FakeServer


TOY_JOB = {
    "name": "job",
    "stages": [{"id": "source", "agent": "toy_source"}, {"id": "analyze", "agent": "toy_analyzer"}],
}


def test_gpu_episode_records_when_its_vllm_was_ready(runs_root: Path, fake_vllm: type[FakeServer]) -> None:
    run = make_run(runs_root, TOY_JOB, overrides={"llm": {"backend": "vllm"}})
    run_episode(run, slurm=FakeSlurm(run.jobs_dir / "fake_slurm.json"), has_gpu=True, episode_id="ep-7")
    (ready,) = EventLog(run.events_path, run.run_id).read(kinds=["llm_server_ready"])
    assert (ready["episode"], ready["stage"]) == ("ep-7", "source")
    data = ready["data"]
    assert data["seconds"] >= 0 and data["model"] == run.config.llm.model
    assert (data["node"], data["gpu"], data["partition"], data["devices"], data["job"]) == (
        "gpu45", "NVIDIA L40S, 46068 MiB", "gpu-l40s", "0", "7",
    )  # fmt: skip
    stats = run_stats(run)
    assert stats.servers[0].node == "gpu45" and stats.episodes[0].server_ready
    assert stats.episodes[0].on_gpu and stats.wall.gpu_episodes == 1


def test_a_vllm_that_fails_to_start_is_recorded(runs_root: Path, fake_vllm: type[FakeServer]) -> None:
    run = make_run(runs_root, TOY_JOB, overrides={"llm": {"backend": "vllm"}})
    fake_vllm.fail = True
    lazy = LazyLLM(run.config, run.jobs_dir / "llm")
    lazy.events = EventLog(run.events_path, run.run_id, "ep-7")
    with pytest.raises(VLLMServerError) as raised:
        lazy.bind(budget=None, on_call=None, stage="source")
    assert not EventLog(run.events_path, run.run_id).read(kinds=["llm_server_failed"])  # run_episode records it
    server_failed(run, "gpu45", str(raised.value), **lazy.failure)  # what run_episode does with the error
    (failed,) = EventLog(run.events_path, run.run_id).read(kinds=["llm_server_failed"])
    assert failed["stage"] == "source" and "exited with code 1" in failed["data"]["error"]
    assert failed["data"]["node"] == "gpu45" and failed["data"]["seconds"] is not None  # one event, every fact


def test_node_facts_without_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "gpu_names", lambda: None)
    facts = node_facts({})
    assert facts["node"] and facts["gpu"] is None and facts["partition"] is None and facts["job"] is None


# ------------------------------------------------------------------ real runs
class Factory:
    def __init__(self, run: RunHandle, llm: MockLLM) -> None:
        self.run, self.llm = run, llm

    def __call__(self) -> LLMClient:
        return LLMClient(self.llm, self.run.config.llm)


def drive_as_slurm_would(run: RunHandle, llm: MockLLM) -> None:
    """Run the toy pipeline as chained episode jobs, approving every gate (as `rtool approve` would)."""
    slurm = FakeSlurm(run.jobs_dir / "fake_slurm.json")
    job = submit_episode(run, slurm, needs_llm=False, reason="rtool run")
    for _ in range(30):
        assert job is not None
        on_gpu = slurm.jobs[job]["spec"]["gpus"] is not None
        result = run_episode(run, slurm=slurm, llm_factory=Factory(run, llm), has_gpu=on_gpu, episode_id=f"ep-{job}")
        if result.outcome is Outcome.DONE:
            return
        if result.outcome is Outcome.WAITING_GATE:
            queue(run, ActionKind.APPROVE)
            job = submit_episode(run, slurm, needs_llm=False, reason="action:approve", self_job_id=job)
            continue
        job = schedule_next(run, slurm, result, self_job_id=job)
        if result.outcome is Outcome.WAITING_JOB:
            slurm.finish_all()
    raise AssertionError("the toy run did not finish")


def invoke(*args: object) -> tuple[int, str]:
    result = CliRunner().invoke(app, [str(a) for a in args])
    return result.exit_code, result.output


def test_stats_of_a_toy_run_of_episode_jobs(runs_root: Path, mock_llm: MockLLM) -> None:
    run = make_run(runs_root, "toy")
    drive_as_slurm_would(run, mock_llm)
    stats = run_stats(run)
    assert [s.stage for s in stats.stages] == ["source", "analyze", "write"]
    assert all(s.last == "finished" for s in stats.stages)
    first = stats.episodes[0]
    assert (first.kind, first.partition, first.reason, first.outcome) == ("cpu", "nodes", "rtool run", "needs_llm")
    assert first.queue_s is not None and first.queue_s < 5
    assert any(e.kind == "gpu" and e.partition == "gpu-l40s" and e.llm_calls for e in stats.episodes)
    assert any(e.after_jobs and e.queue_s is None for e in stats.episodes)  # waited on the experiment job
    assert all(e.job and e.outcome not in ("unfinished", "never_started") for e in stats.episodes)
    assert stats.episodes[-1].outcome == "done"
    assert {g.gate for g in stats.gates} == {"numbers_review", "gpu_runs", "final"}
    assert all(g.result == "approved" and g.decided and g.by for g in stats.gates)
    wall = stats.wall
    parts = wall.running_s + wall.person_s + wall.jobs_s + wall.queue_s + wall.idle_s
    assert abs(parts - wall.total_s) < 0.5 and stats.llm.calls > 0

    code, out = invoke("stats", run.root)
    assert code == 0, out
    assert all(title in out for title in ("where the wall time went", "episodes", "gates", "stages"))
    code, out = invoke("stats", run.root, "--json")
    assert code == 0 and RunStats.model_validate(json.loads(out)) == stats
    code, out = invoke("stats", run.events_path, "--json")  # a copied event log works too
    assert code == 0 and len(json.loads(out)["episodes"]) == len(stats.episodes)


def test_stats_of_a_demo_run_shows_the_literature_phases(runs_root: Path) -> None:
    run = build_demo_run(runs_root, "literature_review")
    stats = run_stats(run)
    assert [p.phase for p in stats.phases][:3] == ["plan", "search", "screen"] and stats.phases[-1].phase == "map"
    assert stats.stages[0].stage == "literature" and stats.stages[0].llm_calls > 0
    assert stats.gates[0].gate == "literature_review" and stats.gates[0].result == "open"
    assert any("ran locally" in note for note in stats.notes)
    code, out = invoke("stats", run.root)
    assert code == 0 and "literature phases" in out
