"""Where a run's wall time went, rebuilt from its event log (``rtool stats``).

A run is a chain of Slurm episode jobs, so at any moment it is doing one of a
few things: an episode is running; a decision waits for a person at a gate; an
episode waits behind cluster jobs it depends on (experiments); an episode waits
in the queue for a node; or nothing is happening. This module rebuilds those
intervals from ``events.jsonl``, plus the processed actions (when a person
actually decided), and reports:

- episodes: queue wait (``episode_submitted`` -> ``episode_started``; the engine's
  episode id is ``ep-<job id>``), CPU or GPU, partition, run time and outcome;
- gates: how long each waited for a person, then for an episode to apply the decision;
- stages and literature phases: time spent running them (time between episodes excluded);
- the LLM: calls, tokens, truncated answers (finish_reason=length), vLLM cold starts
  (``llm_server_ready``) and how busy each episode kept the model: the union of the
  ``[ts - latency_s, ts]`` intervals of its ``llm_call`` events over the episode's length.

Event times have one-second resolution, so short intervals are approximate.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .actions import ActionInbox
from .events import EventLog
from .run import RunHandle

# Events that end a stage's turn on a node. gate_opened: its work is done and it waits for a person;
# a stage_finished that comes later (the gate approved) is not a turn of its own.
STAGE_ENDS = {
    "stage_finished": "finished",
    "stage_suspended": "suspended",
    "stage_deferred": "deferred",
    "gate_opened": "gate",
    "stage_waiting_for_revisions": "revisions",
    "stage_retrying": "retrying",
    "stage_failed": "failed",
}
DECISIONS = {"approve", "reject", "select"}
SLURM_JOB_ID = re.compile(r"\d+(?:[_+]\d+)?")  # an episode job's engine id is ep-<this>; local ones are ep-<hex>


class EpisodeStats(BaseModel):
    episode: str | None = None  # the engine's id; ep-<job id> for an episode job
    job: str | None = None
    attempt: int = 1  # >1: the same job ran again (preempted and requeued)
    kind: str = "local"  # gpu | cpu (as submitted) | local (not an episode job, e.g. `rtool run`)
    reason: str | None = None  # why it was submitted: needs_llm, continue, waiting_job, action:approve, requeued...
    partition: str | None = None
    gpu: str | None = None  # what the node reported when vLLM started, else the GPU asked for
    host: str | None = None
    submitted: str | None = None
    started: str | None = None
    ended: str | None = None
    wait_s: float | None = None  # submitted -> started
    queue_s: float | None = None  # the part of wait_s spent waiting for a node (None: unknown, see after_jobs)
    after_jobs: list[str] = Field(default_factory=list)  # it first waited for these jobs to end
    run_s: float | None = None
    outcome: str = ""  # episode_finished's outcome; unfinished; never_started; not_started (it failed to)
    error: str | None = None
    ran: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: int = 0  # answers cut off at max_tokens (finish_reason=length)
    llm_busy_s: float = 0.0  # time with at least one LLM call in flight
    busy_fraction: float | None = None  # llm_busy_s / run_s
    server_start_s: float | None = None  # vLLM cold start inside this episode
    first_llm_s: float | None = None  # start -> its first LLM call (bounds the cold start in logs without it)

    @property
    def on_gpu(self) -> bool:
        return self.kind == "gpu" or self.server_start_s is not None


class GateStats(BaseModel):
    stage: str
    gate: str
    in_stage: bool = False
    opened: str
    decided: str | None = None  # when the person acted (their action was queued), if known
    resolved: str | None = None  # when an episode applied the decision
    result: str = "open"  # approved | rejected | cancelled (the stage went stale) | open
    by: str | None = None
    wait_s: float = 0.0  # opened -> resolved (or the end of the log)
    human_s: float | None = None  # opened -> decided
    apply_s: float | None = None  # decided -> resolved: the decision waiting for an episode


class StageStats(BaseModel):
    stage: str
    attempts: int = 0
    turns: int = 0  # times it ran on a node (it resumes after jobs, in-stage gates and hand-overs)
    active_s: float = 0.0
    first_started: str
    last_ended: str | None = None
    last: str = ""  # how its latest turn ended: finished, gate, suspended, failed, interrupted, unfinished, ...
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: int = 0
    llm_busy_s: float = 0.0
    server_start_s: float = 0.0  # vLLM cold starts paid while this stage was running


class PhaseStats(BaseModel):
    stage: str
    attempt: int
    phase: str
    seconds: float  # running time since the previous phase ended (time between episodes excluded)
    ended: str
    papers: int | None = None
    included: int | None = None


class ServerStart(BaseModel):
    episode: str | None = None
    stage: str | None = None
    at: str
    seconds: float
    ok: bool = True
    host: str | None = None
    gpu: str | None = None
    partition: str | None = None
    model: str | None = None
    error: str | None = None


class LLMTotals(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: int = 0
    latency_s: float = 0.0  # summed over calls (more than busy_s when calls overlap)
    busy_s: float = 0.0  # time with at least one call in flight
    by_profile: dict[str, int] = Field(default_factory=dict)


class WallTime(BaseModel):
    """The run's wall time split into exclusive parts (an instant counts for the first that applies)."""

    total_s: float = 0.0
    running_s: float = 0.0  # an episode was running
    person_s: float = 0.0  # a gate waited for a decision
    jobs_s: float = 0.0  # an episode waited behind cluster jobs (and then in the queue)
    queue_s: float = 0.0  # an episode waited in the queue for a node
    idle_s: float = 0.0  # none of these (e.g. a decided gate with no episode submitted yet)
    gpu_episodes: int = 0
    gpu_run_s: float = 0.0  # time GPU episodes ran ...
    gpu_cold_start_s: float = 0.0  # ... of which vLLM was starting
    gpu_llm_busy_s: float = 0.0  # ... of which an LLM call was in flight (the rest: CPU work on a GPU node)
    gpu_to_first_llm_s: float = 0.0  # ... of which came before each one's first LLM call


class RunStats(BaseModel):
    run: str = ""
    first: str | None = None
    last: str | None = None
    wall: WallTime = Field(default_factory=WallTime)
    episodes: list[EpisodeStats] = Field(default_factory=list)
    gates: list[GateStats] = Field(default_factory=list)
    auto_gates: int = 0
    stages: list[StageStats] = Field(default_factory=list)
    phases: list[PhaseStats] = Field(default_factory=list)
    servers: list[ServerStart] = Field(default_factory=list)
    llm: LLMTotals = Field(default_factory=LLMTotals)
    notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------- helpers
def epoch(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds")


def union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Total length covered by the intervals (overlaps counted once)."""
    total, reach = 0.0, None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if reach is None or start >= reach:
            total += end - start
            reach = end
        elif end > reach:
            total += end - reach
            reach = end
    return total


def clip(intervals: Iterable[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    return [(max(a, lo), min(b, hi)) for a, b in intervals if min(b, hi) > max(a, lo)]


def split_wall_time(layers: Sequence[Sequence[tuple[float, float]]], start: float, end: float) -> list[float]:
    """Seconds of [start, end] covered by each layer, the earlier layer winning; the last item is the rest."""
    points = sorted({start, end, *(t for layer in layers for a, b in layer for t in (a, b) if start < t < end)})
    totals = [0.0] * (len(layers) + 1)
    for a, b in zip(points, points[1:], strict=False):
        mid = (a + b) / 2
        hit = next((i for i, layer in enumerate(layers) if any(x <= mid < y for x, y in layer)), len(layers))
        totals[hit] += b - a
    return totals


def _r(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds, 1)


# --------------------------------------------------------------- accumulators
@dataclass
class _LLM:
    calls: int = 0
    prompt: int = 0
    completion: int = 0
    truncated: int = 0
    intervals: list[tuple[float, float]] = field(default_factory=list)

    def add(self, t: float, data: Mapping[str, Any]) -> None:
        self.calls += 1
        self.prompt += int(data.get("prompt_tokens") or 0)
        self.completion += int(data.get("completion_tokens") or 0)
        self.truncated += data.get("finish_reason") == "length"
        self.intervals.append((t - float(data.get("latency_s") or 0.0), t))


@dataclass
class _Episode:
    id: str
    attempt: int
    start: float
    last: float
    end: float | None = None
    outcome: str = "unfinished"
    ran: list[str] = field(default_factory=list)
    llm: _LLM = field(default_factory=_LLM)
    server_s: float | None = None
    host: str | None = None
    gpu: str | None = None

    @property
    def stop(self) -> float:
        return self.end if self.end is not None else self.last


@dataclass
class _Turn:
    stage: str
    attempt: int
    start: float
    episode: _Episode | None
    end: float | None = None
    how: str = ""


@dataclass
class _Gate:
    stage: str
    gate: str
    in_stage: bool
    opened: float
    resolved: float | None = None
    result: str = "open"
    decided: float | None = None
    by: str | None = None


@dataclass
class _Phases:
    attempt: int = 0
    carry: float = 0.0
    mark: float | None = None


class _Builder:
    def __init__(self) -> None:
        self.episodes: list[_Episode] = []
        self.open_episodes: dict[str, _Episode] = {}
        self.attempts: Counter[str] = Counter()
        self.submissions: dict[str, tuple[float, dict[str, Any]]] = {}
        self.not_started: dict[str, tuple[float, str]] = {}
        self.turns: list[_Turn] = []
        self.open_turns: dict[str, _Turn] = {}
        self.last_how: dict[str, str] = {}
        self.stage_attempts: dict[str, int] = {}
        self.stage_llm: dict[str, _LLM] = {}
        self.phase_clock: dict[str, _Phases] = {}
        self.phases: list[PhaseStats] = []
        self.gates: list[_Gate] = []
        self.open_gates: dict[tuple[str, str], _Gate] = {}
        self.auto_gates = 0
        self.action_kicks: list[float] = []  # episodes submitted because a person acted
        self.servers: list[ServerStart] = []
        self.all_llm = _LLM()
        self.latency_s = 0.0
        self.profiles: Counter[str] = Counter()

    # ---------------------------------------------------------------- events
    def feed(self, t: float, event: Mapping[str, Any]) -> None:
        kind = event.get("kind")
        data = event.get("data") or {}
        stage = event.get("stage")
        eid = event.get("episode")
        if kind == "episode_started" and eid:
            if eid in self.open_episodes:  # the same job again: it was requeued
                self._end_episode(self.open_episodes.pop(eid), "interrupted")
            self.attempts[eid] += 1
            episode = _Episode(id=eid, attempt=self.attempts[eid], start=t, last=t)
            self.open_episodes[eid] = episode
            self.episodes.append(episode)
            return
        episode = self.open_episodes.get(eid) if eid else None
        if episode is not None:
            episode.last = t
        if kind == "episode_finished" and episode is not None:
            episode.end = t
            episode.outcome = str(data.get("outcome", ""))
            episode.ran = list(data.get("ran") or [])
            self._end_episode(self.open_episodes.pop(episode.id), "interrupted")
        elif kind == "episode_submitted" and data.get("job") is not None:
            self.submissions[str(data["job"])] = (t, dict(data))
            if str(data.get("reason", "")).startswith("action:"):
                self.action_kicks.append(t)
        elif kind == "episode_not_started" and data.get("job") is not None:
            self.not_started[str(data["job"])] = (t, str(data.get("error", "")))
        elif kind == "llm_call":
            self._llm_call(t, stage, data, episode)
        elif kind in ("llm_server_ready", "llm_server_failed"):
            self._server(t, kind, event, data, episode)
        elif kind == "stage_started" and stage:
            self._start_turn(t, stage, data, episode)
        elif kind == "literature_phase_done" and stage:
            self._phase_done(t, stage, data)
        if stage and kind in STAGE_ENDS:
            self.last_how[stage] = STAGE_ENDS[kind]
            if stage in self.open_turns:
                self._end_turn(self.open_turns.pop(stage), t, STAGE_ENDS[kind])
        self._gate_event(t, kind, stage, data)

    def _llm_call(self, t: float, stage: str | None, data: Mapping[str, Any], episode: _Episode | None) -> None:
        self.all_llm.add(t, data)
        self.latency_s += float(data.get("latency_s") or 0.0)
        self.profiles[str(data.get("profile") or "?")] += 1
        if episode is not None:
            episode.llm.add(t, data)
        if stage:
            self.stage_llm.setdefault(stage, _LLM()).add(t, data)

    def _server(
        self, t: float, kind: str, event: Mapping[str, Any], data: Mapping[str, Any], episode: _Episode | None
    ) -> None:
        seconds = float(data.get("seconds") or 0.0)
        start = ServerStart(
            episode=event.get("episode"),
            stage=event.get("stage"),
            at=str(event.get("ts")),
            seconds=round(seconds, 1),
            ok=kind == "llm_server_ready",
            host=data.get("host"),
            gpu=data.get("gpu"),
            partition=data.get("partition"),
            model=data.get("model"),
            error=data.get("error"),
        )
        self.servers.append(start)
        if episode is not None:
            episode.server_s = (episode.server_s or 0.0) + seconds
            episode.host = episode.host or start.host
            episode.gpu = episode.gpu or start.gpu

    def _start_turn(self, t: float, stage: str, data: Mapping[str, Any], episode: _Episode | None) -> None:
        if stage in self.open_turns:  # its episode died mid-stage
            stale = self.open_turns.pop(stage)
            self._end_turn(stale, stale.episode.last if stale.episode else t, "interrupted")
        attempt = int(data.get("attempt") or 0)
        self.stage_attempts[stage] = max(self.stage_attempts.get(stage, 0), attempt)
        turn = _Turn(stage=stage, attempt=attempt, start=t, episode=episode)
        self.open_turns[stage] = turn
        self.turns.append(turn)
        clock = self.phase_clock.setdefault(stage, _Phases())
        if not data.get("resumed") or clock.attempt != attempt:
            clock.carry = 0.0  # a fresh attempt starts its phases again
        clock.attempt = attempt
        clock.mark = t

    def _end_turn(self, turn: _Turn, t: float, how: str) -> None:
        turn.end = max(t, turn.start)
        turn.how = how
        self.last_how[turn.stage] = how
        clock = self.phase_clock.get(turn.stage)
        if clock is not None and clock.mark is not None:
            clock.carry += turn.end - clock.mark
            clock.mark = None

    def _end_episode(self, episode: _Episode, how: str) -> None:
        """Close the stage turns the episode left open (it was killed, or the log ends while it runs)."""
        for stage, turn in list(self.open_turns.items()):
            if turn.episode is episode:
                self._end_turn(self.open_turns.pop(stage), episode.stop, how)

    def _phase_done(self, t: float, stage: str, data: Mapping[str, Any]) -> None:
        clock = self.phase_clock.setdefault(stage, _Phases())
        seconds = clock.carry + (t - clock.mark if clock.mark is not None else 0.0)
        clock.carry, clock.mark = 0.0, t
        self.phases.append(
            PhaseStats(
                stage=stage,
                attempt=clock.attempt,
                phase=str(data.get("phase", "?")),
                seconds=round(seconds, 1),
                ended=iso(t) or "",
                papers=data.get("papers"),
                included=data.get("included"),
            )
        )

    def _gate_event(self, t: float, kind: Any, stage: str | None, data: Mapping[str, Any]) -> None:
        if not stage:
            return
        if kind == "gate_opened":
            gate = str(data.get("gate", "?"))
            previous = self.open_gates.pop((stage, gate), None)
            if previous is not None:
                previous.resolved, previous.result = t, "cancelled"
            opened = _Gate(stage=stage, gate=gate, in_stage=bool(data.get("in_stage")), opened=t)
            self.open_gates[(stage, gate)] = opened
            self.gates.append(opened)
        elif kind == "gate_resolved":
            opened = self.open_gates.pop((stage, str(data.get("gate", "?"))), None)
            if opened is not None:
                opened.resolved = t
                opened.result = "approved" if data.get("approved") else "rejected"
        elif kind == "gate_auto_resolved":
            self.auto_gates += 1
        elif kind == "stage_stale":
            for key in [k for k in self.open_gates if k[0] == stage]:
                cancelled = self.open_gates.pop(key)
                cancelled.resolved, cancelled.result = t, "cancelled"

    # ------------------------------------------------------------- decisions
    def match_decisions(self, actions: Sequence[Mapping[str, Any]]) -> bool:
        """Find when a person decided each gate: their action's created_at, else the episode it kicked off."""
        decided: list[tuple[float, float, Mapping[str, Any]]] = []
        for action in actions:
            outcome = action.get("outcome") or {}
            if action.get("kind") not in DECISIONS or outcome.get("status", "applied") != "applied":
                continue
            try:
                decided.append((epoch(action["created_at"]), epoch(action["applied_at"]), action))
            except (KeyError, TypeError, ValueError):
                continue
        exact = True
        for gate in self.gates:
            if gate.resolved is None or gate.result == "cancelled":
                continue
            candidates = [
                (abs(applied - gate.resolved), max(created, gate.opened), action)
                for created, applied, action in decided
                if created <= gate.resolved + 1
                and abs(applied - gate.resolved) <= 5
                and action.get("stage") in (None, gate.stage)
                and action.get("gate") in (None, gate.gate)
            ]
            if candidates:
                _, gate.decided, action = min(candidates, key=lambda c: c[0])
                gate.by = action.get("by")
                continue
            exact = False
            kicks = [k for k in self.action_kicks if gate.opened <= k <= gate.resolved]
            if kicks:
                gate.decided = kicks[0]
        return exact

    # ----------------------------------------------------------------- build
    def episode_rows(self) -> list[EpisodeStats]:
        ends: dict[str, float] = {}
        for episode in self.episodes:
            ends[episode.id] = max(ends.get(episode.id, 0.0), episode.stop)
        rows: list[tuple[float, EpisodeStats]] = []
        matched: set[str] = set()
        previous: dict[str, _Episode] = {}
        for episode in self.episodes:
            job = episode.id[3:] if episode.id.startswith("ep-") else None
            if job is not None and job not in self.submissions and not SLURM_JOB_ID.fullmatch(job):
                job = None
            submission = self.submissions.get(job) if job and episode.attempt == 1 else None
            row = EpisodeStats(
                episode=episode.id,
                job=job,
                attempt=episode.attempt,
                kind="local",
                started=iso(episode.start),
                ended=iso(episode.stop),
                run_s=_r(episode.stop - episode.start),
                outcome=episode.outcome,
                ran=episode.ran or list(dict.fromkeys(t.stage for t in self.turns if t.episode is episode)),
                host=episode.host,
                gpu=episode.gpu,
                server_start_s=_r(episode.server_s),
            )
            if job and job in self.submissions:
                matched.add(job)
                _, sub = self.submissions[job]
                row.kind = "gpu" if sub.get("needs_llm") else "cpu"
                row.partition = sub.get("partition")
                row.gpu = row.gpu or sub.get("gpu")
                row.reason = sub.get("reason")
            if submission is not None:
                submitted_at, sub = submission
                row.submitted = iso(submitted_at)
                row.wait_s = _r(max(0.0, episode.start - submitted_at))
                deps = [str(d) for d in sub.get("dependency") or []]
                row.after_jobs = [d for d in deps if f"ep-{d}" not in ends]
                if not row.after_jobs:
                    ready = max([submitted_at, *(ends[f"ep-{d}"] for d in deps)])
                    row.queue_s = _r(max(0.0, episode.start - ready))
            elif episode.id in previous:  # back in the queue after a preemption
                row.reason = "requeued"
                row.wait_s = row.queue_s = _r(max(0.0, episode.start - previous[episode.id].stop))
            previous[episode.id] = episode
            llm = episode.llm
            busy = union_seconds(clip(llm.intervals, episode.start, episode.stop))
            row.llm_calls, row.prompt_tokens, row.completion_tokens = llm.calls, llm.prompt, llm.completion
            row.truncated, row.llm_busy_s = llm.truncated, round(busy, 1)
            length = episode.stop - episode.start
            row.busy_fraction = round(busy / length, 3) if length > 0 else None
            if llm.intervals:
                row.first_llm_s = _r(max(0.0, min(a for a, _ in llm.intervals) - episode.start))
            rows.append((episode.start, row))
        for job, (submitted_at, sub) in self.submissions.items():
            if job in matched:
                continue
            failed = self.not_started.get(job)
            row = EpisodeStats(
                episode=f"ep-{job}",
                job=job,
                kind="gpu" if sub.get("needs_llm") else "cpu",
                reason=sub.get("reason"),
                partition=sub.get("partition"),
                gpu=sub.get("gpu"),
                submitted=iso(submitted_at),
                after_jobs=[str(d) for d in sub.get("dependency") or [] if f"ep-{d}" not in ends],
                outcome="not_started" if failed else "never_started",
                error=failed[1] if failed else None,
            )
            if failed:
                row.wait_s = _r(max(0.0, failed[0] - submitted_at))
            rows.append((submitted_at, row))
        return [row for _, row in sorted(rows, key=lambda r: r[0])]

    def stage_rows(self, order: Sequence[str]) -> list[StageStats]:
        by_stage: dict[str, StageStats] = {}
        for turn in self.turns:
            row = by_stage.get(turn.stage)
            if row is None:
                row = by_stage[turn.stage] = StageStats(stage=turn.stage, first_started=iso(turn.start) or "")
            end = turn.end if turn.end is not None else (turn.episode.stop if turn.episode else turn.start)
            row.turns += 1
            row.active_s += max(0.0, end - turn.start)
            row.last_ended = iso(end)
        for stage, row in by_stage.items():
            row.active_s = round(row.active_s, 1)
            row.attempts = self.stage_attempts.get(stage, 0)
            row.last = self.last_how.get(stage, "")
            llm = self.stage_llm.get(stage)
            if llm is not None:
                row.llm_calls, row.prompt_tokens, row.completion_tokens = llm.calls, llm.prompt, llm.completion
                row.truncated, row.llm_busy_s = llm.truncated, round(union_seconds(llm.intervals), 1)
            row.server_start_s = round(sum(s.seconds for s in self.servers if s.stage == stage), 1)
        rank = {sid: i for i, sid in enumerate(order)}
        return sorted(by_stage.values(), key=lambda r: (rank.get(r.stage, len(rank)), r.first_started))

    def gate_rows(self, end: float) -> list[GateStats]:
        rows = []
        for gate in self.gates:
            stop = gate.resolved if gate.resolved is not None else end
            decided = gate.decided if gate.decided is not None and gate.decided <= stop else None
            rows.append(
                GateStats(
                    stage=gate.stage,
                    gate=gate.gate,
                    in_stage=gate.in_stage,
                    opened=iso(gate.opened) or "",
                    decided=iso(decided),
                    resolved=iso(gate.resolved),
                    result=gate.result,
                    by=gate.by,
                    wait_s=round(max(0.0, stop - gate.opened), 1),
                    human_s=_r(None if decided is None else max(0.0, decided - gate.opened)),
                    apply_s=_r(None if decided is None or gate.resolved is None else max(0.0, gate.resolved - decided)),
                )
            )
        return rows

    def wall_time(self, start: float, end: float, episodes: Sequence[EpisodeStats]) -> WallTime:
        running = [(e.start, e.stop) for e in self.episodes]
        person = []
        for gate in self.gates:
            stop = gate.resolved if gate.resolved is not None else end
            if gate.decided is not None and gate.decided <= stop:
                stop = gate.decided
            person.append((gate.opened, stop))
        jobs: list[tuple[float, float]] = []
        queue: list[tuple[float, float]] = []
        starts = {e.id: e.start for e in self.episodes if e.attempt == 1}
        ends: dict[str, float] = {}
        for e in self.episodes:
            if e.id in ends:  # requeued: pending again since its previous attempt ended
                queue.append((ends[e.id], e.start))
            ends[e.id] = max(ends.get(e.id, 0.0), e.stop)
        for job, (submitted_at, sub) in self.submissions.items():
            began = starts.get(f"ep-{job}")
            if began is None:
                continue
            deps = [str(d) for d in sub.get("dependency") or []]
            if any(f"ep-{d}" not in ends for d in deps):
                jobs.append((submitted_at, began))
            else:
                queue.append((max([submitted_at, *(ends[f"ep-{d}"] for d in deps)]), began))
        run_s, person_s, jobs_s, queue_s, idle_s = split_wall_time([running, person, jobs, queue], start, end)
        gpu = [e for e in episodes if e.on_gpu and e.run_s is not None]
        return WallTime(
            total_s=round(end - start, 1),
            running_s=round(run_s, 1),
            person_s=round(person_s, 1),
            jobs_s=round(jobs_s, 1),
            queue_s=round(queue_s, 1),
            idle_s=round(idle_s, 1),
            gpu_episodes=len(gpu),
            gpu_run_s=round(sum(e.run_s or 0.0 for e in gpu), 1),
            gpu_cold_start_s=round(sum(e.server_start_s or 0.0 for e in gpu), 1),
            gpu_llm_busy_s=round(sum(e.llm_busy_s for e in gpu), 1),
            gpu_to_first_llm_s=round(sum(e.first_llm_s or 0.0 for e in gpu), 1),
        )


# ----------------------------------------------------------------------- api
def compute_stats(
    events: Iterable[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]] = (),
    *,
    run_id: str = "",
    stage_order: Sequence[str] = (),
) -> RunStats:
    """The timing report for a run's events (in log order) and its processed actions."""
    builder = _Builder()
    first = last = None
    for event in events:
        try:
            t = epoch(str(event["ts"]))
        except (KeyError, ValueError):
            continue
        first = t if first is None else min(first, t)
        last = t if last is None else max(last, t)
        run_id = run_id or str(event.get("run") or "")
        builder.feed(t, event)
    stats = RunStats(run=run_id)
    if first is None or last is None:
        stats.notes.append("the event log is empty")
        return stats
    for episode in list(builder.open_episodes.values()):
        builder._end_episode(episode, "unfinished")  # still running, or killed without a trace
    exact = builder.match_decisions(actions)
    episodes = builder.episode_rows()
    totals = builder.all_llm
    stats.first, stats.last = iso(first), iso(last)
    stats.episodes = episodes
    stats.gates = builder.gate_rows(last)
    stats.auto_gates = builder.auto_gates
    stats.stages = builder.stage_rows(stage_order)
    stats.phases = builder.phases
    stats.servers = builder.servers
    stats.llm = LLMTotals(
        calls=totals.calls,
        prompt_tokens=totals.prompt,
        completion_tokens=totals.completion,
        truncated=totals.truncated,
        latency_s=round(builder.latency_s, 1),
        busy_s=round(union_seconds(totals.intervals), 1),
        by_profile=dict(sorted(builder.profiles.items(), key=lambda p: -p[1])),
    )
    stats.wall = builder.wall_time(first, last, episodes)
    if not exact and any(g.result in ("approved", "rejected") for g in builder.gates):
        stats.notes.append(
            "some decision times are unknown (no processed action matched): "
            "their gate waits may include the wait for an episode to apply them"
        )
    if builder.episodes and not builder.submissions:
        stats.notes.append("no episode_submitted events: episodes ran locally, so there are no queue waits")
    if builder.all_llm.calls and not builder.servers:
        stats.notes.append(
            "no llm_server_ready events: vLLM start-up was not recorded for this run; "
            "the time from each episode's start to its first LLM call bounds it"
        )
    return stats


def stats_from_files(events_path: Path, actions_dir: Path | None = None, **kwargs: Any) -> RunStats:
    """Stats from an events.jsonl (and, if given, the run's actions directory)."""
    actions = ActionInbox(actions_dir).processed() if actions_dir is not None else []
    return compute_stats(EventLog(events_path, run_id=""), actions, **kwargs)


def run_stats(run: RunHandle) -> RunStats:
    """Stats for a run, its stages in pipeline order."""
    order = [stage.id for stage in run.spec.pipeline.stages]
    return stats_from_files(run.events_path, run.actions_dir, run_id=run.run_id, stage_order=order)
