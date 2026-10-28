"""Printing `rtool stats` (the numbers come from core.stats)."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..core.stats import EpisodeStats, RunStats


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 86400}d{s % 86400 // 3600:02d}h"


def when(ts: str | None) -> str:
    """'2026-10-10T19:14:15+00:00' -> '10-10 19:14' (UTC)."""
    return f"{ts[5:10]} {ts[11:16]}" if ts else "-"


def share(part: float | None, whole: float | None) -> str:
    return f"{100 * part / whole:.0f}%" if part is not None and whole else "-"


def tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 10_000 else str(n)


def startup(e: EpisodeStats, *, recorded: bool) -> str:
    """vLLM start-up time; for a log without it, at most the time to the episode's first LLM call."""
    if e.server_start_s is not None or recorded:
        return duration(e.server_start_s)
    return f"<={duration(e.first_llm_s)}" if e.on_gpu and e.first_llm_s is not None else "-"


def table(title: str, *columns: str, left: int = 1) -> Table:
    """A compact table: the first ``left`` columns are text (which wraps in a narrow terminal), the rest numbers."""
    out = Table(
        title=title,
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        pad_edge=False,
        collapse_padding=True,
    )
    for i, column in enumerate(columns):
        text = i < left
        out.add_column(column, justify="left" if text else "right", no_wrap=i == 0 or not text, overflow="fold")
    return out


def print_stats(stats: RunStats, console: Console) -> None:
    wall = stats.wall
    console.print(
        f"[bold]{escape(stats.run)}[/bold]  {when(stats.first)} -> {when(stats.last)} UTC, "
        f"wall time {duration(wall.total_s)}"
    )
    if not stats.episodes:
        for note in stats.notes:
            console.print(f"[yellow]{escape(note)}[/yellow]")
        return

    t = table("where the wall time went", "", "time", "share")
    for label, seconds in (
        ("an episode was running", wall.running_s),
        ("a gate waited for a person", wall.person_s),
        ("an episode waited behind cluster jobs", wall.jobs_s),
        ("an episode waited in the queue", wall.queue_s),
        ("none of these", wall.idle_s),
    ):
        t.add_row(label, duration(seconds), share(seconds, wall.total_s))
    console.print(t)
    if wall.gpu_episodes:
        before = wall.gpu_cold_start_s if stats.servers else wall.gpu_to_first_llm_s
        other = max(0.0, wall.gpu_run_s - before - wall.gpu_llm_busy_s)
        starting = "vLLM starting" if stats.servers else "before the first LLM call"
        console.print(
            f"GPU episodes: {wall.gpu_episodes}, running {duration(wall.gpu_run_s)}: "
            f"{starting} {share(before, wall.gpu_run_s)}, "
            f"LLM busy {share(wall.gpu_llm_busy_s, wall.gpu_run_s)}, "
            f"other work {share(other, wall.gpu_run_s)}\n"
        )

    t = table("episodes", "job", "where", "why", "outcome", "wait", "queue", "run", "busy", "vllm", left=4)
    for e in stats.episodes:
        where = e.partition or e.kind
        queue = duration(e.queue_s) if e.queue_s is not None else ("jobs" if e.after_jobs else "-")
        t.add_row(
            escape((e.job or e.episode or "-") + (f" #{e.attempt}" if e.attempt > 1 else "")),
            escape(where),
            escape(e.reason or "-"),
            escape(e.outcome),
            duration(e.wait_s),
            queue,
            duration(e.run_s),
            share(e.llm_busy_s, e.run_s) if e.llm_calls else "-",
            startup(e, recorded=bool(stats.servers)),
        )
    console.print(t)
    if any(e.after_jobs for e in stats.episodes):
        console.print("queue 'jobs': it waited for cluster jobs first, so only its total wait is known")
    for e in stats.episodes:
        if e.error:
            console.print(f"job {escape(e.job or '-')} did not start: {escape(e.error)}")
    console.print()

    if stats.gates:
        t = table("gates", "stage", "gate", "opened", "result", "waited", "person", "applied", left=4)
        for g in stats.gates:
            t.add_row(
                escape(g.stage),
                escape(g.gate),
                when(g.opened),
                escape(g.result + (f" by {g.by}" if g.by else "")),
                duration(g.wait_s),
                duration(g.human_s),
                duration(g.apply_s),
            )
        console.print(t)
    if stats.auto_gates:
        console.print(f"{stats.auto_gates} gate(s) approved automatically\n")

    t = table("stages", "stage", "last", "tries", "turns", "ran", "calls", "tokens", "cut", "busy", "vllm", left=2)
    for s in stats.stages:
        llm = s.llm_calls > 0
        t.add_row(
            escape(s.stage),
            s.last,
            str(s.attempts),
            str(s.turns),
            duration(s.active_s),
            str(s.llm_calls) if llm else "-",
            f"{tokens(s.prompt_tokens)}/{tokens(s.completion_tokens)}" if llm else "-",
            str(s.truncated) if llm else "-",
            share(s.llm_busy_s, s.active_s) if llm else "-",
            duration(s.server_start_s) if s.server_start_s else "-",
        )
    console.print(t)

    if stats.phases:
        t = table("literature phases", "stage", "try", "phase", "ended", "running", "papers", "included", left=4)
        for p in stats.phases:
            t.add_row(
                escape(p.stage),
                str(p.attempt),
                escape(p.phase),
                when(p.ended),
                duration(p.seconds),
                "-" if p.papers is None else str(p.papers),
                "-" if p.included is None else str(p.included),
            )
        console.print(t)

    if stats.servers:
        t = table("vLLM starts", "at", "episode", "stage", "node", "gpu", "result", "took", left=6)
        for v in stats.servers:
            t.add_row(
                when(v.at),
                escape(v.episode or "-"),
                escape(v.stage or "-"),
                escape(" ".join(x for x in (v.partition, v.host) if x) or "-"),
                escape((v.gpu or "-").removeprefix("NVIDIA ")),
                "ready" if v.ok else escape(f"failed: {(v.error or '')[:60]}"),
                duration(v.seconds),
            )
        console.print(t)

    llm = stats.llm
    if llm.calls:
        profiles = ", ".join(f"{name} {count}" for name, count in llm.by_profile.items())
        console.print(
            f"LLM: {llm.calls} calls ({profiles}); tokens {llm.prompt_tokens:,} in / {llm.completion_tokens:,} out; "
            f"{llm.truncated} cut off at max_tokens; busy {duration(llm.busy_s)} "
            f"(calls summed: {duration(llm.latency_s)})"
        )
    for note in stats.notes:
        console.print(f"[yellow]note:[/yellow] {escape(note)}")
