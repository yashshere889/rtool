"""`rtool` — create runs, advance them, and make decisions at gates.

Decision commands (approve / reject / select / retry / note) never modify a
run directly: they queue an action that the next episode applies under the
run lease. Pass ``--now`` to run that episode immediately.
"""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .. import __version__
from ..core.actions import Action, ActionInbox, ActionKind
from ..core.artifacts import ArtifactStore
from ..core.config import (
    AppConfig,
    apply_overrides,
    deep_merge,
    default_config_files,
    load_config,
    load_yaml_mapping,
    parse_set_overrides,
)
from ..core.engine import AdvanceResult, Engine, Outcome, load_status
from ..core.errors import RToolError
from ..core.events import EventLog
from ..core.fs import atomic_write_json
from ..core.pipeline import builtin_pipelines, find_pipeline, resolve_pipeline
from ..core.registry import registry
from ..core.run import RunHandle, resolve_run_path
from ..core.secrets import load_secrets
from ..core.services import LLMEndpoint, build_llm, build_services, build_slurm
from ..core.state import StageStatus
from ..hpc.episode import SAFETY_MARGIN_S, llm_needed, run_episode, schedule_next, submit_episode
from ..hpc.fake_slurm import FakeSlurm
from ..hpc.jobspec import parse_time_limit
from ..llm.server import VLLMServer
from ..service import check_edit, create_run, kick, load_app_config, queue_action, run_status, stop_run
from ..service import list_runs as list_all_runs

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Research Tool: a resumable, human-gated multi-agent research pipeline.",
)
console = Console(soft_wrap=True)
err_console = Console(stderr=True, soft_wrap=True)

_OUTCOME_STYLE = {
    Outcome.DONE: "green",
    Outcome.WAITING_GATE: "yellow",
    Outcome.WAITING_JOB: "cyan",
    Outcome.CONTINUE: "cyan",
    Outcome.NEEDS_HUMAN: "red",
    Outcome.NEEDS_LLM: "cyan",
    Outcome.BUSY: "magenta",
}
_STATUS_STYLE = {
    StageStatus.DONE: "green",
    StageStatus.RUNNING: "cyan",
    StageStatus.SUSPENDED: "cyan",
    StageStatus.AWAITING_GATE: "yellow",
    StageStatus.STALE: "magenta",
    StageStatus.FAILED: "red",
    StageStatus.PENDING: "dim",
}


@contextmanager
def _guard() -> Iterator[None]:
    try:
        yield
    except RToolError as exc:
        err_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from None


def _app_config(config: list[Path], sets: list[str], runs_root: Path | None) -> AppConfig:
    return load_app_config(config, sets, runs_root=runs_root)


def _open_run(
    ref: str, runs_root: Path | None, config: list[Path] | None = None, sets: list[str] | None = None
) -> RunHandle:
    """Open a run; ``config``/``sets`` override its config snapshot for this invocation only."""
    root = runs_root or load_config(*default_config_files()).paths.resolved_runs_root()
    run = RunHandle(resolve_run_path(ref, root))
    overrides: dict[str, Any] = {}
    for path in config or []:
        overrides = deep_merge(overrides, load_yaml_mapping(path))
    if sets:
        overrides = deep_merge(overrides, parse_set_overrides(sets))
    if overrides:
        run.config = apply_overrides(run.config, overrides)
    return run


def _ref(run: RunHandle) -> str:
    """Shortest way to refer to the run in printed hints (relative to cwd when possible)."""
    try:
        return str(run.root.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(run.root)


@contextmanager
def _llm_endpoint(cfg: AppConfig, log_dir: Path, serve: bool) -> Iterator[LLMEndpoint | None]:
    """Start a vLLM server on this node for the duration of the block (if asked)."""
    if not serve:
        yield None
        return
    console.print("[dim]starting vLLM on this node (the first start can take several minutes)...[/dim]")
    with VLLMServer(cfg.llm, log_dir).running() as handle:
        console.print(f"[dim]vLLM ready at {handle.url} (log: {handle.log_path})[/dim]")
        yield LLMEndpoint(url=handle.url, api_key=handle.api_key)


def _advance(
    run: RunHandle, *, max_minutes: float | None = None, simulate_jobs: bool = False, serve_llm: bool = False
) -> AdvanceResult:
    if serve_llm and run.config.llm.backend != "vllm":
        raise RToolError("--serve-llm needs llm.backend=vllm (add --set llm.backend=vllm)")
    with _llm_endpoint(run.config, run.jobs_dir / "llm", serve_llm) as endpoint:
        services = build_services(run, llm_endpoint=endpoint)
        max_seconds = (max_minutes or run.config.episode.max_minutes) * 60
        result = AdvanceResult(Outcome.NEEDS_HUMAN)
        for _ in range(50):
            result = Engine(run, services).advance(max_seconds=max_seconds)
            if simulate_jobs and result.outcome is Outcome.WAITING_JOB and isinstance(services.slurm, FakeSlurm):
                console.print(f"[dim]fake Slurm: finishing jobs {', '.join(result.waiting_jobs)}[/dim]")
                services.slurm.finish_all()
                continue
            break
        return result


def _print_gate(run: RunHandle, gate: dict[str, Any], several: bool) -> None:
    summary = gate.get("summary") or {}
    where = f"stage {gate['stage']}" + (", in-stage" if gate.get("in_stage") else "")
    console.print(f"\n[bold yellow]Gate {gate['gate']}[/bold yellow] ({where})")
    for key in ("prompt", "message"):
        if summary.get(key):
            console.print(f"  {escape(str(summary[key]))}")
    for path in summary.get("review_files", []):
        console.print(f"  read:    {path}")
    for name, version in (summary.get("artifacts") or {}).items():
        console.print(f"  review:  rtool show {_ref(run)} {name} --version {version}")
    for key, value in summary.items():
        if key not in {"prompt", "message", "artifacts", "options", "requests", "pending_revisions", "review_files"}:
            console.print(f"  {key}: {escape(str(value))}")
    for request in summary.get("requests", []):
        route = f"{request['from_stage']} -> {request['target_stage']}"
        console.print(f"  request {route}: {escape(request['reason'])}")
    flag = f" --gate {gate['gate']}" if several else ""
    ref = _ref(run)
    if summary.get("options"):
        for option in summary["options"]:
            console.print(f"  \\[{option['index']}] {escape(option['title'])}")
        console.print(f"  choose:  rtool select {ref} <index>{flag}")
    else:
        console.print(f'  decide:  rtool approve {ref}{flag}   |   rtool reject {ref}{flag} -f "why"')


def _print_result(run: RunHandle, result: AdvanceResult) -> None:
    style = _OUTCOME_STYLE.get(result.outcome, "white")
    console.print(f"[{style}]{result.outcome.value}[/{style}]: {escape(result.message)}")
    if result.ran:
        console.print(f"ran: {', '.join(result.ran)}")
    for failure in result.failed:
        console.print(f"[red]{failure['stage']} failed:[/red] {escape(str(failure['error']))}")
    for gate in result.open_gates:
        _print_gate(run, gate, several=len(result.open_gates) > 1)


def _kick(run: RunHandle, reason: str, *, needs_llm: bool = False) -> None:
    """On a real cluster, make sure an episode job will pick up queued work."""
    result = kick(run, reason, needs_llm=needs_llm)
    console.print(result.detail + (f"; follow it with: rtool status {_ref(run)}" if result.submitted else ""))


def _queue(run: RunHandle, action: Action, *, now: bool, simulate_jobs: bool) -> None:
    real = run.config.slurm.backend == "real"
    result = queue_action(run, action, wake=real)
    console.print(f"queued [bold]{action.kind.value}[/bold]")
    if result is not None:
        console.print(result.detail)
    elif now:
        _print_result(run, _advance(run, simulate_jobs=simulate_jobs))
    else:
        console.print(f"it will be applied by the next episode: rtool run {_ref(run)}")


# ---------------------------------------------------------------- commands
def _version_callback(value: bool) -> None:
    if value:
        console.print(f"rtool {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show version."),
) -> None:
    """Research Tool: a resumable, human-gated multi-agent research pipeline."""


@app.command()
def new(
    project: str = typer.Argument(..., help="Project name; runs are grouped under it."),
    topic: str = typer.Option(..., "--topic", "-t", help="The research topic or question."),
    pipeline: str = typer.Option("research", "--pipeline", "-p", help="Built-in pipeline name or a YAML path."),
    question: list[str] = typer.Option([], "--question", "-q", help="A specific research question (repeatable)."),
    constraint: list[str] = typer.Option([], "--constraint", help="A constraint, e.g. 'max 24 GPU-hours'."),
    seed_query: list[str] = typer.Option([], "--seed-query", help="A search query to start from (repeatable)."),
    pack: str = typer.Option("generic", "--pack", help="Domain pack."),
    config: list[Path] = typer.Option([], "--config", "-c", help="Extra config YAML (repeatable)."),
    set_: list[str] = typer.Option([], "--set", help="Override config, e.g. --set gates.default=auto."),
    runs_root: Path | None = typer.Option(None, "--runs-root", help="Where run directories live."),
    run_id: str | None = typer.Option(None, "--run-id", help="Explicit run id (default: timestamp)."),
) -> None:
    """Create a run from a pipeline and a research brief."""
    with _guard():
        cfg = _app_config(config, set_, runs_root)
        run = create_run(
            cfg,
            project,
            topic,
            pipeline=pipeline,
            questions=question,
            constraints=constraint,
            seed_queries=seed_query,
            pack=pack,
            run_id=run_id,
        )
        console.print(f"created run [bold]{run.run_id}[/bold] at {run.root}")
        console.print(f"next: rtool run {_ref(run)}")


@app.command("run")
def run_command(
    run_ref: str = typer.Argument(..., help="Run directory, project/run_id, or run_id."),
    max_minutes: float | None = typer.Option(None, "--max-minutes", help="Episode time budget."),
    simulate_jobs: bool = typer.Option(
        False, "--simulate-jobs", help="Fake Slurm only: finish jobs immediately and keep going."
    ),
    serve_llm: bool = typer.Option(False, "--serve-llm", help="Start vLLM on this (GPU) node for this episode."),
    inline: bool = typer.Option(False, "--inline", help="Run in this process even when Slurm is real."),
    config: list[Path] = typer.Option([], "--config", "-c", help="Config overrides for this invocation."),
    set_: list[str] = typer.Option([], "--set", help="Override a config value for this invocation."),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Advance the run: submit an episode job (real Slurm) or run one episode here."""
    with _guard():
        run = _open_run(run_ref, runs_root, config, set_)
        if run.config.slurm.backend == "real" and not inline:
            _kick(run, "rtool run", needs_llm=llm_needed(run))
            return
        load_secrets()  # the episode runs here, so the agents need the API keys here
        _print_result(run, _advance(run, max_minutes=max_minutes, simulate_jobs=simulate_jobs, serve_llm=serve_llm))


@app.command(hidden=True)
def episode(
    run_dir: str = typer.Argument(..., help="Run directory."),
    config: list[Path] = typer.Option([], "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
) -> None:
    """Body of an episode job (submitted by rtool itself, not meant for interactive use)."""
    with _guard():
        run = _open_run(run_dir, None, config, set_)
        slurm = build_slurm(run)
        self_job = os.environ.get("SLURM_JOB_ID")
        try:
            load_secrets()  # jobs never inherit API keys (sbatch gets a scrubbed environment)
        except RToolError as exc:
            EventLog(run.events_path, run.run_id).emit("episode_not_started", job=self_job, error=str(exc))
            raise

        def time_limit_warning(_signum: int, _frame: object) -> None:
            # Slurm warns before the time limit: queue the continuation to start after this job.
            try:
                submit_episode(
                    run,
                    slurm,
                    needs_llm=llm_needed(run),
                    reason="time_limit",
                    dependency=[self_job] if self_job else [],
                    self_job_id=self_job,
                )
            except RToolError as exc:
                err_console.print(f"[red]could not queue the continuation:[/red] {escape(str(exc))}")

        def terminated(_signum: int, _frame: object) -> None:
            # Preemption or scancel. Unwind now so the vLLM server stops and the lease is released;
            # a preempted job is requeued with the same id and resumes the run where it stopped.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            raise SystemExit(143)

        signal.signal(signal.SIGUSR1, time_limit_warning)
        signal.signal(signal.SIGTERM, terminated)
        budget = parse_time_limit(run.config.slurm.episode.time_limit) - SAFETY_MARGIN_S
        result = run_episode(
            run, slurm=slurm, max_seconds=max(60.0, budget), episode_id=f"ep-{self_job}" if self_job else None
        )
        _print_result(run, result)
        next_job = schedule_next(run, slurm, result, self_job_id=self_job)
        if next_job is None and ActionInbox(run.actions_dir).pending():
            # decisions that arrived after this episode last looked would otherwise wait for the next kick
            next_job = submit_episode(run, slurm, needs_llm=False, reason="pending_actions", self_job_id=self_job)
        if next_job:
            console.print(f"next episode: job {next_job}")


@app.command()
def status(
    run_ref: str = typer.Argument(...),
    events: int = typer.Option(5, "--events", help="How many recent events to show."),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Show stage states, open gates and recent events."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        run_state, stages, pipeline = load_status(run)
        table = Table(title=f"{run.spec.project}/{run.run_id} · pipeline '{run.spec.pipeline.name}'")
        for column in ("stage", "agent", "status", "try", "outputs", "note"):
            table.add_column(column)
        open_gates = [gate.model_dump() for gate in run_status(run).open_gates]
        for sid in pipeline.order:
            st = stages[sid]
            note = ""
            if st.status is StageStatus.AWAITING_GATE and st.gate:
                note = f"gate {st.gate.gate_id}"
            elif st.status is StageStatus.SUSPENDED:
                note = "jobs " + ", ".join(st.waiting_on)
            elif st.status is StageStatus.FAILED:
                note = (st.last_error or "")[:90]
            elif st.status is StageStatus.STALE:
                note = st.stale_reason or ""
            style = _STATUS_STYLE.get(st.status, "white")
            outputs = ", ".join(f"{name} v{v}" for name, v in sorted(st.output_versions.items()))
            table.add_row(
                sid,
                pipeline.stage(sid).spec.agent,
                f"[{style}]{st.status.value}[/{style}]",
                str(st.attempt),
                outputs,
                escape(note),
            )
        console.print(table)
        if run_state is not None:
            usage = run_state.usage
            console.print(
                f"last outcome: {run_state.last_outcome} · episodes: {run_state.episodes} · "
                f"llm calls: {int(usage.get('llm_calls', 0))} · "
                f"tokens: {int(usage.get('prompt_tokens', 0))} in / {int(usage.get('completion_tokens', 0))} out"
            )
        queued = ActionInbox(run.actions_dir).pending()
        if queued:
            console.print(f"[yellow]{len(queued)} queued action(s) will apply on the next `rtool run`[/yellow]")
        for gate in open_gates:
            _print_gate(run, gate, several=len(open_gates) > 1)
        if events:
            console.print("\nrecent events:")
            for event in EventLog(run.events_path, run.run_id).read(last=events):
                stage = f" \\[{event['stage']}]" if event.get("stage") else ""
                data = escape(json.dumps(event.get("data", {}))[:120])
                console.print(f"  {event['ts']} {event['kind']}{stage} {data}")


@app.command()
def approve(
    run_ref: str = typer.Argument(...),
    gate: str | None = typer.Option(None, "--gate", help="Gate id, if several are open."),
    stage: str | None = typer.Option(None, "--stage"),
    note: str = typer.Option("", "--note", "-n", help="Optional comment, kept in the audit log."),
    now: bool = typer.Option(False, "--now", help="Run the next episode immediately."),
    simulate_jobs: bool = typer.Option(False, "--simulate-jobs"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Approve the open gate."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        action = Action(kind=ActionKind.APPROVE, gate=gate, stage=stage, payload={"note": note})
        _queue(run, action, now=now, simulate_jobs=simulate_jobs)


@app.command()
def reject(
    run_ref: str = typer.Argument(...),
    feedback: str = typer.Option(..., "--feedback", "-f", help="What should change; the stage re-runs with it."),
    gate: str | None = typer.Option(None, "--gate"),
    stage: str | None = typer.Option(None, "--stage"),
    now: bool = typer.Option(False, "--now"),
    simulate_jobs: bool = typer.Option(False, "--simulate-jobs"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Reject the open gate; the stage re-runs with your feedback."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        action = Action(kind=ActionKind.REJECT, gate=gate, stage=stage, payload={"feedback": feedback})
        _queue(run, action, now=now, simulate_jobs=simulate_jobs)


@app.command()
def select(
    run_ref: str = typer.Argument(...),
    index: int = typer.Argument(..., help="Option number shown at the gate."),
    edit: Path | None = typer.Option(None, "--edit", help="YAML/JSON file with fields to override in the choice."),
    note: str = typer.Option("", "--note", "-n"),
    gate: str | None = typer.Option(None, "--gate"),
    stage: str | None = typer.Option(None, "--stage"),
    now: bool = typer.Option(False, "--now"),
    simulate_jobs: bool = typer.Option(False, "--simulate-jobs"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Choose an option at a selection gate (e.g. one of the three hypotheses)."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        payload: dict[str, Any] = {"index": index, "note": note}
        if edit is not None:
            changes = yaml.safe_load(edit.read_text(encoding="utf-8"))
            if not isinstance(changes, dict):
                raise RToolError(f"{edit} must contain a mapping of fields to override")
            payload["edit"] = changes
        action = Action(kind=ActionKind.SELECT, gate=gate, stage=stage, payload=payload)
        _queue(run, action, now=now, simulate_jobs=simulate_jobs)


@app.command()
def retry(
    run_ref: str = typer.Argument(...),
    stage: str | None = typer.Option(None, "--stage"),
    resume: bool = typer.Option(False, "--resume", help="Continue from the last checkpoint instead of restarting."),
    note: str = typer.Option("", "--note", "-n", help="Guidance passed to the stage's next attempt."),
    now: bool = typer.Option(False, "--now"),
    simulate_jobs: bool = typer.Option(False, "--simulate-jobs"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Re-run a failed stage."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        action = Action(kind=ActionKind.RETRY, stage=stage, payload={"resume": resume, "note": note})
        _queue(run, action, now=now, simulate_jobs=simulate_jobs)


@app.command()
def note(
    run_ref: str = typer.Argument(...),
    stage: str = typer.Option(..., "--stage", help="Stage that should see the note on its next attempt."),
    text: str = typer.Argument(..., help="The guidance."),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Attach guidance to a stage's next attempt."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        _queue(run, Action(kind=ActionKind.NOTE, stage=stage, payload={"text": text}), now=False, simulate_jobs=False)


@app.command()
def edit(
    run_ref: str = typer.Argument(...),
    artifact: str = typer.Argument(..., help="The output to replace, e.g. experiment_plan."),
    source: Path = typer.Option(
        ..., "--from", help="JSON or YAML file with the whole edited value (start from `rtool show <run> <artifact>`)."
    ),
    note: str = typer.Option("", "--note", "-n", help="Why you changed it (kept with the new version)."),
    now: bool = typer.Option(False, "--now"),
    simulate_jobs: bool = typer.Option(False, "--simulate-jobs"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Replace an output (e.g. the experiment plan) with your edited version.

    The owning agent recomputes what follows from your changes and lists remaining problems;
    later stages that used the old version run again.
    """
    with _guard():
        run = _open_run(run_ref, runs_root)
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        base = ArtifactStore(run.artifacts_dir).head(artifact)
        checked = check_edit(run, artifact, base, value)  # refuse a bad edit now, not in the next episode
        if not checked.changed:
            raise RToolError(f"{source} is the same as {artifact} v{base}; nothing to change")
        console.print(f"changes: {', '.join(checked.changed)}")
        payload = {"artifact": artifact, "base_version": base, "value": value, "note": note}
        action = Action(kind=ActionKind.EDIT, stage=checked.stage, payload=payload)
        _queue(run, action, now=now, simulate_jobs=simulate_jobs)


@app.command()
def reopen(
    run_ref: str = typer.Argument(...),
    stage: str = typer.Option(..., "--stage", help="A finished stage with a checkpoint, e.g. hypotheses."),
    now: bool = typer.Option(False, "--now"),
    simulate_jobs: bool = typer.Option(False, "--simulate-jobs"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Open a finished stage's checkpoint again (choose another hypothesis, add papers, ...).

    Later stages keep their results unless the decision changes this stage's outputs.
    """
    with _guard():
        run = _open_run(run_ref, runs_root)
        _queue(run, Action(kind=ActionKind.REOPEN, stage=stage), now=now, simulate_jobs=simulate_jobs)


@app.command("list")
def list_command(
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """List every run, newest first, with the stage it is at."""
    with _guard():
        root = runs_root or load_config(*default_config_files()).paths.resolved_runs_root()
        table = Table(title=f"runs in {root}")
        for column in ("project/run", "topic", "pipeline", "status", "waiting for you"):
            table.add_column(column)
        for summary in list_all_runs(root):
            if summary.error:
                table.add_row(f"{summary.project}/{summary.run_id}", "", "", "[red]unreadable[/red]", summary.error)
                continue
            current = next((s for s in summary.stages if s.status is not StageStatus.DONE), None)
            status = "done" if current is None else f"{current.id}: {current.status.value}"
            gates = ", ".join(f"{g.gate} ({g.stage})" for g in summary.open_gates)
            table.add_row(
                f"{summary.project}/{summary.run_id}", escape(summary.topic[:60]), summary.pipeline, status, gates
            )
        console.print(table)


@app.command()
def stop(
    run_ref: str = typer.Argument(...),
    experiments: bool = typer.Option(
        False, "--experiments", help="Also cancel running experiment jobs (the Coder reports what finished)."
    ),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Cancel the run's episode job; resume later with `rtool run`."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        result = stop_run(run, cancel_experiments=experiments)
        jobs = ", ".join(result.cancelled) or "none were queued or running"
        console.print(f"cancelled jobs: {jobs}. Resume with: rtool run {_ref(run)}")


@app.command()
def ui(
    mode: str = typer.Option(
        "local",
        "--mode",
        help="local: this machine, with simulated papers, model and cluster (demos, development). "
        "login: on a cluster login node, only for as long as your ssh session lasts.",
    ),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    port: int = typer.Option(8765, "--port"),
    token_file: Path | None = typer.Option(
        None, "--token-file", help="Also write the access token to this file (readable only by you)."
    ),
    demo: bool = typer.Option(False, "--demo", help="local mode: create the demo runs first if there are no runs."),
    open_browser: bool = typer.Option(False, "--open", help="Open the UI in the browser."),
    config: list[Path] = typer.Option([], "--config", "-c"),
) -> None:
    """Start the web UI (needs the `web` extra). It listens on 127.0.0.1 only; reach it through an ssh tunnel."""
    with _guard():
        if mode not in ("local", "login"):
            raise RToolError("--mode must be local or login")
        try:
            from ..web.server import new_token, serve, write_private
            from ..web.settings import UISettings
        except ImportError as exc:
            raise RToolError(f"the web UI needs the web extra: pip install -e '.[web]' ({exc})") from None
        load_secrets()
        root = (runs_root or load_app_config(config).paths.resolved_runs_root()).expanduser()
        if mode == "local" and demo and not list_all_runs(root):
            from ..testing.demo_run import build_demo_runs

            console.print("building the demo runs (about ten seconds)...")
            build_demo_runs(root)
        token = new_token()
        if token_file is not None:
            write_private(token_file.expanduser(), token + "\n")
        settings = UISettings(runs_root=root, token=token, mode=mode, config_files=list(config))  # type: ignore[arg-type]
        serve(settings, host="127.0.0.1", port=port, open_browser=open_browser)


@app.command()
def show(
    run_ref: str = typer.Argument(...),
    name: str | None = typer.Argument(None, help="Artifact name; omit to list all artifacts."),
    version: int | None = typer.Option(None, "--version", "-v"),
    meta: bool = typer.Option(False, "--meta", help="Show the envelope (producer, inputs, hash) too."),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """List artifacts, or print one artifact's content."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        store = ArtifactStore(run.artifacts_dir)
        if name is None:
            table = Table(title="artifacts")
            for column in ("name", "versions", "latest by", "created"):
                table.add_column(column)
            for artifact in store.names():
                env = store.envelope(artifact)
                versions = store.versions(artifact)
                table.add_row(
                    artifact,
                    f"v{versions[0]}..v{versions[-1]}" if len(versions) > 1 else f"v{versions[0]}",
                    f"{env.producer.stage} ({env.producer.agent})",
                    env.created_at,
                )
            console.print(table)
            return
        env = store.envelope(name, version)
        data = env.model_dump(mode="json") if meta else env.payload
        console.print_json(json.dumps(data))


@app.command()
def events(
    run_ref: str = typer.Argument(...),
    last: int = typer.Option(30, "--last", "-n"),
    kind: list[str] = typer.Option([], "--kind", "-k", help="Only these event kinds (repeatable)."),
    stage: str | None = typer.Option(None, "--stage"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Print the run's event log."""
    with _guard():
        run = _open_run(run_ref, runs_root)
        for event in EventLog(run.events_path, run.run_id).read(kinds=kind or None, stage=stage, last=last):
            console.print_json(json.dumps(event))


@app.command()
def agents() -> None:
    """List registered agents and their artifact contracts."""
    table = Table(title="agents")
    for column in ("name", "consumes", "produces", "llm", "description"):
        table.add_column(column)
    for name, cls in registry.items():
        table.add_row(
            name,
            ", ".join(cls.consumes + tuple(f"{o}?" for o in cls.optional_inputs)),
            ", ".join(cls.produces),
            "yes" if cls.needs_llm else "no",
            cls.description,
        )
    console.print(table)


@app.command()
def pipelines(
    check: str | None = typer.Option(None, "--check", help="Validate a pipeline (name or path) and show its order."),
) -> None:
    """List built-in pipelines, or validate one."""
    with _guard():
        if check is None:
            for name in builtin_pipelines():
                console.print(name)
            return
        resolved = resolve_pipeline(find_pipeline(check))
        table = Table(title=f"pipeline '{resolved.spec.name}' (execution order)")
        for column in ("#", "stage", "agent", "after", "gate", "produces"):
            table.add_column(column)
        for i, sid in enumerate(resolved.order, 1):
            rs = resolved.stage(sid)
            table.add_row(str(i), sid, rs.spec.agent, ", ".join(rs.deps), rs.spec.gate or "", ", ".join(rs.produces))
        console.print(table)


def _ints(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise RToolError(f"expected comma-separated integers, got {text!r}") from None


def _print_bench(report: dict[str, Any]) -> None:
    console.print(f"server version: {report.get('server_version')} · healthy: {report.get('healthy')}")
    for key, label in (
        ("structured_reasoning_off", "JSON, reasoning off"),
        ("structured_reasoning_on", "JSON, reasoning on"),
    ):
        if key not in report:
            continue
        r = report[key]
        n = r.get("n")
        console.print(
            f"{label}: first try {r.get('first_try_valid')}/{n}, after repair {r.get('valid_after_repair')}/{n}"
        )
        for bad in r.get("invalid_samples", []):
            console.print(
                f"  [dim]invalid {bad['schema']}: finish={bad['finish_reason']} "
                f"tokens={bad['completion_tokens']} {escape(bad['error'])}[/dim]"
            )
    if "reasoning" in report:
        reasoning = report["reasoning"]
        console.print(
            f"reasoning returned separately: {reasoning.get('reasoning_returned_separately')} · "
            f"tags leaked: {reasoning.get('think_tags_leaked_into_answer')}"
        )
    for case in report.get("long_context", []):
        console.print(
            f"recall at ~{case['target_tokens']} tokens: {'found' if case.get('found') else 'MISSED'} "
            f"{case.get('latency_s', '')}{' ' + case['error'] if case.get('error') else ''}"
        )
    for t in report.get("throughput", []):
        console.print(f"concurrency {t['concurrency']}: {t['tokens_per_s']} tokens/s ({t['errors']} errors)")
    for tip in report.get("recommendations", []):
        console.print(f"[yellow]→ {escape(tip)}[/yellow]")


@app.command("llm-bench")
def llm_bench(
    base_url: str | None = typer.Option(None, "--base-url", help="Server to test, e.g. http://127.0.0.1:8000/v1"),
    serve: bool = typer.Option(False, "--serve", help="Start vLLM on this node from the config first."),
    out: Path = typer.Option(Path("llm_profile.json"), "--out", "-o", help="Where to write the JSON profile."),
    structured: int = typer.Option(30, "--structured", help="Number of structured-output prompts."),
    reasoning: int | None = typer.Option(
        None, "--reasoning", help="Structured prompts with reasoning on (slow; default a third, 0 skips)."
    ),
    contexts: str = typer.Option("8000,32000,60000", "--contexts", help="Long-context sizes in tokens."),
    concurrency: str = typer.Option("1,8,32", "--concurrency", help="Concurrency levels for throughput."),
    config: list[Path] = typer.Option([], "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
) -> None:
    """Measure the served model on what the agents rely on and write a JSON profile."""
    from ..llm.bench import run_bench

    with _guard():
        cfg = _app_config(config, set_, None)
        with _llm_endpoint(cfg, out.parent / "llm-logs", serve) as served:
            endpoint = served
            if endpoint is None:
                url = base_url or cfg.llm.base_url
                if not url:
                    raise RToolError("give --base-url, set llm.base_url, or use --serve")
                endpoint = LLMEndpoint(url=url, api_key=os.environ.get(cfg.llm.api_key_env))
            report = run_bench(
                build_llm(cfg, endpoint),
                structured_n=structured,
                reasoning_n=reasoning,
                context_sizes=_ints(contexts),
                concurrency_levels=_ints(concurrency),
                progress=lambda message: console.print(f"[dim]• {message}[/dim]"),
            )
        atomic_write_json(out, report)
        _print_bench(report)
        console.print(f"profile written to {out}")


@app.command()
def doctor(
    network: bool = typer.Option(True, "--network/--no-network", help="Also test reaching the paper APIs."),
    config: list[Path] = typer.Option([], "--config", "-c"),
    set_: list[str] = typer.Option([], "--set"),
) -> None:
    """Check that this machine can run the pipeline with the current config."""
    from ..hpc.doctor import run_doctor

    with _guard():
        cfg = _app_config(config, set_, None)
        checks = run_doctor(cfg, network=network)
        table = Table(title="rtool doctor")
        for column in ("check", "status", "detail"):
            table.add_column(column)
        style = {"ok": "green", "warn": "yellow", "fail": "red", "skip": "dim"}
        for check in checks:
            table.add_row(
                check.name, f"[{style[check.status]}]{check.status}[/{style[check.status]}]", escape(check.detail)
            )
        console.print(table)
        if any(check.status == "fail" for check in checks):
            raise typer.Exit(code=1)


@app.command()
def archive(
    run_ref: str = typer.Argument(...),
    to: Path | None = typer.Option(None, "--to", help="Folder for the archive (default: paths.archive_root)."),
    max_file_mb: float = typer.Option(200, "--max-file-mb", help="Leave out single files larger than this."),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Pack the run's results, reports, code, state and logs into one .tar.gz (run it on a login node)."""
    from ..core.archive import archive_run

    with _guard():
        run = _open_run(run_ref, runs_root)
        target = to or run.config.paths.archive_root
        if target is None:
            raise RToolError("say where to put it: --to DIR, or set paths.archive_root")
        result = archive_run(
            run.root,
            Path(os.path.expandvars(str(target))).expanduser(),
            name=f"{run.spec.project}-{run.run_id}",
            max_file_bytes=int(max_file_mb * 1e6),
        )
        console.print(f"archived {result.files} files ({result.bytes / 1e6:.1f} MB) to {result.path}")
        for skipped in result.skipped:
            console.print(
                f"[yellow]left out (too large):[/yellow] {skipped['path']} ({int(skipped['bytes']) / 1e6:.0f} MB)"
            )


@app.command("add-paper")
def add_paper(
    run_ref: str = typer.Argument(...),
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="PDF or HTML you obtained yourself."),
    title: str = typer.Option(..., "--title", help="Paper title."),
    doi: str | None = typer.Option(None, "--doi"),
    arxiv_id: str | None = typer.Option(None, "--arxiv"),
    paper_id: str | None = typer.Option(None, "--paper-id", help="Corpus id, if it is already listed (needs_manual)."),
    rerun: bool = typer.Option(False, "--rerun", help="Reject the open literature gate so the stage ingests it now."),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
) -> None:
    """Give the Literature agent a paper it could not download (e.g. paywalled, via your library)."""
    import shutil

    with _guard():
        run = _open_run(run_ref, runs_root)
        stages = [s.id for s in run.spec.pipeline.stages if s.agent == "literature"]
        if not stages:
            raise RToolError("this run's pipeline has no literature stage")
        if file.suffix.lower() not in (".pdf", ".html", ".htm"):
            raise RToolError("only .pdf or .html files can be added")
        inbox = run.work_dir(stages[0]) / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        dest = inbox / file.name
        shutil.copy2(file, dest)
        sidecar = {"title": title, "doi": doi, "arxiv_id": arxiv_id, "paper_id": paper_id}
        atomic_write_json(dest.with_suffix(".json"), {k: v for k, v in sidecar.items() if v})
        console.print(f"added {file.name} to {inbox}")
        if rerun:
            action = Action(
                kind=ActionKind.REJECT, stage=stages[0], payload={"feedback": f"Added a paper by hand: {title}"}
            )
            _queue(run, action, now=False, simulate_jobs=False)
        else:
            console.print("it is ingested the next time the literature stage runs (or pass --rerun)")
