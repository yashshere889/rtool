"""What the CLI and the web UI do with runs: create, list, inspect, decide, wake up, stop.

Nothing here prints. Every decision still goes through the action inbox and
is applied by the next episode under the run lease; this package only writes
action files and submits (or cancels) Slurm jobs.
"""

from .runs import (
    Affected,
    EditCheck,
    KickResult,
    OpenGate,
    RunStatus,
    RunSummary,
    StageSummary,
    StopResult,
    check_edit,
    create_run,
    downstream_impact,
    kick,
    list_runs,
    load_app_config,
    open_run,
    queue_action,
    run_status,
    stop_run,
)

__all__ = [
    "Affected",
    "EditCheck",
    "KickResult",
    "OpenGate",
    "RunStatus",
    "RunSummary",
    "StageSummary",
    "StopResult",
    "check_edit",
    "create_run",
    "downstream_impact",
    "kick",
    "list_runs",
    "load_app_config",
    "open_run",
    "queue_action",
    "run_status",
    "stop_run",
]
