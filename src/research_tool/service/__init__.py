"""What the CLI and the web UI do with runs: create, list, inspect, decide, wake up, stop.

Nothing here prints. Every decision still goes through the action inbox and
is applied by the next episode under the run lease; this package only writes
action files and submits (or cancels) Slurm jobs.
"""

from .runs import (
    KickResult,
    OpenGate,
    RunStatus,
    RunSummary,
    StageSummary,
    StopResult,
    create_run,
    kick,
    list_runs,
    load_app_config,
    open_run,
    queue_action,
    run_status,
    stop_run,
)

__all__ = [
    "KickResult",
    "OpenGate",
    "RunStatus",
    "RunSummary",
    "StageSummary",
    "StopResult",
    "create_run",
    "kick",
    "list_runs",
    "load_app_config",
    "open_run",
    "queue_action",
    "run_status",
    "stop_run",
]
