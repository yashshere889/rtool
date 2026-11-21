"""Result checks of the md pack (listed in pack.yaml)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ...schemas.code import RunRecord
from ...schemas.plan import ExperimentPlan

DRIFT = re.compile(r"energy_drift")
TOLERANCE = 0.01  # relative drift of the total energy over a constant-energy run


def energy_drift_issues(plan: ExperimentPlan, records: Sequence[RunRecord]) -> list[str]:
    """A constant-energy check run whose total energy drifts: the time step or constraints are wrong."""
    ok = [r for r in records if r.status == "ok"]
    names = sorted({name for r in ok for name in r.metrics if DRIFT.search(name)})
    issues = []
    for name in names:
        worst = max(abs(r.metrics[name]) for r in ok if name in r.metrics)
        if worst > TOLERANCE:
            issues.append(
                f"`{name}` reaches {worst:.3g} (above {TOLERANCE:g}): the total energy is not conserved, so the "
                "time step, constraints or integrator are wrong and the trajectories cannot be trusted"
            )
    return issues
