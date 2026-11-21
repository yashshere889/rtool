"""Result checks of the sysbio pack, run on every finished experiment (listed in pack.yaml)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ...schemas.code import RunRecord
from ...schemas.plan import ExperimentPlan

CONSERVATION = re.compile(r"(mass_balance|balance|conservation)_?(error|drift|violation)")
TOLERANCE = 1e-6  # relative change of a conserved total over a run; good stiff solvers stay far below


def conservation_issues(plan: ExperimentPlan, records: Sequence[RunRecord]) -> list[str]:
    """A simulation that loses or creates what its model conserves (a reported *_balance_error above tolerance).

    That is how a too-large step or a non-stiff solver on a stiff model usually shows; the other results
    of such runs cannot be trusted.
    """
    ok = [r for r in records if r.status == "ok"]
    names = sorted({name for r in ok for name in r.metrics if CONSERVATION.search(name)})
    issues = []
    for name in names:
        worst = max(abs(r.metrics[name]) for r in ok if name in r.metrics)
        if worst > TOLERANCE:
            runs = sorted({f"{r.condition}/seed {r.seed}" for r in ok if abs(r.metrics.get(name, 0.0)) > TOLERANCE})
            issues.append(
                f"`{name}` reaches {worst:.3g} (above {TOLERANCE:g}) in {', '.join(runs[:4])}: the simulation does not "
                "conserve what the model conserves, so its other results are suspect (check the solver, its tolerances "
                "and the step size)"
            )
    return issues
