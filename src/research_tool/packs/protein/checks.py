"""Result checks of the protein pack (listed in pack.yaml)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ...schemas.code import RunRecord
from ...schemas.plan import ExperimentPlan

LEAKAGE = re.compile(r"(in_train|train_test_overlap|test_train_overlap)")


def leakage_issues(plan: ExperimentPlan, records: Sequence[RunRecord]) -> list[str]:
    """Test sequences that also appear in the training set (a reported *_in_train count above 0).

    A model scored on sequences it was trained on looks better than it is; with proteins, near-copies
    (homologues) leak the same way, which is why the hints ask for the closest identity too.
    """
    ok = [r for r in records if r.status == "ok"]
    names = sorted({name for r in ok for name in r.metrics if LEAKAGE.search(name)})
    issues = []
    for name in names:
        worst = max(r.metrics[name] for r in ok if name in r.metrics)
        if worst > 0:
            issues.append(
                f"`{name}` is {worst:g}: test sequences also appear in the training data, so the test accuracy "
                "overstates how well the model generalises (split by sequence identity or protein family)"
            )
    return issues
