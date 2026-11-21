"""Plan checks of the ml_dl pack, run on every plan it makes (listed in pack.yaml)."""

from __future__ import annotations

from ...schemas.common import ResearchBrief
from ...schemas.plan import ExperimentPlan


def held_out_issues(plan: ExperimentPlan, brief: ResearchBrief | None) -> list[str]:
    """Every dataset is named by its training split only, so nothing is left to evaluate on.

    A plan that leaves the split empty is not flagged: the split may be made in code.
    """
    datasets = [r for r in plan.resources if r.kind == "dataset"]
    if datasets and all((r.split or "").strip().lower() == "train" for r in datasets):
        return [
            "every dataset is listed with only its training split: evaluate on held-out data (name a test or "
            "validation split, or leave split empty and hold data out in code)"
        ]
    return []
