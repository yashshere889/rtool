"""Plan checks shared by the packs that analyse data across units (omics, tabular)."""

from __future__ import annotations

import re

from ..schemas.common import ResearchBrief
from ..schemas.plan import ExperimentPlan

# Things that repeat within one donor, patient or sample: not independent units.
REPEATED = {
    "cell", "cells", "barcode", "barcodes", "spot", "spots", "read", "reads", "droplet", "droplets", "nucleus",
    "nuclei", "umi", "umis", "transcript", "transcripts", "visit", "visits", "measurement", "measurements",
    "timepoint", "timepoints", "replicate", "replicates", "well", "wells", "image", "images",
}  # fmt: skip


def pseudoreplication_issues(plan: ExperimentPlan, brief: ResearchBrief | None) -> list[str]:
    """The unit of analysis must be independent: a donor or sample, not its cells, reads or visits."""
    if plan.replicates != "units":
        return []
    words = set(re.findall(r"[a-z]+", plan.unit.lower()))
    if not words:
        return ["name the unit of analysis (for example donor or sample): conditions are compared across units"]
    if words & REPEATED:
        return [
            f"the unit of analysis is '{plan.unit}', which repeats within one donor or sample: such measurements are "
            "not independent, so tests over them give p-values far too small. Compare donors or samples instead, "
            "after aggregating to one value each (for single-cell data, a pseudobulk per donor)"
        ]
    return []
