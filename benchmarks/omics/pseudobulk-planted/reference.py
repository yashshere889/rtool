"""Reference for the omics cases: the donor-level analysis, written by hand (standard library only).

    python reference.py '{"donors": 6, "cells": 100, "genes": 40, "fold": 2.0, "donor_sd": 0.3, "seed": 1}'

Sums each donor's cells into a pseudobulk profile, takes log2(CPM + 1) of IFI27, and compares the two
conditions with an exact permutation test over the donors. For contrast it also prints what treating
cells as independent would claim.
"""

import csv
import itertools
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def donor_values(folder):
    sums, totals, condition = {}, {}, {}
    cell_values = {"control": [], "disease": []}
    with open(Path(folder) / "cells.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            counts = [int(v) for k, v in row.items() if k not in ("cell", "donor", "condition", "batch")]
            donor = row["donor"]
            sums[donor] = sums.get(donor, 0) + int(row["IFI27"])
            totals[donor] = totals.get(donor, 0) + sum(counts)
            condition[donor] = row["condition"]
            cell_values[row["condition"]].append(math.log2(1e6 * int(row["IFI27"]) / sum(counts) + 1))
    values = {d: math.log2(1e6 * sums[d] / totals[d] + 1) for d in sums}
    return values, condition, cell_values


def permutation_p(a, b):
    observed = abs(statistics.mean(a) - statistics.mean(b))
    pooled, extreme, total = a + b, 0, 0
    for chosen in itertools.combinations(range(len(pooled)), len(a)):
        x = [pooled[i] for i in chosen]
        y = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        extreme += abs(statistics.mean(x) - statistics.mean(y)) >= observed - 1e-12
        total += 1
    return extreme / total


if __name__ == "__main__":
    args = (
        json.loads(sys.argv[1])
        if len(sys.argv) > 1
        else {"donors": 6, "cells": 100, "genes": 40, "fold": 2.0, "donor_sd": 0.3, "seed": 1}
    )  # noqa: E501
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([sys.executable, "-I", str(HERE / "generate.py"), folder, json.dumps(args)], check=True)
        values, condition, cells = donor_values(folder)
    control = [v for d, v in values.items() if condition[d] == "control"]
    disease = [v for d, v in values.items() if condition[d] == "disease"]
    diff = statistics.mean(disease) - statistics.mean(control)
    print(
        f"{args}: donor-level difference {diff:.3f} (log2), exact permutation p {permutation_p(disease, control):.4f}"
    )
    z = (statistics.mean(cells["disease"]) - statistics.mean(cells["control"])) / math.sqrt(
        statistics.variance(cells["disease"]) / len(cells["disease"])
        + statistics.variance(cells["control"]) / len(cells["control"])
    )
    print(f"treating the {len(cells['control']) + len(cells['disease'])} cells as independent would give z = {z:.1f}")
