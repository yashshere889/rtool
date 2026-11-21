"""Data for the omics cases: single-cell counts of donors in two conditions, with donor-to-donor variation.

    python -I generate.py <output folder> '{"donors": 6, "cells": 100, "genes": 40, "fold": 2.0,
                                             "donor_sd": 0.3, "seed": 1}'

Writes cells.csv (one row per cell: cell, donor, condition, batch, then one count column per gene) and
donors.csv (donor, condition, batch). IFI27 is `fold` times higher in disease donors. Every gene varies
between donors (log-normal, sd `donor_sd`), so the cells of one donor are not independent of each other.
Batches A and B alternate within each condition, so batch is not confounded with condition.
"""

import csv
import json
import math
import random
import sys
from pathlib import Path

TARGET = "IFI27"


def poisson(lam, rng):
    if lam > 30:  # normal approximation, as Knuth's method gets slow
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


if __name__ == "__main__":
    out, args = Path(sys.argv[1]), json.loads(sys.argv[2])
    rng = random.Random(args["seed"])
    genes = [TARGET] + [f"GENE{i:02d}" for i in range(1, args["genes"])]
    weights = [rng.gammavariate(2.0, 1.0) for _ in genes]
    share = [w / sum(weights) for w in weights]  # each gene's share of a cell's counts
    donors = []
    for condition in ("control", "disease"):
        for i in range(args["donors"]):
            donors.append((f"{condition[0]}{i + 1:02d}", condition, "AB"[i % 2]))
    with open(out / "donors.csv", "w", newline="") as handle:
        csv.writer(handle).writerows([("donor", "condition", "batch"), *donors])
    with open(out / "cells.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "donor", "condition", "batch", *genes])
        for donor, condition, batch in donors:
            level = [math.exp(rng.gauss(0, args["donor_sd"])) for _ in genes]  # this donor's own expression
            if condition == "disease":
                level[0] *= args["fold"]
            for c in range(args["cells"]):
                size = rng.uniform(800, 1200)  # counts captured from this cell
                counts = [poisson(size * s * lv, rng) for s, lv in zip(share, level, strict=True)]
                writer.writerow([f"{donor}_{c:03d}", donor, condition, batch, *counts])
