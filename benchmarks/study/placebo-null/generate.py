"""Data for the study cases: a two-arm trial's data.csv, as a researcher would upload it.

    python -I generate.py <output folder> '{"per_arm": 45, "effect_ms": -25, "sd_ms": 40, "seed": 1}'

One row per participant: unit, condition (placebo or caffeine), reaction_time_ms, batch (the test site,
alternating within each arm, so it is not confounded with the condition). Caffeine shifts the mean by
effect_ms.
"""

import csv
import json
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    out, args = Path(sys.argv[1]), json.loads(sys.argv[2])
    rng = random.Random(args["seed"])
    with open(out / "data.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit", "condition", "reaction_time_ms", "batch"])
        number = 0
        for condition, shift in (("placebo", 0.0), ("caffeine", args["effect_ms"])):
            for i in range(args["per_arm"]):
                number += 1
                value = rng.gauss(320 + shift, args["sd_ms"])
                writer.writerow([f"P{number:03d}", condition, round(value, 1), "AB"[i % 2]])
