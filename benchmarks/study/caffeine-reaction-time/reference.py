"""Reference for the study cases: the trial's analysis by hand (standard library only).

    python reference.py '{"per_arm": 45, "effect_ms": -25, "sd_ms": 40, "seed": 1}'

A Welch t-test of caffeine against placebo; with about 88 degrees of freedom, |t| > 1.99 is p < 0.05.
"""

import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"per_arm": 45, "effect_ms": -25, "sd_ms": 40, "seed": 1}
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([sys.executable, "-I", str(HERE / "generate.py"), folder, json.dumps(args)], check=True)
        with open(Path(folder) / "data.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
    arms = {c: [float(r["reaction_time_ms"]) for r in rows if r["condition"] == c] for c in ("placebo", "caffeine")}
    a, b = arms["caffeine"], arms["placebo"]
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    t = (statistics.mean(a) - statistics.mean(b)) / se
    print(f"{args}: difference {statistics.mean(a) - statistics.mean(b):.1f} ms, Welch t {t:.2f}")
