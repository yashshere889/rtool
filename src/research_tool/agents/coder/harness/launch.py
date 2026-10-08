"""Run one entry of a batch file (written by rtool; used by the Slurm array job).

    python launch.py --batch batches/full-1.json --task 3

The batch file lists the runs: {"runs": [{"condition", "seed", "output", "workdir"}, ...]};
--task picks one (the array task id). Runs on Python 3.10 (the experiment image).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one entry of a batch file.")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    batch = json.loads((HERE / args.batch).read_text(encoding="utf-8"))
    entry = batch["runs"][args.task]
    print(f"rtool-run: task {args.task}: condition={entry['condition']} seed={entry['seed']}", flush=True)
    command = [
        sys.executable,
        str(HERE / "run.py"),
        "--condition",
        entry["condition"],
        "--seed",
        str(entry["seed"]),
        "--output",
        entry["output"],
        "--workdir",
        entry["workdir"],
    ]
    return subprocess.call(command, cwd=HERE)


if __name__ == "__main__":
    sys.exit(main())
