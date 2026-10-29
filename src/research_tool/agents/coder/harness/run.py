"""Run one condition of the experiment with one seed (written by rtool; do not edit).

    python run.py --condition NAME --seed N --output PATH [--smoke] [--workdir DIR]

The experiment itself is ``experiment.run(...)`` in experiment.py. This wrapper
seeds every random generator, checks that the returned metrics match
config.json, and writes PATH as JSON::

    {"condition": ..., "seed": ..., "metrics": {...}, "seconds": ..., "smoke": ...}

Contract problems are printed as ``rtool-contract: ...`` and exit with code 3.
Runs on Python 3.10 (the experiment image).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def can_write(path: Path) -> bool:
    """Whether path is writable, or could be created (its nearest existing parent is writable)."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            return os.access(candidate, os.W_OK)
    return False


def writable_caches(workdir: Path) -> None:
    """Move Hugging Face's write caches into this run's scratch space when they are read-only.

    The Nemotron image points HF_HOME at /opt/huggingface_cache, which is read-only: models
    with their own code (trust_remote_code, e.g. vLLM loading Nemotron) fail creating
    HF_HOME/modules, and `datasets` fails writing its cache. Models are still read from HF_HOME.
    """
    home = Path(
        os.environ.get("HF_HOME") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    )
    for variable, folder in (("HF_MODULES_CACHE", "modules"), ("HF_DATASETS_CACHE", "datasets")):
        if not can_write(Path(os.environ.get(variable) or home / folder)):
            os.environ[variable] = str(workdir / "hf" / folder)


def as_number(value: object) -> float | None:
    """float(value) if it is a finite number (numpy and torch scalars included), else None."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def check(metrics: object, names: list[str]) -> list[str]:
    if not isinstance(metrics, dict):
        return [f"experiment.run must return a dict of metrics, got {type(metrics).__name__}"]
    problems = []
    for name in names:
        if name not in metrics:
            problems.append(f"missing metric {name!r} (expected keys: {names})")
            continue
        try:
            value = float(metrics[name])
        except (TypeError, ValueError):
            problems.append(f"metric {name!r} is not a number: {metrics[name]!r}")
            continue
        if not math.isfinite(value):
            problems.append(f"metric {name!r} is {value} (must be a finite number)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one condition of the experiment with one seed.")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny, fast run to check that the code works")
    parser.add_argument("--workdir", default=None, help="scratch space for this run (checkpoints, caches)")
    args = parser.parse_args()

    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if args.condition not in config["conditions"]:
        print(f"rtool-contract: unknown condition {args.condition!r}; known: {sorted(config['conditions'])}")
        return 3
    workdir = Path(args.workdir or HERE.parent / "scratch" / f"{args.condition}__{args.seed}")
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    writable_caches(workdir)  # before the experiment imports transformers, vLLM or datasets
    set_seed(args.seed)
    sys.path.insert(0, str(HERE))
    import experiment

    started = time.time()
    metrics = experiment.run(
        condition=args.condition,
        seed=args.seed,
        resources=config["resources"],
        smoke=args.smoke,
        workdir=str(workdir),
    )
    problems = check(metrics, list(config["metrics"]))
    if problems:
        print("rtool-contract: " + "; ".join(problems))
        return 3
    record = {
        "condition": args.condition,
        "seed": args.seed,
        "metrics": {str(k): v for k, v in ((k, as_number(v)) for k, v in metrics.items()) if v is not None},
        "seconds": round(time.time() - started, 2),
        "smoke": args.smoke,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(partial, output)  # never leave a half-written result
    print("rtool-result: " + json.dumps(record["metrics"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
