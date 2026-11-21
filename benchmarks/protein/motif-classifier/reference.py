"""Reference for motif-classifier: both models trained by hand (standard library only).

    python reference.py        # about half a minute

Logistic regression on sequence length (the baseline) and on the 400 dipeptide counts (per 100
residues), each trained with Adam (learning rate 0.05, 15 epochs, batch 32); the seed sets the
initial weights and the order of the examples.
"""

import csv
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMINO = "ACDEFGHIKLMNPQRSTVWY"
PAIRS = {a + b: i for i, (a, b) in enumerate((a, b) for a in AMINO for b in AMINO)}


def load(path):
    with open(path, newline="") as handle:
        return [(r["sequence"], int(r["label"])) for r in csv.DictReader(handle)]


def features(seq, kind):
    if kind == "length_only":
        return {0: (len(seq) - 80) / 20}
    counts = {}
    for i in range(len(seq) - 1):
        j = PAIRS[seq[i : i + 2]]
        counts[j] = counts.get(j, 0.0) + 100 / len(seq)
    return counts


def train_and_test(train, test, kind, seed):
    rng = random.Random(seed)
    size = 1 if kind == "length_only" else len(PAIRS)
    w, b = [rng.gauss(0, 0.01) for _ in range(size)], 0.0
    m, v, t = [0.0] * (size + 1), [0.0] * (size + 1), 0
    rows = [(features(s, kind), y) for s, y in train]
    for _ in range(15):
        rng.shuffle(rows)
        for start in range(0, len(rows), 32):
            grads = {}
            gb = 0.0
            batch = rows[start : start + 32]
            for x, y in batch:
                z = b + sum(w[j] * value for j, value in x.items())
                d = (1 / (1 + math.exp(-z)) - y) / len(batch)
                gb += d
                for j, value in x.items():
                    grads[j] = grads.get(j, 0.0) + d * value
            t += 1
            for j, g in [*grads.items(), (size, gb)]:
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                step = 0.05 * (m[j] / (1 - 0.9**t)) / (math.sqrt(v[j] / (1 - 0.999**t)) + 1e-8)
                if j == size:
                    b -= step
                else:
                    w[j] -= step
    right = sum((b + sum(w[j] * value for j, value in features(s, kind).items()) > 0) == bool(y) for s, y in test)
    return right / len(test)


if __name__ == "__main__":
    args = {"n_train": 300, "n_test": 200, "seed": 3}
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([sys.executable, "-I", str(HERE / "generate.py"), folder, json.dumps(args)], check=True)
        train, test = load(Path(folder) / "train.csv"), load(Path(folder) / "test.csv")
    base = [train_and_test(train, test, "length_only", s) for s in range(8)]
    pairs = [train_and_test(train, test, "dipeptide", s) for s in range(8)]
    diff = [p - q for p, q in zip(pairs, base, strict=True)]
    t = statistics.mean(diff) / (statistics.stdev(diff) / math.sqrt(8)) if statistics.stdev(diff) else math.inf
    print(f"length_only {statistics.mean(base):.3f}, dipeptide {statistics.mean(pairs):.3f}")
    print(f"difference {statistics.mean(diff):.3f} (sd {statistics.stdev(diff):.3f}), paired t {t:.1f}")
