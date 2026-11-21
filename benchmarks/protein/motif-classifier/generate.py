"""Data for motif-classifier: protein-like sequences, label 1 when they carry the motif HEGH.

    python -I generate.py <output folder> '{"n_train": 300, "n_test": 200, "seed": 3}'

Writes train.csv and test.csv (id, sequence, label), balanced. Lengths (60-100) do not depend on the
label, and no test sequence appears in the training set.
"""

import csv
import json
import random
import sys
from pathlib import Path

AMINO = "ACDEFGHIKLMNPQRSTVWY"
MOTIF = "HEGH"


def sequence(label, rng):
    while True:
        body = "".join(rng.choice(AMINO) for _ in range(rng.randint(60, 100)))
        if label:
            at = rng.randrange(len(body) - len(MOTIF))
            body = body[:at] + MOTIF + body[at + len(MOTIF) :]
        if (MOTIF in body) == bool(label):
            return body


def write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "label"])
        writer.writerows(rows)


if __name__ == "__main__":
    out, args = Path(sys.argv[1]), json.loads(sys.argv[2])
    rng = random.Random(args["seed"])
    train = [(f"tr{i:04d}", sequence(i % 2, rng), i % 2) for i in range(args["n_train"])]
    seen = {s for _, s, _ in train}
    test = []
    while len(test) < args["n_test"]:
        label = len(test) % 2
        body = sequence(label, rng)
        if body not in seen:
            test.append((f"te{len(test):04d}", body, label))
    write(out / "train.csv", train)
    write(out / "test.csv", test)
