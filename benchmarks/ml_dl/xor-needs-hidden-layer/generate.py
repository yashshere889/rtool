"""Data for xor-needs-hidden-layer: two inputs around the corners of a square, labelled by XOR.

    python -I generate.py <output folder> '{"n_train": 2000, "n_test": 1000, "noise": 0.35, "seed": 7}'

Writes train.csv and test.csv (columns x1, x2, label). Points on the diagonal (1, 1)-(-1, -1) have
label 0 and points on the other diagonal label 1, so no straight line separates the classes.
"""

import csv
import json
import random
import sys
from pathlib import Path

CORNERS = (((1, 1), 0), ((-1, -1), 0), ((1, -1), 1), ((-1, 1), 1))


def write(path, n, noise, rng):
    with open(path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["x1", "x2", "label"])
        for _ in range(n):
            (cx, cy), label = rng.choice(CORNERS)
            out.writerow([round(cx + rng.gauss(0, noise), 5), round(cy + rng.gauss(0, noise), 5), label])


if __name__ == "__main__":
    folder, args = Path(sys.argv[1]), json.loads(sys.argv[2])
    rng = random.Random(args["seed"])
    write(folder / "train.csv", args["n_train"], args["noise"], rng)
    write(folder / "test.csv", args["n_test"], args["noise"], rng)
