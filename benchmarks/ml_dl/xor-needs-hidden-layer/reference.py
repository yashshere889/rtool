"""Reference for xor-needs-hidden-layer: both models trained by hand (standard library only).

    python reference.py [noise]        # about a minute per noise level

Logistic regression against one hidden layer of 16 tanh units, each trained with Adam (learning rate
0.01, 30 epochs, batch size 64) on the case's data, 8 seeds (the seed sets the weights and the order).
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


def load(path):
    with open(path, newline="") as handle:
        return [((float(r["x1"]), float(r["x2"])), int(r["label"])) for r in csv.DictReader(handle)]


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


class Adam:
    def __init__(self, params, lr=0.01):
        self.params, self.lr, self.t = params, lr, 0
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            m_hat, v_hat = self.m[i] / (1 - 0.9**self.t), self.v[i] / (1 - 0.999**self.t)
            self.params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + 1e-8)


def train(train_rows, test_rows, hidden, seed):
    rng = random.Random(seed)
    if hidden == 0:  # logistic regression: w1, w2, b
        params = [rng.gauss(0, 0.5), rng.gauss(0, 0.5), 0.0]
    else:  # per hidden unit: w1, w2, b; then output weights and bias
        params = [rng.gauss(0, 1) for _ in range(3 * hidden)] + [rng.gauss(0, 0.5) for _ in range(hidden)] + [0.0]
    opt = Adam(params)

    def forward(x):
        if hidden == 0:
            return sigmoid(params[0] * x[0] + params[1] * x[1] + params[2]), None
        h = [math.tanh(params[3 * j] * x[0] + params[3 * j + 1] * x[1] + params[3 * j + 2]) for j in range(hidden)]
        out = params[3 * hidden : 4 * hidden]
        return sigmoid(sum(o * a for o, a in zip(out, h, strict=True)) + params[-1]), h

    rows = list(train_rows)
    for _ in range(30):
        rng.shuffle(rows)
        for start in range(0, len(rows), 64):
            grads = [0.0] * len(params)
            batch = rows[start : start + 64]
            for x, y in batch:
                p, h = forward(x)
                d = (p - y) / len(batch)  # cross-entropy gradient at the output's input
                if hidden == 0:
                    grads[0] += d * x[0]
                    grads[1] += d * x[1]
                    grads[2] += d
                    continue
                for j in range(hidden):
                    grads[3 * hidden + j] += d * h[j]
                    back = d * params[3 * hidden + j] * (1 - h[j] * h[j])
                    grads[3 * j] += back * x[0]
                    grads[3 * j + 1] += back * x[1]
                    grads[3 * j + 2] += back
                grads[-1] += d
            opt.step(grads)
    return sum((forward(x)[0] >= 0.5) == bool(y) for x, y in test_rows) / len(test_rows)


if __name__ == "__main__":
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    args = {"n_train": 2000, "n_test": 1000, "noise": noise, "seed": 7}
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([sys.executable, "-I", str(HERE / "generate.py"), folder, json.dumps(args)], check=True)
        train_rows, test_rows = load(Path(folder) / "train.csv"), load(Path(folder) / "test.csv")
    linear = [train(train_rows, test_rows, 0, seed) for seed in range(8)]
    mlp = [train(train_rows, test_rows, 16, seed) for seed in range(8)]
    diff = [b - a for a, b in zip(linear, mlp, strict=True)]
    t = statistics.mean(diff) / (statistics.stdev(diff) / math.sqrt(8)) if statistics.stdev(diff) > 0 else math.inf
    print(f"noise {noise}: linear {statistics.mean(linear):.3f}, mlp {statistics.mean(mlp):.3f}")
    print(f"difference {statistics.mean(diff):.3f} (sd {statistics.stdev(diff):.3f}), paired t {t:.1f}")
