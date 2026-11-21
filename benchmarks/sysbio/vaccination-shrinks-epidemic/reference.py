"""Reference for the sysbio epidemic cases: stochastic SIR, written by hand (standard library only).

    python reference.py        # a few seconds

Population 1,000, 10 initial infections, recovery rate 1, transmission rate R0 x recovery rate, run until
no one is infected. The attack rate counts everyone ever infected (index cases included) over 1,000.
Prints the final-size equation's value for comparison: z = 1 - (1 - i0) exp(-R0 z), i0 = 10 / 1000.
"""

import math
import random
import statistics


def direct(n, i0, r0, immune, rng):
    """Gillespie's direct method: one uniform picks the next event, in proportion to the rates."""
    s, i, r = n - i0 - immune, i0, 0
    while i:
        infect, recover = r0 * s * i / n, float(i)
        if rng.random() * (infect + recover) < infect:
            s, i = s - 1, i + 1
        else:
            i, r = i - 1, r + 1
    return r / n


def first_reaction(n, i0, r0, immune, rng):
    """Gillespie's first-reaction method: a waiting time for each reaction, the earliest fires."""
    s, i, r = n - i0 - immune, i0, 0
    while i:
        infect, recover = r0 * s * i / n, float(i)
        t_infect = rng.expovariate(infect) if infect > 0 else math.inf
        if t_infect < rng.expovariate(recover):
            s, i = s - 1, i + 1
        else:
            i, r = i - 1, r + 1
    return r / n


def final_size(r0, i0=0.01):
    z = 0.5
    for _ in range(200):
        z = 1 - (1 - i0) * math.exp(-r0 * z)
    return z


def summary(name, values):
    return f"{name}: mean {statistics.mean(values):.4f}, sd {statistics.stdev(values):.4f}"


if __name__ == "__main__":
    for r0 in (2.0, 3.0):
        runs = [direct(1000, 10, r0, 0, random.Random(s)) for s in range(8)]
        print(summary(f"R0 {r0}", runs), f"(final-size equation {final_size(r0):.4f})")
    none = [direct(1000, 10, 2.0, 0, random.Random(s)) for s in range(8)]
    vaccinated = [direct(1000, 10, 2.0, 300, random.Random(s)) for s in range(8)]
    diff = [b - a for a, b in zip(none, vaccinated, strict=True)]
    t = statistics.mean(diff) / (statistics.stdev(diff) / math.sqrt(8))
    print(
        summary("no vaccination", none),
        "|",
        summary("30 % vaccinated", vaccinated),
        f"| difference {statistics.mean(diff):.4f}, paired t {t:.1f}",
    )
    for r0 in (1.5, 2.0, 3.0):
        a = [direct(1000, 10, r0, 0, random.Random(s)) for s in range(8)]
        b = [first_reaction(1000, 10, r0, 0, random.Random(s)) for s in range(8)]
        d = [y - x for x, y in zip(a, b, strict=True)]
        t = statistics.mean(d) / (statistics.stdev(d) / math.sqrt(8))
        means = f"direct {statistics.mean(a):.4f}, first reaction {statistics.mean(b):.4f}"
        print(f"methods at R0 {r0}: {means}, paired t {t:.2f} (|t| > 2.365 would be p < 0.05)")
