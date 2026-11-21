"""Reference for two-servers-shorter-wait: the case's experiment, written by hand (standard library only).

python reference.py        # about 10 s
"""

import heapq
import math
import random
import statistics


def mean_time_in_system(arrival_rate, service_rate, servers, customers, warmup, rng):
    free = [0.0] * servers  # when each server is next free; a customer takes the first free one
    clock = total = 0.0
    for n in range(customers):
        clock += rng.expovariate(arrival_rate)
        departure = max(clock, free[0]) + rng.expovariate(service_rate)
        heapq.heapreplace(free, departure)
        if n >= warmup:
            total += departure - clock
    return total / (customers - warmup)


if __name__ == "__main__":
    one = [mean_time_in_system(0.8, 1.0, 1, 200_000, 20_000, random.Random(seed)) for seed in range(8)]
    two = [mean_time_in_system(0.8, 1.0, 2, 200_000, 20_000, random.Random(seed)) for seed in range(8)]
    diff = [b - a for a, b in zip(one, two, strict=True)]
    t = statistics.mean(diff) / (statistics.stdev(diff) / math.sqrt(len(diff)))
    print(
        f"one_server {statistics.mean(one):.3f} (formula 5.0), two_servers {statistics.mean(two):.3f} (formula 1.190)"
    )
    print(f"difference {statistics.mean(diff):.3f}, paired t {t:.1f} (|t| > 2.365 is p < 0.05 with 7 df)")
    holds = abs(t) > 2.365 and -4.4 <= statistics.mean(diff) <= -3.2
    print("expectation (significant, lower, difference in [-4.4, -3.2]):", "holds" if holds else "FAILS")
