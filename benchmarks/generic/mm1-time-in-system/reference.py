"""Reference for mm1-time-in-system: the case's experiment, written by hand (standard library only).

python reference.py        # about 5 s
"""

import heapq
import random
import statistics


def mean_time_in_system(arrival_rate, service_rate, servers, customers, warmup, rng):
    free = [0.0] * servers  # when each server is next free
    clock = total = 0.0
    for n in range(customers):
        clock += rng.expovariate(arrival_rate)
        departure = max(clock, free[0]) + rng.expovariate(service_rate)
        heapq.heapreplace(free, departure)
        if n >= warmup:
            total += departure - clock
    return total / (customers - warmup)


if __name__ == "__main__":
    for name, service_rate, truth in (("mm1", 1.0, 5.0), ("mm1_fast_server", 1.25, 1 / 0.45)):
        means = [mean_time_in_system(0.8, service_rate, 1, 200_000, 20_000, random.Random(seed)) for seed in range(8)]
        spread = statistics.stdev(means)
        print(f"{name}: mean {statistics.mean(means):.3f} over 8 seeds (sd {spread:.3f}; formula {truth:.3f})")
    mm1 = statistics.mean(mean_time_in_system(0.8, 1.0, 1, 200_000, 20_000, random.Random(s)) for s in range(8))
    print("expectation (4.5 <= mean of mm1 <= 5.5):", "holds" if 4.5 <= mm1 <= 5.5 else "FAILS")
