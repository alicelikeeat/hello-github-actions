"""Paired noiseless 3-SAT annealing experiment. Units: hbar = 1."""
import csv
import random
from pathlib import Path

import numpy as np


def instance(n, seed):
    rng = random.Random(seed)
    states = np.arange(2**n)
    # Condition on a unique solution; never pass the solution to the schedule.
    while True:
        costs = np.zeros((2, len(states)), dtype=int)
        for i in range(4 * n):
            variables = rng.sample(range(n), 3)
            mask = sum(1 << v for v in variables)
            forbidden = sum(rng.randrange(2) << v for v in variables)
            costs[i % 2] += (states & mask) == forbidden
        total = costs.sum(axis=0)
        if np.count_nonzero(total == 0) == 1:
            return costs, total == 0


def evolve(n, costs, solution, staged, steps):
    duration = 12.0
    dt = duration / steps
    state = np.full(2**n, 2**(-n / 2), dtype=complex)
    total = costs.sum(axis=0)
    imbalance = costs[0] - costs[1]
    for j in range(steps):
        s = (j + 0.5) / steps
        # Weights s +/- s(1-s) are monotone with unchanged sum and endpoints.
        energy = s * total
        if staged:
            energy += s * (1 - s) * imbalance
        phase = np.exp(-0.5j * dt * energy)
        state *= phase
        angle = dt * (1 - s)
        for bit in range(n):
            pairs = state.reshape(-1, 2, 1 << bit)
            pairs[:] = np.cos(angle) * pairs + 1j * np.sin(angle) * pairs[:, ::-1]
        state *= phase
    assert abs(np.vdot(state, state).real - 1) < 1e-9
    return float(np.sum(abs(state[solution])**2))


def main():
    rows = []
    for n in (6, 8, 10):
        pairs = []
        for seed in range(4):
            costs, solution = instance(n, seed)
            probabilities = []
            for staged in (False, True):
                coarse, fine = [evolve(n, costs, solution, staged, k) for k in (120, 240)]
                assert abs(coarse - fine) < 0.002
                probabilities.append(fine)
                rows.append((n, seed, staged, coarse, fine))
            pairs.append(probabilities)
        baseline, staged_mean = np.mean(pairs, axis=0)
        wins = sum(b > a for a, b in pairs)
        print(f"n={n}: linear={baseline:.6f} staged={staged_mean:.6f} wins={wins}/4")
    with Path("results.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("variables", "seed", "staged", "p_120", "p_240"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
