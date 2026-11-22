"""Holdout comparison of schedules chosen using only variable overlap."""
import csv
import random

import numpy as np

from anneal import evolve


def draw_instance(n, seed):
    rng = random.Random(seed)
    states = np.arange(1 << n)
    while True:
        masks, violations = [], []
        for _ in range(4 * n):
            variables = rng.sample(range(n), 3)
            mask = sum(1 << v for v in variables)
            forbidden = sum(rng.randrange(2) << v for v in variables)
            masks.append(mask)
            violations.append((states & mask) == forbidden)
        violations = np.asarray(violations)
        solution = violations.sum(axis=0) == 0
        if solution.sum() == 1:
            return masks, violations, solution


def run():
    names = ("linear", "high_first", "low_first", "random_first")
    with open("structure.csv", "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("n", "seed", "schedule", "p120", "p240"))
        for n in (6, 8, 10):
            scores = []
            for seed in range(100, 112):
                masks, violations, solution = draw_instance(n, seed)
                overlap = [sum((mask & other).bit_count() for other in masks)
                           for mask in masks]
                shuffled = list(range(len(masks)))
                random.Random(10000 * n + seed).shuffle(shuffled)
                ranked = sorted(shuffled, key=overlap.__getitem__, reverse=True)
                half = len(masks) // 2
                orders = (ranked, ranked, ranked[half:] + ranked[:half], shuffled)
                row = []
                for name, order in zip(names, orders):
                    groups = np.array([violations[order[:half]].sum(axis=0),
                                       violations[order[half:]].sum(axis=0)])
                    p, refined = [evolve(n, groups, solution, name != "linear", steps)
                                  for steps in (120, 240)]
                    assert abs(p - refined) < 0.002
                    writer.writerow((n, seed, name, p, refined))
                    row.append(refined)
                scores.append(row)
            scores = np.asarray(scores)
            print(f"variables {n}, 12 paired instances")
            for i, name in enumerate(names):
                delta = scores[:, i] - scores[:, 0]
                print(f"{name} mean={scores[:, i].mean():.6f} "
                      f"wins_vs_linear={sum(delta > 0)}")
            print(f"high beats low: {sum(scores[:, 1] > scores[:, 2])}/12")


if __name__ == "__main__":
    run()
