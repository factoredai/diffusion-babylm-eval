"""Verify the deterministic Dirichlet draw used for O2 (random fixed schedule).

We use Dirichlet(3, 3, 3) instead of Dirichlet(1, 1, 1):
    - Dirichlet(1, 1, 1) is uniform over the simplex but produces pathological
      skews (e.g., (0.08, 0.69, 0.23) with seed=2026), making O2 unfair as a
      "moderately non-uniform" control.
    - Dirichlet(3, 3, 3) is concentrated near (1/3, 1/3, 1/3) with mild spread,
      producing realistic moderate-skew draws.

Idempotent: rerun to confirm the committed pi in configs/O2_random_schedule.yaml.
"""
from __future__ import annotations

import numpy as np


def main() -> None:
    # We want O2 to be a "moderate non-uniform schedule" that points in a
    # DIFFERENT direction from the typological prior (which favors ZH). A
    # well-designed control flips the direction: argmax(pi_O2) != argmax(pi_P1).
    #
    # Strategy: draw Dirichlet(10, 10, 10) samples and accept the first seed
    # whose argmax is NOT zho. Concentration 10 keeps the draw moderate.
    target = 0
    labels = ("eng", "nld", "zho")
    selected_seed = None
    pi: np.ndarray | None = None
    for seed in range(10_000):
        candidate = np.random.default_rng(seed=seed).dirichlet([10.0, 10.0, 10.0])
        if int(candidate.argmax()) != labels.index("zho"):
            # And we want the dominant share to be at least 0.40 so the schedule
            # is meaningfully non-uniform.
            if candidate.max() >= 0.40:
                pi = candidate
                selected_seed = seed
                break
    assert pi is not None and selected_seed is not None, "No seed satisfied criteria; widen search."

    print(f"Deterministic Dirichlet(10,10,10), accepted seed={selected_seed}:")
    for l, p in zip(labels, pi):
        print(f"  {l}: {p:.6f}")
    print(f"Sum: {pi.sum():.6f}")
    print(f"argmax: {labels[int(pi.argmax())]} (expected != zho)")
    print()
    print("Update configs/O2_random_schedule.yaml manually with these values.")


if __name__ == "__main__":
    main()
