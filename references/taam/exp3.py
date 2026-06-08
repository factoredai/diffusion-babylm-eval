"""EXP3 multi-armed bandit for online multilingual data mixing.

Algorithm (Auer, Cesa-Bianchi, Freund & Schapire, 2002):
    w_{t+1}(i) = w_t(i) * exp(eta * r_t(i) / pi_t(i))
    pi_{t+1}(i) = (1 - gamma) * w_{t+1}(i) / sum_j w_{t+1}(j) + gamma / K

Notes vs. standard EXP3:
    - We support a *custom initial distribution* (TAAM's typological prior),
      not just a uniform start. This is encoded as initial log-weights such
      that softmax(log_w0) = pi_0.
    - We support a *floor* on pi to prevent any language from being starved
      (renormalization is conservative: clip then divide by sum).
    - The reward enters as the *full* observed reward for the chosen arm
      under standard EXP3. We treat the mixing problem as observing reward
      for *all* arms each round (since we compute per-language val loss),
      so we use the importance-weighted update for all arms. This is
      EXP3.S / EXP4-style and is the formulation used in ODM (Albalak 2023).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DEFAULT_ETA: float = 0.10
DEFAULT_GAMMA: float = 0.10
DEFAULT_MIN_PI: float = 0.05


def _safe_softmax(log_w: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = log_w - log_w.max()
    e = np.exp(z)
    return e / e.sum()


def _apply_floor(p: np.ndarray, floor: float) -> np.ndarray:
    """Project ``p`` onto the simplex with a per-coordinate floor.

    This is implemented as a linear shrinkage toward the uniform distribution
    scaled by the floor:

        result = floor * 1_K + (1 - K*floor) * p_normalized

    Properties:
        - ``result.sum() == 1`` exactly (no float drift if the input sums to 1).
        - ``result_i >= floor`` exactly (no clip + renormalize side-effect that
          could push some entries back below the floor — the bug the tests catch).
        - If ``floor == 0``, this reduces to normalization.

    Args:
        p: non-negative array of weights (need not sum to 1; we normalize first).
        floor: minimum probability per coordinate; must satisfy ``floor * K <= 1``.

    Raises:
        ValueError on invalid inputs.
    """
    if floor < 0:
        raise ValueError("floor must be >= 0")
    K = len(p)
    if floor * K > 1.0 + 1e-9:
        raise ValueError(
            f"Invalid floor={floor} for K={K} (floor * K must be <= 1)."
        )
    if np.any(p < 0):
        raise ValueError("p must be non-negative.")
    s = p.sum()
    if s <= 0:
        raise ValueError("p must have positive sum.")
    p_norm = p / s
    return floor + (1.0 - floor * K) * p_norm


@dataclass
class EXP3State:
    """Snapshot of EXP3 state, suitable for logging / checkpointing."""

    step: int
    log_w: np.ndarray
    pi: np.ndarray
    last_reward: np.ndarray | None = None


@dataclass
class EXP3MultilingualMixer:
    """EXP3 bandit for K languages with a custom prior and probability floor.

    Args:
        languages: ordered tuple of language codes.
        pi_0: initial sampling probabilities (must sum to 1, no zeros).
        eta: learning rate.
        gamma: exploration mixing parameter (uniform mix-in).
        min_pi: probability floor enforced after each update.
        seed: numpy RNG seed for the categorical sampler.

    Usage:
        mixer = EXP3MultilingualMixer(
            languages=("eng", "nld", "zho"),
            pi_0=np.array([0.30, 0.30, 0.40]),
            eta=0.1, gamma=0.1, min_pi=0.05, seed=42,
        )
        for step in range(N):
            # 1) sample a language for the next batch (optional; for sequential mode)
            lang_idx = mixer.sample()

            # 2) periodically, after computing per-language val losses:
            r = reward.update(losses)  # dict
            r_vec = np.array([r[l] for l in mixer.languages])
            mixer.update(rewards=r_vec)
    """

    languages: tuple[str, ...]
    pi_0: np.ndarray
    eta: float = DEFAULT_ETA
    gamma: float = DEFAULT_GAMMA
    min_pi: float = DEFAULT_MIN_PI
    seed: int | None = None

    # internal state (initialized in __post_init__)
    _log_w: np.ndarray = field(init=False)
    _pi: np.ndarray = field(init=False)
    _step: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False)
    _last_reward: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.pi_0 = np.asarray(self.pi_0, dtype=np.float64)
        if self.pi_0.ndim != 1 or len(self.pi_0) != len(self.languages):
            raise ValueError("pi_0 must be a 1-D array of length len(languages).")
        if not np.isclose(self.pi_0.sum(), 1.0, atol=1e-6):
            raise ValueError(f"pi_0 must sum to 1.0 (got {self.pi_0.sum():.6f}).")
        if np.any(self.pi_0 <= 0.0):
            raise ValueError(
                "pi_0 must be strictly positive everywhere. Apply a floor before "
                "passing it in (e.g., taam.typological_prior._apply_floor)."
            )
        if not (0.0 < self.eta):
            raise ValueError("eta must be > 0")
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma must be in [0, 1)")
        if not (0.0 <= self.min_pi <= 1.0 / len(self.languages)):
            raise ValueError(
                f"min_pi must be in [0, 1/K] = [0, {1.0/len(self.languages):.4f}]."
            )

        # Initialize log-weights so softmax(log_w) = pi_0.
        self._log_w = np.log(self.pi_0)
        self._pi = self._project(self._log_w)
        self._rng = np.random.default_rng(self.seed)

    # ───────────────── public API ─────────────────

    @property
    def pi(self) -> np.ndarray:
        return self._pi.copy()

    @property
    def step(self) -> int:
        return self._step

    def sample(self) -> int:
        """Sample a language index from the current distribution pi."""
        return int(self._rng.choice(len(self.languages), p=self._pi))

    def update(self, rewards: np.ndarray | Sequence[float]) -> EXP3State:
        """Update weights given per-language rewards.

        Args:
            rewards: length-K array of normalized rewards for each language.

        Returns:
            EXP3State snapshot.
        """
        r = np.asarray(rewards, dtype=np.float64)
        if r.shape != (len(self.languages),):
            raise ValueError(
                f"rewards must have shape ({len(self.languages)},), got {r.shape}"
            )

        # Standard EXP3 importance-weighted update.
        # We add eta * r / pi to log-weights and re-project.
        # This is mathematically equivalent to multiplying weights by exp(...)
        # and renormalizing.
        increment = self.eta * r / np.maximum(self._pi, 1e-12)
        self._log_w = self._log_w + increment

        # Re-center log-weights to avoid numerical drift over thousands of steps.
        # Softmax is invariant to constant shifts, so subtracting the max is free.
        self._log_w -= self._log_w.max()

        self._pi = self._project(self._log_w)
        self._step += 1
        self._last_reward = r.copy()
        return self.snapshot()

    def snapshot(self) -> EXP3State:
        return EXP3State(
            step=self._step,
            log_w=self._log_w.copy(),
            pi=self._pi.copy(),
            last_reward=None if self._last_reward is None else self._last_reward.copy(),
        )

    # ───────────────── internals ─────────────────

    def _project(self, log_w: np.ndarray) -> np.ndarray:
        """Compute pi from log_w with uniform mix-in (gamma) and floor (min_pi)."""
        soft = _safe_softmax(log_w)
        K = len(self.languages)
        mixed = (1.0 - self.gamma) * soft + self.gamma / K
        return _apply_floor(mixed, floor=self.min_pi)
