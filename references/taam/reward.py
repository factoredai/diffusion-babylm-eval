"""Reward modules for the EXP3 multilingual bandit.

Two reward variants are provided (selectable via ``config.reward.type``):

1. ``normalized_excess_loss`` (the v1 reward, see research_context §9.2):

       For each language l at eval step t:
           loss_l_t   = CE on held-out 5k-token dev slice
           ref_l_t    = mean over last 5 evals of loss_l (cold-start = first val)
           raw_r_l_t  = ref_l_t - loss_l_t           # positive if loss is decreasing
           sigma_l    = running std of raw_r_l_t over last 20 evals (else 1.0)
           r_l_t      = clip(raw_r_l_t / sigma_l, -CLIP, +CLIP)

   This rewards languages whose loss is *currently going down*. Empirically (run
   ``runs/2026-05-13_TAAM_seed42``) this reward starves languages that plateau
   at high absolute loss: once their delta saturates near 0, EXP3's weight
   collapses to the structural floor (gamma/K projected through min_pi) and the
   language is effectively abandoned, even when it remains the worst-performing.

2. ``cross_lingual_deficit`` (the v2 reward, introduced after the failure-mode
   diagnosis):

       At each eval step t with per-language losses L_l_t:
           target_t   = min_l L_l_t                  # the easiest language's level
           d_l_t      = L_l_t - target_t             # >= 0 by construction
           sigma_t    = running std of d over last 20 evals (cross-language scale)
           r_l_t      = clip(d_l_t / sigma_t, 0, +CLIP)

   This rewards languages whose *absolute level* is high relative to the
   current best language. It cannot drive a hard language's reward to zero
   while easier languages still gain — the deficit is a level-based quantity,
   not a delta. The easiest language gets r=0 (no need for extra budget),
   the hardest gets the largest reward (deserves more attention). This is the
   curriculum-learning behavior the paper's H1 predicts.

   Notes:
     - Reward is non-negative; this makes EXP3 weights non-decreasing in the
       importance-weighted update, which is fine for relative allocation
       (only the *differences* between log-weights matter for softmax).
     - When all losses converge (target ~ everyone), all rewards approach 0
       and EXP3 freezes — a desirable termination property.

Use ``make_reward(type=..., languages=..., **kwargs)`` to instantiate the right
variant from a config dict.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

DEFAULT_REF_WINDOW: int = 5
DEFAULT_STD_WINDOW: int = 20
DEFAULT_CLIP: float = 2.0
DEFAULT_STD_FALLBACK: float = 1.0


@dataclass
class LanguageRewardState:
    """Per-language sliding state for normalized excess-loss reward.

    Attributes:
        history: rolling buffer of past raw rewards (for std estimation).
        ref_buffer: rolling buffer of past *loss* values (for EMA reference).
        last_ref: cached most recent reference loss (lazy/snapshot for logs).
    """

    history: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_STD_WINDOW))
    ref_buffer: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_REF_WINDOW))
    last_ref: float = float("nan")


class NormalizedExcessLossReward:
    """Stateful, multi-language reward computer for EXP3.

    Example:
        reward = NormalizedExcessLossReward(languages=("eng", "nld", "zho"))
        # at every evaluation step, after computing per-language val losses:
        r = reward.update(losses={"eng": 3.21, "nld": 3.45, "zho": 4.12})
        # r is dict[str, float] in same order as languages; pass to EXP3.

    Notes:
        - First call returns zeros for all languages (no history yet).
        - The EMA reference uses *equal weights* over the last N losses; this
          is the simple recommendation in §9.2. Replace with proper EMA if
          desired but keep behavior deterministic.
    """

    def __init__(
        self,
        languages: Iterable[str],
        ref_window: int = DEFAULT_REF_WINDOW,
        std_window: int = DEFAULT_STD_WINDOW,
        clip: float = DEFAULT_CLIP,
        std_fallback: float = DEFAULT_STD_FALLBACK,
    ) -> None:
        self.languages: tuple[str, ...] = tuple(languages)
        self.ref_window = ref_window
        self.std_window = std_window
        self.clip = clip
        self.std_fallback = std_fallback
        self._state: dict[str, LanguageRewardState] = {
            l: LanguageRewardState(
                history=deque(maxlen=std_window),
                ref_buffer=deque(maxlen=ref_window),
            )
            for l in self.languages
        }

    def _step_lang(self, lang: str, loss: float) -> float:
        s = self._state[lang]

        # Cold start: no reference yet -> reference = current loss -> raw_r = 0
        if len(s.ref_buffer) == 0:
            ref = float(loss)
        else:
            ref = float(np.mean(s.ref_buffer))
        s.last_ref = ref

        raw_r = ref - float(loss)  # positive if loss is *decreasing*

        # Update ref buffer AFTER computing raw_r so the current value
        # contributes to future references but not the present comparison.
        s.ref_buffer.append(float(loss))

        # Normalize by running std of raw rewards. If we don't have enough
        # samples yet, fall back to std_fallback (unit scale).
        if len(s.history) >= 2:
            sigma = float(np.std(s.history, ddof=1))
            if sigma < 1e-8:
                sigma = self.std_fallback
        else:
            sigma = self.std_fallback

        s.history.append(raw_r)

        normed = raw_r / sigma
        return float(np.clip(normed, -self.clip, self.clip))

    def update(self, losses: dict[str, float]) -> dict[str, float]:
        """Compute one round of rewards from per-language losses.

        Args:
            losses: dict {lang -> CE loss on held-out dev slice}. Must contain
                every language passed at init.

        Returns:
            dict {lang -> normalized clipped reward}.
        """
        missing = set(self.languages) - set(losses)
        if missing:
            raise KeyError(f"Missing per-language losses for: {sorted(missing)}")
        return {l: self._step_lang(l, losses[l]) for l in self.languages}

    def diagnostics(self) -> dict:
        """Return a snapshot of internal state for logging/debugging."""
        return {
            l: {
                "ref_buffer": list(self._state[l].ref_buffer),
                "history": list(self._state[l].history),
                "last_ref": self._state[l].last_ref,
            }
            for l in self.languages
        }


class CrossLingualDeficitReward:
    """Level-based reward: each language is rewarded by how far above the
    current best language's loss it sits.

    Designed to avoid the failure mode of ``NormalizedExcessLossReward`` that
    is documented in ``runs/2026-05-13_TAAM_seed42`` (the v1 paper run): a
    delta-based reward starves languages once their loss plateaus, even when
    they remain the worst performers.

    Reward formula:
        target_t   = min_l L_l_t
        d_l_t      = max(L_l_t - target_t, 0)            # always >= 0
        sigma_t    = running std of all observed d values over last 20 evals
                     (cross-language sigma; fallback=std_fallback)
        r_l_t      = clip(d_l_t / sigma_t, 0, +CLIP)

    Rationale for choices:
        - Cross-language sigma (single normalizer over the pool of deficits)
          keeps the rewards directly comparable across languages: a deficit of
          1.0 nat means "this language is 1 nat above target", regardless of
          who it is.
        - Reward >= 0: the easiest language gets r=0, EXP3 won't add weight to
          it for being easy; but it won't penalize it either. The next-easiest
          will get a small positive r proportional to its deficit. The hardest
          gets the largest r.
        - No EMA/ref_buffer state: the reward is a snapshot of the current
          eval's spread. This is intentional — we want reactive level-based
          allocation, not history-smoothed delta tracking.

    Args:
        languages: ordered tuple of language codes.
        std_window: how many recent cross-language deficits to use for sigma.
        clip: maximum positive reward magnitude.
        std_fallback: sigma used while history is too short or near zero.

    Example:
        >>> rew = CrossLingualDeficitReward(languages=("eng", "nld", "zho"))
        >>> for _ in range(5):
        ...     rew.update({"eng": 4.0, "nld": 4.0, "zho": 4.0})
        ...     # all rewards ~ 0 (no deficit)
        >>> r = rew.update({"eng": 4.0, "nld": 4.0, "zho": 5.0})
        >>> # r["zho"] > 0, r["eng"] == r["nld"] == 0
    """

    def __init__(
        self,
        languages: Iterable[str],
        std_window: int = DEFAULT_STD_WINDOW,
        clip: float = DEFAULT_CLIP,
        std_fallback: float = DEFAULT_STD_FALLBACK,
    ) -> None:
        self.languages: tuple[str, ...] = tuple(languages)
        self.std_window = std_window
        self.clip = clip
        self.std_fallback = std_fallback
        # Pooled history of all per-language deficits across recent steps.
        # std_window * K entries are tracked so the std estimate sees one full
        # window of cross-lingual deficits.
        self._deficit_history: deque = deque(maxlen=std_window * len(self.languages))
        self._last_target: float = float("nan")
        self._last_deficits: dict[str, float] = {l: float("nan") for l in self.languages}

    def update(self, losses: dict[str, float]) -> dict[str, float]:
        """Compute one round of rewards from per-language losses.

        Args:
            losses: dict {lang -> CE loss on held-out dev slice}.

        Returns:
            dict {lang -> non-negative clipped reward}.
        """
        missing = set(self.languages) - set(losses)
        if missing:
            raise KeyError(f"Missing per-language losses for: {sorted(missing)}")
        target = float(min(losses[l] for l in self.languages))
        deficits = {l: max(float(losses[l]) - target, 0.0) for l in self.languages}

        # Update pooled history before computing sigma so the current values
        # contribute to the normalizer (consistent with v1's semantics).
        for d in deficits.values():
            self._deficit_history.append(d)

        if len(self._deficit_history) >= 2:
            sigma = float(np.std(self._deficit_history, ddof=1))
            if sigma < 1e-8:
                sigma = self.std_fallback
        else:
            sigma = self.std_fallback

        self._last_target = target
        self._last_deficits = deficits
        return {
            l: float(np.clip(deficits[l] / sigma, 0.0, self.clip))
            for l in self.languages
        }

    def diagnostics(self) -> dict:
        """Return a snapshot of internal state for logging/debugging."""
        return {
            "target": self._last_target,
            "deficits": dict(self._last_deficits),
            "n_history": len(self._deficit_history),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────


REWARD_TYPES = ("normalized_excess_loss", "cross_lingual_deficit")


def make_reward(
    reward_type: str,
    languages: Iterable[str],
    *,
    ref_window: int = DEFAULT_REF_WINDOW,
    std_window: int = DEFAULT_STD_WINDOW,
    clip: float = DEFAULT_CLIP,
    std_fallback: float = DEFAULT_STD_FALLBACK,
):
    """Build a reward module from a config-style name.

    Args:
        reward_type: one of ``REWARD_TYPES``.
        languages: ordered tuple of language codes.
        ref_window: only used by ``normalized_excess_loss``.
        std_window: window for the std normalizer (both variants).
        clip: max absolute reward magnitude.
        std_fallback: sigma fallback when history is short.

    Returns:
        A reward instance exposing ``.update(losses) -> dict``.

    Raises:
        ValueError on unknown ``reward_type``.
    """
    if reward_type == "normalized_excess_loss":
        return NormalizedExcessLossReward(
            languages=languages,
            ref_window=ref_window,
            std_window=std_window,
            clip=clip,
            std_fallback=std_fallback,
        )
    if reward_type == "cross_lingual_deficit":
        return CrossLingualDeficitReward(
            languages=languages,
            std_window=std_window,
            clip=clip,
            std_fallback=std_fallback,
        )
    raise ValueError(
        f"Unknown reward_type={reward_type!r}; expected one of {REWARD_TYPES}."
    )
