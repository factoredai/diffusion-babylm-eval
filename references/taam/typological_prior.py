"""URIEL/lang2vec-derived typological prior over initial sampling probabilities.

This module implements the locked recipe from ``improved_research_context_v2.md`` §9.1:

    1. Use lang2vec to get URIEL feature vectors for {eng, nld, zho}
       from syntax + morphology (phonology) + inventory feature families.
    2. Concatenate features into one vector per language; drop features
       marked unknown in any of the 3 languages.
    3. Compute pairwise cosine distances d(i, j).
    4. Per-language isolation: s_l = mean_{l' != l} d(l, l').
    5. Initial probabilities: pi_0(l) = softmax(s_l / T_prior), T_prior = 0.5.
    6. Apply a floor: pi_0(l) >= MIN_FLOOR (renormalize).

The output is the derived ``pi_0`` vector that initializes P1 (typological-prior
fixed) and TAAM (typological-prior + EXP3).

Sanity check: expected ordering is pi_0(zho) > pi_0(eng) ~ pi_0(nld).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

LOG = logging.getLogger(__name__)

# Default URIEL feature families to use when assembling the per-language vector.
# - "syntax_knn"     : k-NN-imputed syntactic features (WALS-derived)
# - "phonology_knn"  : phonological features
# - "inventory_knn"  : phoneme inventory features
# Rationale: syntax is the primary axis of transfer asymmetry between
# Germanic and Sino-Tibetan; phonology + inventory add coverage for features
# that distinguish writing systems / morphology indirectly.
DEFAULT_FEATURE_SETS: tuple[str, ...] = (
    "syntax_knn",
    "phonology_knn",
    "inventory_knn",
)

# Default hyperparameters (locked per spec)
DEFAULT_TEMPERATURE: float = 0.5
DEFAULT_FLOOR: float = 0.10

# Some BabyLM corpora use ISO 639-3 macro-language codes (e.g., "zho" for
# Chinese), but URIEL/lang2vec only stores the specific variety (e.g., "cmn"
# for Mandarin). We map the corpus-side code -> URIEL query code here so the
# rest of the codebase can keep using the BabyBabelLM-aligned label.
URIEL_CODE_ALIAS: dict[str, str] = {
    "zho": "cmn",  # Chinese (macro) -> Mandarin Chinese
}


def _to_uriel_code(lang: str) -> str:
    return URIEL_CODE_ALIAS.get(lang, lang)


@dataclass
class TypologicalPrior:
    """Result of the URIEL prior computation.

    Attributes:
        languages: ordered tuple of ISO 639-3 codes.
        pi_0: same-order numpy array of initial sampling probabilities. Sums to 1.
        isolation: per-language mean cosine distance to the other languages.
        pairwise_distance: |L| x |L| matrix of cosine distances.
        feature_sets: which URIEL feature families were used.
        n_features_used: number of features retained after dropping unknowns.
        temperature: softmax temperature applied.
        floor: probability floor applied (renormalized after).
    """

    languages: tuple[str, ...]
    pi_0: np.ndarray
    isolation: np.ndarray
    pairwise_distance: np.ndarray
    feature_sets: tuple[str, ...]
    n_features_used: int
    temperature: float
    floor: float

    def as_dict(self) -> dict:
        """Render as a plain dict for YAML/JSON serialization."""
        return {
            "languages": list(self.languages),
            "pi_0": {l: float(p) for l, p in zip(self.languages, self.pi_0)},
            "isolation": {l: float(s) for l, s in zip(self.languages, self.isolation)},
            "pairwise_distance": {
                self.languages[i]: {
                    self.languages[j]: float(self.pairwise_distance[i, j])
                    for j in range(len(self.languages))
                }
                for i in range(len(self.languages))
            },
            "feature_sets": list(self.feature_sets),
            "n_features_used": int(self.n_features_used),
            "temperature": float(self.temperature),
            "floor": float(self.floor),
        }


def _load_feature_vectors(
    languages: Sequence[str],
    feature_sets: Sequence[str],
) -> tuple[np.ndarray, int]:
    """Load URIEL features via lang2vec and return a stacked matrix.

    Features marked unknown (``--``) in *any* language are dropped, so the
    returned matrix has no missing values. This is the conservative choice.
    """
    try:
        import lang2vec.lang2vec as l2v
    except ImportError as e:
        raise ImportError(
            "lang2vec is required for typological prior computation. "
            "Install with: pip install lang2vec"
        ) from e

    # Map corpus-side codes (e.g., "zho") to URIEL-side codes (e.g., "cmn")
    uriel_codes = [_to_uriel_code(l) for l in languages]
    LOG.info("URIEL query codes: %s -> %s", list(languages), uriel_codes)

    feature_blocks: list[np.ndarray] = []
    for fs in feature_sets:
        # lang2vec returns a dict {lang_code: list_of_floats_or_'--'}
        raw = l2v.get_features(uriel_codes, fs)
        # Stack into a matrix: rows = languages (in input order), columns = features
        matrix_str = np.array([raw[uc] for uc in uriel_codes], dtype=object)
        # Mask: True where the feature is known across ALL languages
        is_known = np.all(matrix_str != "--", axis=0)
        if not is_known.any():
            LOG.warning("Feature set %s has no features known across all languages.", fs)
            continue
        kept = matrix_str[:, is_known].astype(np.float32)
        LOG.info(
            "Feature set %s: %d / %d features retained after dropping unknowns.",
            fs,
            kept.shape[1],
            matrix_str.shape[1],
        )
        feature_blocks.append(kept)

    if not feature_blocks:
        raise RuntimeError(
            "No usable typological features after dropping unknowns. "
            f"Languages: {languages}, feature_sets: {feature_sets}"
        )

    full = np.concatenate(feature_blocks, axis=1)
    return full, full.shape[1]


def _pairwise_cosine_distance(x: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between rows of x. Result is symmetric, zero diagonal.

    Cosine distance is 1 - cosine_similarity, clipped to [0, 2].
    """
    n = x.shape[0]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero (a zero-norm row would mean all-zero features).
    norms = np.where(norms == 0.0, 1e-12, norms)
    xn = x / norms
    sim = xn @ xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def _softmax(z: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax(z / temperature)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = z / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def _apply_floor(p: np.ndarray, floor: float) -> np.ndarray:
    """Project ``p`` onto the simplex with a per-coordinate floor.

    Uses the linear-shrinkage projection ``floor + (1 - K*floor) * p_normalized``.
    Guarantees ``result.sum() == 1`` and ``result_i >= floor`` exactly. See
    ``taam.exp3._apply_floor`` for the rationale.
    """
    if floor < 0 or floor * len(p) > 1.0 + 1e-9:
        raise ValueError(
            f"Invalid floor={floor} for {len(p)} languages "
            f"(floor * n must be <= 1)."
        )
    if np.any(p < 0):
        raise ValueError("p must be non-negative.")
    s = p.sum()
    if s <= 0:
        raise ValueError("p must have positive sum.")
    K = len(p)
    p_norm = p / s
    return floor + (1.0 - floor * K) * p_norm


def compute_typological_prior(
    languages: Sequence[str] = ("eng", "nld", "zho"),
    feature_sets: Sequence[str] = DEFAULT_FEATURE_SETS,
    temperature: float = DEFAULT_TEMPERATURE,
    floor: float = DEFAULT_FLOOR,
) -> TypologicalPrior:
    """End-to-end URIEL prior derivation.

    Args:
        languages: ISO 639-3 codes; ordering is preserved throughout.
        feature_sets: URIEL feature families to concatenate.
        temperature: softmax temperature applied to isolation scores.
        floor: minimum probability per language (renormalized after clipping).

    Returns:
        TypologicalPrior with derived pi_0 and diagnostics.

    Raises:
        ImportError: if lang2vec is not installed.
        RuntimeError: if no usable features remain after dropping unknowns.
    """
    languages = tuple(languages)
    feature_sets = tuple(feature_sets)

    feats, n_features = _load_feature_vectors(languages, feature_sets)
    LOG.info("Total features assembled: %d (languages=%s).", n_features, languages)

    d = _pairwise_cosine_distance(feats)
    # Isolation: mean distance to other languages (diagonal is 0, so divide by n-1)
    n = len(languages)
    if n < 2:
        raise ValueError("Need >=2 languages to define typological isolation.")
    isolation = d.sum(axis=1) / (n - 1)

    pi = _softmax(isolation, temperature=temperature)
    pi = _apply_floor(pi, floor=floor)

    LOG.info("Derived pi_0: %s", dict(zip(languages, pi.tolist())))

    return TypologicalPrior(
        languages=languages,
        pi_0=pi,
        isolation=isolation,
        pairwise_distance=d,
        feature_sets=feature_sets,
        n_features_used=n_features,
        temperature=temperature,
        floor=floor,
    )
