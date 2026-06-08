"""TAAM — Typology-Aware Adaptive Mixing.

A method combining (a) a URIEL-derived typological prior over initial sampling
probabilities, (b) EXP3 online updates with a normalized excess-loss reward,
and (c) byte-premium-adjusted budgeting, for the BabyLM 2026 Multilingual
Track (EN+NL+ZH).

See ``improved_research_context_v2.md`` for the canonical project spec.
"""

__version__ = "0.1.0"

LANGUAGES = ("eng", "nld", "zho")

BYTE_PREMIUM_HF = {
    "eng": 1.000000,
    "nld": 1.051606,
    "zho": 0.935966,
}

BYTE_PREMIUM_CFP = {
    "eng": 1.0000,
    "nld": 1.0516,
    "zho": 0.9894,
}

DEFAULT_BYTE_PREMIUM = BYTE_PREMIUM_HF
