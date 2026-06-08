"""Masked-Diffusion Language Model (MDLM) for the BabyLM 2026 Strict-Small track.

This package implements a LLaDA / MDLM-style *absorbing-state* discrete diffusion
language model on top of a GPT-2-scale bidirectional Transformer. It is the
research code behind Group #4's submission:

    "Masked-Diffusion BabyLM: Non-Autoregressive Objectives for
     Sample-Efficient Language Learning" (Strict-Small, English only).

Modules
-------
    config   : MaskedDiffusionConfig (HF PretrainedConfig subclass).
    model    : MaskedDiffusionLM (HF PreTrainedModel subclass, bidirectional).
    masking  : the forward (noising) process + the MDLM/LLaDA training loss.
    data     : English text streaming + a synthetic corpus for CPU smoke tests.
    scoring  : minimal-pair sequence scoring (pseudo-log-likelihood / ELBO),
               used by the custom diffusion evaluation backend.

The whole package is import-safe without a GPU and runs a CPU smoke test in
seconds (``python scripts/train.py --smoke-test``).
"""
from __future__ import annotations

# The single language for the Strict-Small track. Kept as a constant so the
# data / eval code reads the same way as the multilingual reference project.
LANGUAGE = "eng"

# Canonical name used in checkpoint directories and HF branch revisions.
TRACK = "strict-small"

from .config import MaskedDiffusionConfig  # noqa: E402,F401
from .masking import MaskingProcess, diffusion_loss  # noqa: E402,F401

__all__ = [
    "LANGUAGE",
    "TRACK",
    "MaskedDiffusionConfig",
    "MaskingProcess",
    "diffusion_loss",
]
