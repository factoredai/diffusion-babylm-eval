"""Sequence scoring for the masked-diffusion model.

The official BabyLM zero-shot tasks (BLiMP, BLiMP-supplement, EWoK, COMPS,
Entity Tracking) are all *minimal-pair* tasks: the model must assign a higher
score to the correct sentence than to the incorrect one. Autoregressive models
score with the chain-rule log-likelihood, but a masked-diffusion model has no
left-to-right factorization, so we provide two diffusion-native scorers:

    pll  : pseudo-log-likelihood (Salazar et al., 2020). Mask each position once,
           read off log p(true_token | rest), and sum. Deterministic, O(T) forward
           passes per sequence, and the strongest minimal-pair scorer in practice.

    elbo : Monte-Carlo estimate of the MDLM negative-ELBO bound on log p(x).
           Cheaper (fixed number of forward passes) but noisier.

Both return a value where **higher == more likely**, so the calling code simply
checks ``score(good) > score(bad)``.

The CFP requires that a submission "provides a function to score a sequence of
words without the need for additional fine-tuning" — that is exactly :func:`score`.
"""
from __future__ import annotations

import torch

from .masking import MaskingProcess


@torch.no_grad()
def score_pll(
    model,
    input_ids: torch.LongTensor,
    *,
    special_ids: set[int] | None = None,
    positions: list[int] | None = None,
    max_positions: int | None = None,
    layer_duplication_factor: int | None = None,
    batch_size: int = 64,
) -> float:
    """Pseudo-log-likelihood of one sequence (sum of per-position log-probs).

    Args:
        model: a :class:`mdlm.model.MaskedDiffusionLM` in eval mode.
        input_ids: (T,) token ids for a single sequence (no batch dim).
        special_ids: ids never scored/masked (bos/eos/pad). Defaults to the
            model config's special tokens.
        positions: optional explicit list of positions to score (e.g. only the
            completion span). If None, all non-special positions are scored.
        max_positions: optionally cap the number of scored positions (speed).
        layer_duplication_factor: inference-time reasoning-depth override.
        batch_size: how many single-position-masked copies to run at once.
    """
    device = next(model.parameters()).device
    cfg = model.config
    mask_id = cfg.mask_token_id
    if special_ids is None:
        special_ids = {cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id}

    ids = input_ids.to(device)
    T = ids.shape[0]
    if positions is None:
        positions = [i for i in range(T) if int(ids[i]) not in special_ids]
    else:
        positions = [i for i in positions if 0 <= i < T]
    if max_positions is not None and len(positions) > max_positions:
        # Evenly subsample positions to bound cost on very long sequences.
        step = len(positions) / max_positions
        positions = [positions[int(i * step)] for i in range(max_positions)]
    if not positions:
        return 0.0

    total_logprob = 0.0
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = ids.unsqueeze(0).repeat(len(chunk), 1)
        for row, pos in enumerate(chunk):
            batch[row, pos] = mask_id
        logits = model(input_ids=batch, layer_duplication_factor=layer_duplication_factor).logits  # (n, T, V)
        log_probs = torch.log_softmax(logits[:, :, : cfg.vocab_size], dim=-1)
        for row, pos in enumerate(chunk):
            total_logprob += float(log_probs[row, pos, ids[pos]].item())
    return total_logprob


@torch.no_grad()
def score_elbo(
    model,
    input_ids: torch.LongTensor,
    *,
    n_samples: int = 16,
    seed: int = 0,
) -> float:
    """Monte-Carlo negative-ELBO score (higher == more likely)."""
    device = next(model.parameters()).device
    cfg = model.config
    ids = input_ids.to(device).unsqueeze(0).repeat(n_samples, 1)
    proc = MaskingProcess(
        mask_token_id=cfg.mask_token_id,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        pad_token_id=cfg.pad_token_id,
    )
    g = torch.Generator(device=device).manual_seed(seed)
    corrupted, labels, weight = proc(ids, generator=g)
    logits = model(input_ids=corrupted).logits
    B, T, V = logits.shape
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="none"
    ).view(B, T)
    masked = labels != -100
    per_seq = (ce * weight.expand(B, T) * masked).sum(dim=1) / masked.sum(dim=1).clamp(min=1)
    # per_seq is a negative-log-likelihood bound; negate so higher == better.
    return float((-per_seq).mean().item())


def score(model, input_ids: torch.LongTensor, method: str = "pll", **kwargs) -> float:
    """Dispatch to the requested scorer (``"pll"`` or ``"elbo"``)."""
    if method == "pll":
        return score_pll(model, input_ids, **kwargs)
    if method == "elbo":
        return score_elbo(model, input_ids, **kwargs)
    raise ValueError(f"Unknown scoring method: {method!r} (use 'pll' or 'elbo').")
