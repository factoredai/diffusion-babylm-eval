"""The forward (noising) process and the masked-diffusion training loss.

We use the *absorbing-state* discrete diffusion formulation (MDLM, Sahoo et al.
2024; LLaDA, Nie et al. 2025): the forward process replaces tokens with a special
``[MASK]`` absorbing state at a rate controlled by a continuous time variable
``t``. The model is trained to denoise — predict the original tokens at the
masked positions — and the per-example loss is reweighted by ``1/t`` so the Monte
Carlo estimate is an unbiased bound on the negative log-likelihood.

Key practical points for the BabyLM Strict-Small budget:
    * Each *forward pass* sees a fresh random masking pattern, so a single pass
      over the 10M-word corpus already trains over many token orderings. This is
      the "implicit data augmentation" that lets diffusion trade compute for data
      (arXiv:2507.15857) — exactly the regime the CFP's 10-epoch / 100M-words-seen
      limit places us in.
    * Optional frequency-informed masking down-weights the masking probability of
      very frequent tokens, focusing the denoising signal on rarer, more
      informative words.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaskingProcess:
    """Samples the absorbing-state forward process for a batch of sequences.

    Args:
        mask_token_id: id of the absorbing ``[MASK]`` state (``config.vocab_size``).
        t_min, t_max: per-sequence masking ratio is sampled ``t ~ U(t_min, t_max)``.
        pad_token_id: padding never gets masked and never contributes to the loss.
        token_log_freq: optional 1-D tensor of per-vocab log-frequencies used for
            frequency-informed masking (None = uniform masking).
        freq_strength: how strongly frequency modulates the per-token mask prob.
    """

    mask_token_id: int
    t_min: float = 1e-3
    t_max: float = 1.0
    pad_token_id: int = 3
    token_log_freq: torch.Tensor | None = None
    freq_strength: float = 0.5

    def __call__(
        self, input_ids: torch.LongTensor, generator: torch.Generator | None = None
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        """Corrupt ``input_ids``.

        Returns:
            corrupted: (B, T) input ids with some positions set to ``mask_token_id``.
            labels:    (B, T) original ids at masked positions, ``-100`` elsewhere.
            weight:    (B, 1) per-sequence loss weight ``1/t`` for the MDLM bound.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # One masking ratio t per sequence.
        t = torch.empty(B, 1, device=device).uniform_(self.t_min, self.t_max, generator=generator)

        prob = t.expand(B, T).clone()
        if self.token_log_freq is not None:
            # Frequent tokens (high log-freq) get masked a bit less often.
            table = self.token_log_freq.to(device)
            ids = input_ids.clamp(min=0, max=table.numel() - 1)
            lf = table[ids]
            lf_norm = (lf - lf.mean()) / (lf.std() + 1e-6)
            prob = (prob * (1.0 - self.freq_strength * torch.tanh(lf_norm))).clamp(0.0, 1.0)

        rand = torch.rand(B, T, device=device, generator=generator)
        mask = rand < prob

        # Never mask padding; always keep at least one masked token per sequence
        # so every example contributes a gradient.
        is_pad = input_ids == self.pad_token_id
        mask = mask & ~is_pad
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            # Force-mask the first non-pad token of any all-unmasked row.
            for b in torch.nonzero(empty_rows, as_tuple=False).flatten().tolist():
                nonpad = torch.nonzero(~is_pad[b], as_tuple=False).flatten()
                if nonpad.numel() > 0:
                    mask[b, nonpad[0]] = True

        corrupted = input_ids.clone()
        corrupted[mask] = self.mask_token_id
        labels = torch.full_like(input_ids, -100)
        labels[mask] = input_ids[mask]
        weight = 1.0 / t  # (B, 1)
        return corrupted, labels, weight


def diffusion_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """MDLM/LLaDA reweighted denoising cross-entropy.

    Args:
        logits: (B, T, V) model outputs.
        labels: (B, T) original tokens at masked positions, ``-100`` elsewhere.
        weight: (B, 1) per-sequence ``1/t`` weight from :class:`MaskingProcess`.

    Returns:
        Scalar loss: mean over masked tokens of ``weight * CE``.
    """
    B, T, V = logits.shape
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="none"
    ).view(B, T)
    masked = labels != -100
    per_seq_weight = weight.expand(B, T)
    num = (ce * per_seq_weight * masked).sum()
    den = masked.sum().clamp(min=1)
    return num / den
