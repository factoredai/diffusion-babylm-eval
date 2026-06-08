"""Bidirectional Transformer for masked-diffusion language modeling.

The model is intentionally simple and self-contained: a token + learned
positional embedding, a stack of pre-norm bidirectional Transformer blocks, and
a tied LM head. There is **no causal mask** — the masked-diffusion objective
lets every position attend to every other position, which is the source of the
"implicit data augmentation over token orderings" that makes diffusion shine in
data-constrained regimes (Prabhudesai et al., 2025, arXiv:2507.15857).

It subclasses ``PreTrainedModel`` so ``save_pretrained`` / ``from_pretrained``
and Hub uploads work out of the box. The forward pass returns logits over the
real vocabulary **plus** the absorbing ``[MASK]`` column (index ``vocab_size``);
the mask column is ignored when computing loss and when scoring.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from .config import MaskedDiffusionConfig


class TransformerBlock(nn.Module):
    """Pre-norm bidirectional Transformer block (MHSA + GEGLU-free MLP)."""

    def __init__(self, config: MaskedDiffusionConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        hidden = config.ffn_mult * config.n_embd
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.ln_1(x)
        # need_weights=False keeps the attention fast; mask is True where padded.
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class MaskedDiffusionLM(PreTrainedModel):
    """A bidirectional Transformer trained with an absorbing-state diffusion loss."""

    config_class = MaskedDiffusionConfig
    base_model_prefix = "mdlm"
    supports_gradient_checkpointing = False
    # Tells HF that lm_head.weight is tied to the input embedding (so
    # save_pretrained does not treat it as an illegal shared tensor).
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MaskedDiffusionConfig) -> None:
        super().__init__(config)
        self.config = config

        # +1 embedding row for the absorbing [MASK] state at index vocab_size.
        self.tok_emb = nn.Embedding(config.num_embeddings, config.n_embd)
        self.pos_emb = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.num_embeddings, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.post_init()

    # ── HF plumbing ────────────────────────────────────────────────────────
    def get_input_embeddings(self) -> nn.Module:
        return self.tok_emb

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.tok_emb = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        layer_duplication_factor: int | None = None,
    ) -> MaskedLMOutput:
        """Run the bidirectional encoder.

        Args:
            input_ids: (B, T) token ids, may contain ``mask_token_id``.
            attention_mask: (B, T) 1 for real tokens, 0 for padding.
            labels: (B, T) original tokens at masked positions, ``-100`` elsewhere.
                If given, a plain (unweighted) cross-entropy is returned in
                ``loss``. The weighted MDLM loss lives in ``masking.diffusion_loss``
                and is used by the training loop instead.
            layer_duplication_factor: optional inference-time "reasoning depth"
                override (repeats the middle blocks). Defaults to the config value.

        Returns:
            ``MaskedLMOutput`` with ``logits`` of shape (B, T, vocab_size + 1).
        """
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        # MultiheadAttention expects True where a position should be *ignored*.
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        dup = layer_duplication_factor or self.config.layer_duplication_factor
        blocks = self._expanded_blocks(dup)
        for block in blocks:
            x = block(x, key_padding_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return MaskedLMOutput(loss=loss, logits=logits)

    def _expanded_blocks(self, dup: int):
        """Return the block sequence, optionally repeating the middle blocks.

        With ``dup == 1`` this is just ``self.blocks``. With ``dup > 1`` the
        interior blocks (all but the first and last) are applied ``dup`` times,
        giving extra "reasoning depth" at no parameter cost — the duplicated-layer
        idea from the proposal's COMPS / entity-tracking hypotheses.
        """
        if dup <= 1 or self.config.n_layer <= 2:
            return self.blocks
        first, *middle, last = list(self.blocks)
        return [first, *(middle * dup), last]
