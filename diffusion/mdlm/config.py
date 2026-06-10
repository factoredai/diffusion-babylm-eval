"""HuggingFace config for the masked-diffusion language model.

We subclass ``PretrainedConfig`` so the model integrates with the HF ecosystem:
``save_pretrained`` / ``from_pretrained``, the Hub, and ``trust_remote_code``
loading from the uploaded modeling file.

Defaults mirror the BabyLM-2026 Strict-Small GPT-2 baseline footprint so the
diffusion vs. autoregressive comparison stays grounded at the same model scale
(see the project proposal). Individual experiment configs override only the keys
they need; see ``configs/*.yaml``.
"""
from __future__ import annotations

from transformers import PretrainedConfig


class MaskedDiffusionConfig(PretrainedConfig):
    """Config for :class:`mdlm.model.MaskedDiffusionLM`.

    The model is a *bidirectional* Transformer (no causal mask) trained with an
    absorbing-state masked-diffusion objective. The extra ``[MASK]`` token is the
    absorbing state of the forward process and lives at index ``vocab_size`` of
    the embedding table (so ``mask_token_id == vocab_size`` by convention).
    """

    model_type = "masked_diffusion_lm"

    # Standard HF aliases (same trick as GPT2Config). The official GLUE
    # fine-tuning harness reads `config.hidden_size` to size its classifier
    # head, so these must resolve even though we store GPT-2-style names.
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size: int = 16_384,
        n_positions: int = 1_024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        # Special tokens. The mask token is appended *after* the real vocab.
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 3,
        mask_token_id: int | None = None,
        # ── Masked-diffusion specific ──────────────────────────────────────
        # Lower bound on the per-sequence masking ratio t ~ U(t_min, t_max).
        # t_min > 0 keeps the 1/t loss weight finite.
        t_min: float = 1e-3,
        t_max: float = 1.0,
        # "Reasoning depth" extension from the proposal: duplicate the middle
        # transformer blocks at *inference* time to add capacity without new
        # parameters. 1 = no duplication (default for the MVP).
        layer_duplication_factor: int = 1,
        # If set, training masks high-frequency tokens less often (frequency-
        # informed masking). None = uniform masking (standard MDLM/LLaDA).
        frequency_informed_masking: bool = False,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.ffn_mult = ffn_mult
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        # By convention the absorbing [MASK] state sits right after the vocab.
        self.mask_token_id = vocab_size if mask_token_id is None else mask_token_id
        self.t_min = t_min
        self.t_max = t_max
        self.layer_duplication_factor = layer_duplication_factor
        self.frequency_informed_masking = frequency_informed_masking

        # Let the Hub load the custom code with trust_remote_code=True. The
        # uploader (scripts/upload_to_hf.py) copies config.py + model.py next to
        # the weights, so these dotted paths resolve on the Hub.
        # - AutoModelForMaskedLM: the zero-shot `mlm` backend scores the denoiser
        #   exactly like a masked LM (logits head).
        # - AutoModel: the GLUE fine-tuning harness loads a *headless* encoder
        #   and expects last_hidden_state, hence the separate class.
        self.auto_map = {
            "AutoConfig": "config.MaskedDiffusionConfig",
            "AutoModel": "model.MaskedDiffusionModel",
            "AutoModelForMaskedLM": "model.MaskedDiffusionLM",
        }

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_embeddings(self) -> int:
        """Embedding rows = real vocab + 1 absorbing [MASK] state."""
        return self.vocab_size + 1
