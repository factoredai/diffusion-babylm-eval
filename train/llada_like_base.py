"""
Minimal LLada-like trainer

Trains a bidirectional denoiser on the official BabyLM corpus, then
saves it so the evaluation pipeline can load it.
"""

import itertools

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DATASET_REPO = "BabyLM-community/BabyLM-2026-Strict-Small"
TOKENIZER = "BabyLM-community/babylm-baseline-10m-gpt-bert-mixed"
SEQ_LEN, BATCH, STEPS, LR = (
    512,
    16,  # 24 is the MPS ceiling at seq-512 on ~19GB unified memory
    200,  # smoke run
    3e-4,
)
OUTPUT_DIR = "ckpt/diffusion-babylm"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def build_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(TOKENIZER)
    assert isinstance(tok.pad_token_id, int)  # someone help pyright
    cfg = AutoConfig.from_pretrained(
        "roberta-base",
        vocab_size=len(tok),
        # roberta-base defaults to pad_token_id=1 but id 1 in this tokenizer is <s>/<cls>
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        # RoBERTa offsets pos embeddings by pad_token_id
        max_position_embeddings=SEQ_LEN + tok.pad_token_id + 1,
        num_hidden_layers=12,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
    )
    model: PreTrainedModel = AutoModelForMaskedLM.from_config(cfg).to(DEVICE)
    return tok, model


def make_loader(tok: PreTrainedTokenizerBase) -> DataLoader:
    ds = load_dataset(DATASET_REPO, split="train")
    assert isinstance(ds, Dataset)  # narrow load_dataset's type

    # Tokenize every row (keeping each <s>...</s> as a
    # natural separator) and pack the token stream into contiguous SEQ_LEN blocks
    # so nothing is dropped except a sub-block remainder at each map-batch boundary.
    ds = ds.map(
        lambda b: tok(b["text"]),
        batched=True,
        remove_columns=ds.column_names,
    )
    cols = ds.column_names

    def pack(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        ids = list(itertools.chain.from_iterable(batch["input_ids"]))
        n = (len(ids) // SEQ_LEN) * SEQ_LEN
        blocks = [ids[i : i + SEQ_LEN] for i in range(0, n, SEQ_LEN)]
        return {"input_ids": blocks, "attention_mask": [[1] * SEQ_LEN for _ in blocks]}

    ds = ds.map(pack, batched=True, batch_size=10_000, remove_columns=cols)
    ds.set_format("torch", columns=["input_ids", "attention_mask"])
    return DataLoader(
        ds,  # pyright: ignore
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
    )


def diffuse(ids: torch.Tensor, mask_token_id: str):
    """mask each token w.p. t, with t ~ U(0, 1] per sequence."""
    t = torch.rand(ids.size(0), 1, device=ids.device).clamp_min(1e-3)
    mask = torch.rand_like(ids, dtype=torch.float) < t
    noisy = torch.where(mask, int(mask_token_id), ids)
    return noisy, mask, t.squeeze(1)


def diffusion_loss(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    ids: torch.Tensor,
    attn: torch.Tensor,
) -> torch.Tensor:
    """LLaDA/MDLM loss: (1/t)-weighted cross-entropy on masked positions only. Normalize by B*L to make it unbiased."""
    mask_id = tok.mask_token_id
    noisy, mask, t = diffuse(ids, mask_id)  # pyright: ignore
    logits = model(input_ids=noisy, attention_mask=attn).logits
    ce = F.cross_entropy(logits[mask], ids[mask], reduction="none")  # per masked token
    w = (1.0 / t).repeat_interleave(mask.sum(dim=1))  # 1/t weight per token
    return (w * ce).sum() / ids.numel()  # normalize by B*L


def main():
    tok, model = build_model()
    loader = make_loader(tok)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    step = 0
    while step < STEPS:
        for batch in loader:
            ids, attn = (
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
            )
            loss = diffusion_loss(model, tok, ids, attn)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"step {step}  loss {loss.item():.4f}")
            step += 1
            if step >= STEPS:
                break

    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
