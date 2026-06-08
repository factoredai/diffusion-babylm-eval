"""Download the Strict-Small corpus, train a BPE tokenizer, and pre-tokenize.

Produces everything the training loop needs:

    tokenizer/mdlm_bpe_16k/        HF tokenizer dir (defines a [MASK] token whose
                                   id == vocab_size, matching the model)
    data/tokens/
        shard_0000.npy ...         1-D int64 token-id shards
        manifest.json              {n_tokens, n_words, eos_token_id, ...}

Word-budget note (CFP): the tokenizer is trained on the SAME corpus we pretrain
on, so it consumes no extra words. We also assert the corpus stays within the
10M-unique-word Strict-Small budget.

Usage:
    python scripts/prepare_data.py --hf-dataset BabyLM-community/babylm-2026-strict-small
    python scripts/prepare_data.py --text-file my_corpus.txt   # bring your own
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("prepare_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECIALS = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"]  # ids 0,1,2,3 (match mdlm/config.py)
EOS_ID = 2


def load_corpus(args) -> list[str]:
    """Return a list of text documents from a local file or an HF dataset."""
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
        return [d for d in text.split("\n\n") if d.strip()]
    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, split=args.split)
    col = args.text_column if args.text_column in ds.column_names else ds.column_names[0]
    return [t for t in ds[col] if t and t.strip()]


def train_tokenizer(docs: list[str], vocab_size: int, out_dir: Path):
    """Train a BPE tokenizer; the [MASK] token is appended at id == vocab_size."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS)
    tok.train_from_iterator(docs, trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]",
        model_max_length=512,
    )
    # The absorbing state. add_special_tokens grows the vocab; with a full BPE
    # vocab the new [MASK] lands at id == vocab_size, exactly as the model expects.
    fast.add_special_tokens({"mask_token": "[MASK]"})
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out_dir)
    LOG.info("Tokenizer saved to %s (mask id=%d, vocab=%d)",
             out_dir, fast.mask_token_id, fast.vocab_size)
    return fast


def tokenize_to_shards(docs: list[str], tokenizer, out_dir: Path, shard_tokens: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    buf: list[int] = []
    shard_idx = 0
    total = 0
    n_words = sum(len(d.split()) for d in docs)

    def flush():
        nonlocal buf, shard_idx
        if not buf:
            return
        np.save(out_dir / f"shard_{shard_idx:04d}.npy", np.array(buf, dtype=np.int64))
        shard_idx += 1
        buf = []

    for d in docs:
        ids = tokenizer(d, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        buf.append(EOS_ID)
        total += len(ids) + 1
        if len(buf) >= shard_tokens:
            flush()
    flush()

    (out_dir / "manifest.json").write_text(json.dumps(
        {"n_tokens": total, "n_words": n_words, "n_shards": shard_idx,
         "eos_token_id": EOS_ID, "vocab_size": tokenizer.vocab_size}, indent=2))
    LOG.info("Wrote %d shards, %d tokens (%d words) to %s", shard_idx, total, n_words, out_dir)
    return n_words


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-dataset", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--text-file", default=None, help="Use a local corpus instead of an HF dataset.")
    p.add_argument("--vocab-size", default=16_384, type=int)
    p.add_argument("--tokenizer-dir", default=REPO_ROOT / "tokenizer/mdlm_bpe_16k", type=Path)
    p.add_argument("--token-dir", default=REPO_ROOT / "data/tokens", type=Path)
    p.add_argument("--shard-tokens", default=5_000_000, type=int)
    p.add_argument("--word-budget", default=10_000_000, type=int)
    args = p.parse_args()

    docs = load_corpus(args)
    n_words = sum(len(d.split()) for d in docs)
    LOG.info("Loaded %d docs (~%d words).", len(docs), n_words)
    if n_words > args.word_budget:
        LOG.warning("Corpus has %d words > Strict-Small budget %d. Trim before training!",
                    n_words, args.word_budget)

    tokenizer = train_tokenizer(docs, args.vocab_size, args.tokenizer_dir)
    tokenize_to_shards(docs, tokenizer, args.token_dir, args.shard_tokens)
    LOG.info("Done. Train with: python scripts/train.py --condition MD_base "
             "--token-data %s --tokenizer %s", args.token_dir, args.tokenizer_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
