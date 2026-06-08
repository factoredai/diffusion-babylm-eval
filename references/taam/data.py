"""Multilingual data pipeline for TAAM.

Components:
    PackedLanguageStream
        Reads tokenized text for ONE language and yields fixed-length blocks
        (LM-packed: documents are concatenated with EOS separators, then sliced
         into ``block_size``-token chunks). Pure-Python; works on CPU.

    MultilingualMixer
        Wraps {language -> PackedLanguageStream} and exposes ``next_batch``
        which uses an external ``pi`` distribution (from EXP3 or a static
        config) to select a language for each batch.

    LanguageBatch
        NamedTuple carrying ``input_ids``, ``labels``, ``language`` for a single
        monolingual batch. Returns torch tensors when ``return_tensors='pt'``.

Why batch-level mixing (not within-batch)?
    Within-batch mixing requires per-sample language IDs and shorter packed
    contexts; it adds engineering complexity for negligible expected gain at
    this scale. Batch-level mixing is the convention in ODM / DoReMi / mmBERT.

CPU smoke test:
    ``python -m taam.data --smoke-test`` generates a tiny synthetic corpus
    in ``data/_synthetic/`` and exercises the loader end-to-end.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

DEFAULT_EOS_TOKEN_ID = 0  # placeholder; overwritten by the tokenizer


@dataclass
class LanguageBatch:
    """A single monolingual batch of language-model packed tokens."""

    input_ids: np.ndarray   # shape (batch_size, block_size), dtype int64
    labels: np.ndarray      # usually input_ids shifted (or identical, depending on model)
    language: str
    step: int               # global training step at which this batch was produced

    def as_torch(self):
        """Convert ``input_ids`` and ``labels`` to torch tensors (lazy import)."""
        import torch
        return {
            "input_ids": torch.as_tensor(self.input_ids, dtype=torch.long),
            "labels": torch.as_tensor(self.labels, dtype=torch.long),
            "language": self.language,
            "step": self.step,
        }


class PackedLanguageStream:
    """Infinite stream of LM-packed blocks for ONE language.

    Args:
        token_id_files: list of paths to .npy files of int32 token ids
            (e.g., produced by ``scripts/train_tokenizer.py`` + tokenization).
        block_size: number of tokens per packed block.
        eos_token_id: token id used as document separator. We append one EOS
            between concatenated documents.
        shuffle_buffer: how many documents to shuffle at once when streaming.
            Larger -> more randomness, more memory. 0 = no shuffling.
        seed: RNG seed.

    Notes:
        - We pre-load ALL .npy files into memory and concatenate. For BabyLM
          scale (~33M tokens per language, ~70MB int32) this is fine. For
          larger scale, replace this with a memory-mapped streaming iterator.
        - The stream wraps around at end-of-data and reshuffles.
    """

    def __init__(
        self,
        token_id_files: Sequence[Path | str],
        block_size: int = 512,
        eos_token_id: int = DEFAULT_EOS_TOKEN_ID,
        shuffle_buffer: int = 1024,
        seed: int = 0,
    ) -> None:
        self.block_size = int(block_size)
        self.eos_token_id = int(eos_token_id)
        self.shuffle_buffer = int(shuffle_buffer)
        self._rng = np.random.default_rng(seed)
        self._token_id_files = [Path(p) for p in token_id_files]
        if not self._token_id_files:
            raise ValueError("Need at least one .npy file of token ids.")
        self._documents: list[np.ndarray] = []
        for p in self._token_id_files:
            if not p.exists():
                raise FileNotFoundError(p)
            arr = np.load(p)
            if arr.ndim != 1:
                # Some pipelines store ragged docs as a single concatenated 1-D
                # array; that's exactly what we want here.
                arr = arr.ravel()
            self._documents.append(arr.astype(np.int64))
        self._doc_index = list(range(len(self._documents)))
        self._build_concat_buffer()

    def _build_concat_buffer(self) -> None:
        """Concatenate all docs with EOS separators into one long buffer."""
        if self.shuffle_buffer > 0:
            self._rng.shuffle(self._doc_index)
        chunks = []
        for i in self._doc_index:
            chunks.append(self._documents[i])
            chunks.append(np.array([self.eos_token_id], dtype=np.int64))
        self._buffer = np.concatenate(chunks)
        self._cursor = 0

    def total_tokens(self) -> int:
        return int(self._buffer.size)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            if self._cursor + self.block_size > self._buffer.size:
                # End of buffer: reshuffle + start over
                self._build_concat_buffer()
            block = self._buffer[self._cursor : self._cursor + self.block_size]
            self._cursor += self.block_size
            yield block

    def get_batch(self, batch_size: int) -> np.ndarray:
        """Return one batch of shape (batch_size, block_size)."""
        rows = []
        it = iter(self)
        for _ in range(batch_size):
            rows.append(next(it))
        return np.stack(rows, axis=0)


class MultilingualMixer:
    """Glue layer: per-language streams + a sampling distribution -> batches.

    Args:
        streams: dict {language: PackedLanguageStream}
        batch_size: number of sequences per batch
        seed: RNG seed for language sampling

    Usage:
        mixer = MultilingualMixer(streams={...}, batch_size=64, seed=42)
        for step in range(N):
            pi = {"eng": 0.32, "nld": 0.32, "zho": 0.36}  # from EXP3 or static
            batch = mixer.next_batch(pi=pi, step=step)
            # batch is a LanguageBatch; use batch.as_torch() to move to GPU
    """

    def __init__(
        self,
        streams: dict[str, PackedLanguageStream],
        batch_size: int,
        seed: int = 0,
    ) -> None:
        self.streams = streams
        self.batch_size = int(batch_size)
        self._rng = np.random.default_rng(seed)
        self.languages = tuple(streams.keys())

    def _sample_language(self, pi: dict[str, float]) -> str:
        # Build pi vector in the same order as self.languages
        p = np.array([pi[l] for l in self.languages], dtype=np.float64)
        if abs(p.sum() - 1.0) > 1e-4:
            LOG.warning("pi does not sum to 1 (sum=%.6f); renormalizing.", p.sum())
            p = p / p.sum()
        idx = self._rng.choice(len(self.languages), p=p)
        return self.languages[idx]

    def next_batch(self, pi: dict[str, float], step: int) -> LanguageBatch:
        lang = self._sample_language(pi)
        ids = self.streams[lang].get_batch(self.batch_size)
        # For causal LM, labels = input_ids (the loss masks the first token internally).
        return LanguageBatch(input_ids=ids, labels=ids.copy(), language=lang, step=step)

    def total_tokens(self) -> dict[str, int]:
        return {l: self.streams[l].total_tokens() for l in self.languages}


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test utilities (no GPU, no HF auth required)
# ──────────────────────────────────────────────────────────────────────────────


def _make_synthetic_tokens(language: str, n_tokens: int, vocab_size: int, seed: int) -> np.ndarray:
    """Generate fake but reproducible token ids for smoke testing.

    Different languages use different token-ID *bands* so a downstream
    test can verify the mixer is honoring the language choice.

    Mapping:
        eng → tokens in [10, 110)
        nld → tokens in [110, 210)
        zho → tokens in [210, 310)
        else → tokens in [310, 410)
    """
    bands = {"eng": (10, 110), "nld": (110, 210), "zho": (210, 310)}
    lo, hi = bands.get(language, (310, 410))
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=n_tokens, dtype=np.int64)


def make_synthetic_corpus(
    output_dir: Path,
    languages: Sequence[str] = ("eng", "nld", "zho"),
    n_tokens_per_lang: int = 50_000,
    vocab_size: int = 32_000,
    seed: int = 0,
) -> dict[str, list[Path]]:
    """Create a tiny synthetic per-language token corpus on disk.

    Returns {language: [Path(.npy file)]}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[Path]] = {}
    for i, lang in enumerate(languages):
        arr = _make_synthetic_tokens(lang, n_tokens_per_lang, vocab_size, seed + i)
        f = output_dir / f"{lang}_synthetic.npy"
        np.save(f, arr)
        out[lang] = [f]
    # Write a small manifest
    manifest = {
        "languages": list(languages),
        "n_tokens_per_lang": n_tokens_per_lang,
        "vocab_size": vocab_size,
        "seed": seed,
        "files": {l: [str(p) for p in v] for l, v in out.items()},
        "purpose": "Synthetic smoke-test corpus. NOT for paper experiments.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def smoke_test() -> int:
    """Exercise the data pipeline end-to-end with synthetic data."""
    print("=== taam.data smoke test ===")
    synth_dir = Path("data/_synthetic")
    files = make_synthetic_corpus(
        synth_dir,
        languages=("eng", "nld", "zho"),
        n_tokens_per_lang=200_000,
        seed=42,
    )
    print(f"Wrote synthetic corpus to {synth_dir}/")
    for l, ps in files.items():
        print(f"  {l}: {[p.name for p in ps]}")

    streams = {
        l: PackedLanguageStream(token_id_files=files[l], block_size=128, eos_token_id=0, seed=42)
        for l in files
    }
    print("\nPer-language total tokens (including EOS separators):")
    for l, s in streams.items():
        print(f"  {l}: {s.total_tokens():,}")

    mixer = MultilingualMixer(streams=streams, batch_size=4, seed=42)
    pi = {"eng": 0.20, "nld": 0.30, "zho": 0.50}
    print(f"\nSampling 1000 batches with pi={pi} ...")
    counts: dict[str, int] = {l: 0 for l in pi}
    for step in range(1000):
        batch = mixer.next_batch(pi=pi, step=step)
        counts[batch.language] += 1
    total = sum(counts.values())
    print("Observed batch-language distribution (should approximate pi):")
    ok = True
    for l, c in counts.items():
        share = c / total
        diff = abs(share - pi[l])
        status = "✓" if diff < 0.03 else "✗"
        if status == "✗":
            ok = False
        print(f"  {status} {l}: {c:4d} batches ({share:.3f})   target {pi[l]:.3f}  diff {diff:.3f}")

    # Also verify that batches from each language stay in the expected ID band
    print("\nVerifying batch-language correctness (token IDs in expected band):")
    bands = {"eng": (10, 110), "nld": (110, 210), "zho": (210, 310)}
    for l in pi:
        deg_pi = {k: (1.0 if k == l else 0.0) for k in pi}
        # _sample_language renormalizes if sum != 1, but a fully zeroed mass on
        # K-1 langs is fine since np.random.choice handles p=0 entries.
        batch = mixer.next_batch(pi=deg_pi, step=0)
        lo, hi = bands[l]
        # Skip EOS tokens (id=0) when checking band membership.
        ids = batch.input_ids
        non_eos = ids[ids != 0]
        in_band = (non_eos >= lo) & (non_eos < hi)
        frac = in_band.mean() if non_eos.size else 1.0
        status = "✓" if frac > 0.95 else "✗"
        if status == "✗":
            ok = False
        print(f"  {status} {l}: {frac:.3%} of non-EOS tokens are in band [{lo}, {hi})")

    print()
    if ok:
        print("smoke test PASSED.")
        return 0
    print("smoke test FAILED.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        return smoke_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
