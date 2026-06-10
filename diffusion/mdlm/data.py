"""English text data pipeline for the Strict-Small masked-diffusion model.

The corpus (token shards produced by ``scripts/prepare_data.py``, which already
terminate every document with EOS in-stream) is concatenated, split into small
chunks, and served as fixed-length LM-packed blocks. Shuffling happens at three
granularities each epoch — chunk order, a random block-boundary offset, and
block serving order — so consecutive batches are decorrelated and no two epochs
present the same blocks in the same order. (A previous version treated each
multi-million-token *shard* as the shuffle unit and served blocks sequentially,
which made batches almost perfectly correlated.)

The dev slice is a stride-sample of chunks across the WHOLE corpus (all
domains), not the last shard.

For CPU smoke tests we generate a tiny synthetic corpus so the whole training /
eval pipeline can be exercised with no data download and no GPU.

Real-data convention:

    token_data_dir/
        shard_0000.npy        # 1-D int32/int64 token ids
        shard_0001.npy
        ...
        manifest.json         # {"n_tokens", "n_words", "eos_token_id", ...}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass
class Batch:
    """A batch of LM-packed token blocks."""

    input_ids: np.ndarray   # (batch_size, block_size) int64
    step: int

    def as_torch(self):
        import torch

        return torch.as_tensor(self.input_ids, dtype=torch.long)


class PackedTextStream:
    """Infinite stream of LM-packed blocks over a set of token chunks.

    ``documents`` may be ``.npy`` paths or in-memory int arrays. EOS separators
    are expected to already be present in the token stream (prepare_data.py
    appends one after every document), so none are inserted here.
    """

    def __init__(
        self,
        documents: Sequence[Path | str | np.ndarray],
        block_size: int = 512,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.block_size = int(block_size)
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(seed)
        self._documents: list[np.ndarray] = []
        for d in documents:
            if isinstance(d, (str, Path)):
                p = Path(d)
                if not p.exists():
                    raise FileNotFoundError(p)
                d = np.load(p)
            self._documents.append(np.asarray(d).ravel().astype(np.int64))
        if not self._documents:
            raise ValueError("Need at least one document of token ids.")
        self._doc_index = list(range(len(self._documents)))
        self._epoch = 0
        self._build_buffer()

    def _build_buffer(self) -> None:
        """(Re)pack the corpus into blocks for one epoch.

        Decorrelation happens at three granularities:
          1. document/chunk order is permuted,
          2. a random offset (< block_size) shifts every block boundary, so the
             model never sees the exact same 1024-grams across epochs,
          3. blocks are served in permuted order, so a batch mixes domains
             instead of being 32 consecutive slices of one document.
        """
        if self.shuffle:
            self._rng.shuffle(self._doc_index)
        buf = np.concatenate([self._documents[i] for i in self._doc_index])
        if self.shuffle and buf.size > self.block_size:
            buf = buf[int(self._rng.integers(0, self.block_size)):]
        n_blocks = buf.size // self.block_size
        if n_blocks == 0:
            raise ValueError(f"Corpus smaller than one block ({self.block_size} tokens).")
        self._blocks = buf[: n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        self._order = self._rng.permutation(n_blocks) if self.shuffle else np.arange(n_blocks)
        self._pos = 0
        self._epoch += 1

    def total_tokens(self) -> int:
        return int(sum(d.size for d in self._documents))

    def _next_block(self) -> np.ndarray:
        if self._pos >= self._order.size:
            self._build_buffer()
        block = self._blocks[self._order[self._pos]]
        self._pos += 1
        return block

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            yield self._next_block()

    def get_batch(self, batch_size: int) -> np.ndarray:
        return np.stack([self._next_block() for _ in range(batch_size)], axis=0)


class BatchProvider:
    """Thin wrapper that yields :class:`Batch` objects for the training loop."""

    def __init__(self, stream: PackedTextStream) -> None:
        self.stream = stream

    def next_batch(self, batch_size: int, step: int) -> Batch:
        return Batch(input_ids=self.stream.get_batch(batch_size), step=step)


# ── Synthetic corpus for CPU smoke tests ────────────────────────────────────


def make_synthetic_corpus(
    output_dir: Path,
    n_tokens: int = 200_000,
    vocab_size: int = 256,
    n_shards: int = 2,
    eos_token_id: int = 2,
    seed: int = 0,
) -> list[Path]:
    """Write a tiny reproducible token corpus to disk and return shard paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    per_shard = max(n_tokens // n_shards, 1)
    paths: list[Path] = []
    for s in range(n_shards):
        # Reserve low ids 0..9 for specials; sample real tokens in [10, vocab).
        arr = rng.integers(10, vocab_size, size=per_shard, dtype=np.int64)
        f = output_dir / f"shard_{s:04d}.npy"
        np.save(f, arr)
        paths.append(f)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "n_tokens": per_shard * n_shards,
                "n_words": int(per_shard * n_shards * 0.78),
                "vocab_size": vocab_size,
                "eos_token_id": eos_token_id,
                "purpose": "Synthetic smoke-test corpus. NOT for real experiments.",
            },
            indent=2,
        )
    )
    return paths


def build_streams(
    token_data_dir: Path | None,
    block_size: int,
    use_synthetic: bool,
    seed: int,
    dev_fraction: float = 0.01,
    chunk_blocks: int = 8,
) -> tuple[PackedTextStream, PackedTextStream]:
    """Build (train_stream, dev_stream).

    The whole corpus is concatenated and split into chunks of
    ``chunk_blocks * block_size`` tokens; ``dev_fraction`` of the chunks are
    held out by STRIDE-sampling, so the dev slice covers every domain of the
    corpus instead of just the tail shard.
    """
    if use_synthetic:
        shard_paths = make_synthetic_corpus(Path("data/_synthetic"), seed=seed)
    else:
        if token_data_dir is None:
            raise FileNotFoundError("token_data_dir is required for non-smoke runs.")
        shard_paths = sorted(Path(token_data_dir).glob("shard_*.npy"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard_*.npy under {token_data_dir}. Run scripts/prepare_data.py first."
            )

    tokens = np.concatenate([np.load(p).ravel().astype(np.int64) for p in shard_paths])
    chunk = max(int(chunk_blocks) * int(block_size), int(block_size))
    n_chunks = max(tokens.size // chunk, 1)
    chunks = [tokens[i * chunk:(i + 1) * chunk] for i in range(n_chunks)]
    tail = tokens[n_chunks * chunk:]
    if tail.size:
        chunks[-1] = np.concatenate([chunks[-1], tail])

    n_dev = min(max(int(round(n_chunks * dev_fraction)), 1), n_chunks - 1) if n_chunks > 1 else 0
    if n_dev > 0:
        stride = max(n_chunks // n_dev, 1)
        dev_idx = set(list(range(stride // 2, n_chunks, stride))[:n_dev])
    else:
        dev_idx = set()
    train_docs = [c for i, c in enumerate(chunks) if i not in dev_idx]
    dev_docs = [c for i, c in enumerate(chunks) if i in dev_idx] or [chunks[-1]]

    LOG.info(
        "Corpus: %d tokens -> %d chunks of ~%d tokens; train=%d chunks, dev=%d chunks "
        "(stride-sampled across the corpus).",
        tokens.size, n_chunks, chunk, len(train_docs), len(dev_docs),
    )
    train = PackedTextStream(train_docs, block_size, seed=seed, shuffle=True)
    dev = PackedTextStream(dev_docs, block_size, seed=seed + 999, shuffle=False)
    return train, dev
