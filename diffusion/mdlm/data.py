"""English text data pipeline for the Strict-Small masked-diffusion model.

Single-language analogue of the multilingual reference loader: documents are
tokenized, concatenated with EOS separators, and sliced into fixed-length blocks
(LM packing). The stream is infinite and reshuffles at every epoch boundary.

For CPU smoke tests we generate a tiny synthetic corpus so the whole training /
eval pipeline can be exercised with no data download and no GPU.

Real-data convention (produced by ``scripts/prepare_data.py``):

    token_data_dir/
        shard_0000.npy        # 1-D int32/int64 token ids
        shard_0001.npy
        ...
        manifest.json         # {"n_tokens", "n_words", "eos_token_id", ...}

The last shard is held out as the dev slice for validation loss.
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
    """Infinite stream of LM-packed blocks from one (English) corpus."""

    def __init__(
        self,
        token_id_files: Sequence[Path | str],
        block_size: int = 512,
        eos_token_id: int = 2,
        shuffle_buffer: int = 1024,
        seed: int = 0,
    ) -> None:
        self.block_size = int(block_size)
        self.eos_token_id = int(eos_token_id)
        self.shuffle_buffer = int(shuffle_buffer)
        self._rng = np.random.default_rng(seed)
        self._files = [Path(p) for p in token_id_files]
        if not self._files:
            raise ValueError("Need at least one .npy file of token ids.")
        self._documents: list[np.ndarray] = []
        for p in self._files:
            if not p.exists():
                raise FileNotFoundError(p)
            arr = np.load(p)
            self._documents.append(arr.ravel().astype(np.int64))
        self._doc_index = list(range(len(self._documents)))
        self._build_buffer()

    def _build_buffer(self) -> None:
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
                self._build_buffer()
            block = self._buffer[self._cursor : self._cursor + self.block_size]
            self._cursor += self.block_size
            yield block

    def get_batch(self, batch_size: int) -> np.ndarray:
        it = iter(self)
        rows = [next(it) for _ in range(batch_size)]
        return np.stack(rows, axis=0)


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
    eos_token_id: int = 2,
) -> tuple[PackedTextStream, PackedTextStream]:
    """Build (train_stream, dev_stream). Last shard is held out for dev."""
    if use_synthetic:
        synth = make_synthetic_corpus(Path("data/_synthetic"), seed=seed, eos_token_id=eos_token_id)
        train = PackedTextStream(synth[:-1] or synth, block_size, eos_token_id, seed=seed)
        dev = PackedTextStream(synth[-1:], block_size, eos_token_id, seed=seed + 999)
        return train, dev

    if token_data_dir is None:
        raise FileNotFoundError("token_data_dir is required for non-smoke runs.")
    token_data_dir = Path(token_data_dir)
    manifest = token_data_dir / "manifest.json"
    if manifest.exists():
        eos_token_id = int(json.loads(manifest.read_text()).get("eos_token_id", eos_token_id))
    shards = sorted(token_data_dir.glob("shard_*.npy"))
    if len(shards) < 2:
        raise FileNotFoundError(
            f"Expected >=2 shards under {token_data_dir}, found {len(shards)}. "
            "Run scripts/prepare_data.py first."
        )
    LOG.info("Found %d shards (eos_id=%d); holding out last shard for dev.", len(shards), eos_token_id)
    train = PackedTextStream(shards[:-1], block_size, eos_token_id, seed=seed)
    dev = PackedTextStream(shards[-1:], block_size, eos_token_id, seed=seed + 999)
    return train, dev
