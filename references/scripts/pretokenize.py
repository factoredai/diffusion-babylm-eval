#!/usr/bin/env python
"""Pre-tokenize the BabyBabelLM corpora into int32 shards on disk.

This is what `taam/data.PackedLanguageStream` consumes at training time.
Pre-tokenizing once and storing shards on disk is ~50x faster at training
time than tokenizing on the fly, and it makes runs perfectly reproducible
(no SentencePiece nondeterminism across epochs).

Output layout:
    data/tokens/{lang}/shard_0000.npy      # 1-D int32 array
    data/tokens/{lang}/shard_0001.npy
    ...
    data/tokens/manifest.json              # per-shard + per-lang stats

Shard layout (within one .npy):
    [doc1_tokens..., EOS, doc2_tokens..., EOS, ..., docN_tokens..., EOS]
    where EOS is the tokenizer's eos_id (we set 2 in SentencePiece training).

We stop appending to a shard once it reaches `--shard-tokens` (default 5M);
the final partial shard is also written. This gives ~7 shards/lang at the
100M-token budget — small enough that `PackedLanguageStream` can pre-load
all of them, large enough that file overhead is negligible.

Usage:
    # Tokenize ALL of EN/NL/ZH:
    python scripts/pretokenize.py

    # Cap per-lang to e.g. 35M tokens (for one specific budget experiment):
    python scripts/pretokenize.py --max-tokens 35000000

    # Different tokenizer:
    python scripts/pretokenize.py --tokenizer tokenizer/spm_8k.model

    # Smoke test (only 50k tokens per lang, ~2 seconds total):
    python scripts/pretokenize.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger("pretokenize")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from taam import LANGUAGES  # noqa: E402
from taam.datasources import ensure_hf_env, iter_documents  # noqa: E402

DEFAULT_SHARD_TOKENS = 5_000_000        # ~20MB int32 per shard
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "tokens"
DEFAULT_TOKENIZER = REPO_ROOT / "tokenizer" / "spm_32k_en_nl_zh.model"


def _display_path(path: Path) -> str:
    """Return a path string relative to REPO_ROOT when possible, else absolute.

    On Colab we symlink data/tokens, tokenizer/, ... to Google Drive (see
    scripts/colab_bootstrap.sh). After Path.resolve() those paths land outside
    REPO_ROOT (e.g. /content/drive/MyDrive/...), so a bare `relative_to`
    raises ValueError. Falling back to the absolute path keeps the manifest
    and logs readable in both layouts.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_tokenizer(model_path: Path):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"tokenizer not found at {model_path}. "
            f"Run scripts/train_tokenizer.py first."
        )
    sp.load(str(model_path))
    return sp


def _write_shard(buffer: list[int], shard_path: Path) -> int:
    """Write a single shard. Returns number of tokens written."""
    arr = np.asarray(buffer, dtype=np.int32)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard_path, arr, allow_pickle=False)
    return int(arr.size)


def pretokenize_language(
    lang: str,
    sp,
    out_dir: Path,
    shard_tokens: int,
    max_tokens: int | None,
    seed: int,
) -> dict:
    """Tokenize one language's corpus into shards on disk.

    Returns per-language statistics for the manifest.
    """
    eos_id = sp.eos_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "[%s] starting pretokenization (eos_id=%d, shard_tokens=%d, max=%s)",
        lang, eos_id, shard_tokens, str(max_tokens),
    )

    # Use HF's dataset shuffle for cheap, deterministic ordering.
    from taam.datasources import get_dataset
    ds = get_dataset(lang, split="train", streaming=False)
    shuffled = ds.shuffle(seed=seed)

    buffer: list[int] = []
    total_tokens = 0
    total_docs = 0
    shard_index = 0
    shard_paths: list[str] = []
    t0 = time.perf_counter()
    last_report_tokens = 0

    for record in shuffled:
        text = record.get("text") or ""
        if not text:
            continue

        ids = sp.encode(text, out_type=int)
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(eos_id)
        total_tokens += len(ids) + 1
        total_docs += 1

        # Flush full shards.
        while len(buffer) >= shard_tokens:
            shard_path = out_dir / f"shard_{shard_index:04d}.npy"
            head, buffer = buffer[:shard_tokens], buffer[shard_tokens:]
            n = _write_shard(head, shard_path)
            shard_paths.append(_display_path(shard_path))
            shard_index += 1
            LOG.info(
                "[%s] wrote %s (%d tokens; %d docs total; %d toks total)",
                lang, shard_path.name, n, total_docs, total_tokens,
            )

        if max_tokens and total_tokens >= max_tokens:
            LOG.info("[%s] reached --max-tokens cap (%d)", lang, max_tokens)
            break

        if total_tokens - last_report_tokens >= 1_000_000:
            elapsed = time.perf_counter() - t0
            rate = total_tokens / max(elapsed, 1e-6)
            print(
                f"  [{lang}] {total_tokens:>12,} tokens "
                f"({rate / 1e6:.2f}M tok/s, {total_docs:,} docs)",
                flush=True,
            )
            last_report_tokens = total_tokens

    if buffer:
        shard_path = out_dir / f"shard_{shard_index:04d}.npy"
        n = _write_shard(buffer, shard_path)
        shard_paths.append(_display_path(shard_path))
        shard_index += 1
        LOG.info(
            "[%s] wrote final %s (%d tokens; total %d shards)",
            lang, shard_path.name, n, shard_index,
        )

    elapsed = time.perf_counter() - t0
    print(
        f"  [{lang}] done: {total_tokens:,} tokens, {total_docs:,} docs, "
        f"{shard_index} shards, in {elapsed:,.1f}s "
        f"({total_tokens / max(elapsed,1e-6) / 1e6:.2f}M tok/s)"
    )
    return {
        "lang": lang,
        "tokenizer_eos_id": int(eos_id),
        "shard_tokens_cap": shard_tokens,
        "max_tokens_cap": max_tokens,
        "num_docs": total_docs,
        "num_tokens": total_tokens,
        "num_shards": shard_index,
        "shard_paths": shard_paths,
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES),
                        choices=list(LANGUAGES))
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS,
        help="max tokens per .npy shard (default 5M)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="optional per-language token cap (e.g. 35000000 for BP-floor)",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="shuffle seed (deterministic doc ordering)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="set max_tokens to 50_000 for a quick CI run")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.smoke_test:
        args.max_tokens = 50_000
        args.shard_tokens = min(args.shard_tokens, 20_000)

    ensure_hf_env()
    args.output_dir = args.output_dir.resolve()
    args.tokenizer = args.tokenizer.resolve()
    sp = _load_tokenizer(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer.name}  vocab_size={sp.vocab_size()}  eos_id={sp.eos_id()}")
    print(f"Output dir: {_display_path(args.output_dir)}")
    print(f"Shard tokens: {args.shard_tokens:,}  Max tokens/lang: "
          f"{args.max_tokens if args.max_tokens else 'all'}")
    print()

    per_lang_stats = {}
    for i, lang in enumerate(args.languages):
        print(f"=== pretokenizing {lang} ===")
        per_lang_stats[lang] = pretokenize_language(
            lang=lang,
            sp=sp,
            out_dir=args.output_dir / lang,
            shard_tokens=args.shard_tokens,
            max_tokens=args.max_tokens,
            seed=args.seed + i,
        )
        print()

    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer_model": _display_path(args.tokenizer),
        "tokenizer_vocab_size": int(sp.vocab_size()),
        "tokenizer_eos_id": int(sp.eos_id()),
        "shard_tokens": args.shard_tokens,
        "max_tokens_per_language": args.max_tokens,
        "per_language": per_lang_stats,
        "totals": {
            "num_docs": sum(p["num_docs"] for p in per_lang_stats.values()),
            "num_tokens": sum(p["num_tokens"] for p in per_lang_stats.values()),
            "num_shards": sum(p["num_shards"] for p in per_lang_stats.values()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {_display_path(manifest_path)}")
    print(
        f"Totals: {manifest['totals']['num_docs']:,} docs, "
        f"{manifest['totals']['num_tokens']:,} tokens, "
        f"{manifest['totals']['num_shards']} shards."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
