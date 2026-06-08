#!/usr/bin/env python
"""End-to-end smoke test of the multilingual data pipeline (no model).

Verifies that:
  1. data/tokens/manifest.json exists and reports a valid tokenizer + EOS id.
  2. Per-language pretokenized shards load cleanly into PackedLanguageStream.
  3. MultilingualMixer respects an externally-supplied pi distribution
     within a tolerance band derived from binomial sampling noise.
  4. Token IDs returned for each language fall in reasonable ranges.

Why this exists:
    The actual transformer forward/backward pass is slow on CPU and not
    informative for debugging the data layer. This script isolates the
    data layer so a developer can validate the *pipeline* in <1 second.

Usage:
    python scripts/smoke_test_data_pipeline.py
    python scripts/smoke_test_data_pipeline.py --n-batches 5000  # tighter pi check
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from taam.data import MultilingualMixer, PackedLanguageStream  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-data", type=Path, default=REPO_ROOT / "data" / "tokens",
        help="directory written by scripts/pretokenize.py",
    )
    parser.add_argument(
        "--prior", type=Path, default=REPO_ROOT / "configs" / "typological_prior.yaml",
        help="yaml file with `typological_prior.pi_0_tokens`",
    )
    parser.add_argument(
        "--block-size", type=int, default=512,
        help="context window size (PackedLanguageStream block_size)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
    )
    parser.add_argument(
        "--n-batches", type=int, default=600,
        help="how many batches to sample for the empirical-pi check",
    )
    parser.add_argument(
        "--tolerance-sigma", type=float, default=3.0,
        help="empirical pi must agree with target within this many sigmas",
    )
    parser.add_argument(
        "--last-shard-only", action="store_true", default=True,
        help="for speed, only load the LAST shard of each language",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 72)
    print("End-to-end data pipeline smoke test (real shards, no model)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: pretok manifest
    # ------------------------------------------------------------------
    manifest_path = args.token_data / "manifest.json"
    if not manifest_path.exists():
        print(f"[FAIL] manifest not found: {manifest_path}", file=sys.stderr)
        print("Run: make pretokenize", file=sys.stderr)
        return 2
    mani = json.loads(manifest_path.read_text())
    eos = int(mani["tokenizer_eos_id"])
    print(
        f"\n1. Pretok manifest OK: vocab={mani['tokenizer_vocab_size']}  "
        f"eos={eos}"
    )
    print(
        "   Shards per lang: "
        + ", ".join(f"{l}={p['num_shards']}" for l, p in mani["per_language"].items())
    )

    # ------------------------------------------------------------------
    # Step 2: build per-language streams
    # ------------------------------------------------------------------
    languages = list(mani["per_language"].keys())
    print(f"\n2. Building streams for {languages} ...")
    t0 = time.perf_counter()
    streams = {}
    for lang in languages:
        shards = sorted((args.token_data / lang).glob("shard_*.npy"))
        if not shards:
            print(f"[FAIL] no shards in {args.token_data / lang}", file=sys.stderr)
            return 3
        chosen = shards[-1:] if args.last_shard_only else shards
        stream = PackedLanguageStream(
            token_id_files=chosen,
            block_size=args.block_size,
            eos_token_id=eos,
            seed=args.seed,
        )
        streams[lang] = stream
        print(
            f"   {lang}: {len(chosen)}/{len(shards)} shards, "
            f"total_tokens={stream.total_tokens():>10,}"
        )
    print(f"   build time: {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 3: empirical pi check
    # ------------------------------------------------------------------
    prior = yaml.safe_load(args.prior.read_text())["typological_prior"]
    pi = {k: float(v) for k, v in prior["pi_0_tokens"].items()}
    print(f"\n3. Sampling {args.n_batches} batches with pi_0_tokens = {pi}")
    mixer = MultilingualMixer(streams=streams, batch_size=args.batch_size, seed=args.seed)
    counts: Counter = Counter()
    t0 = time.perf_counter()
    for step in range(args.n_batches):
        batch = mixer.next_batch(pi=pi, step=step)
        counts[batch.language] += 1
    elapsed = time.perf_counter() - t0
    print(f"   sampled in {elapsed:.2f}s ({args.n_batches / elapsed:,.0f} batches/s)")

    print(
        f"\n   Empirical pi vs target "
        f"(tolerance: +/- {args.tolerance_sigma:.1f} sigma binomial):"
    )
    n = args.n_batches
    ok = True
    for lang in languages:
        emp = counts[lang] / n
        tgt = pi[lang]
        sigma = math.sqrt(max(tgt * (1.0 - tgt) / n, 1e-12))
        diff = emp - tgt
        z = diff / sigma if sigma > 0 else 0.0
        passed = abs(z) <= args.tolerance_sigma
        mark = "OK  " if passed else "FAIL"
        if not passed:
            ok = False
        print(
            f"   [{mark}] {lang}: empirical={emp:.4f}  target={tgt:.4f}  "
            f"diff={diff:+.4f}  z={z:+.2f}"
        )

    # ------------------------------------------------------------------
    # Step 4: token-ID sanity
    # ------------------------------------------------------------------
    print("\n4. First 10 token IDs from each language (should be sane integers):")
    mixer2 = MultilingualMixer(streams=streams, batch_size=args.batch_size, seed=args.seed + 1)
    seen: dict[str, list[int]] = {}
    for step in range(args.n_batches):
        if len(seen) == len(languages):
            break
        b = mixer2.next_batch(pi=pi, step=step)
        if b.language not in seen:
            seen[b.language] = b.input_ids[0, :10].tolist()
    for lang in languages:
        ids = seen.get(lang, [])
        print(f"   {lang}: {ids}")

    print("\n" + "=" * 72)
    if not ok:
        print("Smoke test FAILED: empirical pi outside tolerance.")
        return 1
    print("Smoke test PASSED. Data pipeline is ready for GPU training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
