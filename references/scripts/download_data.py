#!/usr/bin/env python
"""Download BabyBabelLM Tier-1 datasets (EN/NL/ZH) and compute a manifest.

This script:
  1. Forces the HF cache to data/hf_cache/ (workspace-local; survives reboots
     and is easy to delete with `rm -rf data/hf_cache`).
  2. Loads each of the three language datasets via taam.datasources (which
     reuses the same cache layout everywhere downstream).
  3. Walks every document once and tallies:
        - total docs and total `num-tokens` (the field BabyBabelLM ships)
        - breakdown by `category`
        - breakdown by `data-source`
        - breakdown by `age-estimate` (useful for the H4 cognitive analysis)
        - total raw text bytes (for byte-premium sanity check)
  4. Writes data/manifest.json with the aggregated counts. The Markdown
     report is produced by scripts/build_composition_report.py (kept
     separate so we can re-render the report without re-walking the data).

Why no token-budget filtering here:
    The 100M-token budget belongs to the *training mixer*, not to the corpus
    on disk. We keep the full BabyBabelLM corpus locally so we can ablate
    different budgets (TAAM vs B0 vs P2 ...) without redownloading.

Usage:
    python scripts/download_data.py                # all 3 langs
    python scripts/download_data.py --langs eng    # one lang
    python scripts/download_data.py --force        # recount even if manifest exists
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from taam import LANGUAGES  # noqa: E402
from taam.datasources import (  # noqa: E402
    DATASET_REPOS,
    DEFAULT_CACHE_DIR,
    ensure_hf_env,
    iter_documents,
    num_documents,
)


def _top_n_dict(counter: Counter, n: int | None = None) -> dict[str, int]:
    """Return a JSON-friendly dict sorted by count (descending)."""
    items = counter.most_common(n)
    return {str(k): int(v) for k, v in items}


def _display_path(path: Path) -> str:
    """Return a path string relative to REPO_ROOT when possible, else absolute.

    On Colab we symlink data/hf_cache, data/tokens, runs/... to Google Drive
    (see scripts/colab_bootstrap.sh). After Path.resolve() those paths land
    outside REPO_ROOT (e.g. /content/drive/MyDrive/...), so a bare
    `relative_to(REPO_ROOT)` raises ValueError. Falling back to the absolute
    path keeps the manifest/logs readable in both layouts.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def tally_language(lang: str, *, progress_every: int = 5000) -> dict:
    """Walk the full split and compute per-language statistics."""
    repo = DATASET_REPOS[lang]
    total_docs_expected = num_documents(lang)

    print(f"  [{lang}] repo={repo}  expected_docs={total_docs_expected:,}")

    n_docs = 0
    n_tokens = 0
    n_chars = 0
    n_bytes = 0
    n_empty = 0
    by_category: Counter = Counter()
    by_source: Counter = Counter()
    by_age: Counter = Counter()
    by_script: Counter = Counter()
    n_tokens_missing = 0
    sample_doc_ids: list[str] = []
    t0 = time.perf_counter()

    for i, record in enumerate(iter_documents(lang, skip_empty=False)):
        text = record.get("text") or ""
        if not text:
            n_empty += 1
            continue

        tokens = record.get("num-tokens")
        if tokens is None:
            n_tokens_missing += 1
        else:
            n_tokens += int(tokens)

        n_docs += 1
        n_chars += len(text)
        n_bytes += len(text.encode("utf-8"))

        by_category[record.get("category") or "<unknown>"] += 1
        by_source[record.get("data-source") or "<unknown>"] += 1
        by_age[str(record.get("age-estimate") or "<unknown>")] += 1
        by_script[record.get("script") or "<unknown>"] += 1

        if len(sample_doc_ids) < 5:
            sample_doc_ids.append(str(record.get("doc-id") or f"row-{i}"))

        if (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"    ... {i + 1:>8,} docs scanned  "
                f"({rate:,.0f} docs/s)",
                flush=True,
            )

    elapsed = time.perf_counter() - t0

    summary = {
        "lang": lang,
        "repo": repo,
        "num_docs": n_docs,
        "num_docs_empty": n_empty,
        "num_tokens": n_tokens,
        "num_tokens_missing_count": n_tokens_missing,
        "num_chars": n_chars,
        "num_bytes_utf8": n_bytes,
        "bytes_per_token": (n_bytes / n_tokens) if n_tokens else None,
        "chars_per_doc_avg": (n_chars / n_docs) if n_docs else 0,
        "elapsed_seconds": round(elapsed, 2),
        "sample_doc_ids": sample_doc_ids,
        "by_category": _top_n_dict(by_category),
        "by_data_source": _top_n_dict(by_source),
        "by_age_estimate": _top_n_dict(by_age),
        "by_script": _top_n_dict(by_script),
    }
    print(
        f"  [{lang}] done: "
        f"docs={n_docs:,}  tokens={n_tokens:,}  "
        f"bytes={n_bytes:,}  in {elapsed:,.1f}s"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--langs",
        nargs="+",
        default=list(LANGUAGES),
        choices=list(LANGUAGES),
        help="languages to download (default: all)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data" / "manifest.json",
        help="path to write the manifest JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even if the manifest already exists",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="HF cache directory (default: data/hf_cache)",
    )
    args = parser.parse_args()

    cache_dir = ensure_hf_env(args.cache_dir.resolve())

    print("=" * 72)
    print("BabyBabelLM download + manifest")
    print("=" * 72)
    print(f"  cache_dir : {cache_dir}")
    print(f"  manifest  : {args.manifest}")
    print(f"  languages : {args.langs}")
    print()

    if args.manifest.exists() and not args.force:
        existing = json.loads(args.manifest.read_text())
        present = set(existing.get("per_language", {}).keys())
        if set(args.langs).issubset(present):
            print(
                f"[skip] manifest already covers {sorted(present)}. "
                f"Use --force to recompute."
            )
            return 0

    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    per_language: dict[str, dict] = {}
    for lang in args.langs:
        per_language[lang] = tally_language(lang)

    totals = {
        "num_docs": sum(p["num_docs"] for p in per_language.values()),
        "num_tokens": sum(p["num_tokens"] for p in per_language.values()),
        "num_bytes_utf8": sum(p["num_bytes_utf8"] for p in per_language.values()),
    }

    manifest = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_dir": _display_path(cache_dir),
        "languages": args.langs,
        "totals": totals,
        "per_language": per_language,
    }

    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print()
    print(f"Wrote {_display_path(args.manifest)}")
    print(
        f"Totals: {totals['num_docs']:,} docs, "
        f"{totals['num_tokens']:,} tokens, "
        f"{totals['num_bytes_utf8'] / 1e9:.2f} GB UTF-8."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
