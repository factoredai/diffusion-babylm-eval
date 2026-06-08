"""Train a shared SentencePiece BPE tokenizer for EN + NL + ZH.

Recipe (locked, see improved_research_context_v2.md §9):
    - vocab size: 32_000
    - model_type: BPE
    - character_coverage: 0.9995 (high to retain rare Chinese characters)
    - byte-equalized sampling: we take the same number of BYTES from each
      language (scaled by the canonical byte-premium), so the shared vocab does
      not over-allocate sub-words to languages with denser UTF-8 encodings.
    - UNK-rate validation: tokenizer must satisfy per-language UNK rate caps
      (see configs/base.yaml; default eng < 0.01%, nld < 0.05%, zho < 0.10%).

Inputs (default):
    Pulled directly from the cached BabyBabelLM datasets via
    `taam.datasources.iter_documents`. No intermediate raw-text files needed.

Usage:
    # Real training (uses cached HF datasets in data/hf_cache):
    python scripts/train_tokenizer.py \\
        --vocab-size 32000 \\
        --bytes-per-lang 50000000 \\
        --output tokenizer/spm_32k_en_nl_zh.model

    # Smoke test (no data needed; synthetic mini-corpus):
    python scripts/train_tokenizer.py --smoke-test \\
        --output tokenizer/spm_smoke.model
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("train_tokenizer")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from taam import BYTE_PREMIUM_HF, LANGUAGES  # noqa: E402
from taam.datasources import ensure_hf_env, iter_documents  # noqa: E402

DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_CHAR_COVERAGE = 0.9995
DEFAULT_BYTES_PER_LANG = 50 * 1024 * 1024  # 50 MB per language for SP training

DEFAULT_UNK_THRESHOLDS = {
    "eng": 0.0001,
    "nld": 0.0005,
    "zho": 0.0010,
}


@dataclass
class TokenizerArtifact:
    model_path: Path
    vocab_size: int
    unk_rates: dict[str, float]
    byte_equalized: bool
    sample_bytes_per_lang: dict[str, int]


# ---------------------------------------------------------------------------
# Document -> sentence flattening
# ---------------------------------------------------------------------------
# SentencePiece trains on lines, but BabyBabelLM documents can be long (avg
# 3,900 chars for eng). We split on newlines and keep non-empty segments.
def _doc_to_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            out.append(stripped)
    return out


# ---------------------------------------------------------------------------
# Byte-equalized sampling from HF datasets
# ---------------------------------------------------------------------------
def _sample_byte_equalized_from_dataset(
    lang: str,
    target_bytes: int,
    seed: int,
) -> list[str]:
    """Collect lines from the cached HF dataset until the UTF-8 byte budget
    is met, drawing from a deterministically shuffled view of the corpus.

    We use `Dataset.shuffle(seed)` which is efficient on parquet (it builds
    an internal indices map without copying data) and then iterate
    sequentially through the shuffled iterator. This avoids the O(N) random
    parquet reads that `ds[random_idx]` would incur.
    """
    from taam.datasources import get_dataset

    ds = get_dataset(lang, split="train", streaming=False)
    n_docs = len(ds)
    LOG.info("[%s] dataset has %d docs; shuffling with seed=%d", lang, n_docs, seed)

    shuffled = ds.shuffle(seed=seed)

    lines: list[str] = []
    total_bytes = 0
    inspected = 0
    for record in shuffled:
        text = record.get("text") or ""
        if not text:
            continue
        for ln in _doc_to_lines(text):
            lines.append(ln)
            total_bytes += len(ln.encode("utf-8"))
        inspected += 1
        if total_bytes >= target_bytes:
            break

    LOG.info(
        "[%s] sampled %d lines (%d bytes from %d docs) target=%d",
        lang, len(lines), total_bytes, inspected, target_bytes,
    )
    return lines


def _byte_equalize_targets(bytes_per_lang: int, languages: Iterable[str]) -> dict[str, int]:
    """Compute per-language target byte counts.

    The 'byte-equalized' approach asks: if I want each language to contribute
    the same INFORMATION CONTENT (in bytes), how many bytes do I need? Since
    byte_premium tracks how many UTF-8 bytes a language uses per "canonical
    paragraph", we sample MORE bytes from languages with higher BPs and LESS
    from those with lower BPs.

    Concretely: target_bytes(l) = bytes_per_lang * bp(l).
    """
    out = {}
    for lang in languages:
        bp = BYTE_PREMIUM_HF.get(lang, 1.0)
        out[lang] = int(bytes_per_lang * bp)
    return out


# ---------------------------------------------------------------------------
# Synthetic mode (for CI / smoke tests)
# ---------------------------------------------------------------------------
def _generate_synthetic_text(language: str, n_lines: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    inventories = {
        "eng": (
            "the of and to in is it that for with you he be was on as are not".split()
        ),
        "nld": (
            "de het een en van in is dat te niet op met voor wat zijn zich aan om".split()
        ),
        "zho": list("的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家".split()),
    }
    inv = inventories.get(language, inventories["eng"])
    out = []
    for _ in range(n_lines):
        length = rng.randint(5, 25)
        if language == "zho":
            line = "".join(rng.choices(inv, k=length))
        else:
            line = " ".join(rng.choices(inv, k=length))
        out.append(line)
    return out


# ---------------------------------------------------------------------------
# UNK rate measurement
# ---------------------------------------------------------------------------
def _compute_unk_rate(spm_processor, lines: list[str]) -> float:
    total_pieces = 0
    unk_pieces = 0
    unk_id = spm_processor.unk_id()
    for ln in lines:
        ids = spm_processor.encode(ln, out_type=int)
        total_pieces += len(ids)
        unk_pieces += sum(1 for i in ids if i == unk_id)
    return unk_pieces / max(total_pieces, 1)


# ---------------------------------------------------------------------------
# SentencePiece training core
# ---------------------------------------------------------------------------
def train(
    text_per_lang: dict[str, list[str]],
    output: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    character_coverage: float = DEFAULT_CHAR_COVERAGE,
    unk_thresholds: dict[str, float] | None = None,
    sample_bytes_per_lang: dict[str, int] | None = None,
) -> TokenizerArtifact:
    try:
        import sentencepiece as spm
    except ImportError as e:
        raise ImportError(
            "sentencepiece is required for tokenizer training. "
            'Install with: pip install "taam[train]"  or  pip install sentencepiece'
        ) from e

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    unk_thresholds = unk_thresholds or DEFAULT_UNK_THRESHOLDS

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        all_lines_file = td_path / "all.txt"
        held_out: dict[str, list[str]] = {}
        with all_lines_file.open("w", encoding="utf-8") as fout:
            for lang, lines in text_per_lang.items():
                # Hold out the last 5% for UNK measurement; train on the rest.
                split = max(1, int(len(lines) * 0.05))
                held_out[lang] = lines[-split:]
                for ln in lines[:-split]:
                    fout.write(ln + "\n")

        model_prefix = str(output.with_suffix(""))
        LOG.info(
            "Training SentencePiece BPE (vocab=%d, char_cov=%.4f) ...",
            vocab_size, character_coverage,
        )
        spm.SentencePieceTrainer.train(
            input=str(all_lines_file),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=character_coverage,
            byte_fallback=True,
            normalization_rule_name="nmt_nfkc_cf",
            pad_id=3,
            bos_id=1,
            eos_id=2,
            unk_id=0,
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
            num_threads=max(1, (os.cpu_count() or 1) - 1),
        )

    spm_proc = spm.SentencePieceProcessor()
    spm_proc.load(str(output))

    unk_rates: dict[str, float] = {}
    print()
    print("Per-language UNK rate (on held-out 5% of each lang's training text):")
    fail = False
    for lang, lines in held_out.items():
        rate = _compute_unk_rate(spm_proc, lines)
        unk_rates[lang] = rate
        cap = unk_thresholds.get(lang, 0.001)
        status = "OK  " if rate <= cap else "FAIL"
        if status == "FAIL":
            fail = True
        print(f"  [{status}] {lang}: UNK rate = {rate*100:.4f}%   cap = {cap*100:.4f}%")
    if fail:
        raise RuntimeError("One or more languages exceeded their UNK rate cap.")

    return TokenizerArtifact(
        model_path=output,
        vocab_size=vocab_size,
        unk_rates=unk_rates,
        byte_equalized=True,
        sample_bytes_per_lang=sample_bytes_per_lang or {},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Use synthetic mini-text instead of real corpora.",
    )
    parser.add_argument(
        "--languages", nargs="+", default=list(LANGUAGES),
        choices=list(LANGUAGES),
        help="languages to include in the shared vocab",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
        help="target vocabulary size (incl. special tokens)",
    )
    parser.add_argument(
        "--char-coverage", type=float, default=DEFAULT_CHAR_COVERAGE,
    )
    parser.add_argument(
        "--bytes-per-lang", type=int, default=DEFAULT_BYTES_PER_LANG,
        help="baseline byte sample per language (multiplied by BP_l)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "tokenizer" / "spm_32k_en_nl_zh.model",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.smoke_test:
        print("=== SMOKE TEST mode: training tokenizer on synthetic text ===")
        text_per_lang = {
            lang: _generate_synthetic_text(lang, n_lines=16_000, seed=args.seed + i)
            for i, lang in enumerate(args.languages)
        }
        artifact = train(
            text_per_lang=text_per_lang,
            output=args.output,
            vocab_size=min(args.vocab_size, 500),
            character_coverage=args.char_coverage,
            unk_thresholds={lang: 0.05 for lang in args.languages},
        )
    else:
        ensure_hf_env()
        targets = _byte_equalize_targets(args.bytes_per_lang, args.languages)
        print("Byte-equalized targets:")
        for lang in args.languages:
            print(f"  {lang}: {targets[lang]:,} bytes (BP={BYTE_PREMIUM_HF[lang]:.3f})")
        print()
        text_per_lang = {}
        for i, lang in enumerate(args.languages):
            print(f"=== sampling {lang} ===", flush=True)
            text_per_lang[lang] = _sample_byte_equalized_from_dataset(
                lang, targets[lang], seed=args.seed + i,
            )
            print(f"  {lang}: collected {len(text_per_lang[lang]):,} lines")
        print()
        artifact = train(
            text_per_lang=text_per_lang,
            output=args.output,
            vocab_size=args.vocab_size,
            character_coverage=args.char_coverage,
            sample_bytes_per_lang=targets,
        )

    print()
    print("=" * 72)
    print(f"  Tokenizer trained: {artifact.model_path}")
    print(f"  Vocab size: {artifact.vocab_size}")
    print("  UNK rates:")
    for lang, rate in artifact.unk_rates.items():
        print(f"    {lang}: {rate*100:.4f}%")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
