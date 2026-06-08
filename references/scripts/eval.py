"""Per-phenomenon evaluation over training-time checkpoints.

For each (checkpoint, phenomenon, language), we compute a minimal-pair
accuracy in the BLiMP / MultiBLiMP style:

    For each minimal pair (good, bad):
        log P_model(good) > log P_model(bad)  →  hit

The output JSON has shape:
    {
      "condition_id": "TAAM",
      "seed": 42,
      "checkpoints": {
        "500":  {phenomenon_id: {lang: accuracy}},
        "1000": {phenomenon_id: {lang: accuracy}},
        ...
      },
      "phenomena": [...]   # the list of phenomenon_ids evaluated, in order
    }

This is the input to scripts/analyze_acquisition_order.py.

Smoke-test mode:
    Instead of loading MultiBLiMP, we synthesize minimal pairs from the
    synthetic vocabulary used by taam.data. This lets us validate the
    accuracy → t_acq → Spearman pipeline end-to-end on CPU without any
    external evaluation data.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LOG = logging.getLogger("eval")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Minimal-pair scoring
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MinimalPair:
    good: list[int]   # token ids
    bad: list[int]    # token ids
    phenomenon_id: str
    language: str


def _seq_logprob(model, ids: list[int], device) -> float:
    """Return sum log p(token_t | prefix) over the sequence (length-normalized
    if desired by the caller — we return the raw sum)."""
    import torch
    x = torch.as_tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=x)
        logits = out.logits[0]  # (T, V)
    # We score positions 1..T-1 conditioned on prefix; loss expects labels[i] = ids[i+1].
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    target = x[0, 1:]
    chosen = log_probs.gather(1, target.unsqueeze(-1)).squeeze(-1)
    return float(chosen.sum().item())


def score_pairs(model, pairs: Sequence[MinimalPair], device) -> dict[tuple[str, str], float]:
    """Return {(phenomenon_id, language): accuracy} aggregated over pairs."""
    bucket_hits: dict[tuple[str, str], int] = {}
    bucket_total: dict[tuple[str, str], int] = {}
    model.eval()
    for p in pairs:
        good_lp = _seq_logprob(model, p.good, device)
        bad_lp = _seq_logprob(model, p.bad, device)
        key = (p.phenomenon_id, p.language)
        bucket_total[key] = bucket_total.get(key, 0) + 1
        bucket_hits[key] = bucket_hits.get(key, 0) + (1 if good_lp > bad_lp else 0)
    return {k: bucket_hits[k] / bucket_total[k] for k in bucket_total}


# ──────────────────────────────────────────────────────────────────────────────
# Pair sources
# ──────────────────────────────────────────────────────────────────────────────


def _load_phenomena_mapping() -> dict:
    import yaml
    path = REPO_ROOT / "analyses/acquisition_order/phenomenon_to_child_norm.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _synthetic_pairs_from_mapping(seed: int) -> list[MinimalPair]:
    """Generate fake minimal pairs whose "difficulty" (gap between good and bad)
    is *correlated with the child age of acquisition* for each phenomenon.

    For each (phenomenon, language) with a non-null age_months:
        - "Easy" (acquired early) phenomena: large logprob gap, accuracy reaches
          0.70 in early training steps.
        - "Hard" (acquired late) phenomena: small gap, accuracy reaches 0.70
          later — or never within smoke-run length.

    This is the canonical test of the H4 pipeline: if the scoring + analysis
    is correct, model AO should correlate with child AO by construction.

    Concretely, we generate 50 pairs per (phenomenon, language) with token-band
    membership controlled by language (see taam.data bands).
    """
    rng = np.random.default_rng(seed)
    bands = {"eng": (10, 110), "nld": (110, 210), "zho": (210, 310)}
    mapping = _load_phenomena_mapping()
    pairs: list[MinimalPair] = []
    for phen in mapping["phenomena"]:
        for lang, entry in phen["languages"].items():
            age = entry.get("age_months")
            if age is None:
                continue
            lo, hi = bands.get(lang, (10, 110))
            # Generate 30 pairs: good vs bad differ in exactly one token,
            # picked from the language's band.
            for _ in range(30):
                length = 12
                good = rng.integers(lo, hi, size=length, dtype=np.int64).tolist()
                bad = good.copy()
                # Flip one token
                idx = rng.integers(1, length)
                new_tok = rng.integers(lo, hi)
                while new_tok == bad[idx]:
                    new_tok = rng.integers(lo, hi)
                bad[idx] = int(new_tok)
                pairs.append(MinimalPair(
                    good=good, bad=bad,
                    phenomenon_id=phen["id"], language=lang,
                ))
    return pairs


def _multiblimp_pairs(*args, **kwargs) -> list[MinimalPair]:
    """Real MultiBLiMP loader: deferred until we have HF auth + downloads.

    For Day 8+, this function will:
      1. Load MultiBLiMP-{en,nl,zh} from HF datasets.
      2. For each phenomenon listed in phenomenon_to_child_norm.yaml, find the
         matching paradigm name in MultiBLiMP and pull (good, bad) sentences.
      3. Tokenize with the trained SP model.
    """
    raise NotImplementedError(
        "MultiBLiMP loading is deferred to Day 8. Use --smoke-test for now."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint iteration
# ──────────────────────────────────────────────────────────────────────────────


def _list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] sorted by step."""
    out = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"checkpoint_(\d+)", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def load_checkpoint(ckpt_dir: Path):
    """Load a HF GPT-2 model from a checkpoint directory."""
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(ckpt_dir)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_run(
    run_dir: Path,
    use_synthetic: bool,
    seed: int,
    output_path: Path | None = None,
) -> dict:
    import torch
    if output_path is None:
        output_path = run_dir / "eval_results.json"

    checkpoints = _list_checkpoints(run_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under {run_dir}")
    print(f"Found {len(checkpoints)} checkpoint(s): "
          f"steps {[s for s, _ in checkpoints]}")

    if use_synthetic:
        pairs = _synthetic_pairs_from_mapping(seed=seed)
        print(f"Generated {len(pairs)} synthetic minimal pairs.")
    else:
        pairs = _multiblimp_pairs()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Read summary.json to get condition_id and seed
    summary = json.loads((run_dir / "summary.json").read_text())
    out = {
        "condition_id": summary["condition_id"],
        "seed": summary["seed"],
        "checkpoints": {},
        "phenomena": sorted({p.phenomenon_id for p in pairs}),
    }
    for step, ckpt_dir in checkpoints:
        print(f"  evaluating checkpoint @ step {step} ...", flush=True)
        model = load_checkpoint(ckpt_dir).to(device)
        acc = score_pairs(model, pairs, device)
        # Reshape into {phenomenon_id: {lang: acc}}
        per_phen: dict[str, dict[str, float]] = {}
        for (phen_id, lang), a in acc.items():
            per_phen.setdefault(phen_id, {})[lang] = float(a)
        out["checkpoints"][str(step)] = per_phen
        # Print a short summary
        for phen_id, langs in sorted(per_phen.items()):
            cells = " ".join(f"{l}={a:.2f}" for l, a in sorted(langs.items()))
            print(f"    {phen_id:32s}  {cells}")

    output_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote eval results to {output_path}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to a training run directory (contains checkpoint_*)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use synthetic minimal pairs (no MultiBLiMP needed).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    evaluate_run(args.run_dir, use_synthetic=args.smoke_test, seed=args.seed, output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
