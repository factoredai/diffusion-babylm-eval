#!/usr/bin/env python3
"""Upload a TAAM training run to HuggingFace as a BabyLM-2026 multilingual submission.

This script takes a local run directory (e.g. `runs/2026-05-14_TAAM_v2_seed42/`)
that contains `checkpoints/step_NNNNN_tokens_NNNM/` subfolders and pushes each
checkpoint to a single HF repo, exposing the intermediate checkpoints as HF
*branches* (a.k.a. revisions) named exactly `chck_NM` as required by the
BabyLM eval pipeline (`multilingual/scripts/zeroshot_model_fast_all.sh`).

Layout produced on HuggingFace
------------------------------
  https://huggingface.co/<repo_id>            (main branch = final checkpoint)
  https://huggingface.co/<repo_id>/tree/chck_1M
  https://huggingface.co/<repo_id>/tree/chck_2M
  ...
  https://huggingface.co/<repo_id>/tree/chck_600M

Each branch contains: `config.json`, `model.safetensors` (or `pytorch_model.bin`),
the tokenizer files, and a small `ckpt_meta.json`. `main` additionally has a
`README.md` model card with the condition, seed, hyperparameters, and final
metrics.

Requirements
------------
  pip install huggingface_hub safetensors transformers sentencepiece

Authentication
--------------
  export HF_TOKEN="hf_..."           # or pass --hf-token

Typical invocation
------------------
  python scripts/upload_to_hf.py \\
      --run-dir runs/2026-05-14_TAAM_v2_seed42 \\
      --repo-id Amos-Luna/babylm-2026-taam-v2-seed42 \\
      --tokenizer tokenizer/spm_32k_en_nl_zh.model \\
      --condition TAAM_v2 --seed 42

Dry-run (no uploads, just lists what *would* happen)
----------------------------------------------------
  python scripts/upload_to_hf.py [...] --dry-run

Idempotency
-----------
The script is safe to re-run. It uses `exist_ok=True` for the repo and for each
branch, and `upload_folder` overwrites existing files atomically (one commit
per branch). If interrupted, just re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

LOG = logging.getLogger("upload_to_hf")
_HANDLER = logging.StreamHandler(stream=sys.stdout)
_HANDLER.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
)
LOG.addHandler(_HANDLER)
LOG.setLevel(logging.INFO)
LOG.propagate = False


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

_CKPT_NAME_RE = re.compile(r"^step_(\d+)_tokens_(\d+)M$")


def discover_checkpoints(run_dir: Path) -> List[Tuple[Path, int, int]]:
    """Return [(path, step, tokens_M)] for every checkpoint under run_dir/checkpoints/.

    Sorted ascending by token count.
    """
    ckpts_root = run_dir / "checkpoints"
    if not ckpts_root.is_dir():
        LOG.error(
            "No checkpoints/ directory under %s\n"
            "  Training must finish and save under runs/<name>/checkpoints/\n"
            "  with folders like step_00031_tokens_001M/",
            run_dir,
        )
        return []
    out: List[Tuple[Path, int, int]] = []
    for p in sorted(ckpts_root.iterdir()):
        if not p.is_dir():
            continue
        m = _CKPT_NAME_RE.match(p.name)
        if not m:
            LOG.debug("skip non-ckpt dir: %s", p.name)
            continue
        out.append((p, int(m.group(1)), int(m.group(2))))
    out.sort(key=lambda t: t[2])
    return out


def select_final_checkpoint(
    ckpts: List[Tuple[Path, int, int]],
) -> Tuple[Path, int, int]:
    """Return the highest-step checkpoint (used for `main`)."""
    return max(ckpts, key=lambda t: t[1])


def branch_name_for_tokens(tokens_M: int) -> Optional[str]:
    """Return the BabyLM revision name `chck_NM` for the given token count, or None.

    The eval script enumerates {1..9, 10..100 by 10, 200..1000 by 100}. We use the
    *exact* canonical set; if a checkpoint sits at a non-canonical N (e.g. 11M
    because of rounding), we round it to the nearest canonical bucket but
    *only* if there is no other ckpt closer to that bucket. To keep things
    simple here, we accept a checkpoint as canonical iff its `tokens_M` matches
    a value in the canonical set exactly.
    """
    if 1 <= tokens_M <= 9:
        return f"chck_{tokens_M}M"
    if 10 <= tokens_M <= 100 and tokens_M % 10 == 0:
        return f"chck_{tokens_M}M"
    if 200 <= tokens_M <= 1000 and tokens_M % 100 == 0:
        return f"chck_{tokens_M}M"
    return None


def build_canonical_mapping(
    ckpts: List[Tuple[Path, int, int]],
) -> Dict[str, Tuple[Path, int, int]]:
    """Map each canonical branch name -> the best (highest-step) ckpt matching it."""
    out: Dict[str, Tuple[Path, int, int]] = {}
    for ckpt in ckpts:
        path, step, tokens_M = ckpt
        bn = branch_name_for_tokens(tokens_M)
        if bn is None:
            LOG.debug(
                "ckpt %s has non-canonical token count %dM -> not mapped",
                path.name,
                tokens_M,
            )
            continue
        prev = out.get(bn)
        if prev is None or step > prev[1]:
            out[bn] = ckpt
    return out


def parse_summary(run_dir: Path) -> Dict:
    """Read summary.json / meta.json if present."""
    out: Dict = {}
    for fname in ("summary.json", "meta.json"):
        p = run_dir / fname
        if p.is_file():
            try:
                out[fname.replace(".json", "")] = json.loads(p.read_text())
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Could not parse %s: %s", p, exc)
    return out


def write_tokenizer_into_dir(tokenizer_src: Path, dst: Path) -> List[Path]:
    """Copy SentencePiece tokenizer files into dst.

    HF transformers can pick up `tokenizer.model` (SentencePiece raw) but the
    canonical way is to wrap it in a `PreTrainedTokenizerFast` config. To keep
    this script self-contained and lightweight, we just copy the .model and a
    minimal tokenizer_config.json with `model_max_length` and the BOS/EOS we
    set in training (id 1 / id 2 per scripts/train.py).
    """
    import shutil

    written: List[Path] = []
    # tokenizer.model (SentencePiece)
    spm_target = dst / "tokenizer.model"
    if tokenizer_src.is_file():
        shutil.copy2(tokenizer_src, spm_target)
        written.append(spm_target)
    elif tokenizer_src.with_suffix(".model").is_file():
        # User passed just the basename
        shutil.copy2(tokenizer_src.with_suffix(".model"), spm_target)
        written.append(spm_target)
    else:
        raise FileNotFoundError(f"No SentencePiece tokenizer at {tokenizer_src}")

    # Minimal tokenizer_config.json so HF's AutoTokenizer can construct one.
    #
    # IMPORTANT: the raw tokenizer is a SentencePiece .model trained with
    # model_type="bpe" and byte_fallback=True. Do not advertise it as a generic
    # PreTrainedTokenizerFast unless a real tokenizer.json is also present:
    # recent transformers versions will try to parse tokenizer.model as a
    # TikToken/BPE text file and fail. LlamaTokenizer is a slow SP tokenizer
    # wrapper that can read tokenizer.model directly while preserving the same
    # piece ids used during training.
    tok_cfg = {
        "tokenizer_class": "LlamaTokenizer",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 512,
        "add_bos_token": False,
        "add_eos_token": False,
        "legacy": True,
        "clean_up_tokenization_spaces": False,
    }
    (dst / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2))
    written.append(dst / "tokenizer_config.json")

    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    (dst / "special_tokens_map.json").write_text(json.dumps(special_tokens, indent=2))
    written.append(dst / "special_tokens_map.json")

    # Generation config so HF eval doesn't complain about missing eos/pad.
    gen_cfg = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "_from_model_config": True,
    }
    (dst / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))
    written.append(dst / "generation_config.json")

    return written


def render_model_card(
    *,
    condition: str,
    seed: int,
    run_summary: Dict,
    repo_id: str,
    n_branches: int,
) -> str:
    summary = run_summary.get("summary", {})
    final_pi = summary.get("final_pi", {})

    final_pi_str = (
        ", ".join(f"{k}={v:.3f}" for k, v in final_pi.items()) if final_pi else "n/a"
    )
    tokens_seen = summary.get("tokens_seen_total", "n/a")
    elapsed = summary.get("elapsed_sec", "n/a")
    out_dir = summary.get("output_dir", "n/a")

    return f"""---
language:
- en
- nl
- zh
tags:
- babylm-2026
- multilingual
- typology
- taam
- {condition.lower()}
license: apache-2.0
---

# TAAM — Typology-Aware Adaptive Mixing — `{condition}` seed `{seed}`

This is a BabyLM 2026 Multilingual Track submission checkpoint. It was trained
on English + Dutch + Mandarin Chinese under a ≤100M-unique-token budget with
the [TAAM](https://github.com/Amos-Luna/Asymmetric-Multilingual-Acquisition_TAAM)
method.

- **Method**: `{condition}`
- **Seed**: `{seed}`
- **Repo**: `{repo_id}`
- **Final π** (per-language sampling probability): `{final_pi_str}`
- **Total token exposures**: `{tokens_seen}`
- **Training wall-clock**: `{elapsed} s`
- **Source run dir**: `{out_dir}`

## Intermediate checkpoints

This repo exposes `{n_branches}` intermediate checkpoints as branches following
the BabyLM 2026 naming convention: `chck_1M, chck_2M, ..., chck_10M,
chck_20M, ..., chck_100M, chck_200M, ..., chck_600M`. The eval pipeline at
[babylm-org/babylm-eval](https://github.com/babylm-org/babylm-eval) pulls
these revisions automatically with:

```bash
bash multilingual/scripts/zeroshot_model_fast_all.sh \\
    --model_name {repo_id}
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")  # final checkpoint
# Intermediate checkpoint:
# model = AutoModelForCausalLM.from_pretrained("{repo_id}", revision="chck_100M")
```

## Method summary

TAAM combines (a) a URIEL/lang2vec-derived typological prior over initial
sampling probabilities, (b) EXP3 online updates over per-language sampling
probabilities, and (c) byte-premium-aware token budgeting. The two reward
variants are `normalized_excess_loss` (v1, delta-based) and
`cross_lingual_deficit` (v2, level-based).

See the paper and the public repo for full details, including the structural
floor derivation that explains why the v1 reward starves the hardest
language under typological asymmetry.

## Citation

```bibtex
@inproceedings{{taam2026,
  title  = {{Typology-Aware Adaptive Mixing for Multilingual BabyLMs}},
  author = {{Luna, Amos and collaborators}},
  year   = {{2026}},
  booktitle = {{Proceedings of the BabyLM Workshop at EMNLP 2026}}
}}
```
"""


# --------------------------------------------------------------------------- #
# HuggingFace operations                                                      #
# --------------------------------------------------------------------------- #


def _import_hf():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError as exc:  # noqa: F841
        LOG.error(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )
        sys.exit(2)


def ensure_repo(api, repo_id: str, *, private: bool, dry_run: bool) -> None:
    LOG.info("Ensuring repo exists: %s (private=%s)", repo_id, private)
    if dry_run:
        return
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )


def ensure_branch(api, repo_id: str, branch: str, *, dry_run: bool) -> None:
    if branch == "main":
        return
    LOG.info("  → ensure branch %s", branch)
    if dry_run:
        return
    try:
        api.create_branch(
            repo_id=repo_id,
            branch=branch,
            repo_type="model",
            exist_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("create_branch(%s) raised %s — will try upload anyway", branch, exc)


def upload_checkpoint(
    api,
    *,
    repo_id: str,
    branch: str,
    folder_path: Path,
    commit_message: str,
    dry_run: bool,
) -> None:
    LOG.info(
        "  → uploading %s (%.1f MiB) to revision=%s",
        folder_path.name,
        sum(p.stat().st_size for p in folder_path.rglob("*") if p.is_file()) / 1024**2,
        branch,
    )
    if dry_run:
        return
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=commit_message,
        ignore_patterns=["*.tmp", "*.lock", "__pycache__/*"],
    )


# --------------------------------------------------------------------------- #
# Main pipeline                                                               #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to a TAAM training run directory "
        "(e.g. runs/2026-05-14_TAAM_v2_seed42).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination HF repo id, e.g. Amos-Luna/babylm-2026-taam-v2-seed42",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="Path to the SentencePiece tokenizer .model file "
        "(e.g. tokenizer/spm_32k_en_nl_zh.model).",
    )
    parser.add_argument(
        "--condition",
        required=True,
        help="Condition id (TAAM, TAAM_v2, B0, ...) for the model card.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Training seed for the model card.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token. Defaults to env var HF_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private. Default: public (required for BabyLM submission).",
    )
    parser.add_argument(
        "--only-main",
        action="store_true",
        help="Skip intermediate checkpoints; push only the final to main.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If a branch already exists with a non-empty model.safetensors, skip it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan everything but do not contact HuggingFace.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )

    args = parser.parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------- #
    # Setup                                                               #
    # ------------------------------------------------------------------- #

    _import_hf()
    from huggingface_hub import HfApi

    if not args.hf_token and not args.dry_run:
        LOG.error(
            "No HF token. Set --hf-token or env HF_TOKEN, or pass --dry-run."
        )
        return 2
    api = HfApi(token=args.hf_token) if not args.dry_run else HfApi()

    run_dir = args.run_dir.resolve()
    LOG.info("=" * 70)
    LOG.info("TAAM upload-to-HF pipeline")
    LOG.info("  run_dir   : %s", run_dir)
    LOG.info("  repo_id   : %s", args.repo_id)
    LOG.info("  condition : %s", args.condition)
    LOG.info("  seed      : %s", args.seed)
    LOG.info("  dry_run   : %s", args.dry_run)
    LOG.info("  only_main : %s", args.only_main)
    LOG.info("=" * 70)

    if not run_dir.is_dir():
        LOG.error("Run dir does not exist: %s", run_dir)
        return 2

    tok_path = args.tokenizer.resolve()
    if not args.dry_run and not tok_path.is_file():
        LOG.error(
            "Tokenizer not found: %s\n"
            "  The .model file is not in git (see .gitignore). On Colab, symlink\n"
            "  Drive/MyDrive/.../BabyLM/tokenizer/ into the repo (see colab.ipynb\n"
            "  or upload_pipeline.ipynb cell 2), or pass --tokenizer with the full\n"
            "  path to spm_32k_en_nl_zh.model on Drive.",
            tok_path,
        )
        return 2

    # ------------------------------------------------------------------- #
    # Discover checkpoints                                                #
    # ------------------------------------------------------------------- #

    all_ckpts = discover_checkpoints(run_dir)
    if not all_ckpts:
        LOG.error("No checkpoints found under %s/checkpoints/", run_dir)
        return 2
    LOG.info("Found %d checkpoints in %s", len(all_ckpts), run_dir)
    for path, step, tokM in all_ckpts:
        LOG.debug("  - %s  step=%d tokens_M=%d", path.name, step, tokM)

    final_path, final_step, final_tokM = select_final_checkpoint(all_ckpts)
    LOG.info(
        "Final checkpoint (-> main): %s  (step=%d, tokens=%dM)",
        final_path.name,
        final_step,
        final_tokM,
    )

    mapping = build_canonical_mapping(all_ckpts) if not args.only_main else {}
    if mapping:
        LOG.info("Intermediate checkpoints to push:")
        for branch, (path, step, tokM) in sorted(
            mapping.items(), key=lambda kv: kv[1][2]
        ):
            LOG.info("  %-12s  ← %s  (step=%d)", branch, path.name, step)
    else:
        LOG.info("No intermediate checkpoints will be pushed.")

    # ------------------------------------------------------------------- #
    # Ensure repo                                                         #
    # ------------------------------------------------------------------- #

    ensure_repo(api, args.repo_id, private=args.private, dry_run=args.dry_run)

    # ------------------------------------------------------------------- #
    # Prepare local staging copies (add tokenizer + minimal files)        #
    # ------------------------------------------------------------------- #

    # Each ckpt produced by save_pretrained already has: config.json,
    # generation_config.json (sometimes), model.safetensors. We add the
    # tokenizer + tokenizer_config + generation_config to it before pushing.
    # We do this in place (the run dir is on Drive; this is safe).
    LOG.info("Adding tokenizer files to each checkpoint dir (in-place) ...")
    to_decorate: List[Path] = [final_path]
    if not args.only_main:
        to_decorate.extend(p for (p, _s, _t) in mapping.values())

    for ckpt_dir in to_decorate:
        # Don't re-copy if tokenizer.model already exists with same size as src
        src = args.tokenizer.resolve()
        tgt = ckpt_dir / "tokenizer.model"
        if tgt.is_file() and src.is_file() and tgt.stat().st_size == src.stat().st_size:
            LOG.debug("  tokenizer already present in %s", ckpt_dir.name)
            continue
        if args.dry_run:
            LOG.debug("  [dry-run] would copy tokenizer into %s", ckpt_dir)
            continue
        try:
            write_tokenizer_into_dir(src, ckpt_dir)
            LOG.debug("  tokenizer copied into %s", ckpt_dir.name)
        except Exception as exc:  # noqa: BLE001
            LOG.error("Could not write tokenizer into %s: %s", ckpt_dir, exc)
            return 2

    # ------------------------------------------------------------------- #
    # Push final → main                                                   #
    # ------------------------------------------------------------------- #

    run_summary = parse_summary(run_dir)
    card = render_model_card(
        condition=args.condition,
        seed=args.seed,
        run_summary=run_summary,
        repo_id=args.repo_id,
        n_branches=len(mapping),
    )

    # Write the README into the final ckpt dir (it will land on main).
    if not args.dry_run:
        (final_path / "README.md").write_text(card)

    LOG.info("-" * 70)
    LOG.info("Pushing final checkpoint to main ...")
    t0 = time.time()
    upload_checkpoint(
        api,
        repo_id=args.repo_id,
        branch="main",
        folder_path=final_path,
        commit_message=f"main: final checkpoint (step={final_step}, tokens={final_tokM}M)",
        dry_run=args.dry_run,
    )
    LOG.info("  main ✓  (%.1fs)", time.time() - t0)

    # Also drop a copy of the run-level logs onto main for traceability.
    if not args.dry_run:
        for fn in ("config.yaml", "summary.json", "meta.json", "pi_history.csv", "log.jsonl"):
            src = run_dir / fn
            if not src.is_file():
                continue
            try:
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=f"_run/{fn}",
                    repo_id=args.repo_id,
                    repo_type="model",
                    revision="main",
                    commit_message=f"add _run/{fn}",
                )
                LOG.info("  uploaded _run/%s", fn)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("could not upload %s: %s", fn, exc)

    # ------------------------------------------------------------------- #
    # Push intermediate checkpoints to chck_NM branches                   #
    # ------------------------------------------------------------------- #

    if not args.only_main:
        ordered_branches = sorted(mapping.keys(), key=lambda b: int(b.split("_")[1][:-1]))
        LOG.info("-" * 70)
        LOG.info("Pushing %d intermediate checkpoints ...", len(ordered_branches))
        for i, branch in enumerate(ordered_branches, start=1):
            ckpt_dir, step, tokM = mapping[branch]
            LOG.info("[%d/%d] %s", i, len(ordered_branches), branch)
            ensure_branch(api, args.repo_id, branch, dry_run=args.dry_run)
            t0 = time.time()
            upload_checkpoint(
                api,
                repo_id=args.repo_id,
                branch=branch,
                folder_path=ckpt_dir,
                commit_message=f"{branch}: ckpt step={step} tokens={tokM}M",
                dry_run=args.dry_run,
            )
            LOG.info("  %s ✓  (%.1fs)", branch, time.time() - t0)

    # ------------------------------------------------------------------- #
    # Final report                                                        #
    # ------------------------------------------------------------------- #

    LOG.info("=" * 70)
    if args.dry_run:
        LOG.info("DRY-RUN done. Nothing was uploaded.")
    else:
        LOG.info("Upload complete.")
        LOG.info("  Repo:      https://huggingface.co/%s", args.repo_id)
        LOG.info("  main:      step=%d tokens=%dM", final_step, final_tokM)
        if not args.only_main:
            for branch in sorted(mapping.keys(), key=lambda b: int(b.split("_")[1][:-1])):
                _p, step, tokM = mapping[branch]
                LOG.info("  %-12s  step=%d tokens=%dM", branch, step, tokM)
    LOG.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
