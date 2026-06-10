#!/usr/bin/env python3
"""Upload a masked-diffusion run to HuggingFace as a BabyLM-2026 Strict-Small submission.

Takes a local run directory (e.g. ``runs/2026-06-08_MD_base_seed42/``) that
contains ``checkpoints/step_NNNNN_words_NNNM/`` subfolders and pushes each
required checkpoint to a single HF repo, exposing the intermediate checkpoints as
HF *branches* (revisions) named ``chck_NM`` exactly as the BabyLM eval pipeline
expects.

Strict-Small canonical revision set (see strict/evaluation_pipeline/collate_preds.py):
    chck_1M ... chck_9M        (every 1M up to 10M)
    chck_10M, chck_20M ... chck_100M   (every 10M up to 100M)
The final / highest-words checkpoint also lands on ``main``.

Each pushed folder contains: ``config.json`` + ``model.safetensors`` (from
save_pretrained), the **custom modeling code** (``config.py`` + ``model.py``, so
``trust_remote_code=True`` works on the Hub), the tokenizer files, and
``ckpt_meta.json``. ``main`` additionally gets a README model card.

Because a masked-diffusion denoiser is scored like a masked LM, the config
registers ``AutoModelForMaskedLM`` — so the official pipeline's ``mlm`` backend
evaluates the uploaded model directly.

Usage:
    export HF_TOKEN="hf_..."
    python scripts/upload_to_hf.py \
        --run-dir runs/2026-06-08_MD_base_seed42 \
        --repo-id amosluna/babylm-2026-strict-small-mdlm-seed42 \
        --tokenizer-dir tokenizer/mdlm_bpe_16k \
        --condition MD_base --seed 42
    # add --dry-run to plan without uploading.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

LOG = logging.getLogger("upload_to_hf")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

REPO_ROOT = Path(__file__).resolve().parents[1]
MDLM_DIR = REPO_ROOT / "mdlm"
_CKPT_RE = re.compile(r"^step_(\d+)_words_(\d+)M$")

# Files the Hub needs to reconstruct the model with trust_remote_code=True.
CODE_FILES = ["config.py", "model.py"]


def discover_checkpoints(run_dir: Path) -> list[tuple[Path, int, int]]:
    """Return [(path, step, words_M)] for every checkpoint, sorted by words."""
    root = run_dir / "checkpoints"
    if not root.is_dir():
        LOG.error("No checkpoints/ under %s", run_dir)
        return []
    out = []
    for p in sorted(root.iterdir()):
        m = _CKPT_RE.match(p.name)
        if p.is_dir() and m:
            out.append((p, int(m.group(1)), int(m.group(2))))
    out.sort(key=lambda t: t[2])
    return out


def branch_name_for_words(words_M: int) -> str | None:
    """Map a words-in-millions count to its Strict-Small chck_NM branch, or None."""
    if 1 <= words_M <= 9:
        return f"chck_{words_M}M"
    if 10 <= words_M <= 100 and words_M % 10 == 0:
        return f"chck_{words_M}M"
    return None


def build_canonical_mapping(ckpts) -> dict[str, tuple[Path, int, int]]:
    out: dict[str, tuple[Path, int, int]] = {}
    for path, step, words_M in ckpts:
        bn = branch_name_for_words(words_M)
        if bn is None:
            continue
        prev = out.get(bn)
        if prev is None or step > prev[1]:
            out[bn] = (path, step, words_M)
    return out


def decorate_checkpoint(ckpt_dir: Path, tokenizer_dir: Path | None, dry_run: bool) -> None:
    """Copy the custom code (+ tokenizer) next to the weights so the Hub repo is self-contained."""
    for fn in CODE_FILES:
        src = MDLM_DIR / fn
        if src.is_file() and not dry_run:
            shutil.copy2(src, ckpt_dir / fn)
    if tokenizer_dir and tokenizer_dir.is_dir() and not dry_run:
        for f in tokenizer_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, ckpt_dir / f.name)
        _normalize_tokenizer_class(ckpt_dir / "tokenizer_config.json")


def _normalize_tokenizer_class(cfg_path: Path) -> None:
    """Pin a transformers-version-portable tokenizer_class so the official eval
    pipeline (transformers 4.51.x) can load tokenizers saved by transformers>=5
    (which write tokenizer_class="TokenizersBackend", unknown to 4.x)."""
    if not cfg_path.is_file():
        return
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("tokenizer_class") != "PreTrainedTokenizerFast":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.pop("backend", None)
        cfg_path.write_text(json.dumps(cfg, indent=2))


def render_model_card(condition: str, seed: int, run_summary: dict, repo_id: str, n_branches: int) -> str:
    s = run_summary.get("summary", {})
    return f"""---
language: [en]
tags: [babylm-2026, strict-small, masked-diffusion, mdlm, llada, {condition.lower()}]
license: apache-2.0
---

# Masked-Diffusion BabyLM (Strict-Small) — `{condition}` seed `{seed}`

BabyLM 2026 **Strict-Small** (English) submission checkpoint. A LLaDA/MDLM-style
absorbing-state masked-diffusion language model on a GPT-2-scale bidirectional
Transformer, trained on <=10M unique words for <=10 epochs (<=100M words seen).

- **Method**: `{condition}` (masked diffusion)
- **Seed**: `{seed}`
- **Words seen (final)**: `{s.get("words_seen_total", "n/a")}`
- **Intermediate checkpoints**: `{n_branches}` branches `chck_1M ... chck_100M`

## Usage

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForMaskedLM.from_pretrained("{repo_id}", trust_remote_code=True)
# Intermediate checkpoint:
# model = AutoModelForMaskedLM.from_pretrained("{repo_id}", revision="chck_10M", trust_remote_code=True)
```

## Evaluation

A masked-diffusion denoiser is scored like a masked LM (per-token
pseudo-log-likelihood), so the official BabyLM pipeline evaluates it with the
`mlm` backend:

```bash
cd strict
./eval_zero_shot.sh {repo_id} mlm
./eval_finetuning.sh --model_path {repo_id} --seed 42
bash scripts/collate_preds.sh {repo_id} mlm strict-small --fast
```

See the project's `docs/EVALUATION.md` for the diffusion-native ELBO scorer and
the inference-time layer-duplication variant.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--tokenizer-dir", type=Path, default=None,
                   help="Directory with HF tokenizer files (must define a [MASK] token).")
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--private", action="store_true")
    p.add_argument("--only-main", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        LOG.error("pip install huggingface_hub"); return 2

    if not args.hf_token and not args.dry_run:
        LOG.error("No HF token (set --hf-token or env HF_TOKEN), or pass --dry-run."); return 2
    api = HfApi(token=args.hf_token) if not args.dry_run else HfApi()

    run_dir = args.run_dir.resolve()
    ckpts = discover_checkpoints(run_dir)
    if not ckpts:
        LOG.error("No checkpoints under %s/checkpoints/", run_dir); return 2

    final_path, final_step, final_words = max(ckpts, key=lambda t: t[1])
    LOG.info("Final checkpoint (-> main): %s (%dM words)", final_path.name, final_words)
    mapping = {} if args.only_main else build_canonical_mapping(ckpts)
    for bn, (_p, _s, wm) in sorted(mapping.items(), key=lambda kv: kv[1][2]):
        LOG.info("  %-10s <- %s", bn, _p.name)

    if not args.dry_run:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    summary = {}
    sp = run_dir / "summary.json"
    if sp.is_file():
        summary["summary"] = json.loads(sp.read_text())

    # Push main (final checkpoint).
    decorate_checkpoint(final_path, args.tokenizer_dir, args.dry_run)
    if not args.dry_run:
        (final_path / "README.md").write_text(
            render_model_card(args.condition, args.seed, summary, args.repo_id, len(mapping))
        )
    LOG.info("Pushing final -> main ...")
    if not args.dry_run:
        api.upload_folder(folder_path=str(final_path), repo_id=args.repo_id, repo_type="model",
                          revision="main", commit_message=f"main: final ({final_words}M words)",
                          ignore_patterns=["*.tmp", "__pycache__/*", "trainer_state.pt"])

    # Push intermediate checkpoints.
    if not args.only_main:
        for bn in sorted(mapping, key=lambda b: int(b.split("_")[1][:-1])):
            path, step, wm = mapping[bn]
            decorate_checkpoint(path, args.tokenizer_dir, args.dry_run)
            LOG.info("Pushing %s ...", bn)
            if args.dry_run:
                continue
            try:
                api.create_branch(repo_id=args.repo_id, branch=bn, repo_type="model", exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("create_branch(%s): %s", bn, exc)
            t0 = time.time()
            api.upload_folder(folder_path=str(path), repo_id=args.repo_id, repo_type="model",
                              revision=bn, commit_message=f"{bn}: step={step} words={wm}M",
                              ignore_patterns=["*.tmp", "__pycache__/*", "trainer_state.pt"])
            LOG.info("  %s done (%.1fs)", bn, time.time() - t0)

    if args.dry_run:
        LOG.info("DRY-RUN complete; nothing uploaded.")
    else:
        LOG.info("Done. https://huggingface.co/%s", args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
