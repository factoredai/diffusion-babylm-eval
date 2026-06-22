#!/usr/bin/env python3
"""Patch an already-uploaded HF repo so the official BabyLM eval pipeline works,
without re-uploading the multi-hundred-MB weights.

Three fixes, applied to ``main`` AND every ``chck_*`` revision (only tiny text
files are pushed, so it runs in seconds):

1. **tokenizer_config.json** -- tokenizers saved by transformers>=5 carry
   ``"tokenizer_class": "TokenizersBackend"``, which the eval pipeline's
   transformers 4.51.x cannot resolve (``AutoProcessor`` -> "Unrecognized
   processing class"). We rewrite it to the portable ``"PreTrainedTokenizerFast"``.

2. **model.py / config.py** -- the custom modeling code is executed on the Hub
   via ``trust_remote_code=True``. We re-push the latest local copies from
   ``mdlm/`` so fixes (e.g. ``forward`` now tolerating ``token_type_ids`` that the
   ``reading`` task passes) reach already-uploaded checkpoints.

3. **config.json (auto_map)** -- the Auto* classes resolve remote code through
   the ``auto_map`` stored in *config.json* (frozen at training time), NOT
   through config.py. Old checkpoints only registered ``AutoModelForMaskedLM``,
   but the GLUE fine-tuning harness loads encoders with ``AutoModel`` ->
   "Unrecognized configuration class ... for this kind of AutoModel". We add
   ``AutoModel -> model.MaskedDiffusionModel`` (the headless encoder class).

Usage:
    export HF_TOKEN="hf_..."
    python scripts/fix_hub_for_eval.py --repo-id amosluna/babylm-2026-strict-small-mdlm-seed42
    # several repos / dry-run:
    python scripts/fix_hub_for_eval.py --repo-id A --repo-id B --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

LOG = logging.getLogger("fix_hub_for_eval")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

REPO_ROOT = Path(__file__).resolve().parents[1]
MDLM_DIR = REPO_ROOT / "mdlm"
CODE_FILES = ["model.py", "config.py"]  # executed on the Hub via trust_remote_code
PORTABLE_CLASS = "PreTrainedTokenizerFast"


AUTO_MAP = {
    "AutoConfig": "config.MaskedDiffusionConfig",
    "AutoModel": "model.MaskedDiffusionModel",          # GLUE finetune (headless)
    "AutoModelForMaskedLM": "model.MaskedDiffusionLM",  # zero-shot mlm backend
}


def normalize_tokenizer_cfg(cfg_text: str) -> tuple[str, bool]:
    cfg = json.loads(cfg_text)
    changed = cfg.get("tokenizer_class") != PORTABLE_CLASS or "backend" in cfg
    cfg["tokenizer_class"] = PORTABLE_CLASS
    cfg.pop("backend", None)  # tokenizers>=5-only key
    return json.dumps(cfg, indent=2), changed


def normalize_model_cfg(cfg_text: str) -> tuple[str, bool]:
    cfg = json.loads(cfg_text)
    auto_map = dict(cfg.get("auto_map", {}))
    changed = any(auto_map.get(k) != v for k, v in AUTO_MAP.items())
    auto_map.update(AUTO_MAP)
    cfg["auto_map"] = auto_map
    return json.dumps(cfg, indent=2), changed


def revisions_for(api, repo_id: str) -> list[str]:
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    branches = [b.name for b in refs.branches]
    return ["main"] + sorted(b for b in branches if b != "main")


def patch_repo(api, repo_id: str, dry_run: bool) -> None:
    from huggingface_hub import hf_hub_download

    revisions = revisions_for(api, repo_id)
    LOG.info("[%s] revisions to patch (%d): %s", repo_id, len(revisions), ", ".join(revisions))

    # Prepare the patched tokenizer_config.json (downloaded from main, normalized).
    tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", revision="main")
    tok_text, tok_changed = normalize_tokenizer_cfg(Path(tok_path).read_text())
    LOG.info("[%s] tokenizer_class -> %s (changed=%s)", repo_id, PORTABLE_CLASS, tok_changed)

    code_paths = {fn: MDLM_DIR / fn for fn in CODE_FILES if (MDLM_DIR / fn).is_file()}
    LOG.info("[%s] code files to re-push: %s", repo_id, ", ".join(code_paths) or "(none found)")

    if dry_run:
        LOG.info("[%s] DRY-RUN: would upload tokenizer_config.json + %s to each revision.",
                 repo_id, list(code_paths))
        return

    with tempfile.TemporaryDirectory() as td:
        tok_fp = Path(td) / "tokenizer_config.json"
        tok_fp.write_text(tok_text)
        for rev in revisions:
            api.upload_file(path_or_fileobj=str(tok_fp), path_in_repo="tokenizer_config.json",
                            repo_id=repo_id, repo_type="model", revision=rev,
                            commit_message="fix: portable tokenizer_class for eval pipeline")
            for fn, src in code_paths.items():
                api.upload_file(path_or_fileobj=str(src), path_in_repo=fn,
                                repo_id=repo_id, repo_type="model", revision=rev,
                                commit_message=f"fix: refresh {fn} (eval compatibility)")
            # config.json is frozen per checkpoint, so the auto_map must be
            # patched revision by revision.
            cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=rev)
            cfg_text, cfg_changed = normalize_model_cfg(Path(cfg_path).read_text())
            if cfg_changed:
                cfg_fp = Path(td) / f"config_{rev}.json"
                cfg_fp.write_text(cfg_text)
                api.upload_file(path_or_fileobj=str(cfg_fp), path_in_repo="config.json",
                                repo_id=repo_id, repo_type="model", revision=rev,
                                commit_message="fix: register AutoModel in auto_map (GLUE finetune)")
            LOG.info("[%s]   patched %s (auto_map %s)", repo_id, rev,
                     "updated" if cfg_changed else "ok")
    LOG.info("[%s] done. https://huggingface.co/%s", repo_id, repo_id)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, action="append", help="Repeatable.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        LOG.error("pip install huggingface_hub"); return 2

    if not args.hf_token and not args.dry_run:
        LOG.error("No HF token (set --hf-token or env HF_TOKEN), or pass --dry-run."); return 2
    api = HfApi(token=args.hf_token) if not args.dry_run else HfApi()

    for repo_id in args.repo_id:
        patch_repo(api, repo_id, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
