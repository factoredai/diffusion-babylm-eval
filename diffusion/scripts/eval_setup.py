#!/usr/bin/env python3
"""One-shot environment setup for the evaluation notebook (Colab).

Everything that used to live in Cell 2 of ``3_evaluation_pipeline.ipynb``:

1. ``pip install -r strict/requirements.txt`` (official eval pins).
2. Remove ``torchvision`` + ``timm``: Colab preinstalls versions ABI-incompatible
   with the pinned ``torch==2.7.0``; this eval is text-only and never needs them,
   but transformers crashes if they are present-and-broken (torchvision::nms via
   AutoProcessor; ``import timm`` via AutoModel's model-mapping enumeration).
3. ``touch .env`` at the repo root (``eval_finetuning.sh`` sources it; we keep
   secrets in Colab Secrets instead).
4. Download the official eval data snapshot (BLiMP/COMPS/entity_tracking/GLUE/...).
5. Extract ``evaluation_data/fast_eval/ewok_fast.zip`` — the EWoK *fast* set ships
   inside the snapshot as a password-protected zip (password ``BabyLM2025``, see
   the official README). Skipping this made Cell 6 fail with
   ``FileNotFoundError: evaluation_data/fast_eval/ewok_fast``.
6. Best-effort: download + vocab-filter the gated EWoK *full* dataset (requires
   accepting the terms of ewok-core/ewok-core-1.0 with the HF_TOKEN account).

Also exposes ``apply_quiet_env()`` so the notebook kernel can inherit the same
log-silencing environment for every subsequent ``!`` cell.

Usage (from anywhere; paths are resolved relative to this file):
    python diffusion/scripts/eval_setup.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = REPO_ROOT / "strict"

EWOK_FAST_ZIP_PASSWORD = b"BabyLM2025"  # from the official strict/README.md

QUIET_ENV = {
    "TF_CPP_MIN_LOG_LEVEL": "3",          # hide TensorFlow INFO/WARNING banners
    "TF_ENABLE_ONEDNN_OPTS": "0",         # drop the oneDNN notice
    "TRANSFORMERS_VERBOSITY": "error",    # mute "A new version ... was downloaded"
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # no file-download bars
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONWARNINGS": "ignore",           # silence protobuf/other UserWarnings
}


def apply_quiet_env() -> None:
    """Silence TF / HF / protobuf noise for this process and its children."""
    os.environ.update(QUIET_ENV)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=STRICT_DIR, check=True, **kw)


def _step(msg: str) -> None:
    print(f">> {msg}", flush=True)


def install_deps() -> None:
    _step("Installing eval requirements (quiet)...")
    _run([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off",
          "-r", "requirements.txt"])
    _step("Removing torchvision + timm (ABI-broken on Colab, unused by text evals)...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    "torchvision", "timm"], cwd=STRICT_DIR)
    (REPO_ROOT / ".env").touch(exist_ok=True)  # sourced by eval_finetuning.sh


def download_eval_data() -> None:
    _step("Downloading official eval data snapshot...")
    _run([sys.executable, "-m", "scripts.download_evals"])


def extract_ewok_fast() -> None:
    """Unzip the password-protected EWoK fast set (required by Cells 6 and 9)."""
    dest = STRICT_DIR / "evaluation_data/fast_eval/ewok_fast"
    if dest.is_dir() and any(dest.glob("*.jsonl")):
        _step("EWoK fast data already extracted.")
        return
    zip_path = STRICT_DIR / "evaluation_data/fast_eval/ewok_fast.zip"
    if not zip_path.is_file():
        print(f"!! {zip_path} not found — run download step first.", file=sys.stderr)
        return
    _step("Extracting EWoK fast (password-protected zip)...")
    # Member paths inside the zip are rooted at evaluation_data/fast_eval/...,
    # so extraction target is the strict/ directory itself.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path=STRICT_DIR, pwd=EWOK_FAST_ZIP_PASSWORD)
    n = len(list(dest.glob("*.jsonl")))
    _step(f"EWoK fast ready: {n} domain files -> {dest.relative_to(STRICT_DIR)}")


def prepare_ewok_full() -> None:
    """Best-effort: gated EWoK full set (needs accepted terms + HF_TOKEN)."""
    dest = STRICT_DIR / "evaluation_data/full_eval/ewok_filtered"
    if dest.is_dir() and any(dest.glob("*.jsonl")):
        _step("EWoK full data already prepared.")
        return
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        _run([sys.executable, "-m", "evaluation_pipeline.ewok.dl_and_filter"])
        _step("EWoK full-eval data ready -> evaluation_data/full_eval/ewok_filtered")
    except Exception as e:  # gated dataset: skip, the rest of the suite still runs
        print(f"!! Skipping EWoK full eval (gated dataset not accessible): {e}")


def main() -> int:
    apply_quiet_env()
    install_deps()
    download_eval_data()
    extract_ewok_fast()
    prepare_ewok_full()
    _step("Setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
