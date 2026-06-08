#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  TAAM Colab / Linux GPU bootstrap
# ──────────────────────────────────────────────────────────────────────────────
#  One-liner setup that takes a fresh Colab runtime (or any Linux GPU box) to
#  "ready to train" in ~3 minutes. Idempotent: safe to re-run after `git pull`
#  to pick up new dependencies without re-downloading data.
#
#  What it does, in order:
#      1. Records the GPU model + driver version (for the paper's "Hardware" §).
#      2. Installs the project in editable mode with GPU-only extras.
#      3. Reads HF_TOKEN from either:
#           a) Colab Secret named "HF_TOKEN"  (preferred; never logged)
#           b) environment variable HF_TOKEN
#           c) .env file at the repo root
#         and verifies access to the gated BabyBabelLM datasets.
#      4. (Optional) Mounts Google Drive and symlinks data/, tokenizer/, runs/
#         to a persistent location, so re-runs after a runtime disconnect do
#         NOT re-download the 2.6 GB corpus or re-train the tokenizer.
#      5. Runs the data pipeline (idempotent: skipped if already cached).
#      6. Runs the GPU smoke test so any silent install failure surfaces NOW
#         instead of 20 minutes into a 30-minute training run.
#
#  Usage (from a Colab cell or shell):
#      bash scripts/colab_bootstrap.sh                 # full setup
#      bash scripts/colab_bootstrap.sh --skip-data     # if data already cached
#      bash scripts/colab_bootstrap.sh --no-drive      # don't touch Drive
#      bash scripts/colab_bootstrap.sh --drive-root /content/drive/MyDrive/Researchs/BabyLM
#
#  Exit codes:
#      0  success — ready for training
#      2  CUDA not visible
#      3  HF auth failed (missing token or unaccepted gated repo)
#      4  data pipeline failed
#      5  GPU smoke test failed
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRIVE_ROOT="/content/drive/MyDrive/Researchs/BabyLM"
DO_DRIVE=1
DO_DATA=1
DO_SMOKE=1
EXTRAS="train,analysis"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-drive)        DO_DRIVE=0; shift ;;
    --drive-root)      DRIVE_ROOT="$2"; shift 2 ;;
    --skip-data)       DO_DATA=0; shift ;;
    --skip-smoke)      DO_SMOKE=0; shift ;;
    --extras)          EXTRAS="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0" | sed 's/^#\s\{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

step() { echo -e "\n\033[1;36m▶ $*\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $*\033[0m"; }
fail() { echo -e "\033[1;31m✗ $*\033[0m"; exit "${2:-1}"; }
ok()   { echo -e "\033[1;32m✓ $*\033[0m"; }

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hardware fingerprint
# ──────────────────────────────────────────────────────────────────────────────
step "1/6  Hardware"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  warn "nvidia-smi not found — proceeding, but GPU may not be available."
fi
echo "Python: $(python3 --version 2>&1)"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Install
# ──────────────────────────────────────────────────────────────────────────────
step "2/6  Install (taam[$EXTRAS])"
pip install --quiet --upgrade pip
# The "[$EXTRAS]" group pins torch/transformers/datasets/accelerate.
# We do NOT pin a specific CUDA torch wheel here: Colab images ship with a
# torch build that already matches the runtime's CUDA. Reinstalling torch
# from PyPI can break that match. Instead, only install if torch missing.
if ! python3 -c "import torch" &>/dev/null; then
  pip install --quiet "torch>=2.1"
fi
pip install --quiet -e ".[$EXTRAS]"
ok "Install done."

# ──────────────────────────────────────────────────────────────────────────────
# 3. HF auth
# ──────────────────────────────────────────────────────────────────────────────
step "3/6  HuggingFace authentication"
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$REPO_ROOT/.env" ]]; then
  # Pull HF_TOKEN out of .env without exporting other vars.
  HF_TOKEN_FROM_ENV=$(grep -E '^HF_TOKEN=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
  if [[ -n "$HF_TOKEN_FROM_ENV" ]]; then
    export HF_TOKEN="$HF_TOKEN_FROM_ENV"
  fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  fail "HF_TOKEN not set. In Colab, add it via Secrets (key icon, left sidebar) and re-run with:
    import os; from google.colab import userdata; os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')" 3
fi
python3 scripts/verify_hf_auth.py || fail "HF gated-access check failed." 3

# ──────────────────────────────────────────────────────────────────────────────
# 4. Drive persistence (optional)
# ──────────────────────────────────────────────────────────────────────────────
step "4/6  Persistence"
if [[ $DO_DRIVE -eq 1 && -d "/content/drive/MyDrive" ]]; then
  mkdir -p "$DRIVE_ROOT"/{data/hf_cache,data/tokens,tokenizer,runs,wandb}
  for dir in data/hf_cache data/tokens tokenizer runs wandb; do
    src="$DRIVE_ROOT/$dir"
    dst="$REPO_ROOT/$dir"
    # If the dst already exists and is not a symlink, move its contents to
    # Drive (so we don't lose work) and then symlink.
    if [[ -e "$dst" && ! -L "$dst" ]]; then
      warn "moving existing $dst into Drive"
      mkdir -p "$src"
      rsync -a --remove-source-files "$dst/" "$src/" 2>/dev/null || true
      rm -rf "$dst"
    fi
    mkdir -p "$src"
    ln -snf "$src" "$dst"
  done
  ok "Drive symlinks ready at $DRIVE_ROOT"
elif [[ $DO_DRIVE -eq 1 ]]; then
  warn "Google Drive is not mounted (/content/drive/MyDrive missing). Using ephemeral disk."
  warn "To persist between sessions, mount Drive in a previous cell:"
  warn "  from google.colab import drive; drive.mount('/content/drive')"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 5. Data pipeline (idempotent)
# ──────────────────────────────────────────────────────────────────────────────
step "5/6  Data pipeline"
if [[ $DO_DATA -eq 1 ]]; then
  TOKENS_MANIFEST="$REPO_ROOT/data/tokens/manifest.json"
  TOKENIZER_MODEL="$REPO_ROOT/tokenizer/spm_32k_en_nl_zh.model"
  if [[ -f "$TOKENS_MANIFEST" && -f "$TOKENIZER_MODEL" ]]; then
    ok "data/tokens/ + tokenizer present — skipping (delete them to force re-build)."
  else
    python3 scripts/download_data.py || fail "data download failed" 4
    python3 scripts/build_composition_report.py || fail "composition report failed" 4
    if [[ ! -f "$TOKENIZER_MODEL" ]]; then
      python3 scripts/train_tokenizer.py \
        --vocab-size 32000 --bytes-per-lang 50000000 \
        --output "$TOKENIZER_MODEL" || fail "tokenizer training failed" 4
    fi
    python3 scripts/pretokenize.py || fail "pretokenization failed" 4
  fi
  python3 scripts/smoke_test_data_pipeline.py --n-batches 1500 || \
    warn "data pipeline smoke check warned — review above before training."
else
  ok "Data pipeline skipped (--skip-data)."
fi

# ──────────────────────────────────────────────────────────────────────────────
# 6. GPU smoke test
# ──────────────────────────────────────────────────────────────────────────────
step "6/6  GPU smoke test"
if [[ $DO_SMOKE -eq 1 ]]; then
  python3 scripts/gpu_smoke_test.py --condition TAAM --steps 5 || \
    fail "GPU smoke test failed — fix before launching long runs." 5
else
  ok "GPU smoke test skipped (--skip-smoke)."
fi

echo
ok "Bootstrap complete. Suggested next commands:"
cat <<'EOF'

    # Single condition (auto-creates runs/{YYYY-MM-DD}_TAAM_seed42/):
    python scripts/train.py --condition TAAM --seed 42 \
        --token-data data/tokens \
        --tokenizer tokenizer/spm_32k_en_nl_zh.model \
        --total-steps 20000

    # Full matrix (11 conditions x 3 seeds, with retries, resumable):
    python scripts/run_matrix.py --output-dir runs --seeds 13 42 71

EOF
