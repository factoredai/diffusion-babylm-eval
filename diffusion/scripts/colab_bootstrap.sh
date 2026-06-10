#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  Masked-Diffusion BabyLM (Strict-Small) — Colab / Linux GPU bootstrap
# ──────────────────────────────────────────────────────────────────────────────
#  Takes a fresh Colab runtime to "ready to train" in a couple of minutes.
#  Idempotent: safe to re-run after `git pull`. Everything heavy (data, tokenizer,
#  runs) is symlinked onto Google Drive so a runtime disconnect never loses work.
#
#  Usage (from a Colab cell or shell):
#      bash scripts/colab_bootstrap.sh                  # full setup
#      bash scripts/colab_bootstrap.sh --no-drive       # don't touch Drive
#      bash scripts/colab_bootstrap.sh --skip-data      # data already prepared
#      bash scripts/colab_bootstrap.sh --drive-root /content/drive/MyDrive/Researchs/BabyLM_diffusion_G4
#
#  Exit codes: 0 ok | 2 no CUDA | 3 HF auth | 4 data | 5 smoke test
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRIVE_ROOT="/content/drive/MyDrive/Researchs/BabyLM_diffusion_G4"
DO_DRIVE=1
DO_DATA=1
DO_SMOKE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-drive)   DO_DRIVE=0; shift ;;
    --drive-root) DRIVE_ROOT="$2"; shift 2 ;;
    --skip-data)  DO_DATA=0; shift ;;
    --skip-smoke) DO_SMOKE=0; shift ;;
    -h|--help)    sed -n '2,20p' "$0" | sed 's/^#\s\{0,1\}//'; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

step() { echo -e "\n\033[1;36m> $*\033[0m"; }
warn() { echo -e "\033[1;33m! $*\033[0m"; }
fail() { echo -e "\033[1;31mx $*\033[0m"; exit "${2:-1}"; }
ok()   { echo -e "\033[1;32mok $*\033[0m"; }

# 1) Hardware
step "1/5  Hardware"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  warn "nvidia-smi not found — GPU may not be available."
fi
echo "Python: $(python3 --version 2>&1)"

# 2) Install
step "2/5  Install dependencies"
pip install --quiet --upgrade pip
python3 -c "import torch" 2>/dev/null || pip install --quiet "torch>=2.1"
pip install --quiet -r requirements.txt
ok "Install done."

# 3) HF auth (token from Colab Secret or env or .env)
step "3/5  HuggingFace authentication"
if [[ -z "${HF_TOKEN:-}" && -f "$REPO_ROOT/.env" ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d \"\')"
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  warn "HF_TOKEN not set. In Colab add it via Secrets (key icon) and run:"
  warn "  import os; from google.colab import userdata; os.environ['HF_TOKEN']=userdata.get('HF_TOKEN')"
else
  python3 -c "from huggingface_hub import HfApi; print('HF user:', HfApi().whoami(token='${HF_TOKEN}')['name'])" \
    || fail "HF auth failed." 3
fi

# 4) Drive persistence (symlink data/tokenizer/runs onto Drive)
step "4/5  Persistence"
if [[ $DO_DRIVE -eq 1 && -d "/content/drive/MyDrive" ]]; then
  for dir in data tokenizer runs; do
    src="$DRIVE_ROOT/$dir"; dst="$REPO_ROOT/$dir"
    mkdir -p "$src"
    if [[ -e "$dst" && ! -L "$dst" ]]; then
      warn "moving existing $dst into Drive"; rsync -a --remove-source-files "$dst/" "$src/" 2>/dev/null || true; rm -rf "$dst"
    fi
    ln -snf "$src" "$dst"
  done
  ok "Drive symlinks ready at $DRIVE_ROOT"
elif [[ $DO_DRIVE -eq 1 ]]; then
  warn "Drive not mounted. Mount it first with drive.mount('/content/drive') in a Colab cell."
fi

# 5) Data + smoke test
step "5/5  Data pipeline + smoke test"
if [[ $DO_DATA -eq 1 ]]; then
  if [[ -f "$REPO_ROOT/data/tokens/manifest.json" ]]; then
    ok "data/tokens present — skipping (delete to rebuild)."
  else
    python3 scripts/prepare_data.py || fail "data preparation failed" 4
  fi
fi
if [[ $DO_SMOKE -eq 1 ]]; then
  python3 scripts/train.py --smoke-test --condition MD_base --seed 42 \
    || fail "smoke test failed — fix before launching long runs." 5
fi

echo
ok "Bootstrap complete. Next:"
cat <<'EOF'

    # Train the MVP (auto-creates runs/{YYYY-MM-DD}_MD_base_seed42/):
    python scripts/train.py --condition MD_base --seed 42 \
        --token-data data/tokens --tokenizer tokenizer/mdlm_bpe_16k

    # Upload checkpoints to the Hub as chck_NM branches:
    python scripts/upload_to_hf.py --run-dir runs/<run> \
        --repo-id amosluna/babylm-2026-strict-small-mdlm-seed42 \
        --tokenizer-dir tokenizer/mdlm_bpe_16k --condition MD_base --seed 42
EOF
