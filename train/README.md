# LLaDA-like diffusion trainer

The corpus is downloaded and cached on first run.

Setup (uv): `uv venv --python 3.13 && source .venv/bin/activate && uv pip install -r strict/requirements.txt`
Setup (pip): `python3 -m venv .venv && source .venv/bin/activate && pip install -r strict/requirements.txt`
Run (from the repo root): `uv run train/llada_like_base.py` or python -m it
Checkpoints are saved to `ckpt/diffusion-babylm`.
