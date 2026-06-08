# LLaDA-like diffusion trainer

The corpus is downloaded and cached on first run.

Setup (uv): `uv venv --python 3.13 && source .venv/bin/activate && uv pip install -r strict/requirements.txt`
Setup (pip): `python3 -m venv .venv && source .venv/bin/activate && pip install -r strict/requirements.txt`
Run (from the repo root): `uv run train/llada_like_base.py` or python -m it
Checkpoints are saved to `ckpt/diffusion-babylm`.

The checkpoint is a RobertaForMaskedLM, so the whole pipeline scores it with the `mlm` backend (no edits needed, despite strict/README's diffusion note). All commands run from `strict/`:
Get eval data once: `python strict/scripts/download_evals.py`
Zero-shot: `bash strict/scripts/eval_zero_shot.sh ../ckpt/diffusion-babylm mlm`
AoA: `bash strict/scripts/eval_aoa.sh ../ckpt/diffusion-babylm mlm strict-small`
Fine-tuning (GLUE/SuperGLUE): `bash strict/scripts/eval_finetuning.sh --model_path ../ckpt/diffusion-babylm`
HOWEVER it needs a TODO: `eval_finetuning.sh` hardcodes `--take_final --padding_side left` (last-token pooling, for decoders). This is a bidirectional encoder, so edit those two args out of the script's `finetune.run` calls; the defaults then pool the first/CLS token (`encoding[:, 0]`) with right padding. They're baked into the script, not flags you can pass to it.
TODO: a valid Challenge submission also scores every `chck_*M` checkpoint on the fast suite (`eval_zero_shot_fast_all_revisions.sh`) then collates with `collate_preds.sh`, but this trainer saves only the final model — add periodic checkpoint saving.

I ran a 10-step training run and validated that all validation scripts work except the eval_finetuning (per above) and AOA needs the checkpoints.
