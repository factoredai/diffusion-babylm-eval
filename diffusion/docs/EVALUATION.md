# Evaluating a Masked-Diffusion Model

The official BabyLM pipeline ships backends for `causal`, `mlm`, `mntp`, and
encoder-decoder models. A masked-diffusion model is **not** autoregressive, but
it *is* a masked denoiser — so it slots into the existing `mlm` backend with no
pipeline edits. This document explains why, and how to run the full submission.

## 1. Why `mlm` is the right backend

All Strict-Small zero-shot tasks (BLiMP, BLiMP-supplement, EWoK, COMPS, Entity
Tracking) are **minimal-pair** tasks: assign a higher score to the correct
sentence. The `mlm` backend scores a sentence by masking each target token in
turn and reading off `log p(token | bidirectional context)`, then summing — the
**pseudo-log-likelihood** (Salazar et al., 2020).

That is precisely how a LLaDA/MDLM denoiser is meant to be scored: our model was
trained to recover masked tokens from bidirectional context. We register the
model under `AutoModelForMaskedLM` (via `auto_map` in `MaskedDiffusionConfig`),
so:

```bash
cd ../strict
./eval_zero_shot.sh amosluna/babylm-2026-strict-small-mdlm-seed42 mlm
```

works out of the box (the eval pipeline loads it with `trust_remote_code=True`).

> **Prerequisite.** The uploaded tokenizer must define a `[MASK]` token whose id
> equals `vocab_size` (this is what `scripts/prepare_data.py` produces and what
> the model expects as its absorbing state). `scripts/upload_to_hf.py` bundles
> the tokenizer and the modeling code into every branch.

## 2. Full submission checklist (Strict-Small)

Run from `../strict/` after the model is public on the Hub.

```bash
MODEL=amosluna/babylm-2026-strict-small-mdlm-seed42

# (a) FULL zero-shot on the final model (main)
./eval_zero_shot.sh $MODEL mlm

# (b) AoA across checkpoints (required for all tracks)
./eval_aoa.sh $MODEL mlm strict-small

# (c) FAST zero-shot on every chck_NM checkpoint
./eval_zero_shot_fast_all_revisions.sh $MODEL mlm strict-small

# (d) GLUE fine-tuning on the final model
./eval_finetuning.sh --model_path $MODEL --seed 42

# (e) Collate into the submission file (server-side scoring)
bash scripts/collate_preds.sh $MODEL mlm strict-small
```

The result is `strict/results/<model>/all_full_preds_and_fast_scores_mlm.json` —
upload it to the [leaderboard](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026).
See [`STORAGE.md`](STORAGE.md) for exactly where every file lands.

> The fast-revision script assumes checkpoints run up to the full budget. If you
> trained for fewer words (e.g. stopped at 20M), edit its revision loop and the
> assumptions in `evaluation_pipeline/collate_preds.py` accordingly — incomplete
> evaluation is allowed and missing tasks count as 0.

## 3. Diffusion-native scorer (ELBO + layer duplication)

`scripts/diffusion_eval_backend.py` is an alternative scorer for two cases the
`mlm` backend cannot express:

* **ELBO scoring** — a Monte-Carlo negative-ELBO bound instead of single-token
  PLL (`--scoring elbo`).
* **Inference-time layer duplication** — the "reasoning depth" extension from the
  proposal, repeating the middle Transformer blocks (`--layer_duplication_factor 2`).
  Targets the COMPS / Entity Tracking hypotheses.

It reads the official task data and writes `predictions.json` in the **same**
layout (with `--backend mlm`), so its output is interchangeable with the official
backend and collates the same way:

```bash
python scripts/diffusion_eval_backend.py \
    --model_path_or_name $MODEL --revision_name chck_10M \
    --task comps --data_path ../strict/evaluation_data/full_eval/comps \
    --scoring elbo --layer_duplication_factor 2 \
    --backend mlm --save_predictions --output_dir ../strict/results
```

Supported tasks: `blimp`, `ewok`, `comps`, `entity_tracking`.

## 4. Comparing against the autoregressive baseline

`configs/AR_baseline_ref.yaml` points at the matched-scale official baseline
(`BabyLM-community/gpt2-baseline-BabyLM-2026-Strict-Small`). Evaluate it with the
**`causal`** backend through the same pipeline so the only variable is the
training objective:

```bash
./eval_zero_shot.sh BabyLM-community/gpt2-baseline-BabyLM-2026-Strict-Small causal
```

The published baseline scores (BLiMP 65.08, BLiMP-supp 57.25, COMPS 51.81,
Entity Tracking 21.07; GLUE in the repo README) are the numbers our diffusion
model must beat to confirm the proxy hypothesis.
