# Masked-Diffusion BabyLM — Strict-Small (English)

Non-autoregressive, sample-efficient language learning for the **BabyLM 2026
Strict-Small** track. We train a LLaDA/MDLM-style **absorbing-state masked
diffusion** language model on a GPT-2-scale bidirectional Transformer and
compare it head-to-head against the matched-scale autoregressive GPT-2 baseline.

> **Hypothesis.** Masked diffusion trades *compute for data*: its randomized
> masking objective is an implicit data augmentation over token orderings, so it
> keeps improving from repeated passes over a tiny corpus where AR models
> saturate (Prabhudesai et al., 2025, [arXiv:2507.15857](https://arxiv.org/abs/2507.15857)).
> The 10M-word Strict-Small budget is exactly the data-constrained regime where
> diffusion is predicted to win.

This folder is the **research project**. The official BabyLM evaluation
pipelines live in the sibling `../strict/` (Strict + Strict-Small) and
`../multilingual/` directories.

## Track constraints (encoded in `configs/base.yaml`)

| Constraint | Strict-Small value |
| --- | --- |
| Unique training words | ≤ **10M** |
| Max epochs | **10** |
| Total exposure (input + generated) | ≤ **100M** words (final checkpoint) |
| Required checkpoints | every **1M** up to 10M, then every **10M** up to 100M |
| Hub revisions | `chck_1M … chck_10M, chck_20M … chck_100M` |
| Model | public on HuggingFace |

## Repository layout

```
diffusion/
├── README.md                  ← you are here
├── requirements.txt
├── configs/                   experiment configs (base + conditions)
│   ├── base.yaml              shared defaults + all CFP constraints
│   ├── MD_base.yaml           MVP: uniform masked diffusion
│   ├── MD_freq_mask.yaml      ablation: frequency-informed masking
│   ├── MD_layerdup.yaml       extension: inference-time layer duplication
│   └── AR_baseline_ref.yaml   autoregressive control (eval-only reference)
├── mdlm/                      the model package (importable, smoke-testable)
│   ├── config.py              MaskedDiffusionConfig (HF, with auto_map)
│   ├── model.py               MaskedDiffusionLM (bidirectional, HF PreTrainedModel)
│   ├── masking.py             forward noising process + MDLM/LLaDA loss
│   ├── data.py                English text streaming + synthetic smoke corpus
│   └── scoring.py             minimal-pair scoring (PLL / ELBO)
├── scripts/
│   ├── prepare_data.py        download corpus, train BPE tokenizer, pre-tokenize
│   ├── train.py               the training loop (CFP checkpoint schedule)
│   ├── upload_to_hf.py        push run → Hub as chck_NM branches
│   ├── diffusion_eval_backend.py   diffusion-native zero-shot scorer
│   └── colab_bootstrap.sh     one-command Colab setup (+ Drive persistence)
├── notebooks/                 simple, logged Colab notebooks
│   ├── colab.ipynb            train
│   ├── evaluation.ipynb       evaluate (official `mlm` backend) + collate
│   └── upload_pipeline.ipynb  upload checkpoints
├── docs/
│   ├── EXPERIMENTS.md         research protocol: hypotheses, conditions, success criteria
│   ├── STORAGE.md             ← where checkpoints / predictions / evals are saved
│   └── EVALUATION.md          how to evaluate a diffusion model + submit
└── checkpoints/               placeholder (real ckpts live in runs/, see STORAGE.md)
```

## Quickstart

```bash
pip install -r requirements.txt

# 0) Sanity check the whole pipeline on CPU (synthetic data, ~30s):
python scripts/train.py --smoke-test --condition MD_base --seed 42

# 1) Prepare data + tokenizer (Strict-Small corpus):
python scripts/prepare_data.py

# 2) Train the MVP:
python scripts/train.py --condition MD_base --seed 42 \
    --token-data data/tokens --tokenizer tokenizer/mdlm_bpe_16k

# 3) Upload checkpoints to the Hub (final → main, intermediates → chck_NM):
python scripts/upload_to_hf.py --run-dir runs/<run> \
    --repo-id <user>/babylm-2026-strict-small-mdlm-seed42 \
    --tokenizer-dir tokenizer/mdlm_bpe_16k --condition MD_base --seed 42

# 4) Evaluate + collate the submission (see docs/EVALUATION.md).
```

On Colab, run `notebooks/colab.ipynb` instead — it bootstraps everything and
persists data/checkpoints to Google Drive.

## Why this is scored as a masked LM

A masked-diffusion denoiser predicts masked tokens from bidirectional context,
which is exactly what the official pipeline's `mlm` backend evaluates (per-token
pseudo-log-likelihood). So we submit with `--backend mlm` and only need the
diffusion-native scorer for the ELBO / layer-duplication variants. Details in
[`docs/EVALUATION.md`](docs/EVALUATION.md).
