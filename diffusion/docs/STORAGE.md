# Storage & Folder Structure

Everything an experiment produces — checkpoints, logs, predictions, evaluation
results, and the final submission — has a fixed home. Because all training runs
on **Google Colab**, the durable artifacts live on **Google Drive** and are
symlinked into the repo by `scripts/colab_bootstrap.sh`, so a runtime disconnect
never loses work.

## 1. Google Drive root

`scripts/colab_bootstrap.sh --drive-root <ROOT>` symlinks three repo folders to
Drive (default `ROOT = /content/drive/MyDrive/Researchs/BabyLM_diffusion_G4`):

```
/content/drive/MyDrive/Researchs/BabyLM_diffusion_G4/
├── data/
│   ├── tokens/                     pre-tokenized shards (shard_0000.npy, … + manifest.json)
│   └── _synthetic/                 throwaway smoke-test corpus
├── tokenizer/
│   └── mdlm_bpe_16k/               HF tokenizer dir (defines [MASK] at id == vocab_size)
├── runs/                           one subfolder per (condition, seed) training run
└── results/                        evaluation runs persisted from Colab (see §4)
```

`data/` and `tokenizer/` are produced once by `scripts/prepare_data.py` and
reused across runs. `data/tokens/manifest.json` records the unique-word count so
we can confirm the **≤10M Strict-Small budget**.

## 2. A training run: `runs/<run>/`

`scripts/train.py` creates `runs/{YYYY-MM-DD}_{condition}_seed{S}/`, e.g.
`runs/2026-06-08_MD_base_seed42/`:

```
runs/2026-06-08_MD_base_seed42/
├── config.yaml                 merged base + condition snapshot (full provenance)
├── meta.json                   git SHA, GPU, start/finish time, status
├── checkpoint_schedule.json    the CFP word→step schedule actually used
├── log.jsonl                   one JSON line per logged step / eval
├── train_loss.csv              per-logged-step training loss
├── summary.json                final summary (words seen, #checkpoints, wall-clock)
└── checkpoints/
    ├── step_00781_words_001M/   ← 1M words seen
    ├── step_01562_words_002M/
    │   …                        every 1M up to 10M
    ├── step_07812_words_010M/   ← 10M words seen
    ├── step_15625_words_020M/
    │   …                        every 10M up to 100M
    └── step_78125_words_100M/   ← final (also pushed to `main`)
```

### Checkpoint folder naming

`step_{step:05d}_words_{N:03d}M` where `N` is **words seen in millions** (input
words counted with repeats — the CFP unit). The exact step is derived from the
word budget at runtime: `words_per_step = batch_size × block_size ×
words_per_token`. Each checkpoint contains a standard
`save_pretrained` dump plus `ckpt_meta.json`:

```
step_07812_words_010M/
├── config.json                 MaskedDiffusionConfig (incl. auto_map)
├── model.safetensors           weights
├── trainer_state.pt            optimizer + LR scheduler + RNG + step (resume; not uploaded)
└── ckpt_meta.json              {step, words_seen, words_m, saved_at}
```

The set of required checkpoints is fixed by the CFP for Strict-Small:
`1M, 2M, …, 10M, 20M, 30M, …, 100M` (no 200M–1000M).

**Auto-resume.** Because `runs/` lives on Drive, every checkpoint (weights +
`trainer_state.pt`) survives a Colab disconnect. Re-launching
`train.py --condition X --seed S` without `--output-dir` finds the existing run
for that `(condition, seed)`, loads the latest checkpoint (model + optimizer +
RNG) and continues at the saved step — the data stream is deterministic in the
step index, so the order is preserved. Pass `--no-resume` to force a fresh run.
`trainer_state.pt` is for local resume only and is **not** pushed to the Hub.

## 3. On the HuggingFace Hub

`scripts/upload_to_hf.py` maps each run checkpoint to a Hub **branch / revision**
the evaluation pipeline understands:

| Run checkpoint | Hub revision |
| --- | --- |
| `step_*_words_001M` … `step_*_words_009M` | `chck_1M` … `chck_9M` |
| `step_*_words_010M`, `020M`, … `100M` | `chck_10M`, `chck_20M`, … `chck_100M` |
| highest words (final) | `main` |

Each branch additionally gets the **custom code** (`config.py`, `model.py`) and
the **tokenizer** files, so `AutoModelForMaskedLM.from_pretrained(repo,
revision="chck_10M", trust_remote_code=True)` works for the evaluators. `main`
also carries the `README.md` model card.

```
https://huggingface.co/amosluna/babylm-2026-strict-small-mdlm-seed42
├── (main)        config.json, model.safetensors, config.py, model.py,
│                 tokenizer.json, tokenizer_config.json, …, README.md, ckpt_meta.json
├── (chck_1M)     same files for the 1M-word checkpoint
├── (chck_2M)     …
└── (chck_100M)   …
```

## 4. Evaluation results & prediction JSONs

Evaluation is run from `../strict/` (official pipeline) and/or the
diffusion-native `scripts/diffusion_eval_backend.py`. Both write into the same
official layout, rooted at `strict/results/`:

```
strict/results/
└── <model_stem>/
    ├── main/                              ← FULL evaluation (final model)
    │   ├── zero_shot/mlm/
    │   │   ├── blimp/blimp_filtered/predictions.json
    │   │   ├── blimp/supplement_filtered/predictions.json
    │   │   ├── ewok/ewok_filtered/predictions.json
    │   │   ├── comps/comps/predictions.json
    │   │   ├── entity_tracking/entity_tracking/predictions.json
    │   │   ├── reading/predictions.json
    │   │   └── AoA_word/surprisal.json
    │   └── finetune/
    │       ├── boolq/predictions.json   (+ results.txt)
    │       ├── mnli/ …  mrpc/ multirc/ qqp/ rte/ wsc/
    ├── chck_1M/  …  chck_100M/            ← FAST evaluation (each checkpoint)
    │   └── zero_shot/mlm/
    │       ├── blimp/blimp_fast/predictions.json
    │       ├── blimp/supplement_fast/predictions.json
    │       ├── ewok/ewok_fast/predictions.json
    │       └── entity_tracking/entity_tracking_fast/predictions.json
    └── all_full_preds_and_fast_scores_mlm.json   ← THE SUBMISSION FILE
```

* **`predictions.json`** shape: `{ "<UID>": { "predictions": [ {"id": "<UID>_0",
  "pred": <sentence-or-label>}, … ] } }`. For zero-shot minimal-pair tasks `pred`
  is the candidate the model scored highest; scores are computed **server-side**
  against held-out targets, so we only upload predictions.
* **Full** evaluation (zero-shot + GLUE fine-tuning) runs on `main`.
* **Fast** evaluation (subsampled zero-shot, no fine-tuning) runs on every
  `chck_NM` revision — this is what makes the submission a valid Challenge entry.
* **`all_full_preds_and_fast_scores_mlm.json`** is produced by
  `collate_preds.sh <model> mlm strict-small` and is the file uploaded to the
  [leaderboard](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026).

### Persisting eval results to Drive (append-only)

`strict/results/` lives on the **ephemeral** Colab disk. Cell 10 of
`3_evaluation_pipeline.ipynb` copies everything to Drive under **one immutable
directory per eval run** (the standard MLflow/W&B-style layout): grouped by
model, named with an ISO-8601 timestamp so runs sort chronologically and can
never collide — multiple evals of the same model on the same day coexist, and
**nothing is ever overwritten** (the cell aborts if the target dir exists).

```
{DRIVE_ROOT}/results/
└── amosluna__babylm-2026-strict-small-mdlm-seed42/     ← model (/" -> "__")
    ├── 2026-06-08_143052/                              ← one eval = one folder
    │   ├── eval_meta.json        model_id, backend, track, eval date,
    │   │                         git SHA of the eval code, tasks covered
    │   ├── results_summary.csv   flattened scores (split, task, metric, score)
    │   ├── results/              full copy of strict/results/ (reports + predictions)
    │   └── *.zip                 submission file, if collate_preds was run
    └── 2026-06-10_091500/                              ← a later eval, untouched
        └── …
```

`eval_meta.json` is the provenance record: months later it tells you exactly
which model, code version, and task set produced each number in the paper.

## 5. One-glance data flow

```
prepare_data.py ─► data/tokens + tokenizer/        (≤10M unique words)
        │
train.py ───────► runs/<run>/checkpoints/step_*_words_*M  + logs/summary
        │
upload_to_hf.py ─► Hub: main + chck_1M … chck_100M  (code + tokenizer bundled)
        │
eval (mlm backend / diffusion_eval_backend.py) ─► strict/results/<model>/…/predictions.json
        │
collate_preds.sh ─► all_full_preds_and_fast_scores_mlm.json ─► leaderboard
        │
notebook 3, Cell 10 ─► Drive: results/<model>/<timestamp>/  (append-only archive)
```
