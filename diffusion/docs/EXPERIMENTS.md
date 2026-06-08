# Experiment Protocol

This is the research plan for the paper: *Masked-Diffusion BabyLM:
Non-Autoregressive Objectives for Sample-Efficient Language Learning*
(BabyLM 2026, **Strict-Small**, English). It fixes the hypotheses, conditions,
metrics, and success/failure criteria **before** running, so results are
interpretable regardless of outcome.

## 1. Research question

The CMU paper (Prabhudesai et al., 2025, arXiv:2507.15857) shows diffusion beats
AR in data-constrained settings **when compute is abundant** (it benefits from
repeating data up to ~500 effective epochs; AR saturates near ~15). BabyLM
Strict-Small **caps compute at ≤10 epochs / ≤100M words seen**. So we are not
replicating their regime — we are testing a sharper question:

> **Within the BabyLM Strict-Small compute budget (≤10 epochs over 10M words),
> is a masked-diffusion LM already competitive with — or better than — a
> matched-scale autoregressive baseline on linguistic-competence evaluations?**

The paper is informative either way: a win is evidence for diffusion on small
curated corpora; a loss localizes the compute threshold where diffusion would
overtake AR, which is itself a useful, honest result.

## 2. Hypotheses (pre-registered)

A masked-diffusion LM trained on 10M words will, relative to the GPT-2
Strict-Small baseline, show:

| ID | Claim | Primary metric |
| --- | --- | --- |
| H1 | Better syntactic knowledge | **BLiMP** (+ supplement) accuracy |
| H2 | Better compositional/property reasoning | **COMPS** accuracy |
| H3 | Generalizable semantic/pragmatic reps | **GLUE** (fine-tuned) macro |
| H4 | Better referential tracking (esp. with layer duplication) | **Entity Tracking** accuracy |

Baseline numbers to beat (official `gpt2-baseline-BabyLM-2026-Strict-Small`):
BLiMP **65.08**, BLiMP-supplement **57.25**, COMPS **51.81**,
Entity Tracking **21.07**; GLUE: boolq 65.87, mnli 49.80, mrpc 83.49,
multirc 64.52, qqp 60.86, rte 60.43.

## 3. Conditions

All conditions are 98M-parameter models (matched to the baseline; see
`configs/base.yaml`). Run each with **3 seeds** (13, 42, 71); report median and
min–max.

| Condition | Config | What it tests |
| --- | --- | --- |
| **AR baseline** | `AR_baseline_ref.yaml` | control (causal); published / re-evaluated |
| **MD_base** | `MD_base.yaml` | MVP: uniform masked diffusion (H1–H3) |
| **MD_freq_mask** | `MD_freq_mask.yaml` | ablation: frequency-informed masking |
| **MD_layerdup** | `MD_layerdup.yaml` | extension: inference-time reasoning depth (H4) |

## 4. The headline result: a compute↔performance curve

The single most informative figure. Because every intermediate checkpoint is
saved (`chck_1M … chck_100M`), we get the curve for free from one training run:

* **x-axis**: words seen (≈ epochs over the 10M corpus): 1M, 2M, …, 10M, 20M, …, 100M.
* **y-axis**: BLiMP / COMPS / Entity Tracking accuracy (fast eval per checkpoint).
* **two lines**: MD_base vs. AR baseline (the AR baseline also publishes
  intermediate checkpoints, so the comparison is checkpoint-for-checkpoint).

Read off: (i) does diffusion overtake AR, and (ii) at how many words/epochs —
i.e. is the crossover **inside** the Strict-Small budget?

## 5. Metrics & success / failure criteria

Map directly to the proposal's failure modes so we can call the result early:

| Failure mode | Early signal (monitored in `log.jsonl` + fast eval) | Decision |
| --- | --- | --- |
| **Overfitting** | val-loss rises while train-loss falls; BLiMP/COMPS drop on later checkpoints | reduce epochs / add dropout / smaller model |
| **Loss of coherence ("marginal trap")** | manual inspection of denoised samples; flat BLiMP-syntax | revisit masking schedule / sampling steps |
| **Compute-for-data trade-off fails** | after 100M words seen, MD ≤ AR on the macro-average | report the crossover analysis; this *is* the finding |

**Success (MVP):** MD_base beats the AR baseline macro-average at the 100M-words
checkpoint, by more than the seed min–max spread.

## 6. Phased plan

1. **Setup & sanity** — `scripts/train.py --smoke-test` (CPU), then
   `scripts/prepare_data.py` (real 10M corpus + tokenizer). *(done / validated)*
2. **MVP** — train one `MD_base` seed end-to-end; upload; full eval; compare to
   baseline. Go/no-go on the hypotheses.
3. **Curve** — use MD_base intermediate checkpoints + the AR baseline checkpoints
   to plot §4. This is the core figure.
4. **Robustness** — add seeds 13 & 71 for MD_base; report median ± min–max.
5. **Ablations / extension** — `MD_freq_mask`, `MD_layerdup`.
6. *(stretch)* Energy-based variant (arXiv:2410.21357) if the MVP succeeds.

## 7. Reproducibility

Every run records: merged `config.yaml`, git SHA + GPU (`meta.json`), per-step
loss (`train_loss.csv`), step/eval log (`log.jsonl`), and the CFP checkpoint
schedule. Seeds are fixed in `experiment.seed_pool`. See `docs/STORAGE.md` for
where everything lands and `docs/EVALUATION.md` for the scoring protocol.

## 8. Compute budget (Colab, practical)

98M params @ ctx 1024. Masked diffusion does one forward per micro-batch in
training (like MLM), but PLL eval costs O(tokens) forward passes — budget eval
time accordingly (use the fast eval set for checkpoints; full eval only on
`main`). If a run OOMs, lower `training.batch_size` and raise
`training.grad_accum_steps` (now wired) to keep the effective batch fixed.

## 9. Reproduce commands

```bash
# Data (once)
python scripts/prepare_data.py

# MVP run (seed 42)
python scripts/train.py --condition MD_base --seed 42 \
    --token-data data/tokens --tokenizer tokenizer/mdlm_bpe_16k -v

# Upload + evaluate + collate  (see docs/EVALUATION.md)
python scripts/upload_to_hf.py --run-dir runs/<run> --repo-id <user>/... \
    --tokenizer-dir tokenizer/mdlm_bpe_16k --condition MD_base --seed 42
```
