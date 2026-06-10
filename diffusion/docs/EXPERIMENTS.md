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

## 6. Phased plan (with status)

1. **Setup & sanity** — smoke test + `prepare_data.py`. ***(done)***
2. **MVP** — train one `MD_base` seed end-to-end; upload; full eval; compare to
   baseline. ***(done — see §6.1 for the outcome and §6.2 for the pipeline
   fixes it triggered)***
3. **Curve** — fast eval on every `chck_*` checkpoint to plot §4. ***(running)***
4. **Robustness** — add seeds 13 & 71 for MD_base; report median ± min–max.
5. **Ablations** — low-`t` masking emphasis (new, motivated by the MVP result),
   `MD_freq_mask`, `MD_layerdup` (via the ELBO scorer's duplication knob).

Current results vs. the leaderboard and the prioritized next steps live in
[`NEXT_STEPS.md`](NEXT_STEPS.md).

### 6.1 MVP outcome (go/no-go call)

Verdict: **the MVP "succeeded" as an experiment, not as a leaderboard win** —
exactly the informative-either-way design of §1. With a clean, verified
training pipeline (v2 below), `MD_base` seed 42 at the full 10-epoch budget:

* **H4 (Entity Tracking): supported.** ~2× the official GPT-2 baseline and at
  the level of the best leaderboard entries — attributable to bidirectionality,
  *without* layer duplication yet.
* **H1 (BLiMP) / H2 (COMPS): not supported at this budget.** BLiMP sits well
  below the AR/hybrid baselines and was *unchanged* by the v2 training fixes,
  which rules out "training bug" as the explanation: the gap is attributable to
  the objective at ≤10 epochs. This is the pre-registered
  "compute-for-data trade-off fails" branch of §5 — the crossover analysis (§4
  curve) is now the headline deliverable.
* **H3 (GLUE): pending.**
* Working mechanistic hypothesis for the BLiMP↔EntityTracking split: PLL
  scoring probes the low-masking regime (`t→0`), but training spreads compute
  over `t ~ U(0,1)`; models trained *only* at low masking ratios (GPT-BERT)
  excel at BLiMP. Motivates the low-`t` ablation in phase 5.

### 6.2 Training pipeline v2 (fixes applied after the first MVP run)

The first MVP run exposed three implementation problems; all are fixed and the
MVP was re-trained from scratch on the corrected pipeline:

| Fix | Was | Now |
| --- | --- | --- |
| **Data shuffling** (`mdlm/data.py`) | shuffle unit = multi-million-token *shard*; blocks served sequentially → batches were 32 consecutive slices of one domain, identical block boundaries every epoch | per-epoch shuffling at three granularities: chunk order, random block-boundary offset, served-block order. Loss oscillations (domain cycling) disappeared |
| **Loss normalization** (`mdlm/masking.py`) | divided by the batch's masked-token count → loss scale fluctuated with the sampled `t`'s (≈2× CE, very noisy) | exact LLaDA Eq. 3: per-sequence `1/(t·L)`, batch-averaged. Loss ≈ CE, ~5× less noise |
| **Word accounting** (`scripts/train.py`) | hardcoded `words_per_token=0.78` → the run stopped at ~8 true epochs, leaving budget unused | true ratio read from `manifest.json` (~0.61), budget capped at `min(100M, 10 × n_words)`, `floor` not `ceil` on steps → full 10 epochs, guaranteed CFP-compliant |
| **Dev split** (`mdlm/data.py`) | last shard only (single domain) | stride-sampled chunks across the whole corpus |

Net effect: val CE dropped from ~7.5–9 (noisy, domain-biased) to a stable
~5.9 with val ≈ train (no overfitting). Downstream zero-shot scores were
**unchanged within noise** — which is itself the §6.1 finding.

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
python scripts/upload_to_hf.py --run-dir runs/<run> --repo-id amosluna/... \
    --tokenizer-dir tokenizer/mdlm_bpe_16k --condition MD_base --seed 42
```
