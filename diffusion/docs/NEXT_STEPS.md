# Results so far & Next Steps

Status as of **2026-06-09**. Model: `MD_base` [see configs/ folder], seed 42, trained on the
**v2 pipeline** (see `EXPERIMENTS.md` §6.2) for the full Strict-Small budget
(10 epochs / ~61M words-seen cap from the manifest). Scores from the official
`mlm` backend (PLL), zero-shot full sets.

## 1. Results obtained so far (vs. the leaderboard)

| Task | MD_base (ours) | GPT-2 baseline | Best leaderboard (masked/hybrid) | Read |
| --- | --- | --- | --- | --- |
| **Entity Tracking** | **41.64** | 21.07 | ~42.1 (rwkv7), 39–40 (GPT-BERT) | **top-tier — ~2× the baseline, at the level of the best entries** |
| BLiMP | 54.89 | 65.08 | ~69–71 (GPT-BERT) | below baseline; the pre-registered "≤10-epoch" failure mode |
| BLiMP supplement | 48.57 | 57.25 | ~63 | same as BLiMP |
| COMPS | 51.02 | 51.81 | ~52–55 | at chance-ish level — same as nearly everyone |
| GLUE (H3) | *running* | ~64 macro | GPT-BERT ~66 | prediction: competitive (bidirectional encoder helps fine-tuning) |

What this means against the pre-registered hypotheses:

* **H4 (Entity Tracking): supported.** Our single unambiguous win, attributable
  to bidirectionality — and we haven't used layer duplication yet.
* **H1 (BLiMP) / H2 (COMPS): not supported at this budget.** Crucially, the
  scores were *unchanged* after the v2 training fixes → the gap is a property
  of the uniform-`t` diffusion objective at ≤10 epochs, **not** a training bug.
  This is the informative "compute-for-data trade-off" branch of the protocol.
* Working explanation for the split: PLL/BLiMP probes the low-masking regime
  (`t→0`), but uniform `t ~ U(0,1)` training spends most compute at high
  masking ratios. GPT-BERT (trained *only* at low masking) gets ~70 BLiMP and
  ~40 Entity Tracking; we get ~55 and ~41.6. The masking-ratio spectrum looks
  like the knob that controls the syntax ↔ tracking trade-off.

**Verdict: the MVP passed as an experiment.** We will not win Text Average with
`MD_base` — but the project promised a rigorous answer to "can diffusion trade
compute for data at 10 epochs?", and we now have the clean pipeline, the
positive result (H4), and the localized negative result (H1) to write it.

## 2. Next steps (in order of value)

Already running (eval notebook cells 6–8):

1. **Checkpoint curve (cell 6)** — fast eval on every `chck_1M…chck_100M`.
   Decides the headline figure: if BLiMP still has positive slope at the budget
   cap, the crossover lies *outside* Strict-Small (compute-limited, not
   structural).
2. **GLUE fine-tuning (cell 7)** — settles H3.
3. **ELBO scorer (cell 8)** — diffusion-native scoring
   (`diffusion_eval_backend.py`); may recover +2–5 BLiMP if part of the gap is
   PLL itself, and gives the `MD_layerdup` knob for free.

Then, in order:

4. **Low-`t` masking ablation** *(the key experiment)* — bias training toward
   low masking ratios (e.g. `t ∈ (0, 0.5)`). If BLiMP climbs toward the 60s
   while Entity Tracking holds ~40, that trade-off is the central result of
   the paper.
5. **Seeds 13 & 71 for MD_base** — robustness; report median ± min–max.
6. **`MD_freq_mask`** — (see configs/ folder) frequency-informed masking ablation.
7. **Submission** — `collate_preds.sh` on the best configuration → upload
   `all_full_preds_and_fast_scores_mlm.json` to the
   [leaderboard](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026).
8. *(if time allows)* matched AR baseline re-trained on our exact data pipeline
   for a checkpoint-for-checkpoint §4 curve.
