# Results so far & Next Steps

Status as of **2026-06-12**. Model: `MD_base` [see configs/ folder], seed 42, trained on the
**v2 pipeline** (see `EXPERIMENTS.md` §6.2) for the full Strict-Small budget
(10 epochs / ~61M words-seen cap from the manifest). Zero-shot via the official
`mlm` backend (PLL); GLUE via official fine-tuning harness (`eval_finetuning.sh`).

### What is `MD_base`?

**LLaDA-style masked diffusion at GPT-2 scale on BabyLM** — not the 8B LLaDA model.
Same training recipe (absorbing `[MASK]` state, uniform masking ratio `t ~ U(0,1)`,
LLaDA Eq. 3 loss weighting), but ~98M parameters, bidirectional Transformer matched
to the official GPT-2 Strict-Small baseline, trained on ≤10M unique words / ≤10 epochs.
This matches the `grupo4.md` plan: *"a diffusion variant on top of GPT-2 BabyLM's
baseline"*.

## 1. Results obtained so far (vs. the leaderboard)

| Task | MD_base (ours) | GPT-2 baseline | Best leaderboard (masked/hybrid) | Read |
| --- | --- | --- | --- | --- |
| **Entity Tracking** | **41.64** | 21.07 | ~42.1 (rwkv7), 39–40 (GPT-BERT) | **top-tier — ~2× the baseline, at the level of the best entries** |
| BLiMP | 54.89 | 65.08 | ~69–71 (GPT-BERT) | below baseline; the pre-registered "≤10-epoch" failure mode |
| BLiMP supplement | 48.57 | 57.25 | ~63 | same as BLiMP |
| COMPS | 51.02 | 51.81 | ~52–55 | at chance-ish level — same as nearly everyone |
| **(Super)GLUE macro (H3)** | **60.8** | 63.8 | 66.0 (GPT-BERT masked) | below AR baseline; bidirectional encoder helps on boolq only |

### GLUE per-task breakdown (primary metric per task)

| Task | Metric | Ours | GPT-2 baseline | Read |
| --- | --- | --- | --- | --- |
| boolq | accuracy | **66.97** | 65.87 | **win** — bidirectional context pays off |
| mnli | accuracy | 40.10 | 49.80 | **big loss** — main drag on macro (−9.7) |
| mrpc | f1 | 81.37 | 83.49 | slight loss |
| multirc | accuracy | 58.37 | 64.52 | loss |
| qqp | f1 | 60.66 | 60.86 | tie |
| rte | accuracy | 56.83 | 60.43 | slight loss |
| wsc | accuracy | 61.54 | — | accuracy OK but f1/mcc = 0 → likely one-class collapse |

Macro = mean of the seven primary metrics above (same rule as the leaderboard
`about.py` / `print_results_table.py`).

### Diffusion-native scorer (Cell 8 — ELBO + layer duplication)

Optional ablation via `diffusion_eval_backend.py` (not the official submission
scorer). BLiMP only, `main` checkpoint:

| Scorer | BLiMP | Notes |
| --- | --- | --- |
| Official PLL (`mlm` backend, Cell 5) | **54.89** | submission path |
| ELBO + `layer_duplication_factor=2` (Cell 8) | **51.90** | ~3 pts *worse* |

**Finding:** ELBO scoring and inference-time layer duplication did **not** close
the BLiMP gap; the PLL deficit is not an evaluation-artifact story. For
leaderboard submission, use Cell 5 predictions only — Cell 8 overwrites the same
`predictions.json` path if run afterwards.

What this means against the pre-registered hypotheses:

* **H4 (Entity Tracking): supported.** Our single unambiguous win, attributable
  to bidirectionality. Layer duplication at inference (Cell 8) did not help
  BLiMP; Entity Tracking with ELBO+dup remains untested.
* **H1 (BLiMP) / H2 (COMPS): not supported at this budget.** Crucially, the
  scores were *unchanged* after the v2 training fixes → the gap is a property
  of the uniform-`t` diffusion objective at ≤10 epochs, **not** a training bug.
  This is the informative "compute-for-data trade-off" branch of the protocol.
* **H3 (GLUE): not supported.** Macro ~60.8 vs GPT-2 baseline 63.8 and GPT-BERT
  66.0. The bidirectional encoder helps on boolq (+1.1) but cannot compensate
  for weak semantic representations on mnli (−9.7) and multirc (−6.2). Same
  pattern as BLiMP: strong global context, weak fine-grained syntax/semantics.
* Working explanation for the split: PLL/BLiMP probes the low-masking regime
  (`t→0`), but uniform `t ~ U(0,1)` training spends most compute at high
  masking ratios. GPT-BERT (trained *only* at low masking) gets ~70 BLiMP and
  ~40 Entity Tracking; we get ~55 and ~41.6. The masking-ratio spectrum looks
  like the knob that controls the syntax ↔ tracking trade-off.

**Verdict: the MVP passed as an experiment.** We will not win Text Average with
`MD_base` — but the project promised a rigorous answer to "can diffusion trade
compute for data at 10 epochs?", and we now have the clean pipeline, one clear
win (H4), and three localized negatives (H1, H2, H3) to write it.

### What we achieved (team summary)

| Deliverable | Status |
| --- | --- |
| Train MD_base end-to-end (v2 pipeline, CFP-compliant) | **done** |
| Upload to Hub + official eval (zero-shot + GLUE) | **done** |
| Compare vs GPT-2 baseline & leaderboard | **done** |
| Pre-registered failure mode ("compute-for-data trade-off") | **confirmed** |
| ELBO / layer-dup ablation on BLiMP | **done** — no gain |
| Checkpoint learning curve (Cell 6) | in progress / verify |
| Low-`t` masking ablation | **not started** — key follow-up |
| Seeds 13 & 71 | not started |
| Leaderboard submission zip | pending (needs full fast eval + Cell 5 preds) |

**Aligned with `grupo4.md`?** Yes — the MVP asked to train a diffusion LM and
compare to the AR baseline under Strict-Small. We did that rigorously. We did
**not** beat the baseline on macro-average benchmarks, but we **did** produce a
publishable, pre-registered result: diffusion loses on syntax/GLUE at ≤10 epochs,
wins on entity tracking via bidirectionality, and the masking-ratio spectrum is
the leading mechanistic explanation.

## 2. Next steps (in order of value)

Done:

1. ~~**GLUE fine-tuning (Cell 7)**~~ — macro **60.8**; H3 not supported.
2. ~~**ELBO scorer (Cell 8)**~~ — BLiMP **51.9** with ELBO + layer dup ×2; no
   improvement over PLL (54.9). Hypothesis that PLL underrates diffusion:
   **rejected** for BLiMP.

In progress / verify:

3. **Checkpoint curve (Cell 6)** — fast eval on every `chck_1M…chck_100M`. Decides
   the headline figure: if BLiMP still has positive slope at the budget cap, the
   crossover lies *outside* Strict-Small.

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
