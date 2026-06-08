"""Smoke test for the H4 pipeline (both modes).

Mode 1 (preferred): synthetic `eval_results/chck_NM/<slug>/results_*.json`
files, exercising the raw-lm-eval-output ingestion path.

Mode 2 (fallback): synthetic `*_predictions.json` files, exercising the
aggregated-keys ingestion path used by `notebooks/evaluation.ipynb` Cell 7.

Both modes go through the full pipeline (bridge → analyzer → Δρ → Figure 6)
and assert that all output CSVs and the figure exist.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure REPO_ROOT is importable
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.h4.predictions_to_eval_results import (   # noqa: E402
    FAST_EVAL_REVISIONS, load_task_map, DEFAULT_TASK_MAP,
)
from scripts.h4.run_h4_pipeline import run_pipeline    # noqa: E402


def _synthetic_record(
    cond: str,
    revision_idx: int,
    n_revisions: int,
    task_map: dict,
) -> dict:
    """Produce a dict {task: {"acc,none": float}} where accuracy depends on
    (a) the task's language, (b) the condition, and (c) how far into training
    we are (revision_idx/n_revisions in [0, 1])."""
    prog = revision_idx / max(1, n_revisions - 1)  # 0..1

    # Per-condition "speed multiplier" per language. TAAM-v2 is faster on ZH;
    # B0 is faster on EN/NL.
    if cond == "taam_v2":
        speed = {"eng": 0.85, "nld": 0.85, "zho": 1.30}
    else:
        speed = {"eng": 1.05, "nld": 1.00, "zho": 0.70}

    # Per-phenomenon difficulty (higher = later mastery). Mirrors the YAML's
    # child norms loosely (lex 24m / morph 30m / hard syntax 60m).
    base_difficulty = {
        "sv_agreement_number": 0.20,
        "negation":             0.18,
        "det_noun_number":      0.20,
        "wh_matrix":            0.30,
        "anaphor_binding":      0.55,
        "passive":              0.55,
        "relative_clauses":     0.65,
        "npi_licensing":        0.70,
        "v2_word_order":        0.30,
        "past_tense":           0.30,
        "argument_omission":    0.25,
        "quantifier_scope":     0.70,
        "classifier_noun":      0.30,
        "aspect_le_zhe":        0.25,
    }

    record: dict[str, dict[str, float]] = {}
    for phen_id, lang_map in task_map.items():
        diff = base_difficulty.get(phen_id, 0.40)
        for lang, task_names in lang_map.items():
            if not task_names:
                continue
            s = speed.get(lang, 1.0)
            # Sigmoid-ish trajectory in `prog`, centred at `diff`, scaled by `s`.
            effective_prog = prog * s
            # Mastery curve: 0.50 baseline (chance) -> 0.95 asymptote
            #  accuracy(t) = 0.50 + 0.45 / (1 + exp(-8 * (effective_prog - diff)))
            import math
            acc = 0.50 + 0.45 / (1.0 + math.exp(-8.0 * (effective_prog - diff)))
            # Add a tiny per-task offset to break ties (deterministic via task name hash)
            acc += 0.005 * ((hash(task_names[0]) % 7) - 3) / 7
            acc = max(0.0, min(1.0, acc))
            for t in task_names:
                record[t] = {"acc,none": float(acc), "acc_stderr,none": 0.01}
    return record


def _write_synthetic_predictions(pred_dir, task_map, n_revisions):
    """Mode 2 (fallback) — `*_predictions.json` with aggregated structure."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    for cond in ("taam_v2", "b0"):
        fast = [
            _synthetic_record(cond, i, n_revisions, task_map)
            for i in range(n_revisions)
        ]
        zeroshot = fast[-1]
        pred = {
            "zeroshot": zeroshot,
            "finetune": {},
            "fast_eval_results": [
                {t: {t: v["acc,none"]} for t, v in entry.items()}
                for entry in fast
            ],
        }
        path = pred_dir / f"babylm-2026-{cond}-seed42_predictions.json"
        path.write_text(json.dumps(pred, indent=2), encoding="utf-8")


def _write_synthetic_eval_results_dir(eval_dir, task_map, n_revisions, revisions):
    """Mode 1 (preferred) — `eval_results/chck_NM/<slug>/results_*.json`."""
    for cond in ("taam_v2", "b0"):
        slug = f"amosluna__babylm-2026-{cond}-seed42"
        for i, rev in enumerate(revisions[:n_revisions]):
            rev_dir = eval_dir / rev / slug
            rev_dir.mkdir(parents=True, exist_ok=True)
            record = _synthetic_record(cond, i, n_revisions, task_map)
            data = {
                "model_name": slug,
                "config": {
                    "model_args": f"pretrained=amosluna/babylm-2026-{cond}-seed42,"
                                  f"revision={rev}",
                },
                "results": {
                    task: {"acc,none": float(stats["acc,none"]),
                            "acc_stderr,none": 0.01}
                    for task, stats in record.items()
                },
            }
            path = rev_dir / "results_2026-05-28T01-00-00.000000.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("smoke")

    task_map = load_task_map(DEFAULT_TASK_MAP)

    smoke_dir = REPO_ROOT / "runs/_smoke_h4"
    n_revisions = 24
    revisions = FAST_EVAL_REVISIONS[:n_revisions]

    # ──────────────────────────────────────────────────────────────────────
    # Mode 1: eval-results-dir (preferred)
    # ──────────────────────────────────────────────────────────────────────
    eval_dir = smoke_dir / "eval_results"
    log.info("[mode 1] writing synthetic eval_results dir at %s", eval_dir)
    _write_synthetic_eval_results_dir(eval_dir, task_map, n_revisions, revisions)

    log.info("[mode 1] running pipeline...")
    s1 = run_pipeline(
        eval_results_dir=eval_dir,
        h4_models=["taam_v2-seed42", "b0-seed42"],
        out_runs_dir=smoke_dir / "mode1" / "runs",
        out_results_dir=smoke_dir / "mode1" / "results",
        out_figures_dir=smoke_dir / "mode1" / "figures",
        task_map_path=DEFAULT_TASK_MAP,
        n_min=4,
        n_bootstrap=2_000,
        mirror_dirs=[smoke_dir / "mode1" / "mirror"],
    )
    assert (smoke_dir / "mode1" / "results" / "rho_table.csv").exists()
    assert (smoke_dir / "mode1" / "results" / "delta_rho_table.csv").exists()
    assert (smoke_dir / "mode1" / "figures" / "figure6_acquisition_order.png").exists()
    assert (smoke_dir / "mode1" / "mirror" / "delta_rho_table.csv").exists(), \
        "mirror_dirs did not copy outputs"
    log.info("[mode 1] OK (%d ρ rows, %d Δρ rows)",
             len(s1["rho_rows"]), len(s1["delta_rows"]))

    # ──────────────────────────────────────────────────────────────────────
    # Mode 2: predictions-glob (fallback)
    # ──────────────────────────────────────────────────────────────────────
    pred_dir = smoke_dir / "mode2" / "predictions"
    log.info("\n[mode 2] writing synthetic predictions dir at %s", pred_dir)
    _write_synthetic_predictions(pred_dir, task_map, n_revisions)
    log.info("[mode 2] running pipeline...")
    s2 = run_pipeline(
        predictions_paths=sorted(pred_dir.glob("*_predictions.json")),
        out_runs_dir=smoke_dir / "mode2" / "runs",
        out_results_dir=smoke_dir / "mode2" / "results",
        out_figures_dir=smoke_dir / "mode2" / "figures",
        task_map_path=DEFAULT_TASK_MAP,
        n_min=4,
        n_bootstrap=2_000,
    )
    assert (smoke_dir / "mode2" / "results" / "rho_table.csv").exists()
    assert (smoke_dir / "mode2" / "results" / "delta_rho_table.csv").exists()
    log.info("[mode 2] OK (%d ρ rows, %d Δρ rows)",
             len(s2["rho_rows"]), len(s2["delta_rows"]))

    # ──────────────────────────────────────────────────────────────────────
    # Verify the empty-rho regression: read delta_rho_table.csv with pandas
    # to confirm we do NOT raise EmptyDataError even when rows is empty.
    # ──────────────────────────────────────────────────────────────────────
    import pandas as pd
    for mode in ("mode1", "mode2"):
        df = pd.read_csv(smoke_dir / mode / "results" / "delta_rho_table.csv")
        assert list(df.columns), f"{mode}/delta_rho_table.csv has no columns"
        log.info("[%s] pd.read_csv(delta_rho_table.csv): %d cols, %d rows",
                 mode, len(df.columns), len(df))

    print()
    print("=" * 72)
    print("Δρ table from mode 1 (eval-results-dir):")
    print("=" * 72)
    print(f"  {'lang':4s} | {'rho(taam_v2)':>12s} | {'rho(b0)':>8s} | "
          f"{'delta':>8s} | {'p_holm':>7s}")
    for d in s1["delta_rows"]:
        print(f"  {d['language']:4s} | {d['rho_a']:>12s} | {d['rho_b']:>8s} | "
              f"{d['delta']:>8s} | {d['p_holm']:>7s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
