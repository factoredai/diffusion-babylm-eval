"""H4 analysis: Spearman ρ between model acquisition order and child norms.

Inputs (per condition × seed):
    runs/{cond}_seed{S}/eval_results.json  produced by scripts/eval.py
    analyses/acquisition_order/phenomenon_to_child_norm.yaml

Algorithm:
    For each (condition, seed, language, phenomenon):
        Read accuracy(step) from eval_results.json.
        Define t_acq(c, S, ℓ, p) = the smallest step at which
              accuracy(step') ≥ THRESHOLD for two consecutive evals.
        If no such step exists, t_acq = ∞ (recorded as -1 sentinel).

    For each (condition, seed, language):
        Pair every phenomenon p with t_acq and child age_months.
        Drop pairs where t_acq is ∞.
        Compute Spearman ρ on at least N_min pairs; report bootstrap CI.

Outputs:
    analyses/results/rho_table.csv          (consumed by Figure 6 panel A)
    analyses/results/zh_scatter.csv          (consumed by Figure 6 panel B)
    analyses/results/t_acq_table.csv         (per (cond, seed, lang, phen) t_acq)

Usage:
    python scripts/analyze_acquisition_order.py \
        --runs runs/_smoke_TAAM_seed42 runs/_smoke_B0_seed42 \
        --output-dir analyses/results/

If a (cond, lang) pair has fewer than N_min phenomena that ever cross the
threshold, we record ρ as NaN with a note in the row.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from scipy.stats import spearmanr

LOG = logging.getLogger("h4_analysis")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_THRESHOLD = 0.70
DEFAULT_CONSECUTIVE = 2
DEFAULT_N_BOOTSTRAP = 10_000

# Evanson et al. 2023 (§2.3) method defaults
DEFAULT_T_ACQ_METHOD = "relative"   # "relative" (Evanson) or "absolute" (legacy)
DEFAULT_RELATIVE_FRAC = 0.90        # reach 90% of final accuracy
DEFAULT_CHANCE = 0.50               # binary minimal-pair chance level
DEFAULT_ABOVE_CHANCE_MARGIN = 0.05  # a probe must finish >= chance + margin to count


# ──────────────────────────────────────────────────────────────────────────────
# Acquisition step extraction
# ──────────────────────────────────────────────────────────────────────────────


def _final_accuracy(accuracy_by_step: dict[int, float], tail: int = 3) -> float:
    """Robust 'final accuracy' = mean of the last `tail` checkpoints (Evanson use
    the converged value; the tail-mean is a low-variance estimator of it)."""
    steps = sorted(accuracy_by_step.keys())
    if not steps:
        return float("nan")
    tail_steps = steps[-tail:] if len(steps) >= tail else steps
    return sum(accuracy_by_step[s] for s in tail_steps) / len(tail_steps)


def extract_t_acq_absolute(
    accuracy_by_step: dict[int, float],
    threshold: float = DEFAULT_THRESHOLD,
    consecutive: int = DEFAULT_CONSECUTIVE,
) -> int | None:
    """Legacy method: first step with accuracy >= `threshold` for `consecutive`
    consecutive evals. Returns the first step of the qualifying run, or None.

    This CENSORS any probe that never sustains the absolute threshold, which is
    why it produced N<6 on most languages. Kept for the appendix robustness row.
    """
    steps = sorted(accuracy_by_step.keys())
    run = 0
    run_start = None
    for s in steps:
        a = accuracy_by_step[s]
        if a >= threshold:
            if run == 0:
                run_start = s
            run += 1
            if run >= consecutive:
                return run_start
        else:
            run = 0
            run_start = None
    return None


def extract_t_acq_relative(
    accuracy_by_step: dict[int, float],
    frac: float = DEFAULT_RELATIVE_FRAC,
    consecutive: int = DEFAULT_CONSECUTIVE,
    chance: float = DEFAULT_CHANCE,
    above_chance_margin: float = DEFAULT_ABOVE_CHANCE_MARGIN,
) -> int | None:
    """Evanson et al. 2023 (§2.3) method.

    'Acquisition time' = first step at which accuracy >= frac * final_accuracy,
    sustained for `consecutive` evals, where final_accuracy is the tail-mean.

    A probe is INCLUDED only if it is learned above chance level
    (final_accuracy >= chance + margin); otherwise return None (floor-locked /
    never-learned probe, excluded exactly as Evanson §3.1 does).

    This never censors an above-chance probe, which is the key fix for the
    N<6 NaN problem.
    """
    steps = sorted(accuracy_by_step.keys())
    if not steps:
        return None
    final = _final_accuracy(accuracy_by_step)
    if math.isnan(final) or final < (chance + above_chance_margin):
        return None  # not learned above chance -> excluded
    target = frac * final
    run = 0
    run_start = None
    for s in steps:
        if accuracy_by_step[s] >= target:
            if run == 0:
                run_start = s
            run += 1
            if run >= consecutive:
                return run_start
        else:
            run = 0
            run_start = None
    # If sustained-run never satisfied (rare for relative), fall back to the
    # first single crossing.
    for s in steps:
        if accuracy_by_step[s] >= target:
            return s
    return None


def extract_t_acq(
    accuracy_by_step: dict[int, float],
    threshold: float = DEFAULT_THRESHOLD,
    consecutive: int = DEFAULT_CONSECUTIVE,
    method: str = DEFAULT_T_ACQ_METHOD,
    relative_frac: float = DEFAULT_RELATIVE_FRAC,
) -> int | None:
    """Dispatch to the relative (Evanson, default) or absolute (legacy) method."""
    if method == "absolute":
        return extract_t_acq_absolute(accuracy_by_step, threshold, consecutive)
    return extract_t_acq_relative(
        accuracy_by_step, frac=relative_frac, consecutive=consecutive,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Spearman with bootstrap
# ──────────────────────────────────────────────────────────────────────────────


def spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    """Return (rho, ci_lo, ci_hi, n).

    Uses paired-resample bootstrap (each iteration resamples N indices with
    replacement). If N < 3, returns NaN for rho and CI.
    """
    n = len(x)
    if n < 3:
        return (math.nan, math.nan, math.nan, n)
    rho, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            r, _ = spearmanr(x[idx], y[idx])
        except Exception:
            continue
        if not math.isnan(r):
            rhos.append(r)
    if len(rhos) < 100:
        # Bootstrap mostly returned NaN (e.g., all ties); abstain.
        return (float(rho), math.nan, math.nan, n)
    rhos = np.array(rhos)
    lo = float(np.percentile(rhos, 100 * (alpha / 2)))
    hi = float(np.percentile(rhos, 100 * (1 - alpha / 2)))
    return (float(rho), lo, hi, n)


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ──────────────────────────────────────────────────────────────────────────────


def _load_phenomena_mapping(norms_csv: Path | None = None) -> dict:
    """Load child norms. Prefer the v2 CSV dataset; fall back to the legacy YAML.

    If `norms_csv` is given (or the default CSV exists), use it as the single
    source of truth (see docs/h4_child_norms.md). Otherwise read the old YAML.
    """
    csv_path = norms_csv or (REPO_ROOT / "analyses/acquisition_order/child_norms_dataset.csv")
    if csv_path and Path(csv_path).exists():
        try:
            from scripts.h4.child_norms import load_child_norms
        except ImportError:
            sys.path.insert(0, str(REPO_ROOT / "scripts" / "h4"))
            from child_norms import load_child_norms  # type: ignore
        return load_child_norms(Path(csv_path))
    path = REPO_ROOT / "analyses/acquisition_order/phenomenon_to_child_norm.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class TAcqRecord:
    condition_id: str
    seed: int
    language: str
    phenomenon_id: str
    t_acq_step: int | None
    child_age_months: float | None


def collect_t_acq_records(
    run_dirs: Sequence[Path],
    mapping: dict,
    threshold: float,
    consecutive: int,
    method: str = DEFAULT_T_ACQ_METHOD,
    relative_frac: float = DEFAULT_RELATIVE_FRAC,
) -> list[TAcqRecord]:
    """For each run × language × phenomenon, compute t_acq."""
    out: list[TAcqRecord] = []
    # Map phenomenon_id -> {lang: age_months}
    phen_to_age: dict[str, dict[str, float | None]] = {}
    for phen in mapping["phenomena"]:
        phen_to_age[phen["id"]] = {
            l: entry.get("age_months") for l, entry in phen["languages"].items()
        }

    for run_dir in run_dirs:
        eval_path = run_dir / "eval_results.json"
        if not eval_path.exists():
            LOG.warning("No eval_results.json in %s; skipping.", run_dir)
            continue
        eval_data = json.loads(eval_path.read_text())
        cond_id = eval_data["condition_id"]
        seed = eval_data["seed"]
        checkpoints = eval_data["checkpoints"]  # {step_str: {phen: {lang: acc}}}
        # Per (lang, phenomenon), collect accuracy curve
        acc_curves: dict[tuple[str, str], dict[int, float]] = {}
        for step_str, per_phen in checkpoints.items():
            step = int(step_str)
            for phen_id, per_lang in per_phen.items():
                for lang, acc in per_lang.items():
                    key = (lang, phen_id)
                    acc_curves.setdefault(key, {})[step] = float(acc)
        for (lang, phen_id), curve in acc_curves.items():
            t_acq = extract_t_acq(
                curve, threshold=threshold, consecutive=consecutive,
                method=method, relative_frac=relative_frac,
            )
            age = phen_to_age.get(phen_id, {}).get(lang)
            out.append(TAcqRecord(
                condition_id=cond_id,
                seed=seed,
                language=lang,
                phenomenon_id=phen_id,
                t_acq_step=t_acq,
                child_age_months=age,
            ))
    return out


def compute_rho_table(
    records: list[TAcqRecord],
    n_min: int = 8,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> list[dict]:
    """For each (condition, language), pool over seeds and compute Spearman."""
    # Group by (condition_id, language); within each group, aggregate per-seed
    # t_acq into a single value (median across seeds) per phenomenon.
    grouped: dict[tuple[str, str], dict[str, list[TAcqRecord]]] = {}
    for r in records:
        if r.child_age_months is None:
            continue
        grouped.setdefault((r.condition_id, r.language), {}).setdefault(r.phenomenon_id, []).append(r)

    out: list[dict] = []
    for (cond, lang), phen_map in sorted(grouped.items()):
        xs = []
        ys = []
        notes = []
        for phen_id, recs in sorted(phen_map.items()):
            ages = {rec.child_age_months for rec in recs if rec.child_age_months is not None}
            if len(ages) != 1:
                LOG.warning(
                    "Inconsistent ages for %s/%s/%s: %s; using min.",
                    cond, lang, phen_id, ages,
                )
            age = min(ages) if ages else None
            # Pool t_acq across seeds: use median over those that DID acquire;
            # if no seed acquired, treat as right-censored (skip from correlation
            # but note).
            t_vals = [rec.t_acq_step for rec in recs if rec.t_acq_step is not None]
            if not t_vals:
                notes.append(f"{phen_id}:censored")
                continue
            t_median = float(np.median(t_vals))
            xs.append(t_median)
            ys.append(float(age))
        x = np.array(xs)
        y = np.array(ys)
        if len(x) < n_min:
            row = {
                "condition": cond,
                "language": lang,
                "rho": math.nan,
                "ci_lo": math.nan,
                "ci_hi": math.nan,
                "n": len(x),
                "note": f"N<{n_min}; ungeneralizable. censored: {','.join(notes)}",
            }
        else:
            rho, ci_lo, ci_hi, n = spearman_with_bootstrap(x, y, n_bootstrap=n_bootstrap)
            row = {
                "condition": cond,
                "language": lang,
                "rho": rho,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": n,
                "note": (f"censored: {','.join(notes)}" if notes else ""),
            }
        out.append(row)
    return out


def compute_zh_scatter(records: list[TAcqRecord]) -> list[dict]:
    """Per-phenomenon scatter rows for ZH. One row per (phenomenon, condition, seed)."""
    out = []
    for r in records:
        if r.language != "zho":
            continue
        if r.child_age_months is None or r.t_acq_step is None:
            continue
        out.append({
            "phenomenon_id": r.phenomenon_id,
            "condition": r.condition_id,
            "seed": r.seed,
            "t_acq_step": int(r.t_acq_step),
            "child_age_months": float(r.child_age_months),
        })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="One or more run directories (must contain eval_results.json).")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analyses/results")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--consecutive", type=int, default=DEFAULT_CONSECUTIVE)
    parser.add_argument("--t-acq-method", choices=["relative", "absolute"],
                        default=DEFAULT_T_ACQ_METHOD,
                        help="relative = Evanson 2023 (90%% of final acc, default); "
                             "absolute = legacy fixed threshold (robustness).")
    parser.add_argument("--relative-frac", type=float, default=DEFAULT_RELATIVE_FRAC)
    parser.add_argument("--norms-csv", type=Path, default=None,
                        help="Child norms CSV (defaults to child_norms_dataset.csv).")
    parser.add_argument("--n-min", type=int, default=8, help="Minimum N for reporting rho")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    mapping = _load_phenomena_mapping(args.norms_csv)
    records = collect_t_acq_records(
        args.runs, mapping=mapping,
        threshold=args.threshold, consecutive=args.consecutive,
        method=args.t_acq_method, relative_frac=args.relative_frac,
    )
    print(f"Collected {len(records)} (cond, seed, lang, phen) t_acq records.")

    rho_rows = compute_rho_table(records, n_min=args.n_min, n_bootstrap=args.n_bootstrap)
    scatter_rows = compute_zh_scatter(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rho_rows, args.output_dir / "rho_table.csv")
    write_csv(scatter_rows, args.output_dir / "zh_scatter.csv")
    write_csv(
        [{
            "condition": r.condition_id,
            "seed": r.seed,
            "language": r.language,
            "phenomenon_id": r.phenomenon_id,
            "t_acq_step": "" if r.t_acq_step is None else r.t_acq_step,
            "child_age_months": "" if r.child_age_months is None else r.child_age_months,
        } for r in records],
        args.output_dir / "t_acq_table.csv",
    )

    print()
    print("Rho table (per condition × language):")
    print(f"  {'condition':10s} {'lang':4s}  {'rho':>7s} {'ci_lo':>7s} {'ci_hi':>7s} {'N':>3s}  note")
    for r in rho_rows:
        rho_s = f"{r['rho']:.3f}" if not math.isnan(r['rho']) else "  NaN"
        lo_s = f"{r['ci_lo']:.3f}" if not math.isnan(r['ci_lo']) else "  NaN"
        hi_s = f"{r['ci_hi']:.3f}" if not math.isnan(r['ci_hi']) else "  NaN"
        print(f"  {r['condition']:10s} {r['language']:4s}  {rho_s:>7s} {lo_s:>7s} {hi_s:>7s} {r['n']:>3d}  {r['note']}")

    print()
    print(f"Outputs written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
