"""Orchestrator for the H4 (Acquisition-Order Alignment) analysis.

End-to-end:
    1. Convert `*_predictions.json` -> `runs/_h4/<cond>_seed<S>/eval_results.json`
       (via scripts.h4.predictions_to_eval_results.convert_one_file)
    2. Run the Spearman ρ / bootstrap analysis on the converted runs
       (via scripts.analyze_acquisition_order)
    3. Compute the paired-bootstrap CI on `ρ_TAAM-v2 − ρ_B0` per language
       and append a `delta_rho_table.csv` to analyses/results/
    4. Render Figure 6 (per-language scatter + condition-comparison bar chart)
       to analyses/figures/figure6_acquisition_order.png

This script is designed to be called either:
    * From CLI:    python -m scripts.h4.run_h4_pipeline --predictions-glob ...
    * From Colab:  via notebooks/h4_analysis.ipynb (imports `run_pipeline`)

Outputs (all relative to REPO_ROOT):
    runs/_h4/<cond>_seed<S>/eval_results.json
    runs/_h4/audit.csv
    analyses/results/t_acq_table.csv
    analyses/results/rho_table.csv
    analyses/results/zh_scatter.csv
    analyses/results/delta_rho_table.csv          <-- NEW (this orchestrator)
    analyses/figures/figure6_acquisition_order.png <-- NEW (this orchestrator)
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

import numpy as np
import yaml

LOG = logging.getLogger("h4.pipeline")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Local imports (after sys.path patch)
from scripts.h4.predictions_to_eval_results import (        # noqa: E402
    convert_one_file, convert_from_eval_dir, expand_paths,
    load_task_map, write_audit_csv,
    DEFAULT_TASK_MAP,
)
from scripts.analyze_acquisition_order import (              # noqa: E402
    collect_t_acq_records, compute_rho_table, compute_zh_scatter,
    spearman_with_bootstrap, _load_phenomena_mapping,
    DEFAULT_THRESHOLD, DEFAULT_CONSECUTIVE, DEFAULT_N_BOOTSTRAP,
    DEFAULT_T_ACQ_METHOD, DEFAULT_RELATIVE_FRAC,
)
from scripts.h4.child_norms import (                         # noqa: E402
    load_task_map as load_task_map_csv,
    DEFAULT_NORMS_CSV,
)


# ──────────────────────────────────────────────────────────────────────────────
# CSV writers that emit headers even when rows is empty (so pd.read_csv works)
# ──────────────────────────────────────────────────────────────────────────────


RHO_TABLE_FIELDS = [
    "condition", "language", "rho", "ci_lo", "ci_hi", "n", "note",
]
DELTA_RHO_FIELDS = [
    "language", "cond_a", "cond_b", "rho_a", "rho_b", "delta",
    "ci_lo", "ci_hi", "n_pairs", "p_two_sided", "p_holm", "verdict",
]
ZH_SCATTER_FIELDS = [
    "phenomenon_id", "condition", "seed", "t_acq_step", "child_age_months",
]
T_ACQ_FIELDS = [
    "condition", "seed", "language", "phenomenon_id",
    "t_acq_step", "child_age_months",
]


def _write_csv_safe(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """CSV writer that always writes headers — even when rows is empty —
    so downstream `pd.read_csv(path)` does not raise EmptyDataError.
    """
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ──────────────────────────────────────────────────────────────────────────────
# Paired bootstrap on `ρ_A − ρ_B`
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DeltaRho:
    language: str
    cond_a: str
    cond_b: str
    rho_a: float
    rho_b: float
    delta: float            # rho_a - rho_b
    ci_lo: float
    ci_hi: float
    n_pairs: int
    p_two_sided: float


def paired_delta_rho_bootstrap(
    pairs: list[tuple[float, float, float]],
    n_bootstrap: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float, float, float, int]:
    """Bootstrap CI on `Spearman(x_a, y) − Spearman(x_b, y)`.

    `pairs` is a list of (t_acq_a, t_acq_b, child_age) triples — one per
    phenomenon. We resample with replacement at the *phenomenon* level
    (matched across the two conditions).

    Returns: (rho_a, rho_b, delta, ci_lo, ci_hi, n_pairs)
    """
    from scipy.stats import spearmanr  # local import

    n = len(pairs)
    if n < 3:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, n)
    xa = np.array([p[0] for p in pairs])
    xb = np.array([p[1] for p in pairs])
    y = np.array([p[2] for p in pairs])

    rho_a, _ = spearmanr(xa, y)
    rho_b, _ = spearmanr(xb, y)
    delta = float(rho_a) - float(rho_b)

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            ra, _ = spearmanr(xa[idx], y[idx])
            rb, _ = spearmanr(xb[idx], y[idx])
        except Exception:
            continue
        if not (math.isnan(ra) or math.isnan(rb)):
            deltas.append(float(ra) - float(rb))
    if len(deltas) < 100:
        return (float(rho_a), float(rho_b), delta, math.nan, math.nan, n)
    arr = np.array(deltas)
    lo = float(np.percentile(arr, 100 * (alpha / 2)))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return (float(rho_a), float(rho_b), delta, lo, hi, n)


def two_sided_pvalue_from_bootstrap(
    pairs: list[tuple[float, float, float]],
    n_bootstrap: int = 10_000,
    seed: int = 1,
) -> float:
    """Permutation-style p-value: fraction of bootstrap deltas with opposite sign.

    Crude but standard; Holm-Bonferroni is applied downstream.
    """
    from scipy.stats import spearmanr  # local import

    n = len(pairs)
    if n < 3:
        return math.nan
    xa = np.array([p[0] for p in pairs])
    xb = np.array([p[1] for p in pairs])
    y = np.array([p[2] for p in pairs])
    ra_obs, _ = spearmanr(xa, y)
    rb_obs, _ = spearmanr(xb, y)
    if math.isnan(ra_obs) or math.isnan(rb_obs):
        return math.nan
    delta_obs = float(ra_obs) - float(rb_obs)

    rng = np.random.default_rng(seed)
    n_extreme = 0
    n_valid = 0
    for _ in range(n_bootstrap):
        # Resample y to break the structure (null hypothesis: delta == 0)
        y_shuffled = rng.permutation(y)
        try:
            ra, _ = spearmanr(xa, y_shuffled)
            rb, _ = spearmanr(xb, y_shuffled)
        except Exception:
            continue
        if math.isnan(ra) or math.isnan(rb):
            continue
        n_valid += 1
        if abs(float(ra) - float(rb)) >= abs(delta_obs):
            n_extreme += 1
    if n_valid == 0:
        return math.nan
    return (n_extreme + 1) / (n_valid + 1)  # +1 smoothing


def compute_delta_rho_table(
    records: list,
    n_bootstrap: int,
    cond_a: str = "taam_v2",
    cond_b: str = "b0",
) -> list[DeltaRho]:
    """For each language: pair phenomena across cond_a and cond_b, bootstrap delta."""
    # Build {(cond, lang, phen): median_t_acq, child_age}
    per_key: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for r in records:
        if r.child_age_months is None or r.t_acq_step is None:
            continue
        key = (r.condition_id, r.language, r.phenomenon_id)
        bucket = per_key.setdefault(key, {"t_vals": [], "age": r.child_age_months})
        bucket["t_vals"].append(r.t_acq_step)

    # Median across seeds
    per_phen_a: dict[tuple[str, str], float] = {}
    per_phen_b: dict[tuple[str, str], float] = {}
    child_age: dict[tuple[str, str], float] = {}
    for (cond, lang, phen), info in per_key.items():
        if not info["t_vals"]:
            continue
        med = float(np.median(info["t_vals"]))
        if cond == cond_a:
            per_phen_a[(lang, phen)] = med
        elif cond == cond_b:
            per_phen_b[(lang, phen)] = med
        child_age[(lang, phen)] = float(info["age"])

    out: list[DeltaRho] = []
    for lang in sorted({k[0] for k in child_age.keys()}):
        shared = sorted([phen for (l, phen) in per_phen_a if l == lang
                         and (l, phen) in per_phen_b])
        if len(shared) < 3:
            LOG.warning("Not enough shared phenomena on %s (N=%d)", lang, len(shared))
            continue
        pairs = [
            (per_phen_a[(lang, phen)], per_phen_b[(lang, phen)],
             child_age[(lang, phen)])
            for phen in shared
        ]
        rho_a, rho_b, delta, ci_lo, ci_hi, n = paired_delta_rho_bootstrap(
            pairs, n_bootstrap=n_bootstrap,
        )
        pval = two_sided_pvalue_from_bootstrap(pairs, n_bootstrap=n_bootstrap)
        out.append(DeltaRho(
            language=lang,
            cond_a=cond_a, cond_b=cond_b,
            rho_a=rho_a, rho_b=rho_b, delta=delta,
            ci_lo=ci_lo, ci_hi=ci_hi,
            n_pairs=n,
            p_two_sided=pval,
        ))
    return out


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, NaN-safe."""
    finite = [(i, p) for i, p in enumerate(pvals) if not math.isnan(p)]
    if not finite:
        return pvals
    finite_sorted = sorted(finite, key=lambda x: x[1])
    m = len(finite_sorted)
    out = list(pvals)
    prev = 0.0
    for rank, (orig_i, p) in enumerate(finite_sorted):
        adj = min(1.0, max(prev, p * (m - rank)))
        out[orig_i] = adj
        prev = adj
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6
# ──────────────────────────────────────────────────────────────────────────────


def render_figure6(
    records: list,
    rho_rows: list[dict],
    delta_rows: list[DeltaRho],
    out_path: Path,
) -> None:
    """Render the canonical H4 figure to `out_path`.

    Panel layout:
        Row 1: 3 panels, one per language. Per-phenomenon scatter
                (t_acq_model_x median across seeds, child_age_months_y),
                with TAAM-v2 in one colour and B0 in another, regression
                lines per condition.
        Row 2: 1 wide panel. Bar chart of `ρ_TAAM-v2 − ρ_B0` per language,
                with error bars from the bootstrap CI.
    """
    try:
        import os, tempfile
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        LOG.warning("matplotlib not installed; skipping figure render. "
                    "Install with `pip install matplotlib`.")
        return

    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })

    # Aggregate per-(cond, lang, phen) median t_acq across seeds
    grouped: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    seed_groups: dict[tuple[str, str, str], list[float]] = {}
    ages: dict[tuple[str, str], float] = {}
    for r in records:
        if r.child_age_months is None or r.t_acq_step is None:
            continue
        seed_groups.setdefault((r.condition_id, r.language, r.phenomenon_id), []).append(r.t_acq_step)
        ages[(r.language, r.phenomenon_id)] = float(r.child_age_months)
    for (cond, lang, phen), tvals in seed_groups.items():
        grouped.setdefault((cond, lang), []).append((float(np.median(tvals)), ages[(lang, phen)], phen))

    langs_order = ["eng", "nld", "zho"]
    lang_titles = {"eng": "English", "nld": "Dutch", "zho": "Chinese"}
    colors = {"taam_v2": "#d62728", "b0": "#1f77b4"}
    labels = {"taam_v2": "TAAM-v2", "b0": "B0 (uniform)"}

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.40, wspace=0.30)

    # Row 1 — per-language scatter
    for j, lang in enumerate(langs_order):
        ax = fig.add_subplot(gs[0, j])
        for cond in ["b0", "taam_v2"]:
            pts = grouped.get((cond, lang), [])
            if not pts:
                continue
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            ax.scatter(xs, ys, c=colors[cond], alpha=0.7, s=40,
                       edgecolors="white", linewidths=0.5, label=labels[cond])
            # Best-fit line if N>=3
            if len(xs) >= 3:
                slope, intercept = np.polyfit(xs, ys, 1)
                xline = np.linspace(xs.min(), xs.max(), 50)
                ax.plot(xline, slope * xline + intercept,
                        c=colors[cond], lw=1.0, alpha=0.5)
        ax.set_xlabel("Model acquisition step (M tokens)")
        ax.set_ylabel("Child age of acquisition (months)")
        # Pull ρ from rho_rows for the title
        rho_summary = []
        for row in rho_rows:
            if row["language"] == lang and not math.isnan(row.get("rho", math.nan)):
                rho_summary.append(f"{row['condition'][:8]}: ρ={row['rho']:.2f}")
        title = lang_titles[lang]
        if rho_summary:
            title += "\n" + "  |  ".join(rho_summary)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)

    # Row 2 — Δρ bar chart
    ax2 = fig.add_subplot(gs[1, :])
    if delta_rows:
        xs = np.arange(len(delta_rows))
        deltas = [d.delta for d in delta_rows]
        errs_lo = [d.delta - d.ci_lo for d in delta_rows]
        errs_hi = [d.ci_hi - d.delta for d in delta_rows]
        ax2.bar(xs, deltas, yerr=[errs_lo, errs_hi], capsize=4,
                color=["#d62728" if d.delta > 0 else "#7f7f7f" for d in delta_rows],
                edgecolor="black", linewidth=0.5)
        ax2.axhline(0, color="black", lw=0.7)
        ax2.set_xticks(xs)
        ax2.set_xticklabels([
            f"{lang_titles[d.language]}\n(N={d.n_pairs}, p={d.p_two_sided:.3f})"
            for d in delta_rows
        ])
        ax2.set_ylabel("ρ(TAAM-v2) − ρ(B0)\n[positive = TAAM closer to child]")
        ax2.set_title("Δρ = Spearman correlation (TAAM-v2 vs child norms) − "
                      "(B0 vs child norms)", fontsize=9)
        ax2.grid(True, alpha=0.2, axis="y")
    else:
        ax2.text(0.5, 0.5, "No Δρ rows computed (need both `taam_v2` and `b0` runs).",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    fig.suptitle("Figure 6 — Acquisition-order alignment (H4)\n"
                 "Model `t_acq` (chck_NM) vs. child age-of-acquisition, "
                 "per phenomenon and language.",
                 fontsize=11, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    predictions_paths: list[Path] | None = None,
    eval_results_dir: Path | None = None,
    h4_models: list[str] | None = None,
    out_runs_dir: Path = Path("runs/_h4"),
    out_results_dir: Path = Path("analyses/results"),
    out_figures_dir: Path = Path("analyses/figures"),
    task_map_path: Path | None = None,
    norms_csv: Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    consecutive: int = DEFAULT_CONSECUTIVE,
    t_acq_method: str = DEFAULT_T_ACQ_METHOD,
    relative_frac: float = DEFAULT_RELATIVE_FRAC,
    n_min: int = 6,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    cond_a: str = "taam_v2",
    cond_b: str = "b0",
    mirror_dirs: list[Path] | None = None,
) -> dict:
    """Execute the H4 pipeline. Returns a summary dict for logging.

    Provide EITHER:
        eval_results_dir + h4_models  (preferred — full sub-task granularity)
    OR:
        predictions_paths             (fallback — aggregated tasks only)

    Data sources (v2, single source of truth):
        `norms_csv` (default: analyses/acquisition_order/child_norms_dataset.csv)
        provides BOTH the task→phenomenon map AND the child ages. If a legacy
        `task_map_path` YAML is passed explicitly, it overrides the CSV task map.

    `t_acq_method`:
        "relative" (Evanson 2023, default) = step to reach 90% of final acc,
        only for above-chance probes. "absolute" = legacy fixed-0.70 threshold.

    `mirror_dirs` is an optional list of directories to which we will copy
    every result CSV + Figure 6 + audit CSV after the pipeline finishes
    (e.g., a Google Drive path so outputs survive Colab disconnects).
    """
    norms_csv = Path(norms_csv) if norms_csv else DEFAULT_NORMS_CSV
    # Task map: prefer the v2 CSV; allow explicit legacy YAML override.
    if task_map_path is not None:
        task_map = load_task_map(Path(task_map_path))
        LOG.info("Using legacy YAML task map: %s", task_map_path)
    elif Path(norms_csv).exists():
        task_map = load_task_map_csv(Path(norms_csv))
        LOG.info("Using CSV task map (v2): %s", norms_csv)
    else:
        task_map = load_task_map(DEFAULT_TASK_MAP)
        LOG.info("Falling back to default YAML task map.")
    all_audit: list[dict] = []
    converted_paths: list[Path] = []

    # ── Step 1: build per-model eval_results.json ────────────────────────────
    if eval_results_dir is not None:
        if not h4_models:
            raise ValueError("h4_models is required when eval_results_dir is set.")
        LOG.info("Step 1/4: converting %d model(s) from %s",
                 len(h4_models), eval_results_dir)
        for tok in h4_models:
            try:
                res, audit = convert_from_eval_dir(
                    eval_results_dir, tok, task_map, out_runs_dir,
                )
            except Exception as e:
                LOG.exception("Failed to convert %s: %s", tok, e)
                continue
            all_audit.extend(audit)
            converted_paths.append(
                out_runs_dir / f"{res['condition_id']}_seed{res['seed']}"
            )
    elif predictions_paths:
        LOG.info("Step 1/4: converting %d predictions file(s)",
                 len(predictions_paths))
        for p in predictions_paths:
            try:
                res, audit = convert_one_file(p, task_map, out_runs_dir)
            except Exception as e:
                LOG.exception("Failed to convert %s: %s", p, e)
                continue
            all_audit.extend(audit)
            converted_paths.append(
                out_runs_dir / f"{res['condition_id']}_seed{res['seed']}"
            )
    else:
        raise ValueError("Must provide either predictions_paths or eval_results_dir+h4_models")

    write_audit_csv(all_audit, out_runs_dir / "audit.csv")
    LOG.info("Audit CSV: %s", out_runs_dir / "audit.csv")

    # ── Step 2: extract t_acq records and per-(cond, lang) Spearman ρ ────────
    LOG.info("Step 2/4: computing t_acq + Spearman ρ on %d run(s)",
             len(converted_paths))
    mapping = _load_phenomena_mapping(norms_csv)
    records = collect_t_acq_records(
        converted_paths, mapping=mapping,
        threshold=threshold, consecutive=consecutive,
        method=t_acq_method, relative_frac=relative_frac,
    )
    rho_rows = compute_rho_table(records, n_min=n_min, n_bootstrap=n_bootstrap)
    scatter_rows = compute_zh_scatter(records)

    out_results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_safe(rho_rows, out_results_dir / "rho_table.csv", RHO_TABLE_FIELDS)
    _write_csv_safe(scatter_rows, out_results_dir / "zh_scatter.csv", ZH_SCATTER_FIELDS)
    _write_csv_safe(
        [{
            "condition": r.condition_id, "seed": r.seed, "language": r.language,
            "phenomenon_id": r.phenomenon_id,
            "t_acq_step": "" if r.t_acq_step is None else r.t_acq_step,
            "child_age_months": "" if r.child_age_months is None else r.child_age_months,
        } for r in records],
        out_results_dir / "t_acq_table.csv",
        T_ACQ_FIELDS,
    )

    # ── Step 3: paired Δρ bootstrap, Holm-Bonferroni p_holm ─────────────────
    LOG.info("Step 3/4: computing Δρ (paired bootstrap) %s vs %s",
             cond_a, cond_b)
    delta_rows = compute_delta_rho_table(
        records, n_bootstrap=n_bootstrap, cond_a=cond_a, cond_b=cond_b,
    )
    pvals = [d.p_two_sided for d in delta_rows]
    p_holm = holm_bonferroni(pvals)
    delta_dicts = []
    for d, ph in zip(delta_rows, p_holm):
        # Verdict logic, in priority order:
        #   1. underpowered      -> N below n_min (don't over-interpret)
        #   2. undefined         -> ρ could not be computed (constant/tied input)
        #   3. Holm-significant  -> cond_a closer to child norms than cond_b
        #   4. null              -> no significant difference
        if d.n_pairs < n_min:
            verdict = f"underpowered (N={d.n_pairs} < {n_min})"
        elif math.isnan(d.rho_a) or math.isnan(d.rho_b) or math.isnan(d.delta):
            verdict = "undefined (constant/tied input)"
        elif not math.isnan(ph) and ph < 0.05 and d.delta > 0:
            verdict = f"{cond_a} beats {cond_b} (Holm-significant)"
        elif not math.isnan(ph) and ph < 0.05 and d.delta < 0:
            verdict = f"{cond_b} beats {cond_a} (Holm-significant)"
        else:
            verdict = "no significant difference (null)"
        delta_dicts.append({
            "language": d.language,
            "cond_a": d.cond_a, "cond_b": d.cond_b,
            "rho_a": f"{d.rho_a:+.4f}" if not math.isnan(d.rho_a) else "NaN",
            "rho_b": f"{d.rho_b:+.4f}" if not math.isnan(d.rho_b) else "NaN",
            "delta": f"{d.delta:+.4f}" if not math.isnan(d.delta) else "NaN",
            "ci_lo": "NaN" if math.isnan(d.ci_lo) else f"{d.ci_lo:+.4f}",
            "ci_hi": "NaN" if math.isnan(d.ci_hi) else f"{d.ci_hi:+.4f}",
            "n_pairs": d.n_pairs,
            "p_two_sided": "NaN" if math.isnan(d.p_two_sided) else f"{d.p_two_sided:.4f}",
            "p_holm":      "NaN" if math.isnan(ph) else f"{ph:.4f}",
            "verdict": verdict,
        })
    _write_csv_safe(delta_dicts, out_results_dir / "delta_rho_table.csv", DELTA_RHO_FIELDS)

    # ── Step 4: Figure 6 ────────────────────────────────────────────────────
    LOG.info("Step 4/4: rendering Figure 6")
    render_figure6(records, rho_rows, delta_rows,
                   out_figures_dir / "figure6_acquisition_order.png")

    outputs = {
        "rho_table_csv":   out_results_dir / "rho_table.csv",
        "delta_rho_csv":   out_results_dir / "delta_rho_table.csv",
        "zh_scatter_csv":  out_results_dir / "zh_scatter.csv",
        "t_acq_table_csv": out_results_dir / "t_acq_table.csv",
        "audit_csv":       out_runs_dir / "audit.csv",
        "figure6_png":     out_figures_dir / "figure6_acquisition_order.png",
    }

    # ── Optional: mirror outputs to Drive (or any extra dir) ─────────────────
    if mirror_dirs:
        import shutil
        for dest in mirror_dirs:
            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=True)
            for name, src in outputs.items():
                src = Path(src)
                if src.exists():
                    try:
                        shutil.copy2(src, dest / src.name)
                        LOG.info("Mirrored %s -> %s", src.name, dest / src.name)
                    except Exception as e:
                        LOG.warning("Could not mirror %s -> %s: %s",
                                    src, dest, e)
            # Also mirror the per-run eval_results.json + audit
            for run_dir in converted_paths:
                rel = run_dir.name  # e.g. "taam_v2_seed42"
                src_json = run_dir / "eval_results.json"
                if src_json.exists():
                    sub = dest / "runs" / rel
                    sub.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(src_json, sub / src_json.name)
                    except Exception as e:
                        LOG.warning("Could not mirror %s: %s", src_json, e)

    return {
        "n_predictions": len(predictions_paths) if predictions_paths else 0,
        "n_runs_converted": len(converted_paths),
        "n_records": len(records),
        "rho_rows": rho_rows,
        "delta_rows": delta_dicts,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "mirrored_to": [str(d) for d in (mirror_dirs or [])],
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--eval-results-dir", type=Path, default=None,
        help="Drive path with chck_NM/<slug>/results_*.json (preferred mode).",
    )
    src.add_argument(
        "--predictions-glob", "-p", nargs="+",
        help="Glob(s) of *_predictions.json files (fallback mode).",
    )
    parser.add_argument(
        "--h4-models", nargs="+", default=None,
        help="Required with --eval-results-dir (e.g. taam_v2-seed42 b0-seed42).",
    )
    parser.add_argument("--out-runs-dir",     type=Path,
                        default=REPO_ROOT / "runs/_h4")
    parser.add_argument("--out-results-dir",  type=Path,
                        default=REPO_ROOT / "analyses/results")
    parser.add_argument("--out-figures-dir",  type=Path,
                        default=REPO_ROOT / "analyses/figures")
    parser.add_argument(
        "--mirror-dir", action="append", default=None,
        help="Extra directory to mirror all outputs to (e.g. a Drive path). "
             "Can be repeated.",
    )
    parser.add_argument("--task-map", type=Path, default=None,
                        help="Legacy YAML task map override. Default: use the v2 CSV.")
    parser.add_argument("--norms-csv", type=Path, default=None,
                        help="Child norms CSV (default: child_norms_dataset.csv). "
                             "Provides BOTH task map and child ages.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--consecutive", type=int, default=DEFAULT_CONSECUTIVE)
    parser.add_argument("--t-acq-method", choices=["relative", "absolute"],
                        default=DEFAULT_T_ACQ_METHOD,
                        help="relative = Evanson 2023 (default); absolute = legacy.")
    parser.add_argument("--relative-frac", type=float, default=DEFAULT_RELATIVE_FRAC)
    parser.add_argument("--n-min", type=int, default=6)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--cond-a", default="taam_v2")
    parser.add_argument("--cond-b", default="b0")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.eval_results_dir is not None:
        if not args.h4_models:
            LOG.error("--h4-models is required when using --eval-results-dir")
            return 2
        summary = run_pipeline(
            predictions_paths=None,
            eval_results_dir=args.eval_results_dir,
            h4_models=args.h4_models,
            out_runs_dir=args.out_runs_dir,
            out_results_dir=args.out_results_dir,
            out_figures_dir=args.out_figures_dir,
            task_map_path=args.task_map,
            norms_csv=args.norms_csv,
            threshold=args.threshold, consecutive=args.consecutive,
            t_acq_method=args.t_acq_method, relative_frac=args.relative_frac,
            n_min=args.n_min, n_bootstrap=args.n_bootstrap,
            cond_a=args.cond_a, cond_b=args.cond_b,
            mirror_dirs=[Path(d) for d in (args.mirror_dir or [])],
        )
    else:
        paths = expand_paths(args.predictions_glob)
        if not paths:
            LOG.error("No predictions files matched; aborting.")
            return 2
        summary = run_pipeline(
            predictions_paths=paths,
            out_runs_dir=args.out_runs_dir,
            out_results_dir=args.out_results_dir,
            out_figures_dir=args.out_figures_dir,
            task_map_path=args.task_map,
            norms_csv=args.norms_csv,
            threshold=args.threshold, consecutive=args.consecutive,
            t_acq_method=args.t_acq_method, relative_frac=args.relative_frac,
            n_min=args.n_min, n_bootstrap=args.n_bootstrap,
            cond_a=args.cond_a, cond_b=args.cond_b,
            mirror_dirs=[Path(d) for d in (args.mirror_dir or [])],
        )

    print()
    print("=" * 72)
    print("H4 PIPELINE — SUMMARY")
    print("=" * 72)
    print(f"  predictions in           : {summary['n_predictions']}")
    print(f"  runs converted           : {summary['n_runs_converted']}")
    print(f"  t_acq records collected  : {summary['n_records']}")
    print()
    print("ρ table (per cond × lang):")
    if not summary["rho_rows"]:
        print("  (empty — likely no phenomena crossed the threshold; widen --threshold "
              "or check --task-map)")
    for r in summary["rho_rows"]:
        rho_s  = "NaN" if math.isnan(r.get("rho", math.nan))     else f"{r['rho']:+.3f}"
        lo_s   = "NaN" if math.isnan(r.get("ci_lo", math.nan))    else f"{r['ci_lo']:+.3f}"
        hi_s   = "NaN" if math.isnan(r.get("ci_hi", math.nan))    else f"{r['ci_hi']:+.3f}"
        print(f"  {r['condition']:10s} {r['language']:4s}  ρ={rho_s}  "
              f"95%CI=[{lo_s}, {hi_s}]  N={r['n']}")
    print()
    print(f"Δρ table ({args.cond_a} vs {args.cond_b}):")
    for d in summary["delta_rows"]:
        print(f"  {d['language']:4s}  ρ_a={d['rho_a']}  ρ_b={d['rho_b']}  "
              f"Δ={d['delta']}  CI=[{d['ci_lo']}, {d['ci_hi']}]  "
              f"N={d['n_pairs']}  p={d['p_two_sided']}  p_holm={d['p_holm']}  "
              f"-> {d['verdict']}")
    print()
    print("Output files:")
    for k, v in summary["outputs"].items():
        print(f"  {k:18s}: {v}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
