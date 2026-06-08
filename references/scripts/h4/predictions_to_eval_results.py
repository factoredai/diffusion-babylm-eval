"""Bridge: lm-eval raw results / `*_predictions.json` -> `eval_results.json` (H4 format).

This module supports TWO input modes:

1) **eval-results-dir mode (preferred for H4)** — reads the RAW per-checkpoint
   lm-eval-harness `results_*.json` files that `notebooks/evaluation.ipynb`
   Cell 6 writes under e.g.::

       /content/drive/MyDrive/Researchs/BabyLM/eval_results/chck_NM/<slug>/results_*.json

   These files contain ALL ~110 sub-tasks at full per-phenomenon granularity
   (`blimp_babylm_filtered_passive_1`, `zhoblimp_BEI_construction_a`, ...),
   which is what H4 needs.

2) **predictions-glob mode (fallback, low-fidelity)** — reads the
   `_predictions.json` files emitted by Cell 7. Those only contain the 16
   aggregated leaderboard tasks under `fast_eval_results`, NOT the sub-tasks,
   so H4 ρ will be NaN unless your `phenomenon_to_task_map.yaml` lists those
   aggregated names. Kept as a fallback when the raw eval dir is unavailable.

The H4 analyzer `scripts/analyze_acquisition_order.py` expects, per run::

    runs/<cond>_seed<S>/eval_results.json   with structure::

    {
        "condition_id": "<cond>",     # e.g. "taam_v2", "b0"
        "seed":         <int>,        # e.g. 42
        "checkpoints":  {
            "<step>": {                 # str(int) of cumulative tokens (M)
                "<phenomenon_id>": {
                    "<lang>": <accuracy>
                }
            },
            ...
        }
    }

Phenomena and their task lists are defined in
    analyses/acquisition_order/phenomenon_to_task_map.yaml
For each (checkpoint, phenomenon, language) we average accuracy across the
listed tasks, dropping NaN/None values.

Usage (eval-results-dir mode, preferred)::

    python -m scripts.h4.predictions_to_eval_results \
        --eval-results-dir /content/drive/MyDrive/Researchs/BabyLM/eval_results \
        --h4-models taam_v2-seed42 b0-seed42 \
        --out-dir runs/_h4/

Usage (predictions-glob mode)::

    python -m scripts.h4.predictions_to_eval_results \
        --predictions-glob "submissions/babylm-2026-{taam_v2,b0}-seed42_predictions.json" \
        --out-dir runs/_h4/

Output per model::

    runs/_h4/<cond>_seed<S>/eval_results.json

And an audit CSV at runs/_h4/audit.csv with the per-cell averaging trace.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml

LOG = logging.getLogger("h4.bridge")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_MAP = REPO_ROOT / "analyses/acquisition_order/phenomenon_to_task_map.yaml"

# Revisions evaluated in Cell 6 of evaluation.ipynb (the "Fast Eval" loop).
# Order matters: index in fast_eval_results[i] corresponds to FAST_EVAL_REVISIONS[i].
FAST_EVAL_REVISIONS: list[str] = (
    [f"chck_{i}M" for i in range(1, 10)]              # chck_1M ... chck_9M
    + [f"chck_{i*10}M" for i in range(1, 10)]         # chck_10M ... chck_90M
    + [f"chck_{i*100}M" for i in range(1, 11)]        # chck_100M ... chck_1000M
)

# Regex to parse "babylm-2026-<cond>-seed<S>_predictions.json"
FNAME_RE = re.compile(
    r"babylm-2026-(?P<cond>[a-z0-9_]+)-seed(?P<seed>\d+)_predictions\.json$"
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_filename(path: Path) -> tuple[str, int]:
    """Extract (condition_id, seed) from a predictions filename."""
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse cond/seed from filename: {path.name}")
    return m.group("cond"), int(m.group("seed"))


def revision_to_step(rev: str) -> int:
    """Convert e.g. "chck_50M" -> 50.

    The integer is the cumulative training tokens (in millions). It is also the
    natural sort key, which is what `extract_t_acq` needs for `t_acq`.
    """
    m = re.match(r"chck_(\d+)M$", rev)
    if not m:
        raise ValueError(f"Cannot parse revision: {rev!r}")
    return int(m.group(1))


def load_task_map(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load phenomenon_to_task_map.yaml -> {phen_id: {lang: [task_name, ...]}}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[str]]] = {}
    for phen in data["phenomena"]:
        pid = phen["id"]
        out[pid] = {
            "eng": list(phen.get("tasks_eng", []) or []),
            "nld": list(phen.get("tasks_nld", []) or []),
            "zho": list(phen.get("tasks_zho", []) or []),
        }
    return out


def extract_accuracy(record: dict, task_name: str) -> float | None:
    """Pull a single accuracy from either the `zeroshot` block or a fast-eval entry.

    The Colab notebook stores values in two slightly different shapes:

      a) `zeroshot[task]`:
            {"acc,none": float, "acc_stderr,none": float, ...}

      b) `fast_eval_results[i][task]`:
            {task: float}              # built by `build_submission` in Cell 7

    Both shapes are handled. Returns None if the key/task is missing or the
    value is NaN-like.
    """
    if task_name not in record:
        return None
    val = record[task_name]
    # Shape (a): zeroshot dict from lm-eval-harness raw output.
    if isinstance(val, dict):
        if "acc,none" in val:
            v = val["acc,none"]
        elif task_name in val:
            v = val[task_name]
        else:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(v):
            return None
        return v
    # Shape (b): scalar after a previous reshape — be lenient.
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def mean_or_none(values: Iterable[float | None]) -> float | None:
    """Mean of non-None floats; None if the iterable contains no valid values."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: read raw lm-eval `results_*.json` files from Drive's eval_results dir
# ──────────────────────────────────────────────────────────────────────────────


def parse_model_token(token: str) -> tuple[str, int]:
    """Parse "taam_v2-seed42" or "amosluna/babylm-2026-taam_v2-seed42" → (cond, seed).

    Accepts either the bare condition+seed token or the full HF model id.
    """
    # Try the file-name regex first (works for "babylm-2026-<cond>-seed<S>")
    m = re.search(r"babylm-2026-(?P<cond>[a-z0-9_]+)-seed(?P<seed>\d+)", token)
    if m:
        return m.group("cond"), int(m.group("seed"))
    # Try a looser pattern: "<cond>-seed<N>"
    m = re.match(r"(?P<cond>[a-z0-9_]+)-seed(?P<seed>\d+)$", token)
    if not m:
        raise ValueError(
            f"Cannot parse model token {token!r}; expected "
            f"'<cond>-seed<N>' or 'amosluna/babylm-2026-<cond>-seed<N>'"
        )
    return m.group("cond"), int(m.group("seed"))


def discover_results_files_for_model(
    revision_dir: Path,
    model_id_substrings: list[str],
) -> list[Path]:
    """Find every `results_*.json` under `revision_dir` that belongs to a given
    model.

    Heuristics tried (in order):
        1. Substring of the model id appears in the file path.
        2. Substring appears in the JSON's `model_name` / `model_args` /
           `model_source` field.

    Returns the matching paths sorted by mtime (oldest first), so concatenation
    is reproducible.
    """
    if not revision_dir.exists():
        return []
    all_results = sorted(revision_dir.glob("**/results_*.json"))
    matches: list[Path] = []
    for cand in all_results:
        # Heuristic 1: path-based
        if any(sub in str(cand) for sub in model_id_substrings):
            matches.append(cand)
            continue
        # Heuristic 2: read top-level metadata
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
        except Exception as e:
            LOG.debug("Could not parse %s: %s", cand, e)
            continue
        for k in ("model_name", "model_args", "model_source", "model"):
            blob = data.get(k)
            if isinstance(blob, str) and any(sub in blob for sub in model_id_substrings):
                matches.append(cand)
                break
            # Some lm-eval versions store config in a nested dict
            if isinstance(blob, dict):
                args = blob.get("model_args", "") or ""
                if isinstance(args, str) and any(sub in args for sub in model_id_substrings):
                    matches.append(cand)
                    break
        # Newer lm-eval versions:
        cfg = data.get("config", {})
        if isinstance(cfg, dict):
            args = cfg.get("model_args", "")
            if isinstance(args, str) and any(sub in args for sub in model_id_substrings):
                if cand not in matches:
                    matches.append(cand)
    # Sort by mtime for reproducibility
    matches.sort(key=lambda p: p.stat().st_mtime)
    return matches


def discover_revisions_on_disk(
    eval_results_dir: Path,
    include_main: bool = False,
) -> list[str]:
    """Return the list of ``chck_NM`` directories that actually exist on disk,
    sorted by step (the numeric prefix). Skips ``main`` by default since it
    has no numeric token count and would break Spearman.
    """
    if not eval_results_dir.exists():
        return []
    revs = []
    for sub in eval_results_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        if name == "main":
            if include_main:
                revs.append(name)
            continue
        if re.match(r"^chck_\d+M$", name):
            revs.append(name)
    revs.sort(key=lambda r: revision_to_step(r) if r != "main" else 10**9)
    return revs


def load_results_for_model_from_eval_dir(
    eval_results_dir: Path,
    model_id_substrings: list[str],
    revisions: list[str] | None = None,
) -> dict[int, dict[str, float]]:
    """Walk eval_results_dir/chck_NM/.../results_*.json and merge per checkpoint.

    Returns: {step (int M tokens): {task_name: acc_value}}.

    If ``revisions`` is None we auto-discover from the filesystem (so absent
    checkpoints simply do not appear, instead of being treated as missing).

    All ``results_*.json`` files belonging to the same model at the same
    revision are merged (one per language is the normal case). Later writes
    win on duplicate task keys (should not happen since eng/nld/zho task sets
    are disjoint).
    """
    if revisions is None:
        revisions = discover_revisions_on_disk(eval_results_dir)
        LOG.info(
            "Auto-discovered %d revisions on disk: %s",
            len(revisions), revisions,
        )

    out: dict[int, dict[str, float]] = {}
    for rev in revisions:
        rev_dir = eval_results_dir / rev
        files = discover_results_files_for_model(rev_dir, model_id_substrings)
        if not files:
            LOG.debug("  %s: no results files matching model substrings", rev)
            continue
        merged: dict[str, float] = {}
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                LOG.warning("Could not parse %s: %s", f, e)
                continue
            results = data.get("results") or {}
            for task, stats in results.items():
                if not isinstance(stats, dict):
                    continue
                acc = stats.get("acc,none")
                if acc is None:
                    acc = stats.get("acc")
                if acc is None:
                    continue
                try:
                    merged[task] = float(acc)
                except (TypeError, ValueError):
                    pass
        if merged:
            step = revision_to_step(rev) if rev != "main" else 10**9
            out[step] = merged
            LOG.info(
                "  %s: merged %d tasks from %d file(s)", rev, len(merged), len(files),
            )
    return out


def convert_from_eval_dir(
    eval_results_dir: Path,
    model_token: str,
    task_map: dict[str, dict[str, list[str]]],
    out_root: Path,
    revisions: list[str] | None = None,
    extra_substrings: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    """Build `eval_results.json` for one model by reading the raw lm-eval
    results dir.

    `model_token` is either the bare condition+seed (e.g. ``taam_v2-seed42``)
    or the full HF id (``amosluna/babylm-2026-taam_v2-seed42``). Both work.
    """
    cond_id, seed = parse_model_token(model_token)

    # Build substring list for file matching. We try several variations so we
    # match both path-based and JSON-body-based.
    subs = list(extra_substrings or [])
    base = f"babylm-2026-{cond_id}-seed{seed}"
    subs.extend([
        f"amosluna/{base}",
        f"amosluna__{base}",
        base,
        f"{cond_id}-seed{seed}",
    ])
    subs = list(dict.fromkeys(subs))  # de-dup, preserve order

    LOG.info("Looking for %s under %s with substrings: %s",
             model_token, eval_results_dir, subs)
    if revisions is None:
        revisions = discover_revisions_on_disk(eval_results_dir)
    per_step_raw = load_results_for_model_from_eval_dir(
        eval_results_dir, model_id_substrings=subs, revisions=revisions,
    )
    if not per_step_raw:
        LOG.error(
            "No results_*.json found for %s under %s. "
            "Checked %d revisions; verify path layout chck_NM/<slug>/results_*.json.",
            model_token, eval_results_dir, len(revisions),
        )

    # Build the per-step phenomenon×lang dict
    checkpoints: dict[str, dict[str, dict[str, float]]] = {}
    audit_rows: list[dict] = []
    for step in sorted(per_step_raw.keys()):
        record = per_step_raw[step]
        per_phen: dict[str, dict[str, float]] = {}
        for phen_id, lang_map in task_map.items():
            for lang, task_names in lang_map.items():
                if not task_names:
                    continue
                values: list[float | None] = []
                for t in task_names:
                    v = record.get(t)
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        values.append(None)
                    else:
                        try:
                            values.append(float(v))
                        except (TypeError, ValueError):
                            values.append(None)
                n_valid = sum(1 for v in values if v is not None)
                acc = mean_or_none(values)
                audit_rows.append({
                    "condition": cond_id,
                    "seed": seed,
                    "step": step,
                    "phenomenon_id": phen_id,
                    "language": lang,
                    "n_tasks_listed": len(task_names),
                    "n_tasks_valid": n_valid,
                    "accuracy": "" if acc is None else f"{acc:.6f}",
                })
                if acc is not None:
                    per_phen.setdefault(phen_id, {})[lang] = acc
        checkpoints[str(step)] = per_phen

    eval_results = {
        "condition_id": cond_id,
        "seed": seed,
        "checkpoints": checkpoints,
        "source": str(eval_results_dir),
        "source_mode": "eval-results-dir",
        "revisions_used": [r for r in revisions
                            if revision_to_step(r) in per_step_raw],
    }

    run_dir = out_root / f"{cond_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "eval_results.json"
    out_path.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    LOG.info(
        "Wrote %s  (%d checkpoints, %d audit rows)",
        out_path, len(checkpoints), len(audit_rows),
    )
    return eval_results, audit_rows


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: read aggregated `*_predictions.json` (fallback, low fidelity)
# ──────────────────────────────────────────────────────────────────────────────


def convert_one_file(
    predictions_path: Path,
    task_map: dict[str, dict[str, list[str]]],
    out_root: Path,
    revisions: list[str] = FAST_EVAL_REVISIONS,
) -> tuple[dict, list[dict]]:
    """Convert one `*_predictions.json` into the H4 `eval_results.json` shape.

    Returns:
        (eval_results, audit_rows) where audit_rows is the list of per-cell
        dicts (one per phen × lang × checkpoint) written to audit.csv.
    """
    cond_id, seed = parse_filename(predictions_path)
    raw = json.loads(predictions_path.read_text(encoding="utf-8"))
    fast = raw.get("fast_eval_results") or []
    zs = raw.get("zeroshot") or {}

    if not fast:
        LOG.warning(
            "No fast_eval_results in %s (workshop-only). Building eval_results.json "
            "from `zeroshot` only — H4 will not be computable for this model.",
            predictions_path,
        )

    # Per-checkpoint records: index `i` -> record dict (task -> raw value)
    # Plus the "final" / main checkpoint from `zeroshot`.
    per_step: dict[int, dict] = {}
    # Fast-eval entries (intermediate checkpoints)
    for i, entry in enumerate(fast):
        if i >= len(revisions):
            LOG.warning(
                "fast_eval_results has %d entries but only %d revisions "
                "defined; truncating.", len(fast), len(revisions),
            )
            break
        step = revision_to_step(revisions[i])
        per_step[step] = entry
    # Main / final checkpoint (full 1000M tokens by convention; use 1000 step)
    if zs:
        # Use a sentinel step > any fast-eval step. The training ran ~655M tokens,
        # but for ranking purposes the only requirement is that this step is the
        # LARGEST step in the trajectory. We use 1000M (= 1e9 tokens).
        per_step[1000] = zs

    # Build the per-step phenomenon×lang dict
    checkpoints: dict[str, dict[str, dict[str, float]]] = {}
    audit_rows: list[dict] = []
    for step in sorted(per_step.keys()):
        record = per_step[step]
        per_phen: dict[str, dict[str, float]] = {}
        for phen_id, lang_map in task_map.items():
            phen_lang_accs: dict[str, float] = {}
            for lang, task_names in lang_map.items():
                if not task_names:
                    continue
                values = [extract_accuracy(record, t) for t in task_names]
                n_valid = sum(1 for v in values if v is not None)
                acc = mean_or_none(values)
                audit_rows.append({
                    "condition": cond_id,
                    "seed": seed,
                    "step": step,
                    "phenomenon_id": phen_id,
                    "language": lang,
                    "n_tasks_listed": len(task_names),
                    "n_tasks_valid": n_valid,
                    "accuracy": "" if acc is None else f"{acc:.6f}",
                })
                if acc is not None:
                    phen_lang_accs[lang] = acc
            if phen_lang_accs:
                per_phen[phen_id] = phen_lang_accs
        checkpoints[str(step)] = per_phen

    eval_results = {
        "condition_id": cond_id,
        "seed": seed,
        "checkpoints": checkpoints,
        "source": str(predictions_path.relative_to(predictions_path.anchor)),
        "revisions_used": revisions[: len(fast)] + (["main"] if zs else []),
    }

    # Write per-run eval_results.json
    run_dir = out_root / f"{cond_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "eval_results.json"
    out_path.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")

    LOG.info(
        "Wrote %s  (%d checkpoints, %d phen×lang cells, %d audit rows)",
        out_path, len(checkpoints),
        sum(len(p) * 3 for p in checkpoints.values()),  # rough cell count
        len(audit_rows),
    )
    return eval_results, audit_rows


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def expand_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        # glob.glob accepts both literal paths and "*" wildcards.
        matches = sorted(glob.glob(p))
        if not matches:
            LOG.warning("Pattern matched zero files: %s", p)
        out.extend(Path(m) for m in matches)
    return out


DEFAULT_AUDIT_FIELDS = [
    "condition", "seed", "step", "phenomenon_id", "language",
    "n_tasks_listed", "n_tasks_valid", "accuracy",
]


def write_audit_csv(
    rows: list[dict], path: Path, fieldnames: list[str] | None = None,
) -> None:
    """Write rows to CSV. If rows is empty, write headers only so downstream
    `pd.read_csv` does not raise `EmptyDataError`.
    """
    fields = fieldnames or (list(rows[0].keys()) if rows else DEFAULT_AUDIT_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--eval-results-dir", type=Path, default=None,
        help="Drive path with chck_NM/<slug>/results_*.json subtrees "
             "(preferred mode — full sub-task granularity).",
    )
    src.add_argument(
        "--predictions-glob", "-p", nargs="+",
        help="One or more glob patterns or exact paths to predictions JSON files. "
             "Fallback mode — only aggregated tasks, NOT full sub-tasks.",
    )
    parser.add_argument(
        "--h4-models", nargs="+", default=None,
        help="Required with --eval-results-dir. List of model tokens to convert "
             "(e.g. taam_v2-seed42 b0-seed42).",
    )
    parser.add_argument(
        "--out-dir", "-o", type=Path, default=REPO_ROOT / "runs/_h4",
        help="Where to write per-run eval_results.json (default: %(default)s).",
    )
    parser.add_argument(
        "--task-map", type=Path, default=DEFAULT_TASK_MAP,
        help="Path to phenomenon_to_task_map.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--audit-csv", type=Path, default=None,
        help="Override output path for the audit CSV "
             "(default: <out-dir>/audit.csv).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    task_map = load_task_map(args.task_map)
    print(f"Loaded task map with {len(task_map)} phenomena from {args.task_map}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_audit: list[dict] = []
    summaries: list[tuple[str, int, int, int]] = []

    if args.eval_results_dir is not None:
        if not args.h4_models:
            LOG.error("--h4-models is required when using --eval-results-dir")
            return 2
        print(f"Mode: eval-results-dir ({args.eval_results_dir})")
        print(f"Converting {len(args.h4_models)} model(s)...")
        for tok in args.h4_models:
            try:
                res, audit = convert_from_eval_dir(
                    args.eval_results_dir, tok, task_map, args.out_dir,
                )
            except Exception as e:
                LOG.exception("Failed to convert %s: %s", tok, e)
                continue
            all_audit.extend(audit)
            n_chck = len(res["checkpoints"])
            n_filled = sum(
                1 for cells in res["checkpoints"].values() for phen in cells.values()
                for v in phen.values() if v is not None
            )
            n_total = len(res["checkpoints"]) * len(task_map) * 3
            summaries.append((f"{res['condition_id']}_seed{res['seed']}",
                              n_chck, n_filled, n_total))
    else:
        paths = expand_paths(args.predictions_glob)
        if not paths:
            LOG.error("No predictions files matched any pattern. Aborting.")
            return 2
        print("Mode: predictions-glob (fallback — aggregated tasks only)")
        print(f"Converting {len(paths)} predictions file(s)...")
        for p in paths:
            try:
                res, audit = convert_one_file(p, task_map, args.out_dir)
            except Exception as e:
                LOG.exception("Failed to convert %s: %s", p, e)
                continue
            all_audit.extend(audit)
            n_chck = len(res["checkpoints"])
            n_filled = sum(
                1 for cells in res["checkpoints"].values() for phen in cells.values()
                for v in phen.values() if v is not None
            )
            n_total = len(res["checkpoints"]) * len(task_map) * 3
            summaries.append((f"{res['condition_id']}_seed{res['seed']}",
                              n_chck, n_filled, n_total))

    audit_path = args.audit_csv or (args.out_dir / "audit.csv")
    write_audit_csv(all_audit, audit_path)

    print()
    print("Per-run summary:")
    print(f"  {'run':30s}  {'#chck':>6s}  {'#cells_filled':>14s}  {'#cells_max':>11s}  {'coverage':>9s}")
    for label, n_chck, n_filled, n_total in summaries:
        cov = (n_filled / n_total * 100) if n_total else 0.0
        print(f"  {label:30s}  {n_chck:>6d}  {n_filled:>14d}  {n_total:>11d}  {cov:>8.1f}%")
    print()
    print(f"Audit CSV:                    {audit_path}")
    print(f"Per-run eval_results.json in: {args.out_dir}/")
    print()
    print("Next: run `python -m scripts.analyze_acquisition_order --runs "
          f"{args.out_dir}/*_seed*/`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
