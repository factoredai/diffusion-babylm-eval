"""Single source of truth for H4 child-acquisition norms.

Reads `analyses/acquisition_order/child_norms_dataset.csv` and exposes it in the
two shapes the rest of the pipeline needs:

1. `load_task_map(csv)` -> {phenomenon_id: {lang: [lm-eval task names]}}
   (the same shape `predictions_to_eval_results.load_task_map` returns for the
   old YAML, so it is a drop-in replacement).

2. `load_child_norms(csv)` -> mapping dict in the YAML "phenomena" shape that
   `analyze_acquisition_order._load_phenomena_mapping` returns, i.e.
   {"phenomena": [{"id": ..., "languages": {lang: {"age_months": ...}}}, ...]}.

The CSV is the canonical data; this module just parses it. Comment lines
(starting with '#') and blank lines are ignored. The `benchmark_tasks` column is
a comma-separated list of EXACT lm-eval task names.

See docs/h4_child_norms.md for the provenance of every number.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NORMS_CSV = REPO_ROOT / "analyses/acquisition_order/child_norms_dataset.csv"


def _read_rows(csv_path: Path) -> list[dict]:
    """Read the CSV, skipping comment (#) and blank lines, return list of dicts."""
    text = csv_path.read_text(encoding="utf-8")
    # Strip comment lines but keep the header (first non-comment line).
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        kept.append(line)
    if not kept:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(kept)))
    rows = []
    for r in reader:
        # Normalize whitespace in keys/values
        rows.append({(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                     for k, v in r.items()})
    return rows


def _split_tasks(cell: str | None) -> list[str]:
    if not cell:
        return []
    return [t.strip() for t in cell.split(",") if t.strip()]


def load_task_map(csv_path: Path = DEFAULT_NORMS_CSV) -> dict[str, dict[str, list[str]]]:
    """Return {phenomenon_id: {lang: [task names]}} from the norms CSV."""
    rows = _read_rows(csv_path)
    out: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        phen = r.get("phenomenon_id")
        lang = r.get("language")
        if not phen or not lang:
            continue
        tasks = _split_tasks(r.get("benchmark_tasks"))
        out.setdefault(phen, {})[lang] = tasks
    return out


def load_child_norms(csv_path: Path = DEFAULT_NORMS_CSV) -> dict:
    """Return the mapping dict in the YAML "phenomena" shape used by the analyzer.

    Shape::

        {"phenomena": [
            {"id": <phen>, "languages": {<lang>: {"age_months": <float>,
                                                  "confidence": <str>,
                                                  "age_min_months": <float|None>,
                                                  "age_max_months": <float|None>}}},
            ...
        ]}
    """
    rows = _read_rows(csv_path)
    phen_order: list[str] = []
    by_phen: dict[str, dict[str, dict]] = {}
    for r in rows:
        phen = r.get("phenomenon_id")
        lang = r.get("language")
        if not phen or not lang:
            continue
        if phen not in by_phen:
            by_phen[phen] = {}
            phen_order.append(phen)

        def _num(key: str):
            v = r.get(key)
            if v in (None, "", "null", "n/a"):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        by_phen[phen][lang] = {
            "age_months": _num("age_months"),
            "age_min_months": _num("age_min_months"),
            "age_max_months": _num("age_max_months"),
            "confidence": r.get("confidence") or "",
            "mastery_criterion": r.get("mastery_criterion") or "",
            "primary_source": r.get("primary_source") or "",
        }
    phenomena = [{"id": p, "languages": by_phen[p]} for p in phen_order]
    return {"phenomena": phenomena, "meta": {"source": str(csv_path),
                                              "source_format": "csv"}}


def summary(csv_path: Path = DEFAULT_NORMS_CSV) -> str:
    """Human-readable summary: phenomena per language, distinct ages, low-conf."""
    norms = load_child_norms(csv_path)
    per_lang: dict[str, list[float]] = {}
    low_conf: dict[str, list[str]] = {}
    for phen in norms["phenomena"]:
        for lang, e in phen["languages"].items():
            if e.get("age_months") is not None:
                per_lang.setdefault(lang, []).append(e["age_months"])
                if (e.get("confidence") or "").lower() == "low":
                    low_conf.setdefault(lang, []).append(phen["id"])
    lines = ["Child norms summary:"]
    for lang in sorted(per_lang):
        ages = per_lang[lang]
        lines.append(
            f"  {lang}: {len(ages)} phenomena, "
            f"{len(set(ages))} distinct ages "
            f"[{int(min(ages))}-{int(max(ages))} mo], "
            f"{len(low_conf.get(lang, []))} low-confidence"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    tm = load_task_map()
    total_tasks = sum(len(t) for ph in tm.values() for t in ph.values())
    print(f"\nTask map: {len(tm)} phenomena, {total_tasks} (phen,lang)->task lists")
