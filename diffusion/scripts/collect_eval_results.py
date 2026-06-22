#!/usr/bin/env python3
"""Collect eval scores from ``strict/results/`` and archive them to Drive.

Everything that used to live in Cell 10 of ``3_evaluation_pipeline.ipynb``.
Works after the full zero-shot eval alone (GLUE / submission zip optional).

Persistence layout (one immutable directory per eval run, like MLflow / W&B):
    {drive_root}/{model}/{YYYY-MM-DD_HHMMSS}/
        eval_meta.json        model, backend, track, eval date, git SHA, tasks
        results_summary.csv   flattened scores (split, task, metric, score)
        results/              full strict/results/ tree (reports + predictions)
        *.zip                 submission file, if collate_preds was run
Append-only: the timestamped folder is never overwritten; multiple evals of the
same model coexist and sort chronologically.

Usage:
    python diffusion/scripts/collect_eval_results.py \
        --model-id amosluna/babylm-2026-strict-small-mdlm-seed42 \
        --backend mlm --track strict-small \
        --drive-root /content/drive/MyDrive/Researchs/BabyLM_diffusion_G4/results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = REPO_ROOT / "strict"


def parse_zero_shot(backend: str) -> list[tuple]:
    """One row per task: the '### AVERAGE' line of each best_temperature_report."""
    rows = []
    pattern = str(STRICT_DIR / f"results/*/main/zero_shot/{backend}/*/*/best_temperature_report.txt")
    for rep in sorted(glob.glob(pattern)):
        parts = rep.split("/")
        lines = Path(rep).read_text().splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("### AVERAGE"):
                try:
                    rows.append(("zero_shot", parts[-3], parts[-2], float(lines[i + 1].strip())))
                except (IndexError, ValueError):
                    pass
                break
    return rows


def parse_finetune() -> list[tuple]:
    rows = []
    for res in sorted(glob.glob(str(STRICT_DIR / "results/*/main/finetune/*/results.txt"))):
        task = res.split("/")[-2]
        for line in Path(res).read_text().splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("accuracy", "f1", "mcc"):
                try:
                    rows.append(("finetune", task, k.strip(), float(v.strip()) * 100))
                except ValueError:
                    pass
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--backend", default="mlm")
    p.add_argument("--track", default="strict-small")
    p.add_argument("--drive-root", required=True, help="Drive results/ root to archive into.")
    args = p.parse_args()

    rows = parse_zero_shot(args.backend) + parse_finetune()
    if not rows:
        print("!! No scores found under strict/results/ — run an eval first.", file=sys.stderr)
        return 1

    print(f"{'split':<10} {'task':<18} {'metric':<24} score")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<18} {r[2]:<24} {r[3]:.2f}")

    # Unique, append-only eval-run directory: {model}/{timestamp}.
    safe = args.model_id.replace("/", "__")
    eval_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dst = Path(args.drive_root) / safe / eval_id
    assert not dst.exists(), f"Eval dir already exists (re-run): {dst}"
    dst.mkdir(parents=True)

    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=STRICT_DIR,
                             capture_output=True, text=True).stdout.strip()
    (dst / "eval_meta.json").write_text(json.dumps({
        "model_id": args.model_id, "backend": args.backend, "track": args.track,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_code_git_sha": git_sha, "n_scores": len(rows),
        "tasks": sorted({r[1] for r in rows}),
    }, indent=2))

    with open(dst / "results_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "task", "metric", "score"])
        w.writerows(rows)
    if (STRICT_DIR / "results").is_dir():
        shutil.copytree(STRICT_DIR / "results", dst / "results")
    for z in STRICT_DIR.glob("*.zip"):  # collate_preds writes the submission zip here
        shutil.copy2(z, dst)

    print(f"\nSaved {len(rows)} scores -> {dst}/")
    print("Previous evals of this model:")
    for d in sorted((Path(args.drive_root) / safe).glob("*/")):
        print("  ", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
