#!/usr/bin/env python
"""Sequential launcher for the full experimental matrix (conditions x seeds).

Design goals
------------
1. **Resumable**: if a previous run completed (`DONE` marker file present)
   the job is skipped on re-launch. Crashed runs are auto-detected (no
   `DONE`) and re-attempted up to `--retries` times.
2. **Ordered for sanity**: TAAM (the headline) goes FIRST per seed so a
   broken pipeline surfaces immediately, not after 15 baseline runs.
3. **Per-run isolation**: each job has its own output_dir + log file. No
   shared state. A SIGKILL on one run never corrupts another.
4. **Wall-clock honesty**: prints an ETA after the first run completes,
   so the user can decide whether to keep Colab open or come back later.
5. **Drive-safe**: works with output_dir symlinked to Google Drive (the
   default with `colab_bootstrap.sh --drive-root ...`).

The matrix (locked in improved_research_context_v2.md §5 and README):

    Static baselines:   B0, B1, B2, P1, P2, O2
    Online:             O1, TAAM
    Monolingual:        M_EN, M_NL, M_ZH

By default we run 11 conditions x 3 seeds = 33 runs.

Usage
-----
    # Full matrix, 3 seeds, 20k steps each (~16h on H100):
    python scripts/run_matrix.py

    # Quick screening pass (10k steps, half the time):
    python scripts/run_matrix.py --total-steps 10000

    # Only the online conditions:
    python scripts/run_matrix.py --conditions TAAM O1 --seeds 42

    # Resume after a crash (skips DONE runs automatically):
    python scripts/run_matrix.py   # same command; just re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Priority order: things we want to see first if something is going to break.
# TAAM is the headline; the static baselines are the safest sanity checks.
DEFAULT_CONDITIONS: tuple[str, ...] = (
    "TAAM",     # headline (typology prior + EXP3 online updates)
    "B0",       # uniform tokens (simplest sanity baseline)
    "P1",       # static typological prior (head-to-head vs TAAM, tests H1)
    "O1",       # EXP3 from uniform init (online-only; tests H2)
    "B1",       # uniform bytes (byte-premium corrected)
    "B2",       # proportional to data
    "P2",       # reverse prior (negative control)
    "O2",       # random fixed schedule
    "M_EN",     # monolingual English oracle
    "M_NL",     # monolingual Dutch oracle
    "M_ZH",     # monolingual Chinese oracle
)

# Default seed pool matches configs/base.yaml -> experiment.seed_pool.
DEFAULT_SEEDS: tuple[int, ...] = (13, 42, 71)


@dataclass
class Job:
    condition: str
    seed: int
    output_dir: Path
    total_steps: int
    eval_every: int | None
    token_data: Path
    tokenizer: Path
    extra_args: list[str] = field(default_factory=list)

    @property
    def log_path(self) -> Path:
        return self.output_dir / "train.log"

    @property
    def done_marker(self) -> Path:
        return self.output_dir / "DONE"

    @property
    def fail_marker(self) -> Path:
        return self.output_dir / "FAILED"

    def is_done(self) -> bool:
        return self.done_marker.exists()

    def is_failed(self) -> bool:
        return self.fail_marker.exists() and not self.is_done()

    def build_cmd(self) -> list[str]:
        cmd: list[str] = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--condition", self.condition,
            "--seed", str(self.seed),
            "--token-data", str(self.token_data),
            "--tokenizer", str(self.tokenizer),
            "--output-dir", str(self.output_dir),
            "--total-steps", str(self.total_steps),
        ]
        if self.eval_every is not None:
            cmd += ["--eval-every", str(self.eval_every)]
        cmd += self.extra_args
        return cmd


def fmt_hms(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _short(path: Path) -> str:
    """Show a path as repo-relative when possible, else absolute.

    Required because output dirs are often symlinked into Drive, which puts
    them outside REPO_ROOT — `Path.relative_to` would raise.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_one(job: Job, retries: int) -> tuple[bool, float]:
    """Run a single job with retries. Returns (success, wall_seconds)."""
    job.output_dir.mkdir(parents=True, exist_ok=True)
    if job.fail_marker.exists():
        job.fail_marker.unlink()

    attempt = 0
    total_wall = 0.0
    while attempt <= retries:
        attempt += 1
        cmd = job.build_cmd()
        print(f"      cmd: {' '.join(cmd)}")
        t0 = time.perf_counter()
        # Always tail the log to the parent process so the user sees progress.
        # We use Popen so we can stream both to file and stdout.
        with job.log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write(f"\n\n===== attempt {attempt} at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log_fh.write(" ".join(cmd) + "\n")
            log_fh.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                text=True,
                bufsize=1,
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_fh.write(line)
                    log_fh.flush()
                    # Forward only short status lines to terminal to avoid noise.
                    stripped = line.rstrip()
                    if any(tag in stripped for tag in ("step ", "eval ", "ERROR", "Trace", "checkpoint")):
                        print(f"        {stripped[:140]}")
                rc = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
        elapsed = time.perf_counter() - t0
        total_wall += elapsed
        if rc == 0:
            job.done_marker.write_text(json.dumps({
                "condition": job.condition,
                "seed": job.seed,
                "total_steps": job.total_steps,
                "wall_seconds": round(elapsed, 1),
                "attempts": attempt,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2))
            print(f"      ✓ done in {fmt_hms(elapsed)} (attempt {attempt})")
            return True, total_wall
        print(f"      ✗ exit {rc} after {fmt_hms(elapsed)} (attempt {attempt})")
        if attempt > retries:
            job.fail_marker.write_text(json.dumps({
                "exit_code": rc,
                "attempts": attempt,
                "wall_seconds": round(total_wall, 1),
                "log": _short(job.log_path),
            }, indent=2))
            return False, total_wall
        backoff = min(30 * attempt, 120)
        print(f"      retrying in {backoff}s ...")
        time.sleep(backoff)
    return False, total_wall


def _resolve_run_dir(output_root: Path, condition: str, seed: int, today: str) -> Path:
    """Pick the output_dir for a (condition, seed) pair, honoring resume.

    Resume semantics: if any directory matching ``*_{condition}_seed{seed}``
    already exists under ``output_root`` (from a previous matrix launch,
    possibly on a different day), reuse it. This is required so that DONE
    markers from a previous session are still found.

    First launch: create a new directory named ``{today}_{condition}_seed{seed}``.
    """
    pattern = f"*_{condition}_seed{seed}"
    matches = sorted(p for p in output_root.glob(pattern) if p.is_dir())
    if matches:
        # ISO-date prefix orders lex == chrono; pick most recent.
        return matches[-1]
    return output_root / f"{today}_{condition}_seed{seed}"


def build_jobs(args: argparse.Namespace) -> list[Job]:
    """Build the ordered list of (condition, seed) jobs."""
    conditions = list(args.conditions)
    seeds = list(args.seeds)
    today = datetime.now().strftime("%Y-%m-%d")

    # Interleave: TAAM seed1, TAAM seed2, TAAM seed3, then next condition, ...
    # This way if you only have time for half the runs, you get full seed
    # coverage on the highest-priority conditions instead of all 11 conditions
    # on a single seed.
    jobs: list[Job] = []
    for cond in conditions:
        for seed in seeds:
            run_dir = _resolve_run_dir(args.output_dir, cond, seed, today)
            jobs.append(Job(
                condition=cond,
                seed=seed,
                output_dir=run_dir,
                total_steps=args.total_steps,
                eval_every=args.eval_every,
                token_data=args.token_data,
                tokenizer=args.tokenizer,
                extra_args=args.train_arg or [],
            ))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions", nargs="+", default=list(DEFAULT_CONDITIONS),
        help="ordered list of condition names (without .yaml). "
             "Default: TAAM first, then baselines, then monolingual.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="seeds to run for each condition (default: 13 42 71).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "runs",
        help="parent directory for per-run output_dir (default: runs/).",
    )
    parser.add_argument(
        "--token-data", type=Path, default=REPO_ROOT / "data" / "tokens",
        help="pretokenized-shards directory (default: data/tokens).",
    )
    parser.add_argument(
        "--tokenizer", type=Path,
        default=REPO_ROOT / "tokenizer" / "spm_32k_en_nl_zh.model",
        help="shared SentencePiece model path.",
    )
    parser.add_argument(
        "--total-steps", type=int, default=20_000,
        help="train.py --total-steps (default: 20000 per CFP).",
    )
    parser.add_argument(
        "--eval-every", type=int, default=None,
        help="train.py --eval-every (default: read from base.yaml).",
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="number of retries per job on non-zero exit (default: 2).",
    )
    parser.add_argument(
        "--train-arg", action="append", default=None,
        help="extra arg forwarded to train.py (repeatable, e.g. --train-arg --verbose).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the matrix and skip actual training.",
    )
    args = parser.parse_args()

    # Quick sanity checks before launching anything.
    if not args.token_data.exists():
        print(f"[FAIL] token_data does not exist: {args.token_data}", file=sys.stderr)
        print("       Run scripts/pretokenize.py (or `bash scripts/colab_bootstrap.sh`).",
              file=sys.stderr)
        return 2
    if not args.tokenizer.exists():
        print(f"[FAIL] tokenizer does not exist: {args.tokenizer}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)

    print("=" * 72)
    print(f"  Matrix: {len(args.conditions)} conditions x {len(args.seeds)} seeds "
          f"= {len(jobs)} runs")
    print(f"  Output dir : {_short(args.output_dir)}")
    print(f"  Steps/run  : {args.total_steps:,}")
    print(f"  Retries    : {args.retries}")
    print("=" * 72)

    skip_done = [j for j in jobs if j.is_done()]
    todo = [j for j in jobs if not j.is_done()]
    if skip_done:
        print(f"\nSkipping {len(skip_done)} job(s) already marked DONE:")
        for j in skip_done:
            print(f"  - {j.condition:<24s} seed={j.seed}")
    print(f"\nWill run {len(todo)} job(s):")
    for j in todo:
        print(f"  - {j.condition:<24s} seed={j.seed}  -> {_short(j.output_dir)}")

    if args.dry_run:
        print("\n--dry-run set; not launching anything.")
        return 0

    overall_t0 = time.perf_counter()
    results: dict[str, dict] = {}
    succeeded = 0
    failed = 0
    for i, job in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] {job.condition} seed={job.seed}")
        ok, wall = run_one(job, retries=args.retries)
        results[f"{job.condition}_seed{job.seed}"] = {
            "ok": ok,
            "wall_seconds": round(wall, 1),
            "output_dir": _short(job.output_dir),
        }
        if ok:
            succeeded += 1
        else:
            failed += 1

        # ETA from average wall-time of completed jobs.
        if succeeded > 0:
            elapsed_total = time.perf_counter() - overall_t0
            avg = elapsed_total / i
            remaining = len(todo) - i
            eta = avg * remaining
            print(f"   running ETA: {fmt_hms(eta)} ({remaining} run(s) left, avg {fmt_hms(avg)}/run)")

    overall_wall = time.perf_counter() - overall_t0

    # Persist a top-level matrix manifest for downstream analysis scripts.
    manifest_path = args.output_dir / "matrix_manifest.json"
    manifest = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions": args.conditions,
        "seeds": args.seeds,
        "total_steps_per_run": args.total_steps,
        "results": results,
        "skipped_done": [f"{j.condition}_seed{j.seed}" for j in skip_done],
        "wall_seconds_total": round(overall_wall, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 72)
    print(f"  Matrix complete in {fmt_hms(overall_wall)}")
    print(f"  Succeeded: {succeeded}  Failed: {failed}  Skipped: {len(skip_done)}")
    print(f"  Manifest:  {_short(manifest_path)}")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Already-completed runs remain DONE.")
        sys.exit(130)
