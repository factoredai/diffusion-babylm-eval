"""Masked-diffusion training loop for the BabyLM 2026 Strict-Small track.

Single entry point for every diffusion experiment. The condition config
(e.g. configs/MD_base.yaml) is deep-merged onto configs/base.yaml and selects the
masking variant and model size.

What happens at each step:
    1. Pull a packed block batch from the English stream.
    2. Apply the absorbing-state forward process (mask some tokens).
    3. Forward (bidirectional), compute the reweighted MDLM loss, backward, step.
    4. Every eval_every_steps: compute validation loss on the held-out dev slice.
    5. Save a checkpoint whenever the CFP word schedule says so.

Outputs (Drive-friendly; see docs/STORAGE.md):
    runs/{YYYY-MM-DD}_{condition}_seed{S}/
        config.yaml                 merged base + condition snapshot
        meta.json                   git SHA + GPU + timing
        log.jsonl                   one line per logged step / eval
        train_loss.csv              per-logged-step train loss
        summary.json                final summary
        checkpoint_schedule.json    the CFP word→step schedule actually used
        checkpoints/
            step_{step:05d}_words_{N}M/   HF model + ckpt_meta.json (+ tokenizer at upload time)

Usage:
    # CPU smoke test (synthetic data, tiny model, ~30s)
    python scripts/train.py --smoke-test --condition MD_base --seed 42

    # Real run (after scripts/prepare_data.py)
    python scripts/train.py --condition MD_base --seed 42 \
        --token-data data/tokens --tokenizer tokenizer/spm_16k.model
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG = logging.getLogger("train")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mdlm.config import MaskedDiffusionConfig  # noqa: E402
from mdlm.data import BatchProvider, build_streams  # noqa: E402
from mdlm.masking import MaskingProcess, diffusion_loss  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# CFP checkpoint schedule (expressed in WORDS seen, Strict-Small)
# ──────────────────────────────────────────────────────────────────────────────


def compute_cfp_checkpoint_schedule(
    words_per_step: float,
    total_steps: int,
    *,
    phase1_every: int,
    phase1_max: int,
    phase2_every: int,
    phase2_max: int,
) -> dict[int, int]:
    """Map {1-indexed step -> words_seen} per the Strict-Small CFP requirement.

    Checkpoints at every ``phase1_every`` words up to ``phase1_max`` (1M..10M),
    then every ``phase2_every`` up to ``phase2_max`` (10M..100M). The final step
    is always included. Strict-Small has no phase beyond 100M words.
    """
    schedule: dict[int, int] = {}

    def _add(threshold_words: int) -> None:
        step = int(np.ceil(threshold_words / words_per_step))
        if 0 < step <= total_steps:
            schedule[step] = int(round(step * words_per_step))

    w = phase1_every
    while w <= phase1_max:
        _add(w)
        w += phase1_every
    w = phase1_max + phase2_every
    while w <= phase2_max:
        _add(w)
        w += phase2_every

    schedule[total_steps] = int(round(total_steps * words_per_step))
    return dict(sorted(schedule.items()))


# ──────────────────────────────────────────────────────────────────────────────
# Config loading + run metadata
# ──────────────────────────────────────────────────────────────────────────────


def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_condition_config(condition_id: str) -> dict:
    """Load configs/{condition_id}.yaml merged onto configs/base.yaml."""
    base = yaml.safe_load((REPO_ROOT / "configs/base.yaml").read_text())
    for c in (REPO_ROOT / "configs").glob("*.yaml"):
        if c.name == "base.yaml":
            continue
        data = yaml.safe_load(c.read_text())
        cid = (data.get("condition") or {}).get("id")
        if cid == condition_id or c.stem == condition_id:
            data.pop("include", None)
            merged = _deep_merge(base, data)
            merged["_meta"] = {"condition_config_path": str(c.relative_to(REPO_ROOT))}
            return merged
    raise FileNotFoundError(f"No config with condition id {condition_id!r}")


def collect_run_metadata() -> dict:
    meta: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "git": {},
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            meta["cuda_available"] = True
            meta["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    def _git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            ).decode().strip() or None
        except Exception:  # noqa: BLE001
            return None

    sha = _git(["rev-parse", "HEAD"])
    if sha:
        meta["git"] = {"sha": sha, "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"])}
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────


def build_model(config: dict, vocab_size: int):
    from mdlm.model import MaskedDiffusionLM

    m = config["model"]
    cfg = MaskedDiffusionConfig(
        vocab_size=vocab_size,
        n_positions=m["n_positions"],
        n_embd=m["n_embd"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        ffn_mult=m["ffn_mult"],
        dropout=m["dropout"],
        layer_norm_eps=m.get("layer_norm_eps", 1e-5),
        # Training always runs with no layer duplication; duplication is an
        # inference-time-only knob, set by the eval backend.
        layer_duplication_factor=1,
        t_min=config["diffusion"]["t_min"],
        t_max=config["diffusion"]["t_max"],
        frequency_informed_masking=config["diffusion"]["frequency_informed_masking"],
    )
    model = MaskedDiffusionLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("MaskedDiffusionLM: %d params (~%.1fM)", n_params, n_params / 1e6)
    return model, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (validation loss on the held-out dev slice)
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(model, dev_stream, masking: MaskingProcess, n_batches: int, batch_size: int, device):
    import torch

    model.eval()
    total, count = 0.0, 0
    g = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for _ in range(n_batches):
            ids = torch.as_tensor(dev_stream.get_batch(batch_size), dtype=torch.long, device=device)
            corrupted, labels, weight = masking(ids, generator=g)
            logits = model(input_ids=corrupted).logits
            total += float(diffusion_loss(logits, labels, weight).item())
            count += 1
    model.train()
    return total / max(count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────


def find_latest_checkpoint(checkpoints_dir: Path) -> tuple[Path, int] | None:
    """Return (dir, completed_steps) of the resumable checkpoint with the highest
    step, or None. Only checkpoints carrying a trainer_state.pt are resumable."""
    best: tuple[Path, int] | None = None
    if not checkpoints_dir.is_dir():
        return None
    for d in checkpoints_dir.glob("step_*"):
        if not (d / "trainer_state.pt").exists():
            continue
        try:
            step = json.loads((d / "ckpt_meta.json").read_text())["step"]
        except Exception:
            continue
        if best is None or step > best[1]:
            best = (d, int(step))
    return best


def train(
    config: dict,
    seed: int,
    token_data_dir: Path | None,
    output_dir: Path,
    use_synthetic: bool,
    total_steps_override: int | None = None,
    vocab_size_override: int | None = None,
    resume: bool = True,
) -> dict:
    import torch
    from torch.optim import AdamW

    cond_id = config["condition"]["id"]
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    meta = collect_run_metadata()
    meta.update({"condition_id": cond_id, "seed": seed, "output_dir": str(output_dir)})
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    torch.manual_seed(seed)
    np.random.seed(seed)

    block_size = int(config["model"]["n_positions"])
    batch_size = int(config["training"]["batch_size"])
    train_stream, dev_stream = build_streams(
        token_data_dir, block_size=block_size, use_synthetic=use_synthetic, seed=seed
    )
    provider = BatchProvider(train_stream)

    vocab_size = vocab_size_override or int(config["tokenizer"]["vocab_size"])
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    LOG.info("Using device: %s", device)

    model, model_cfg = build_model(config, vocab_size=vocab_size)
    model = model.to(device)
    optim = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(float(config["training"]["beta1"]), float(config["training"]["beta2"])),
    )

    masking = MaskingProcess(
        mask_token_id=model_cfg.mask_token_id,
        t_min=model_cfg.t_min,
        t_max=model_cfg.t_max,
        pad_token_id=model_cfg.pad_token_id,
    )

    # ── Resume from the latest checkpoint on Drive (survives Colab restarts) ──
    # Each checkpoint stores model weights + a trainer_state.pt (optimizer, step,
    # RNG). On a fresh runtime we reload all of it and continue at the saved step;
    # the data stream is deterministic in `step`, so the order is preserved.
    start_step = 0
    resumed_from = None
    found = find_latest_checkpoint(checkpoints_dir) if resume else None
    if found is not None:
        ckpt_dir, start_step = found
        # weights_only=False: this is our own trainer_state (optimizer + RNG
        # objects), not an untrusted download.
        state = torch.load(ckpt_dir / "trainer_state.pt", map_location=device, weights_only=False)
        try:
            from safetensors.torch import load_file
            sd = load_file(str(ckpt_dir / "model.safetensors"))
        except Exception:
            sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        optim.load_state_dict(state["optimizer"])
        try:
            torch.set_rng_state(state["torch_rng"].cpu())
            np.random.set_state(state["numpy_rng"])
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        except Exception as e:
            LOG.warning("Could not restore RNG state: %s", e)
        resumed_from = ckpt_dir.name
        LOG.info("Resuming from %s at step %d.", resumed_from, start_step)
        print(f"** Resuming from checkpoint {resumed_from} (step {start_step}). **")

    # Derive total_steps from the word budget unless overridden.
    words_per_token = float(config["data"]["words_per_token"])
    words_per_step = batch_size * block_size * words_per_token
    if total_steps_override is not None:
        total_steps = total_steps_override
    else:
        total_steps = int(np.ceil(config["training"]["total_word_budget"] / words_per_step))

    ckpt_cfg = config["checkpointing"]
    ckpt_schedule = compute_cfp_checkpoint_schedule(
        words_per_step=words_per_step,
        total_steps=total_steps,
        phase1_every=ckpt_cfg["phase1_every_words"],
        phase1_max=ckpt_cfg["phase1_max_words"],
        phase2_every=ckpt_cfg["phase2_every_words"],
        phase2_max=ckpt_cfg["phase2_max_words"],
    )
    (output_dir / "checkpoint_schedule.json").write_text(json.dumps(
        {"words_per_step": words_per_step, "total_steps": total_steps, "schedule": ckpt_schedule},
        indent=2,
    ))
    LOG.info("CFP schedule: %d checkpoints over %d steps (%.1f words/step).",
             len(ckpt_schedule), total_steps, words_per_step)

    file_mode = "a" if start_step > 0 else "w"
    log_file = (output_dir / "log.jsonl").open(file_mode, encoding="utf-8")
    loss_csv_path = output_dir / "train_loss.csv"
    write_header = file_mode == "w" or not loss_csv_path.exists() or loss_csv_path.stat().st_size == 0
    loss_csv = loss_csv_path.open(file_mode, newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_csv)
    if write_header:
        loss_writer.writerow(["step", "loss"])

    log_every = int(config["logging"]["log_every_steps"])
    eval_every = int(config["logging"]["eval_every_steps"])
    n_eval_batches = int(config["logging"]["n_eval_batches"])
    grad_clip = float(config["training"]["gradient_clip"])
    # Gradient accumulation: process `grad_accum` micro-batches per optimizer
    # update so the *effective* batch is batch_size * grad_accum, while peak GPU
    # memory stays at one micro-batch. Word accounting is per micro-batch, so the
    # CFP checkpoint schedule is unaffected.
    grad_accum = max(int(config["training"].get("grad_accum_steps", 1)), 1)
    LOG.info("Effective batch size: %d (batch_size=%d x grad_accum=%d)",
             batch_size * grad_accum, batch_size, grad_accum)

    if start_step >= total_steps:
        print(f"Run already complete ({start_step}/{total_steps} steps). Nothing to do.")
    print(f"\n=== Training {cond_id} (seed={seed}): steps {start_step}->{total_steps} on {device} ===")
    t0 = time.time()
    last = t0
    run_loss, run_count = 0.0, 0
    optim.zero_grad(set_to_none=True)

    for step in range(start_step, total_steps):
        batch = provider.next_batch(batch_size, step)
        ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=device)
        corrupted, labels, weight = masking(ids)

        model.train()
        logits = model(input_ids=corrupted).logits
        loss = diffusion_loss(logits, labels, weight)
        # Scale so accumulated grads average (not sum) over the window.
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == total_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)

        cur = float(loss.detach().item())
        run_loss += cur
        run_count += 1
        loss_writer.writerow([step + 1, cur])

        if (step + 1) % log_every == 0 or step == 0:
            now = time.time()
            avg = run_loss / max(run_count, 1)
            elapsed = now - last
            last = now
            log_file.write(json.dumps({
                "step": step + 1, "phase": "train", "avg_train_loss": avg,
                "words_seen": int(round((step + 1) * words_per_step)), "elapsed_sec": elapsed,
            }) + "\n")
            log_file.flush()
            run_loss, run_count = 0.0, 0
            print(f"  step {step+1:6d}/{total_steps}  loss={avg:.4f}  "
                  f"words={int((step+1)*words_per_step):,}  ({elapsed:.1f}s)")

        if (step + 1) % eval_every == 0:
            val = evaluate(model, dev_stream, masking, n_eval_batches, batch_size, device)
            log_file.write(json.dumps({"step": step + 1, "phase": "eval", "val_loss": val}) + "\n")
            log_file.flush()
            print(f"  eval@{step+1}  val_loss={val:.4f}")

        if (step + 1) in ckpt_schedule:
            words_seen = ckpt_schedule[step + 1]
            words_m = max(int(round(words_seen / 1_000_000)), 1)
            ckpt_dir = checkpoints_dir / f"step_{step+1:05d}_words_{words_m:03d}M"
            ckpt_dir.mkdir(exist_ok=True)
            model.save_pretrained(ckpt_dir)
            # Trainer state for full resume (optimizer + RNG + completed step).
            torch.save({
                "step": step + 1,
                "optimizer": optim.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
            }, ckpt_dir / "trainer_state.pt")
            (ckpt_dir / "ckpt_meta.json").write_text(json.dumps({
                "step": step + 1, "words_seen": words_seen, "words_m": words_m,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }, indent=2))
            print(f"  ckpt @{step+1} (words={words_seen:,}) -> {ckpt_dir.name}")

    log_file.close()
    loss_csv.close()

    elapsed_total = time.time() - t0
    summary = {
        "condition_id": cond_id, "seed": seed, "total_steps": total_steps,
        "words_per_step": words_per_step,
        "words_seen_total": int(round(total_steps * words_per_step)),
        "elapsed_sec": round(elapsed_total, 1),
        "n_checkpoints": len(ckpt_schedule),
        "output_dir": str(output_dir),
        "resumed_from": resumed_from,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    meta.update({"finished_at": datetime.now().isoformat(timespec="seconds"),
                 "elapsed_sec": round(elapsed_total, 1), "status": "success"})
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {elapsed_total:.1f}s.  {summary['words_seen_total']:,} words seen.")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, help="Condition id (e.g. MD_base)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Synthetic data + tiny model; runs in ~30s on CPU.")
    parser.add_argument("--token-data", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoints and start a fresh run.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = load_condition_config(args.condition)
    method_type = config["condition"]["method"]["type"]
    print(f"Loaded merged config for '{args.condition}' (method={method_type}).")
    if method_type != "masked_diffusion":
        print(f"ERROR: '{args.condition}' has method.type={method_type!r}. This trainer only "
              "trains masked-diffusion conditions. (AR_baseline_ref is evaluation-only.)")
        return 2

    today = datetime.now().strftime("%Y-%m-%d")
    run_id = f"{today}_{args.condition}_seed{args.seed}"
    vocab_override = None
    if args.smoke_test:
        output_dir = args.output_dir or (REPO_ROOT / "runs" / f"_smoke_{run_id}")
        total_steps = args.total_steps or 60
        # Downsize for CPU.
        config["model"].update({"n_layer": 2, "n_head": 4, "n_embd": 64, "n_positions": 64})
        config["training"]["batch_size"] = 8
        config["logging"].update({"log_every_steps": 10, "eval_every_steps": 20, "n_eval_batches": 2})
        # Tiny vocab matching the synthetic corpus.
        vocab_override = 256
    else:
        output_dir = args.output_dir
        if output_dir is None:
            # Reuse an existing run for this (condition, seed) if one has
            # checkpoints, so a re-launch on another day continues it instead of
            # starting a fresh dated run. --no-resume forces a new dir.
            runs_root = REPO_ROOT / "runs"
            candidates = sorted(
                (d for d in runs_root.glob(f"*_{args.condition}_seed{args.seed}")
                 if (d / "checkpoints").is_dir() and any((d / "checkpoints").glob("step_*"))),
                key=lambda d: d.stat().st_mtime,
            )
            if candidates and not args.no_resume:
                output_dir = candidates[-1]
                print(f"Found existing run with checkpoints -> {output_dir.name} "
                      f"(use --no-resume to start fresh).")
            else:
                output_dir = runs_root / run_id
        total_steps = args.total_steps

    summary = train(
        config=config, seed=args.seed, token_data_dir=args.token_data,
        output_dir=output_dir, use_synthetic=args.smoke_test,
        total_steps_override=total_steps, vocab_size_override=vocab_override,
        resume=not args.no_resume,
    )
    print("\n" + "=" * 65)
    print(f"  Run complete: {summary['output_dir']}")
    print(f"  Words seen:   {summary['words_seen_total']:,}")
    print(f"  Checkpoints:  {summary['n_checkpoints']}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
