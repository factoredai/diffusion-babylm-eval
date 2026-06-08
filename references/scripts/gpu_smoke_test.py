#!/usr/bin/env python
"""Fast GPU smoke test (<30 seconds on H100, <2 minutes on T4).

Validates the full training stack on the actual GPU before committing to a
long run:
    1. CUDA / device visibility
    2. dtype support (bf16 on Ampere+, fp16 fallback)
    3. Model build (GPT-2-tiny from the production config)
    4. One forward pass + one backward pass on a real packed batch
    5. Optimizer step
    6. Throughput estimate (tokens/sec) to project wall-clock for 20k steps

This is purely a "did everything install correctly and does the GPU work"
check. It does NOT update model weights in any meaningful way (only 5
steps). It deliberately uses the production config so the throughput
estimate is realistic.

Exits non-zero on any failure so it can be wired into Makefile/CI.

Usage:
    python scripts/gpu_smoke_test.py
    python scripts/gpu_smoke_test.py --condition TAAM --steps 10
    python scripts/gpu_smoke_test.py --no-real-data    # synthetic batch only
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check_cuda() -> dict:
    """Verify CUDA visibility and device specs."""
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if not info["cuda_available"]:
        return info
    info.update({
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": torch.cuda.get_device_capability(0),
        "device_total_mem_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    })
    return info


def pick_dtype(cuda_info: dict, requested: str):
    """Choose a runtime dtype that the current GPU actually supports.

    bf16 needs Ampere+ (compute capability >= 8.0). T4/V100 fall back to fp16.
    fp32 always works but is 2-3x slower on H100/A100.
    """
    import torch

    if requested == "auto":
        if cuda_info.get("bf16_supported", False):
            return torch.bfloat16, "bfloat16"
        if cuda_info.get("cuda_available", False):
            return torch.float16, "float16"
        return torch.float32, "float32"
    return {
        "bfloat16": (torch.bfloat16, "bfloat16"),
        "float16":  (torch.float16, "float16"),
        "fp32":     (torch.float32, "float32"),
    }[requested]


def build_model_from_config(cfg_path: Path):
    """Build GPT-2 tiny using the production config (merged base + condition)."""
    import yaml
    from transformers import GPT2Config, GPT2LMHeadModel

    base = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())
    if cfg_path.name != "base.yaml":
        cond = yaml.safe_load(cfg_path.read_text())
        # Deep-merge (1 level deep is enough for our schema)
        for k, v in cond.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v

    m = base["model"]
    vocab_size = base["tokenizer"]["vocab_size"]
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=m["block_size"],
        n_embd=m["n_embd"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        bos_token_id=1,
        eos_token_id=2,
    )
    return GPT2LMHeadModel(cfg), base


def make_synthetic_batch(batch_size: int, block_size: int, vocab_size: int, device):
    """One random packed batch (no real data needed)."""
    import torch

    return torch.randint(
        low=4,  # skip the 0..3 special-token range
        high=vocab_size,
        size=(batch_size, block_size),
        dtype=torch.long,
        device=device,
    )


def make_real_batch(batch_size: int, block_size: int, device, pi: dict[str, float] | None):
    """One real batch from data/tokens/ via PackedLanguageStream.

    Returns (input_ids, language). Falls back to synthetic if shards missing.
    """
    import json
    import torch
    from taam.data import MultilingualMixer, PackedLanguageStream

    token_dir = REPO_ROOT / "data" / "tokens"
    if not (token_dir / "manifest.json").exists():
        return None, None

    mani = json.loads((token_dir / "manifest.json").read_text())
    eos = int(mani["tokenizer_eos_id"])
    streams = {}
    for lang in mani["per_language"]:
        shards = sorted((token_dir / lang).glob("shard_*.npy"))
        if not shards:
            return None, None
        streams[lang] = PackedLanguageStream(
            token_id_files=shards[-1:],  # last shard only for speed
            block_size=block_size,
            eos_token_id=eos,
            seed=0,
        )
    pi = pi or {l: 1.0 / len(streams) for l in streams}
    mixer = MultilingualMixer(streams=streams, batch_size=batch_size, seed=0)
    batch = mixer.next_batch(pi=pi, step=0)
    return (
        torch.as_tensor(batch.input_ids, dtype=torch.long, device=device),
        batch.language,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="TAAM",
                        help="condition config under configs/ to use for hparams")
    parser.add_argument("--steps", type=int, default=5,
                        help="forward+backward iterations to run")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override config's batch_size (useful on small GPUs)")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "fp32"])
    parser.add_argument("--no-real-data", action="store_true",
                        help="skip data/tokens/ check; always use synthetic batch")
    args = parser.parse_args()

    _print_header("GPU smoke test")
    print(f"Python: {sys.version.split()[0]}")

    # 1. CUDA visibility ----------------------------------------------------
    cuda = check_cuda()
    print(f"torch version: {cuda['torch_version']}")
    if not cuda["cuda_available"]:
        print("\n[FAIL] CUDA is not available. Running on CPU will be too slow.")
        print("       In Colab: Runtime > Change runtime type > GPU.")
        return 2
    print(
        f"GPU: {cuda['device_name']}  cap={cuda['device_capability']}  "
        f"mem={cuda['device_total_mem_gb']:.1f} GB  bf16={cuda['bf16_supported']}"
    )

    # 2. dtype choice -------------------------------------------------------
    import torch
    dtype, dtype_name = pick_dtype(cuda, args.dtype)
    print(f"Using dtype: {dtype_name}")

    # 3. Model build --------------------------------------------------------
    _print_header("Build model")
    cfg_path = REPO_ROOT / "configs" / f"{args.condition}.yaml"
    if not cfg_path.exists():
        # Fall back to base.yaml only
        cfg_path = REPO_ROOT / "configs" / "base.yaml"
    print(f"Config: {cfg_path.relative_to(REPO_ROOT)}")

    t0 = time.perf_counter()
    model, cfg = build_model_from_config(cfg_path)
    device = torch.device("cuda")
    model = model.to(device=device, dtype=dtype)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built in {time.perf_counter() - t0:.2f}s: "
          f"{n_params/1e6:.1f}M params, on {device}, dtype={dtype_name}")

    block_size = cfg["model"]["block_size"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    vocab_size = cfg["tokenizer"]["vocab_size"]
    print(f"batch_size={batch_size}  block_size={block_size}  vocab={vocab_size}")

    # 4. Build one batch ----------------------------------------------------
    _print_header("Prepare batch")
    real_batch, lang = (None, None)
    if not args.no_real_data:
        real_batch, lang = make_real_batch(batch_size, block_size, device,
                                           pi=None)
    if real_batch is not None:
        print(f"Real batch from data/tokens/ (sampled lang={lang}, "
              f"shape={tuple(real_batch.shape)})")
        input_ids = real_batch
    else:
        if not args.no_real_data:
            print("data/tokens/ not found; using synthetic batch.")
        input_ids = make_synthetic_batch(batch_size, block_size, vocab_size, device)
        print(f"Synthetic batch shape={tuple(input_ids.shape)}")

    # 5. Optimizer ----------------------------------------------------------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        weight_decay=cfg["training"]["weight_decay"],
    )

    # 6. Train loop ---------------------------------------------------------
    _print_header(f"Run {args.steps} forward+backward steps")
    # First step often pays a JIT/cublas warmup tax; time the post-warmup steps.
    torch.cuda.synchronize()
    warmup_start = time.perf_counter()
    losses: list[float] = []
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
        opt.step()
        losses.append(float(loss.detach().item()))
        if step == 0:
            torch.cuda.synchronize()
            warmup_elapsed = time.perf_counter() - warmup_start
            print(f"  step 1 (incl. warmup): {warmup_elapsed:.2f}s  loss={losses[-1]:.4f}")
            measure_start = time.perf_counter()
        else:
            print(f"  step {step + 1}: loss={losses[-1]:.4f}")

    torch.cuda.synchronize()
    measure_elapsed = time.perf_counter() - measure_start if args.steps > 1 else 0.0
    steady_steps = max(args.steps - 1, 0)

    # 7. Memory usage -------------------------------------------------------
    _print_header("Memory & throughput")
    mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(f"Peak GPU memory: {mem_allocated:.2f} GB allocated, "
          f"{mem_reserved:.2f} GB reserved")

    if steady_steps > 0:
        tokens_per_step = batch_size * block_size
        steps_per_sec = steady_steps / measure_elapsed
        tokens_per_sec = tokens_per_step * steps_per_sec
        print(
            f"Steady-state: {steps_per_sec:.2f} steps/s  "
            f"({tokens_per_sec/1e3:.1f}k tokens/s)"
        )
        # Project wall-clock for the canonical 20k-step run.
        canonical_steps = int(cfg["training"]["total_steps"])
        eta_seconds = canonical_steps / steps_per_sec
        eta_min = eta_seconds / 60.0
        print(
            f"Projected wall-clock for {canonical_steps:,} steps: "
            f"{eta_seconds/3600:.2f} h ({eta_min:.1f} min)"
        )
        print(
            f"Projected for 33 runs (11 conditions x 3 seeds): "
            f"{33 * eta_seconds / 3600:.1f} h of GPU time"
        )

    _print_header("Result")
    if not losses:
        print("[FAIL] no steps executed.")
        return 1
    if losses[0] < losses[-1] - 1e-3:
        print(f"[WARN] loss increased from {losses[0]:.4f} to {losses[-1]:.4f}.")
    print("[OK]  GPU smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
