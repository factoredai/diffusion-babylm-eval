"""GPT-2-tiny multilingual training loop with TAAM mixing.

This is the SINGLE training entry point used by all 11 MVP conditions. The
condition config (e.g., configs/TAAM.yaml) selects:
    - method.type ∈ {static, online}
    - method.pi (static) or method.pi_0 + method.online (online)

What happens at each step:
    1. Sample language from current pi (EXP3 or static).
    2. Pull a packed block batch from that language's stream.
    3. Forward, compute CE loss, backward, optimizer step.
    4. Every eval_every_steps:
        - Compute per-language val loss on the held-out reward_dev slice.
        - If online: compute normalized excess-loss reward, call EXP3 update.
        - Log pi(t) + val losses to a JSONL run log.
    5. Save intermediate checkpoints at scheduled steps.

Outputs:
    runs/{condition_id}_seed{S}/
        config.yaml           (merged base + condition snapshot)
        log.jsonl             (one line per logged step)
        pi_history.csv        (consumed by Figure 1)
        checkpoint_{step}/    (HF model + tokenizer)

Usage:
    # Smoke test (CPU, 50 steps, synthetic data)
    python scripts/train.py --smoke-test --condition TAAM --seed 42

    # Real run (Day 5+, after data + tokenizer are ready)
    python scripts/train.py --condition TAAM --seed 42 \
        --tokenizer tokenizer/spm_32k_en_nl_zh.model \
        --token-data data/tokenized/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG = logging.getLogger("train")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from taam import LANGUAGES  # noqa: E402
from taam.data import (  # noqa: E402
    MultilingualMixer,
    PackedLanguageStream,
    make_synthetic_corpus,
)
from taam.exp3 import EXP3MultilingualMixer  # noqa: E402
from taam.reward import (  # noqa: E402
    CrossLingualDeficitReward,
    NormalizedExcessLossReward,
    make_reward,
)


# ──────────────────────────────────────────────────────────────────────────────
# CFP-compliant checkpoint schedule (improved_research_context_v2.md §1 / C3)
# ──────────────────────────────────────────────────────────────────────────────


def compute_cfp_checkpoint_schedule(
    tokens_per_step: int,
    total_steps: int,
    *,
    phase1_every: int = 1_000_000,
    phase2_every: int = 10_000_000,
    phase3_every: int = 100_000_000,
    phase1_max: int = 10_000_000,
    phase2_max: int = 100_000_000,
    phase3_max: int = 1_000_000_000,
) -> dict[int, int]:
    """Map {1-indexed step -> tokens_seen} per the BabyLM 2026 CFP.

    The CFP mandates intermediate checkpoints at:
        - every 1M tokens while tokens_seen <= 10M
        - every 10M tokens while 10M < tokens_seen <= 100M
        - every 100M tokens while 100M < tokens_seen <= 1B

    We translate "tokens" to "step" using `tokens_per_step = batch_size *
    block_size`. The final step is always included.

    Args:
        tokens_per_step: Number of tokens consumed by one optimizer step.
        total_steps: The training loop's total step count.

    Returns:
        Ordered dict {step: tokens_seen_at_end_of_step}. Both keys and
        values are 1-indexed (step 1 = "after first optimizer step").
    """
    schedule: dict[int, int] = {}

    def _add(threshold_tokens: int) -> None:
        step = (threshold_tokens + tokens_per_step - 1) // tokens_per_step
        if 0 < step <= total_steps:
            schedule[step] = step * tokens_per_step

    t = phase1_every
    while t <= phase1_max:
        _add(t)
        t += phase1_every
    t = phase1_max + phase2_every
    while t <= phase2_max:
        _add(t)
        t += phase2_every
    t = phase2_max + phase3_every
    while t <= phase3_max:
        _add(t)
        t += phase3_every

    schedule[total_steps] = total_steps * tokens_per_step

    return dict(sorted(schedule.items()))


# ──────────────────────────────────────────────────────────────────────────────
# Run metadata (git SHA + GPU + env timing)
# ──────────────────────────────────────────────────────────────────────────────


def collect_run_metadata() -> dict:
    """Capture host + git + torch metadata at run start.

    Best-effort: any individual probe may fail silently (e.g. on a system
    with no git or no CUDA). The returned dict always has every key.
    """
    meta: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "git": {},
    }

    try:
        import torch
        meta["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            meta["cuda_available"] = True
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                meta["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                )
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        pass

    def _git(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
            ).decode().strip()
            return out or None
        except Exception:  # noqa: BLE001
            return None

    sha = _git(["rev-parse", "HEAD"])
    if sha:
        meta["git"] = {
            "sha": sha,
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain"])),
        }
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────


def _deep_merge(a: dict, b: dict) -> dict:
    """Recursive dict merge (b overrides a). Returns a new dict."""
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_condition_config(condition_id: str) -> dict:
    """Load configs/{condition_id}.yaml merged with configs/base.yaml."""
    base = yaml.safe_load((REPO_ROOT / "configs/base.yaml").read_text())
    # Find the file by condition id, supporting either name match or id match.
    candidates = list((REPO_ROOT / "configs").glob("*.yaml"))
    cond_path = None
    for c in candidates:
        if c.name in {"base.yaml", "typological_prior.yaml"}:
            continue
        data = yaml.safe_load(c.read_text())
        cid = (data.get("condition") or {}).get("id")
        if cid == condition_id or c.stem == condition_id:
            cond_path = c
            break
    if cond_path is None:
        raise FileNotFoundError(f"No config with condition.id == {condition_id!r}")
    cond_data = yaml.safe_load(cond_path.read_text())
    cond_data.pop("include", None)
    merged = _deep_merge(base, cond_data)
    merged["_meta"] = {
        "condition_config_path": str(cond_path.relative_to(REPO_ROOT)),
        "merged_keys": sorted(set(base) | set(cond_data)),
    }
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Mixer construction
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MixerWrapper:
    """Unified interface over static or online (EXP3) mixers."""

    languages: tuple[str, ...]
    is_online: bool
    static_pi: dict[str, float] | None = None
    exp3: EXP3MultilingualMixer | None = None
    reward: NormalizedExcessLossReward | CrossLingualDeficitReward | None = None

    def get_pi(self) -> dict[str, float]:
        if self.is_online:
            return dict(zip(self.languages, self.exp3.pi.tolist()))
        return dict(self.static_pi)

    def update(self, val_losses: dict[str, float]) -> dict[str, float] | None:
        """If online, returns the per-language reward; else returns None."""
        if not self.is_online:
            return None
        r_dict = self.reward.update(val_losses)
        r_vec = np.array([r_dict[l] for l in self.languages])
        self.exp3.update(rewards=r_vec)
        return r_dict


def build_mixer(config: dict, seed: int) -> MixerWrapper:
    cond = config["condition"]
    method = cond["method"]
    languages = tuple(config["data"]["languages"])

    if method["type"] == "static":
        pi = method["pi"]
        # Filter to active languages (for monolingual conditions).
        pi_active = {l: pi[l] for l in languages}
        s = sum(pi_active.values())
        if abs(s - 1.0) > 1e-4:
            raise ValueError(
                f"Static pi for languages {languages} does not sum to 1: {pi_active}"
            )
        return MixerWrapper(languages=languages, is_online=False, static_pi=pi_active)

    if method["type"] == "online":
        pi_0 = method["pi_0"]
        on = method["online"]
        algo = on.get("algorithm", "exp3")
        if algo != "exp3":
            raise NotImplementedError(f"Online algorithm {algo} not implemented")
        pi_0_vec = np.array([pi_0[l] for l in languages], dtype=np.float64)
        if abs(pi_0_vec.sum() - 1.0) > 1e-4:
            raise ValueError(f"pi_0 must sum to 1 (got {pi_0_vec.sum():.6f}).")
        exp3 = EXP3MultilingualMixer(
            languages=languages,
            pi_0=pi_0_vec,
            eta=on.get("eta", 0.1),
            gamma=on.get("gamma", 0.1),
            min_pi=on.get("min_pi", 0.05),
            seed=seed,
        )
        reward_cfg = config["reward"]
        # The condition-local reward name (e.g. method.online.reward) wins over
        # the global default, so ablations can swap rewards without editing
        # base.yaml.
        reward_type = on.get("reward", reward_cfg.get("type", "normalized_excess_loss"))
        rw = make_reward(
            reward_type=reward_type,
            languages=languages,
            ref_window=reward_cfg.get("ref_window", 5),
            std_window=reward_cfg.get("std_window", 20),
            clip=reward_cfg.get("clip", 2.0),
            std_fallback=reward_cfg.get("std_fallback", 1.0),
        )
        return MixerWrapper(languages=languages, is_online=True, exp3=exp3, reward=rw)

    raise ValueError(f"Unknown method type: {method['type']}")


# ──────────────────────────────────────────────────────────────────────────────
# Data streams
# ──────────────────────────────────────────────────────────────────────────────


def build_streams(
    config: dict,
    token_data_dir: Path | None,
    use_synthetic: bool,
    seed: int,
) -> tuple[dict[str, PackedLanguageStream], dict[str, PackedLanguageStream]]:
    """Build per-language train + reward-dev streams.

    Real-data convention (produced by scripts/pretokenize.py):
        token_data_dir/
            {lang}/shard_0000.npy
            {lang}/shard_0001.npy
            ...
            manifest.json
    The LAST shard per language is reserved for dev/reward evaluation; the
    rest are used for training. We also honor the EOS token id stored in
    the pretokenization manifest (default 2) so the EOS separator that
    pretokenize wrote between docs is preserved at pack time.
    """
    languages = tuple(config["data"]["languages"])
    block_size = int(config["model"]["block_size"])

    if use_synthetic:
        synth_dir = Path("data/_synthetic")
        files_train = make_synthetic_corpus(
            output_dir=synth_dir / "train",
            languages=languages,
            n_tokens_per_lang=block_size * 200,
            seed=seed,
        )
        files_dev = make_synthetic_corpus(
            output_dir=synth_dir / "dev",
            languages=languages,
            n_tokens_per_lang=block_size * 40,
            seed=seed + 9999,
        )
        train_streams = {
            l: PackedLanguageStream(
                token_id_files=files_train[l],
                block_size=block_size,
                eos_token_id=0,
                seed=seed,
            )
            for l in languages
        }
        dev_streams = {
            l: PackedLanguageStream(
                token_id_files=files_dev[l],
                block_size=block_size,
                eos_token_id=0,
                seed=seed + 9999,
            )
            for l in languages
        }
        return train_streams, dev_streams

    if token_data_dir is None:
        raise FileNotFoundError("token_data_dir is required for non-smoke runs.")

    # Discover the EOS id from the pretok manifest (falls back to 2 = SP default).
    eos_id = 2
    manifest_path = token_data_dir / "manifest.json"
    if manifest_path.exists():
        try:
            mani = json.loads(manifest_path.read_text())
            eos_id = int(mani.get("tokenizer_eos_id", eos_id))
            LOG.info("Loaded EOS id %d from %s", eos_id, manifest_path.name)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not parse %s: %s; using EOS=%d",
                        manifest_path.name, exc, eos_id)

    train_streams = {}
    dev_streams = {}
    for l in languages:
        shards = sorted((token_data_dir / l).glob("shard_*.npy"))
        if len(shards) < 2:
            raise FileNotFoundError(
                f"Expected >=2 shards under {token_data_dir / l}, found "
                f"{len(shards)}. Did you run scripts/pretokenize.py?"
            )
        # Hold out the last shard for evaluation; train on the rest.
        train_shards, dev_shards = shards[:-1], shards[-1:]
        LOG.info(
            "[%s] %d train shards, %d dev shards (eos_id=%d)",
            l, len(train_shards), len(dev_shards), eos_id,
        )
        train_streams[l] = PackedLanguageStream(
            token_id_files=train_shards,
            block_size=block_size, eos_token_id=eos_id, seed=seed,
        )
        dev_streams[l] = PackedLanguageStream(
            token_id_files=dev_shards,
            block_size=block_size, eos_token_id=eos_id, seed=seed + 9999,
        )
    return train_streams, dev_streams


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────


def build_model(config: dict, vocab_size: int):
    """Build a HuggingFace GPT-2 model with config-derived hyperparams."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    m = config["model"]
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=m["block_size"],
        n_embd=m["n_embd"],
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        bos_token_id=1,
        eos_token_id=2,
    )
    model = GPT2LMHeadModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("GPT-2 model: %d params (~%.1fM)", n_params, n_params / 1e6)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (per-language val loss on a fixed dev slice)
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(model, dev_streams: dict[str, PackedLanguageStream],
             n_eval_batches: int, batch_size: int, device) -> dict[str, float]:
    """Compute mean cross-entropy loss per language on a fixed eval slice."""
    import torch
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for lang, stream in dev_streams.items():
            total_loss = 0.0
            for _ in range(n_eval_batches):
                ids = torch.as_tensor(stream.get_batch(batch_size), dtype=torch.long, device=device)
                out = model(input_ids=ids, labels=ids)
                total_loss += float(out.loss.detach().item())
            losses[lang] = total_loss / max(n_eval_batches, 1)
    model.train()
    return losses


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────


def train(
    config: dict,
    seed: int,
    token_data_dir: Path | None,
    output_dir: Path,
    use_synthetic: bool,
    total_steps_override: int | None = None,
    eval_every_override: int | None = None,
    n_eval_batches: int = 4,
    vocab_size_override: int | None = None,
) -> dict:
    """Run training end-to-end. Returns the final log summary dict."""
    import torch
    from torch.optim import AdamW

    cond_id = config["condition"]["id"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Snapshot the merged config
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    # Capture host/git/env metadata at run start (finalized again at run end).
    meta = collect_run_metadata()
    meta["condition_id"] = cond_id
    meta["seed"] = seed
    meta["output_dir"] = str(output_dir)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Mixer (static or online)
    mixer_state = build_mixer(config, seed=seed)
    train_streams, dev_streams = build_streams(
        config, token_data_dir=token_data_dir, use_synthetic=use_synthetic, seed=seed
    )
    data_mixer = MultilingualMixer(
        streams=train_streams,
        batch_size=int(config["training"]["batch_size"]),
        seed=seed,
    )

    # Determine vocab_size
    if use_synthetic:
        # Synthetic ids are in [10, 410); add headroom.
        vocab_size = vocab_size_override or 512
    else:
        vocab_size = vocab_size_override or int(config["tokenizer"]["vocab_size"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    LOG.info("Using device: %s", device)

    model = build_model(config, vocab_size=vocab_size).to(device)
    optim = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(float(config["training"]["beta1"]), float(config["training"]["beta2"])),
    )

    total_steps = total_steps_override or int(config["training"]["total_steps"])
    eval_every = eval_every_override or int(config["online"]["eval_every_steps"])
    log_every = int(config["online"]["log_every_steps"])
    grad_clip = float(config["training"]["gradient_clip"])

    # CFP §1 (C3): build the mandatory intermediate-checkpoint schedule from
    # tokens_per_step rather than a fixed step interval. The legacy
    # `checkpoint_every_steps` key is kept readable for debugging only.
    batch_size = int(config["training"]["batch_size"])
    block_size = int(config["model"]["block_size"])
    tokens_per_step = batch_size * block_size
    ckpt_schedule = compute_cfp_checkpoint_schedule(
        tokens_per_step=tokens_per_step, total_steps=total_steps,
    )
    LOG.info(
        "CFP checkpoint schedule: %d intermediate checkpoints across %d steps "
        "(%d tokens/step).",
        len(ckpt_schedule), total_steps, tokens_per_step,
    )
    (output_dir / "checkpoint_schedule.json").write_text(json.dumps(
        {"tokens_per_step": tokens_per_step,
         "total_steps": total_steps,
         "schedule": ckpt_schedule},
        indent=2,
    ))

    # Per-step train-loss CSV (consumed by the figure scripts).
    train_loss_csv = (output_dir / "train_loss.csv").open("w", newline="", encoding="utf-8")
    train_loss_writer = csv.writer(train_loss_csv)
    train_loss_writer.writerow(["step", "loss", "language"])

    log_file = (output_dir / "log.jsonl").open("w", encoding="utf-8")
    pi_history_rows = [["step"] + list(mixer_state.languages)]

    print(f"\n=== Training {cond_id} (seed={seed}) for {total_steps} steps on {device} ===")
    t0 = time.time()
    last_log_time = t0
    train_loss_running = 0.0
    train_loss_count = 0

    for step in range(total_steps):
        pi = mixer_state.get_pi()
        batch = data_mixer.next_batch(pi=pi, step=step)
        ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=device)

        model.train()
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optim.step()

        cur_loss = float(loss.detach().item())
        train_loss_running += cur_loss
        train_loss_count += 1
        train_loss_writer.writerow([step + 1, cur_loss, batch.language])

        if (step + 1) % log_every == 0 or step == 0:
            now = time.time()
            avg_loss = train_loss_running / max(train_loss_count, 1)
            elapsed = now - last_log_time
            last_log_time = now
            record = {
                "step": step + 1,
                "phase": "train",
                "avg_train_loss": avg_loss,
                "batch_language": batch.language,
                "elapsed_sec": elapsed,
                "pi": pi,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            train_loss_running = 0.0
            train_loss_count = 0
            print(
                f"  step {step+1:5d}/{total_steps}  "
                f"loss={avg_loss:.4f}  lang={batch.language}  "
                f"pi={[f'{v:.3f}' for v in pi.values()]}  ({elapsed:.2f}s)"
            )

        if (step + 1) % eval_every == 0:
            val_losses = evaluate(
                model, dev_streams, n_eval_batches=n_eval_batches,
                batch_size=int(config["training"]["batch_size"]), device=device
            )
            reward_dict = mixer_state.update(val_losses)
            pi_after = mixer_state.get_pi()
            record = {
                "step": step + 1,
                "phase": "eval",
                "val_loss": val_losses,
                "reward": reward_dict,
                "pi_after_update": pi_after,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            pi_history_rows.append([step + 1] + [pi_after[l] for l in mixer_state.languages])
            print(
                f"  eval@{step+1}  val_loss={ {l: round(v, 4) for l, v in val_losses.items()} }  "
                f"pi'={[f'{v:.3f}' for v in pi_after.values()]}"
            )

        if (step + 1) in ckpt_schedule:
            tokens_seen = ckpt_schedule[step + 1]
            tokens_m = max(int(round(tokens_seen / 1_000_000)), 1)
            ckpt_name = f"step_{step+1:05d}_tokens_{tokens_m:03d}M"
            ckpt_dir = checkpoints_dir / ckpt_name
            ckpt_dir.mkdir(exist_ok=True)
            model.save_pretrained(ckpt_dir)
            (ckpt_dir / "ckpt_meta.json").write_text(json.dumps({
                "step": step + 1,
                "tokens_seen": tokens_seen,
                "tokens_per_step": tokens_per_step,
                "pi_at_save": mixer_state.get_pi(),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }, indent=2))
            LOG.info("Saved checkpoint to %s (tokens_seen=%d)", ckpt_dir, tokens_seen)
            print(f"  ckpt @{step+1} (tokens={tokens_seen:,}) -> {ckpt_name}")

    log_file.close()
    train_loss_csv.close()

    # Write pi history CSV (consumed by Figure 1)
    pi_csv = output_dir / "pi_history.csv"
    with pi_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in pi_history_rows:
            w.writerow(row)

    elapsed_total = time.time() - t0
    out_str = str(
        output_dir.relative_to(REPO_ROOT) if output_dir.is_relative_to(REPO_ROOT) else output_dir
    )
    summary = {
        "condition_id": cond_id,
        "seed": seed,
        "total_steps": total_steps,
        "tokens_per_step": tokens_per_step,
        "tokens_seen_total": total_steps * tokens_per_step,
        "elapsed_sec": elapsed_total,
        "final_pi": mixer_state.get_pi(),
        "n_checkpoints": len(ckpt_schedule),
        "output_dir": out_str,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Finalize meta.json with end-of-run fields.
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["elapsed_sec"] = round(elapsed_total, 2)
    meta["status"] = "success"
    meta["total_steps"] = total_steps
    meta["tokens_seen_total"] = total_steps * tokens_per_step
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone in {elapsed_total:.1f}s.  Final pi: {mixer_state.get_pi()}")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, help="Condition id (e.g., TAAM, B0, P1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use synthetic data + tiny model; runs in ~30s on CPU.")
    parser.add_argument("--token-data", type=Path, default=None,
                        help="Path to tokenized data directory (required if not --smoke-test).")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Path to .model file (for vocab size); inferred if not given.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory; default = runs/{cond}_seed{S}/")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override total_steps in config (useful for smoke tests)")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = load_condition_config(args.condition)
    print(f"Loaded merged config for condition '{args.condition}' "
          f"(method.type={config['condition']['method']['type']}).")

    today = datetime.now().strftime("%Y-%m-%d")
    default_run_id = f"{today}_{args.condition}_seed{args.seed}"
    if args.smoke_test:
        output_dir = args.output_dir or (
            REPO_ROOT / "runs" / f"_smoke_{default_run_id}"
        )
        total_steps = args.total_steps or 50
        eval_every = args.eval_every or 10
        # Downsize model + batch for CPU smoke test
        config["model"]["n_layer"] = 2
        config["model"]["n_head"] = 4
        config["model"]["n_embd"] = 128
        config["model"]["block_size"] = 64
        config["training"]["batch_size"] = 4
        # Logging cadence
        config["online"]["log_every_steps"] = 10
    else:
        output_dir = args.output_dir or (REPO_ROOT / "runs" / default_run_id)
        total_steps = args.total_steps
        eval_every = args.eval_every

    summary = train(
        config=config,
        seed=args.seed,
        token_data_dir=args.token_data,
        output_dir=output_dir,
        use_synthetic=args.smoke_test,
        total_steps_override=total_steps,
        eval_every_override=eval_every,
        n_eval_batches=2 if args.smoke_test else 8,
    )

    print()
    print("=================================================================")
    print(f"  Run complete: {summary['output_dir']}")
    print(f"  Final pi:     {summary['final_pi']}")
    print(f"  Elapsed:      {summary['elapsed_sec']:.1f}s")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
