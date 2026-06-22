"""Diffusion-native zero-shot evaluation backend (minimal-pair tasks).

The official BabyLM pipeline supports causal / mlm / mntp / enc_dec backends. A
masked-diffusion denoiser is naturally scored by *masked* pseudo-log-likelihood,
which is what the official `mlm` backend already does — so for a standard
submission you can simply run the official pipeline with `--backend mlm` (see
docs/EVALUATION.md).

This script is the diffusion-*native* alternative. Use it when you want:
    * the ELBO scorer instead of single-token PLL, or
    * the inference-time layer-duplication ("reasoning depth") variant, which
      the official pipeline cannot express.

It reads the official task data directories (jsonl files) and writes
``predictions.json`` in the exact directory layout the official
``collate_preds.py`` expects, so the two paths are interchangeable:

    {output_dir}/{model_stem}/{revision}/zero_shot/{backend}/{task}/{dataset}/predictions.json

Supported tasks (English, minimal-pair): blimp, ewok, comps, entity_tracking.

Usage:
    python scripts/diffusion_eval_backend.py \
        --model_path_or_name <repo_or_local> --revision_name chck_10M \
        --task blimp --data_path ../strict/evaluation_data/full_eval/blimp_filtered \
        --scoring pll --backend mlm --save_predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ── Task decoding (compact mirror of the official read_files.decode_*) ───────


def decode_line(task: str, raw: dict, file_stem: str) -> list[dict]:
    """Return a list of {sentences, prefixes, completions, label, UID} pairs."""
    if task == "blimp":
        uid = raw.get("UID", file_stem)
        return [{
            "sentences": [raw["sentence_good"], raw["sentence_bad"]],
            "prefixes": [None, None],
            "completions": [raw["sentence_good"], raw["sentence_bad"]],
            "label": 0, "UID": uid,
        }]
    if task == "comps":
        prefixes = [raw["prefix_acceptable"], raw["prefix_unacceptable"]]
        sents = [" ".join([raw["prefix_acceptable"], raw["property_phrase"]]),
                 " ".join([raw["prefix_unacceptable"], raw["property_phrase"]])]
        subset = {"comps_base": "base", "comps_wugs": "wugs",
                  "comps_wugs_dist-before": "wugs_dist_before"}.get(file_stem, "wugs_dist_in_between")
        return [{"sentences": sents, "prefixes": prefixes,
                 "completions": [raw["property_phrase"], raw["property_phrase"]],
                 "label": 0, "UID": subset}]
    if task == "entity_tracking":
        subset = f'{file_stem}_{raw["numops"]}_ops'
        return [{"sentences": [raw["input_prefix"] + o for o in raw["options"]],
                 "prefixes": [raw["input_prefix"] for _ in raw["options"]],
                 "completions": list(raw["options"]), "label": 0, "UID": subset}]
    if task == "ewok":
        out = []
        for tgt, c_good, c_bad in [
            (raw["Target1"], raw["Context1"], raw["Context2"]),
            (raw["Target2"], raw["Context2"], raw["Context1"]),
        ]:
            out.append({
                "sentences": [" ".join([c_good, tgt]), " ".join([c_bad, tgt])],
                "prefixes": [c_good, c_bad],
                "completions": [" " + tgt, " " + tgt],
                "label": 0, "UID": raw["Domain"],
            })
        return out
    raise NotImplementedError(f"task {task!r} not supported by this backend.")


def read_task(task: str, data_path: Path) -> list[dict]:
    items: list[dict] = []
    for f in sorted(data_path.iterdir()):
        if f.suffix != ".jsonl":
            continue
        for line in f.open():
            line = line.strip()
            if line:
                items.extend(decode_line(task, json.loads(line), f.stem))
    return items


# ── Scoring ──────────────────────────────────────────────────────────────────


def completion_positions(tokenizer, sentence: str, prefix: str | None) -> tuple[list[int], "object"]:
    """Tokenize ``sentence`` and return (positions_to_score, input_ids)."""
    import torch

    enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                    max_length=tokenizer.model_max_length)
    ids = enc["input_ids"][0]
    if prefix is None:
        return list(range(ids.shape[0])), ids
    plen = tokenizer(prefix, return_tensors="pt")["input_ids"].shape[1]
    positions = list(range(min(plen, ids.shape[0]), ids.shape[0]))
    return positions or list(range(ids.shape[0])), ids


def score_candidate(model, tokenizer, sentence, prefix, args) -> float:
    from mdlm.scoring import score_elbo, score_pll

    positions, ids = completion_positions(tokenizer, sentence, prefix)
    if args.scoring == "elbo":
        return score_elbo(model, ids, n_samples=args.elbo_n_samples)
    return score_pll(
        model, ids, positions=positions, max_positions=args.pll_max_positions,
        layer_duplication_factor=args.layer_duplication_factor,
    )


# ── Main ───────────────────────────────────────────────────────────────────


def load_model(name: str, revision: str | None):
    import torch
    from mdlm.model import MaskedDiffusionLM

    try:
        model = MaskedDiffusionLM.from_pretrained(name, revision=revision)
    except Exception:  # noqa: BLE001 — fall back to the Hub auto-class (custom code)
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(name, revision=revision, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path_or_name", required=True)
    p.add_argument("--task", required=True, choices=["blimp", "ewok", "comps", "entity_tracking"])
    p.add_argument("--data_path", required=True, type=Path)
    p.add_argument("--output_dir", default=Path("results"), type=Path)
    p.add_argument("--revision_name", default=None)
    p.add_argument("--backend", default="mlm", help="Results-folder name (keep 'mlm' for collate compatibility).")
    p.add_argument("--scoring", default="pll", choices=["pll", "elbo"])
    p.add_argument("--pll_max_positions", default=256, type=int)
    p.add_argument("--elbo_n_samples", default=16, type=int)
    p.add_argument("--layer_duplication_factor", default=1, type=int)
    p.add_argument("--save_predictions", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer

    revision = args.revision_name or "main"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path_or_name, revision=args.revision_name, trust_remote_code=True
    )
    model = load_model(args.model_path_or_name, args.revision_name)

    items = read_task(args.task, args.data_path)
    print(f"[{args.task}] {len(items)} minimal-pair items from {args.data_path}")

    predictions: dict[str, list] = defaultdict(list)
    n_correct = 0
    for item in items:
        scores = [score_candidate(model, tokenizer, s, pre, args)
                  for s, pre in zip(item["sentences"], item["prefixes"])]
        chosen = int(max(range(len(scores)), key=lambda i: scores[i]))
        n_correct += int(chosen == item["label"])
        uid = item["UID"]
        idx = len(predictions[uid])
        # comps reports the full sentence; everything else reports the completion.
        pred = item["sentences"][chosen] if args.task == "comps" else item["completions"][chosen]
        predictions[uid].append({"id": f"{uid}_{idx}", "pred": pred})

    print(f"[{args.task}] accuracy = {100.0 * n_correct / max(len(items), 1):.2f}%")

    if args.save_predictions:
        model_stem = Path(args.model_path_or_name).stem
        dataset = args.data_path.stem
        out_dir = args.output_dir / model_stem / revision / "zero_shot" / args.backend / args.task / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {uid: {"predictions": preds} for uid, preds in predictions.items()}
        (out_dir / "predictions.json").write_text(json.dumps(payload))
        print(f"Wrote {out_dir / 'predictions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
