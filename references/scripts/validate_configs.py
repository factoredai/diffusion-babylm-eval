"""Validate every condition config in configs/.

Checks:
  - YAML parses without error.
  - ``include: "base.yaml"`` is present (sentinel for the merge convention).
  - ``condition.id`` is unique across all configs.
  - For static methods: pi sums to 1 (within 1e-6) and entries are in [0, 1].
  - For online methods: pi_0 sums to 1 (within 1e-6) and entries are positive.
  - Each language in pi/pi_0 is in {eng, nld, zho}.

This does NOT load the merged base + condition into the training stack; that
happens in scripts/train.py. We just want to catch dumb typos before any GPU
time is burned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

ALLOWED_LANGS = {"eng", "nld", "zho"}
TOLERANCE = 1e-6


def _check_pi(pi: dict, ctx: str, errors: list[str], allow_zero: bool = False) -> None:
    if not isinstance(pi, dict):
        errors.append(f"{ctx}: pi must be a dict, got {type(pi).__name__}")
        return
    bad_langs = set(pi.keys()) - ALLOWED_LANGS
    if bad_langs:
        errors.append(f"{ctx}: unknown languages in pi: {bad_langs}")
    s = sum(pi.values())
    if abs(s - 1.0) > TOLERANCE:
        errors.append(f"{ctx}: pi sums to {s:.6f}, expected 1.0")
    for l, p in pi.items():
        if not isinstance(p, (int, float)):
            errors.append(f"{ctx}: pi[{l}] = {p!r} is not numeric")
            continue
        if p < 0 or p > 1:
            errors.append(f"{ctx}: pi[{l}] = {p} out of [0, 1]")
        if not allow_zero and p == 0:
            errors.append(
                f"{ctx}: pi[{l}] = 0 not allowed for this condition type "
                f"(only monolingual conditions can have zeros)"
            )


def main() -> int:
    files = sorted(p for p in CONFIGS_DIR.glob("*.yaml") if p.name not in {"base.yaml", "typological_prior.yaml"})
    if not files:
        print("No condition configs found in configs/.", file=sys.stderr)
        return 1

    errors: list[str] = []
    ids_seen: dict[str, str] = {}

    print(f"Validating {len(files)} condition config(s):")
    for f in files:
        ctx = f.relative_to(REPO_ROOT).as_posix()
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{ctx}: YAML parse error: {e}")
            continue

        if data.get("include") != "base.yaml":
            errors.append(f"{ctx}: missing ``include: \"base.yaml\"``")

        cond = data.get("condition") or {}
        cid = cond.get("id")
        if not cid:
            errors.append(f"{ctx}: missing condition.id")
        else:
            if cid in ids_seen:
                errors.append(f"{ctx}: duplicate condition.id '{cid}' (also in {ids_seen[cid]})")
            ids_seen[cid] = ctx

        method = cond.get("method") or {}
        mtype = method.get("type")
        allow_zero = cid in {"M_EN", "M_NL", "M_ZH"} if cid else False

        if mtype == "static":
            if "pi" not in method:
                errors.append(f"{ctx}: method.type=static requires method.pi")
            else:
                _check_pi(method["pi"], f"{ctx}.method.pi", errors, allow_zero=allow_zero)
            if method.get("online") not in (None, "null"):
                errors.append(f"{ctx}: static condition should have online: null, got {method.get('online')!r}")
        elif mtype == "online":
            if "pi_0" not in method:
                errors.append(f"{ctx}: method.type=online requires method.pi_0")
            else:
                _check_pi(method["pi_0"], f"{ctx}.method.pi_0", errors, allow_zero=False)
            if not method.get("online"):
                errors.append(f"{ctx}: method.type=online requires method.online block")
        else:
            errors.append(f"{ctx}: unknown method.type={mtype!r}")

        status = "✓" if not any(e.startswith(ctx) for e in errors) else "✗"
        print(f"  {status} {ctx}  (id={cid})")

    print()
    if errors:
        print(f"FAIL: {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    print(f"PASS: {len(files)} condition configs valid. {len(ids_seen)} unique condition IDs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
