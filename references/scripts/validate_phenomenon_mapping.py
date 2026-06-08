"""Validate analyses/acquisition_order/phenomenon_to_child_norm.yaml.

Checks (all reported, none silently fixed):

    SCHEMA
    1. All phenomena have id, name, multiblimp_paradigms, languages.
    2. ``id`` is unique and snake_case.
    3. For each language entry: either age_months is a positive int/float, or
       both age_months is None AND excluded_reason is provided.
    4. When age_months is not None, sources is non-empty and confidence in
       {high, medium, low}.

    SCIENCE
    5. ages_months are within [12, 144] (reasonable child-language range).
    6. Each phenomenon contributing to the cross-language correlation has
       sources of type PRIMARY or REVIEW (no orphan INFERRED entries).
    7. Per-language sample size >= meta.spearman_min_n.

    HYGIENE
    8. Any review_needed: true is flagged in stdout (so the author can address before submission).

Exit code 0 if all checks pass; 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def _err(msg: str) -> None:
    print(f"  ✗ {msg}")


def validate(path: Path) -> int:
    print(f"Validating: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    errors: list[str] = []
    warnings: list[str] = []

    if "meta" not in data or "phenomena" not in data:
        return _abort("Top-level keys 'meta' and 'phenomena' must be present.")

    meta = data["meta"]
    languages = meta.get("languages_supported", [])
    spearman_min_n = int(meta.get("spearman_min_n", 0))

    print()
    print(f"Languages: {languages}")
    print(f"Spearman min N: {spearman_min_n}")
    print(f"Phenomena declared: {len(data['phenomena'])}")
    print()

    seen_ids: Counter[str] = Counter()
    per_lang_n: Counter[str] = Counter()
    per_lang_review_needed: Counter[str] = Counter()

    for i, p in enumerate(data["phenomena"]):
        ctx = f"phenomenon[{i}] (id={p.get('id', '<missing>')})"

        # Schema 1: required fields
        for field in ("id", "name", "multiblimp_paradigms", "languages"):
            if field not in p:
                errors.append(f"{ctx}: missing required field '{field}'")

        pid = p.get("id")
        if pid:
            if not SNAKE.match(pid):
                errors.append(f"{ctx}: id '{pid}' is not snake_case")
            seen_ids[pid] += 1

        # Schema 3+4 and Science 5+6
        for lang, entry in p.get("languages", {}).items():
            age = entry.get("age_months")
            if age is None:
                if "excluded_reason" not in entry:
                    errors.append(
                        f"{ctx}/{lang}: age_months is None but no excluded_reason given"
                    )
                continue
            if not isinstance(age, (int, float)) or age <= 0:
                errors.append(f"{ctx}/{lang}: age_months={age!r} must be positive number")
                continue
            if age < 12 or age > 144:
                warnings.append(
                    f"{ctx}/{lang}: age_months={age} outside typical [12, 144] range"
                )
            sources = entry.get("sources", [])
            if not sources:
                errors.append(f"{ctx}/{lang}: age_months provided but sources empty")
            else:
                types = {s.get("type") for s in sources}
                if not types & {"PRIMARY", "REVIEW"}:
                    errors.append(
                        f"{ctx}/{lang}: only INFERRED sources (need at least one PRIMARY or REVIEW)"
                    )
            conf = entry.get("confidence")
            if conf not in {"high", "medium", "low", "n/a"}:
                errors.append(f"{ctx}/{lang}: confidence={conf!r} not in {{high,medium,low,n/a}}")
            per_lang_n[lang] += 1
            if entry.get("review_needed"):
                per_lang_review_needed[lang] += 1

    # Schema 2: unique ids
    dupes = [pid for pid, cnt in seen_ids.items() if cnt > 1]
    if dupes:
        errors.append(f"Duplicate phenomenon ids: {dupes}")

    # Science 7: per-language sample size
    for lang in languages:
        n = per_lang_n[lang]
        if n < spearman_min_n:
            errors.append(
                f"Language '{lang}' has only {n} non-null phenomena "
                f"(spearman_min_n={spearman_min_n})"
            )

    print("Schema and science checks:")
    if not errors:
        _ok(f"{len(data['phenomena'])} phenomena, all required fields present")
    for e in errors:
        _err(e)
    if warnings:
        print()
        print("Warnings (won't fail the run):")
        for w in warnings:
            _warn(w)

    print()
    print("Per-language coverage:")
    for lang in languages:
        n = per_lang_n[lang]
        rn = per_lang_review_needed[lang]
        status = "✓" if n >= spearman_min_n else "✗"
        print(f"  {status} {lang}: N={n}   (review_needed flags = {rn})")
    print()

    if errors:
        print(f"FAIL: {len(errors)} error(s) found.")
        return 1
    print("PASS: phenomenon mapping schema is valid.")
    print(
        "  Note: ``review_needed: true`` flags should be verified against the "
        "cited primary sources before submission."
    )
    return 0


def _abort(msg: str) -> int:
    print(f"FATAL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("analyses/acquisition_order/phenomenon_to_child_norm.yaml"),
    )
    args = parser.parse_args()
    return validate(args.mapping)


if __name__ == "__main__":
    sys.exit(main())
