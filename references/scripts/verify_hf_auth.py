#!/usr/bin/env python
"""Verify that the HuggingFace token is valid and that the BabyBabelLM gated
datasets (EN/NL/ZH) are accessible.

Usage:
    python scripts/verify_hf_auth.py

What this script checks, in order:
    1.  Loads HF_TOKEN from .env (preferred) or from the environment.
    2.  Calls /api/whoami-v2 to confirm the token is valid and reports the
        username. Never prints the token itself.
    3.  For each of the three target datasets (babylm-eng/nld/zho), queries
        /api/datasets/<repo> with the token and inspects the response:
          - 200 + has tags => accessible (you accepted the agreement).
          - 401 / 403      => not accessible yet (need to click "Agree").
          - 404            => repo not found (typo in URL or moved).
    4.  Prints a final summary table and exits non-zero if anything failed,
        so this is safe to call from `make` / CI.

Why this matters:
    The training pipeline depends on these three datasets. A token that works
    for `whoami` but has not been used to accept the per-dataset agreement
    will silently fail at `load_dataset()` with a confusing 401. Catching
    that here saves hours of debugging later.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The three Tier-1 BabyBabelLM datasets we need for the EN+NL+ZH track.
# Source: https://huggingface.co/collections/BabyLM-community/babybabellm
TARGET_DATASETS: list[tuple[str, str]] = [
    ("eng", "BabyLM-community/babylm-eng"),
    ("nld", "BabyLM-community/babylm-nld"),
    ("zho", "BabyLM-community/babylm-zho"),
]


@dataclass
class DatasetStatus:
    lang: str
    repo: str
    ok: bool
    message: str
    fix_url: str | None = None


def _load_env_token() -> str | None:
    """Load HF_TOKEN from .env (if present) or from the environment.

    Order of precedence: process env > .env file. This matches what most
    shells already do via direnv/python-dotenv and avoids surprising the user
    when they have HF_TOKEN exported in their shell.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    # Also accept HUGGING_FACE_HUB_TOKEN, which is the canonical name used by
    # the huggingface_hub library itself.
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return os.environ["HUGGING_FACE_HUB_TOKEN"]

    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    try:
        from dotenv import dotenv_values  # type: ignore
    except ImportError:
        # Manual fallback: parse KEY=VALUE lines, skip comments and blanks.
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                return value.strip().strip('"').strip("'")
        return None
    parsed = dotenv_values(env_path)
    return parsed.get("HF_TOKEN") or parsed.get("HUGGING_FACE_HUB_TOKEN")


def _check_whoami(token: str) -> tuple[bool, str]:
    """Return (ok, message_or_username)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi(token=token)
    try:
        info = api.whoami()
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", "?")
        return False, f"whoami failed (HTTP {status}). Token may be invalid or revoked."
    except Exception as exc:  # noqa: BLE001
        return False, f"whoami failed: {exc!s}"
    username = info.get("name") or info.get("fullname") or "<unknown>"
    return True, str(username)


def _check_dataset(token: str, lang: str, repo: str, cache_dir: Path) -> DatasetStatus:
    """Two-step check:
    1. dataset_info  -> repo exists and is visible.
    2. hf_hub_download of README.md (small) -> agreement actually accepted,
       not just visible metadata. Gated repos let you read metadata but
       refuse file downloads until you click "Agree".
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    api = HfApi(token=token)
    url = f"https://huggingface.co/datasets/{repo}"
    try:
        api.dataset_info(repo, token=token)
    except GatedRepoError:
        return DatasetStatus(
            lang=lang,
            repo=repo,
            ok=False,
            message="GATED: you need to click 'Agree and access repository' on the dataset page",
            fix_url=url,
        )
    except RepositoryNotFoundError:
        return DatasetStatus(
            lang=lang,
            repo=repo,
            ok=False,
            message="repo not found (was it renamed/moved?)",
            fix_url=url,
        )
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", "?")
        if status in (401, 403):
            return DatasetStatus(
                lang=lang,
                repo=repo,
                ok=False,
                message=f"HTTP {status}: agreement not accepted or token lacks gated-repo access",
                fix_url=url,
            )
        return DatasetStatus(
            lang=lang,
            repo=repo,
            ok=False,
            message=f"HTTP {status}: {exc!s}",
            fix_url=url,
        )
    except Exception as exc:  # noqa: BLE001
        return DatasetStatus(
            lang=lang,
            repo=repo,
            ok=False,
            message=f"unexpected error: {exc!s}",
            fix_url=url,
        )

    # Step 2: actually try to download a small file (README.md) to confirm
    # that the agreement is accepted, not just that the repo is visible.
    try:
        hf_hub_download(
            repo_id=repo,
            filename="README.md",
            repo_type="dataset",
            token=token,
            cache_dir=str(cache_dir),
        )
        return DatasetStatus(lang=lang, repo=repo, ok=True, message="accessible (download verified)")
    except GatedRepoError:
        return DatasetStatus(
            lang=lang,
            repo=repo,
            ok=False,
            message="GATED at download: visible but you must click 'Agree' first",
            fix_url=url,
        )
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", "?")
        # 404 on README is harmless: some repos use README.MD or no README.
        if status == 404:
            return DatasetStatus(
                lang=lang,
                repo=repo,
                ok=True,
                message="accessible (metadata ok; README.md absent, this is fine)",
            )
        return DatasetStatus(
            lang=lang,
            repo=repo,
            ok=False,
            message=f"download blocked (HTTP {status})",
            fix_url=url,
        )


def main() -> int:
    print("=" * 72)
    print("HuggingFace authentication & gated-access verification")
    print("=" * 72)

    token = _load_env_token()
    if not token:
        print("\n[FAIL] HF_TOKEN not found.")
        print("  Add it to .env (HF_TOKEN=hf_...) or export it in your shell.")
        print("  Create a token at: https://huggingface.co/settings/tokens")
        return 2

    if not token.startswith(("hf_", "api_")):
        print("\n[WARN] HF_TOKEN does not start with 'hf_' or 'api_'. ")
        print("       Make sure you copied the full token (no extra quotes/spaces).")

    print(f"\nStep 1/2: validating token ({len(token)} chars, hidden) ...")
    ok, user_or_msg = _check_whoami(token)
    if not ok:
        print(f"  [FAIL] {user_or_msg}")
        print("  Fix: regenerate a token at https://huggingface.co/settings/tokens")
        print("       with permission: 'Read access to contents of all public gated")
        print("       repos you can access'.")
        return 3
    print(f"  [OK]  authenticated as: {user_or_msg}")

    # Use a workspace-local cache so we don't pollute ~/.cache and so the
    # check works inside restricted environments (CI, sandboxes).
    cache_dir = REPO_ROOT / "data" / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))

    print(f"\nStep 2/2: checking access to BabyBabelLM Tier-1 datasets ...")
    print(f"  (cache dir: {cache_dir.relative_to(REPO_ROOT)})")
    statuses: list[DatasetStatus] = []
    for lang, repo in TARGET_DATASETS:
        st = _check_dataset(token, lang, repo, cache_dir)
        statuses.append(st)
        mark = "OK  " if st.ok else "FAIL"
        print(f"  [{mark}] {lang.upper():3s}  {repo:32s}  {st.message}")

    print("\n" + "=" * 72)
    failed = [s for s in statuses if not s.ok]
    if not failed:
        print("All checks passed. You can now run:  make data  (or scripts/download_data.py)")
        return 0

    print("Some checks failed. To fix:")
    for s in failed:
        if s.fix_url:
            print(f"  - {s.lang.upper()}: open {s.fix_url} and click 'Agree and access repository'")
        else:
            print(f"  - {s.lang.upper()}: {s.message}")
    print("\nThen re-run:  python scripts/verify_hf_auth.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
