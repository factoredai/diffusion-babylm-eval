"""Data sources: thin wrapper around the BabyBabelLM HuggingFace datasets.

Provides `iter_documents(lang)` and `get_dataset(lang)` so the rest of the
codebase (tokenizer training, pretokenization, statistics) never has to talk
to `datasets.load_dataset` directly. This module handles:

* Loading HF_TOKEN from .env (and exporting it for huggingface_hub).
* Forcing HF cache to a workspace-local directory (data/hf_cache) so we do
  not pollute the user's home dir and we can ship the cache across machines.
* Mapping our internal language codes (eng/nld/zho) to dataset repo names.

Document schema (from BabyLM-community/babylm-*):
    text:         str   — document text
    doc-id:       str   — internal id
    category:     str   — high-level source category (e.g., 'CDS', 'books')
    data-source:  str   — specific corpus / project
    script:       str   — Latin, Hans, etc.
    age-estimate: str   — coarse age band the doc targets (when known)
    license:     str
    misc:        dict
    num-tokens:  int   — pre-computed by the BabyBabelLM team
    language:    str   — ISO-3 lang code (matches our internal code)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "hf_cache"

# Internal lang code -> HuggingFace repo id.
DATASET_REPOS: dict[str, str] = {
    "eng": "BabyLM-community/babylm-eng",
    "nld": "BabyLM-community/babylm-nld",
    "zho": "BabyLM-community/babylm-zho",
}

# Documents shorter than this many characters are likely empty/garbage rows.
MIN_DOC_CHARS = 1


def ensure_hf_env(cache_dir: Path | None = None, dotenv_path: Path | None = None) -> Path:
    """Configure huggingface_hub environment variables for this process.

    Order of precedence for HF_TOKEN:
        1. existing env var (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)
        2. .env file at the repo root (loaded via python-dotenv if available,
           otherwise a manual KEY=VALUE parse).

    The cache directory is always pinned to `cache_dir` (or DEFAULT_CACHE_DIR)
    so different scripts share the same downloaded blobs.

    Returns the resolved cache directory.
    """
    cache_dir = (cache_dir or DEFAULT_CACHE_DIR).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token = _read_token_from_dotenv(dotenv_path or REPO_ROOT / ".env")
        if token:
            os.environ["HF_TOKEN"] = token
            # huggingface_hub respects HUGGING_FACE_HUB_TOKEN too; setting both
            # avoids surprises when libraries check different names.
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    return cache_dir


def _read_token_from_dotenv(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        from dotenv import dotenv_values  # type: ignore

        values = dotenv_values(path)
        return values.get("HF_TOKEN") or values.get("HUGGING_FACE_HUB_TOKEN")
    except ImportError:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                return value.strip().strip('"').strip("'")
    return None


def repo_for(lang: str) -> str:
    """Map internal lang code to HF dataset repo id."""
    if lang not in DATASET_REPOS:
        raise KeyError(
            f"unknown language code: {lang!r}; expected one of {list(DATASET_REPOS)}"
        )
    return DATASET_REPOS[lang]


def get_dataset(lang: str, split: str = "train", streaming: bool = False):
    """Load (and cache) a BabyBabelLM dataset.

    Returns a `datasets.Dataset` (or `IterableDataset` if streaming=True).
    Importing `datasets` lazily keeps `taam` cheap to import in CPU-only
    contexts that don't need data access.
    """
    ensure_hf_env()
    from datasets import load_dataset  # noqa: WPS433

    return load_dataset(repo_for(lang), split=split, streaming=streaming)


def iter_documents(
    lang: str,
    split: str = "train",
    *,
    fields: Iterable[str] | None = None,
    skip_empty: bool = True,
) -> Iterator[dict]:
    """Yield documents one at a time from the named language's corpus.

    Parameters
    ----------
    lang:
        One of "eng", "nld", "zho".
    split:
        HF split. BabyBabelLM only ships "train" today.
    fields:
        If given, only return these keys per document (e.g. ("text",)
        for tokenizer training, which keeps memory low). If None, returns
        the full record.
    skip_empty:
        Drop records whose text is empty or shorter than MIN_DOC_CHARS.
    """
    ds = get_dataset(lang, split=split, streaming=False)
    field_set = set(fields) if fields else None
    for record in ds:
        text = record.get("text", "")
        if skip_empty and (not text or len(text) < MIN_DOC_CHARS):
            continue
        if field_set is None:
            yield record
        else:
            yield {k: record.get(k) for k in field_set}


def num_documents(lang: str, split: str = "train") -> int:
    """Return the number of (non-streaming) documents in a corpus."""
    ds = get_dataset(lang, split=split, streaming=False)
    return len(ds)
