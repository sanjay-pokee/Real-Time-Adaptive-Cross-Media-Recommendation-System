"""Generate and incrementally update Sentence-BERT embeddings for the content catalog.

Behavior
--------
1. Load ``data/processed/content_catalog.csv``.
2. Load ``embeddings/content_embedding_index.csv`` (if it exists).
3. For each catalog row, check whether an index entry already exists with the
   same ``global_id``, ``embedding_model``, ``embedding_version``, and
   ``text_hash``.
4. Reuse the existing embedding vector if a match is found; generate a new one
   otherwise.
5. Write the complete embedding matrix to ``embeddings/content_embeddings.npy``
   and the updated index to ``embeddings/content_embedding_index.csv``.

Usage
-----
    python -m embeddings.build_embeddings
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
NPY_PATH = EMBEDDINGS_DIR / "content_embeddings.npy"
INDEX_PATH = EMBEDDINGS_DIR / "content_embedding_index.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_VERSION = "1"

INDEX_COLUMNS = [
    "global_id",
    "content_type",
    "source",
    "source_id",
    "embedding_model",
    "embedding_version",
    "text_hash",
    "embedding_row",
    "created_at",
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_embeddings(
    catalog_path: Path = CATALOG_PATH,
    npy_path: Path = NPY_PATH,
    index_path: Path = INDEX_PATH,
    embedding_model: str = EMBEDDING_MODEL,
    embedding_version: str = EMBEDDING_VERSION,
    batch_size: int = 256,
) -> None:
    """Build or incrementally update the embedding store."""
    # --- Load catalog ---
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Content catalog not found: {catalog_path}\n"
            "Run: python -m preprocessing.build_content_catalog"
        )
    catalog = pd.read_csv(catalog_path)
    _assert_catalog_columns(catalog)

    print(f"Catalog loaded: {len(catalog)} rows")

    # --- Load existing index ---
    existing_index = _load_existing_index(index_path)
    existing_vectors = _load_existing_vectors(npy_path, existing_index)

    # Build a lookup keyed by (global_id, model, version, text_hash) → old row idx
    reuse_map: dict[tuple[str, str, str, str], int] = {}
    if existing_index is not None:
        for _, row in existing_index.iterrows():
            key = (
                row["global_id"],
                row["embedding_model"],
                row["embedding_version"],
                row["text_hash"],
            )
            reuse_map[key] = int(row["embedding_row"])

    # --- Decide which rows need new embeddings ---
    needs_embedding_mask = []
    for _, row in catalog.iterrows():
        key = (row["global_id"], embedding_model, embedding_version, row["text_hash"])
        needs_embedding_mask.append(key not in reuse_map)

    needs_new = sum(needs_embedding_mask)
    reused = len(catalog) - needs_new
    print(f"  Reusing : {reused} embeddings")
    print(f"  Generating: {needs_new} new embeddings")

    # --- Generate new embeddings ---
    new_texts: list[str] = []
    new_indices: list[int] = []
    for i, (needs, (_, row)) in enumerate(
        zip(needs_embedding_mask, catalog.iterrows())
    ):
        if needs:
            new_texts.append(str(row["embedding_text"]))
            new_indices.append(i)

    new_vectors: np.ndarray | None = None
    if new_texts:
        new_vectors = _encode_texts(new_texts, embedding_model, batch_size)

    # --- Assemble full embedding matrix (one row per catalog row, in order) ---
    embedding_dim = _get_embedding_dim(existing_vectors, new_vectors, embedding_model)
    full_matrix = np.zeros((len(catalog), embedding_dim), dtype=np.float32)

    # Fill reused rows
    new_vec_cursor = 0
    for i, (needs, (_, row)) in enumerate(
        zip(needs_embedding_mask, catalog.iterrows())
    ):
        key = (row["global_id"], embedding_model, embedding_version, row["text_hash"])
        if not needs and existing_vectors is not None:
            old_row = reuse_map[key]
            full_matrix[i] = existing_vectors[old_row]
        else:
            assert new_vectors is not None
            full_matrix[i] = new_vectors[new_vec_cursor]
            new_vec_cursor += 1

    # --- Build new index ---
    now = datetime.now(timezone.utc).isoformat()
    index_records = []
    for i, (_, row) in enumerate(catalog.iterrows()):
        index_records.append({
            "global_id": row["global_id"],
            "content_type": row["content_type"],
            "source": row["source"],
            "source_id": row["source_id"],
            "embedding_model": embedding_model,
            "embedding_version": embedding_version,
            "text_hash": row["text_hash"],
            "embedding_row": i,
            "created_at": now,
        })
    new_index = pd.DataFrame(index_records, columns=INDEX_COLUMNS)

    # --- Validate before saving ---
    _validate_embedding_output(full_matrix, new_index)

    # --- Save ---
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, full_matrix)
    new_index.to_csv(index_path, index=False)

    print(f"\nEmbeddings saved:")
    print(f"  Matrix : {npy_path}  shape={full_matrix.shape}")
    print(f"  Index  : {index_path}  rows={len(new_index)}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_embedding_output(matrix: np.ndarray, index: pd.DataFrame) -> None:
    """Raise ValueError if the output does not satisfy integrity constraints."""
    errors: list[str] = []

    # 1. Index has all required columns — bail early so later checks don't crash
    #    on missing columns (e.g. duplicated(subset=[...'text_hash'...]) KeyError).
    missing = [c for c in INDEX_COLUMNS if c not in index.columns]
    if missing:
        errors.append(f"Index missing columns: {missing}")
        _fail_embedding(errors)

    # 2. Row counts match
    if len(index) != matrix.shape[0]:
        errors.append(
            f"Row count mismatch: index has {len(index)} rows, "
            f"matrix has {matrix.shape[0]} rows."
        )

    # 3. embedding_row is a contiguous 0-based range
    if not errors and list(index["embedding_row"]) != list(range(len(index))):
        errors.append("embedding_row is not a contiguous 0-based sequence.")

    # 4. No duplicate (global_id, model, version, hash)
    dup_cols = ["global_id", "embedding_model", "embedding_version", "text_hash"]
    dup_count = index.duplicated(subset=dup_cols).sum()
    if dup_count > 0:
        errors.append(
            f"Duplicate (global_id, model, version, hash) in index: {dup_count} row(s)."
        )

    if errors:
        _fail_embedding(errors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail_embedding(errors: list[str]) -> None:
    message = "\n".join(f"  - {e}" for e in errors)
    raise ValueError(f"Embedding output validation failed:\n{message}")


def _assert_catalog_columns(catalog: pd.DataFrame) -> None:
    required = ["global_id", "content_type", "source", "source_id", "embedding_text", "text_hash"]
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise ValueError(
            f"Content catalog is missing required columns: {missing}\n"
            "Run: python -m preprocessing.build_content_catalog"
        )


def _load_existing_index(index_path: Path) -> pd.DataFrame | None:
    if not index_path.exists():
        return None
    idx = pd.read_csv(index_path)
    print(f"Existing index loaded: {len(idx)} rows")
    return idx


def _load_existing_vectors(
    npy_path: Path, existing_index: pd.DataFrame | None
) -> np.ndarray | None:
    if existing_index is None or not npy_path.exists():
        return None
    vectors = np.load(npy_path)
    print(f"Existing embeddings loaded: shape={vectors.shape}")
    return vectors


def _encode_texts(
    texts: list[str], model_name: str, batch_size: int
) -> np.ndarray:
    """Encode a list of texts using Sentence-BERT and return a float32 matrix."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "Install it with: pip install sentence-transformers"
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("Using CPU (CUDA is not enabled in PyTorch environment)")

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    print(f"Encoding {len(texts)} texts (batch_size={batch_size}) ...")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )
    return vectors.astype(np.float32)


def _get_embedding_dim(
    existing_vectors: np.ndarray | None,
    new_vectors: np.ndarray | None,
    model_name: str,
) -> int:
    """Determine the embedding dimension from available data or a test encode."""
    if existing_vectors is not None:
        return existing_vectors.shape[1]
    if new_vectors is not None:
        return new_vectors.shape[1]

    # Fallback: run a single encode to discover dimension.
    probe = _encode_texts(["probe"], model_name, batch_size=1)
    return probe.shape[1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        build_embeddings()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
