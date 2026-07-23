"""Unit tests for the embeddings pipeline.

All tests use small synthetic data — no real file I/O, no sentence-transformers
model download. The embedding validation logic is tested in isolation.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings.build_embeddings import (
    INDEX_COLUMNS,
    EMBEDDING_MODEL,
    EMBEDDING_VERSION,
    _validate_embedding_output,
)
from preprocessing.cleaners import make_text_hash
from preprocessing.content_schema import make_global_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_index(n: int = 3, model: str = EMBEDDING_MODEL, version: str = EMBEDDING_VERSION) -> pd.DataFrame:
    """Build a synthetic embedding index with n rows."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    media = [
        ("movie", "tmdb_5000_movies", "101"),
        ("book", "google_books_dataset", "b202"),
        ("music", "spotify_songs", "s303"),
    ]
    for i in range(n):
        ctype, source, sid = media[i % len(media)]
        gid = make_global_id(ctype, source, sid + str(i))
        embed_text = f"{ctype} title {i} description {i}"
        rows.append({
            "global_id": gid,
            "content_type": ctype,
            "source": source,
            "source_id": sid + str(i),
            "embedding_model": model,
            "embedding_version": version,
            "text_hash": make_text_hash(embed_text),
            "embedding_row": i,
            "created_at": now,
        })
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def _make_matrix(n: int = 3, dim: int = 8) -> np.ndarray:
    """Build a synthetic float32 embedding matrix."""
    rng = np.random.default_rng(seed=42)
    return rng.random((n, dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Index column tests
# ---------------------------------------------------------------------------

class TestIndexColumns:
    def test_index_has_all_required_columns(self):
        idx = _make_index()
        for col in INDEX_COLUMNS:
            assert col in idx.columns, f"Missing column: {col}"

    def test_index_column_order(self):
        assert INDEX_COLUMNS == [
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
# _validate_embedding_output tests
# ---------------------------------------------------------------------------

class TestValidateEmbeddingOutput:
    def test_valid_output_passes(self):
        idx = _make_index(3)
        mat = _make_matrix(3)
        _validate_embedding_output(mat, idx)  # should not raise

    def test_row_count_mismatch_raises(self):
        idx = _make_index(3)
        mat = _make_matrix(4)  # wrong number of rows
        with pytest.raises(ValueError, match="Row count mismatch"):
            _validate_embedding_output(mat, idx)

    def test_non_contiguous_embedding_row_raises(self):
        idx = _make_index(3)
        idx.loc[1, "embedding_row"] = 99  # break sequence
        mat = _make_matrix(3)
        with pytest.raises(ValueError, match="embedding_row is not a contiguous"):
            _validate_embedding_output(mat, idx)

    def test_duplicate_global_id_model_version_hash_raises(self):
        idx = _make_index(3)
        # Duplicate the first row with a different embedding_row to avoid
        # triggering the contiguous check first.
        dup = idx.iloc[[0]].copy()
        dup["embedding_row"] = 3
        idx2 = pd.concat([idx, dup], ignore_index=True)
        mat = _make_matrix(4)
        with pytest.raises(ValueError, match="Duplicate"):
            _validate_embedding_output(mat, idx2)

    def test_missing_column_raises(self):
        idx = _make_index(3).drop(columns=["text_hash"])
        mat = _make_matrix(3)
        with pytest.raises(ValueError, match="Index missing columns"):
            _validate_embedding_output(mat, idx)


# ---------------------------------------------------------------------------
# Embedding row ordering
# ---------------------------------------------------------------------------

class TestEmbeddingRowOrdering:
    def test_embedding_row_is_zero_based(self):
        idx = _make_index(5)
        assert list(idx["embedding_row"]) == list(range(5))

    def test_npy_row_count_matches_index(self):
        n = 4
        idx = _make_index(n)
        mat = _make_matrix(n)
        assert mat.shape[0] == len(idx)

    def test_npy_row_count_matches_index_after_concat(self):
        """Simulates rebuilding: old 3 rows + 2 new = 5 total."""
        idx = _make_index(5)
        mat = _make_matrix(5)
        assert mat.shape[0] == len(idx)
        _validate_embedding_output(mat, idx)


# ---------------------------------------------------------------------------
# Reuse-map key tests (logic parity with build_embeddings)
# ---------------------------------------------------------------------------

class TestReuseMapKey:
    def test_same_key_considered_reusable(self):
        """If the key matches, the row should be reused (not re-embedded)."""
        idx = _make_index(3)
        reuse_map = {
            (
                row["global_id"],
                row["embedding_model"],
                row["embedding_version"],
                row["text_hash"],
            ): int(row["embedding_row"])
            for _, row in idx.iterrows()
        }
        for _, row in idx.iterrows():
            key = (
                row["global_id"],
                row["embedding_model"],
                row["embedding_version"],
                row["text_hash"],
            )
            assert key in reuse_map

    def test_changed_text_hash_not_in_reuse_map(self):
        """A row whose text_hash changed must NOT be found in the old reuse map."""
        idx = _make_index(1)
        reuse_map = {
            (
                row["global_id"],
                row["embedding_model"],
                row["embedding_version"],
                row["text_hash"],
            ): int(row["embedding_row"])
            for _, row in idx.iterrows()
        }
        # Same global_id but different hash → not reusable
        old_row = idx.iloc[0]
        new_key = (
            old_row["global_id"],
            old_row["embedding_model"],
            old_row["embedding_version"],
            make_text_hash("completely different text"),
        )
        assert new_key not in reuse_map
