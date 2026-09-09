"""Upload content embeddings to Qdrant.

Usage:
    python -m embeddings.build_qdrant_collection
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
NPY_PATH = PROJECT_ROOT / "embeddings" / "content_embeddings.npy"
INDEX_PATH = PROJECT_ROOT / "embeddings" / "content_embedding_index.csv"


def build_qdrant_collection(
    catalog_path: Path = CATALOG_PATH,
    npy_path: Path = NPY_PATH,
    index_path: Path = INDEX_PATH,
    batch_size: int = 256,
) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required.\n"
            "Install it with: pip install qdrant-client"
        ) from exc

    _assert_inputs(catalog_path, npy_path, index_path)
    catalog = pd.read_csv(catalog_path)
    embedding_index = pd.read_csv(index_path)
    vectors = np.load(npy_path).astype(np.float32, copy=False)
    _validate_inputs(catalog, embedding_index, vectors)

    settings = get_settings()
    if settings.qdrant_path:
        client = QdrantClient(path=settings.qdrant_path)
    else:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    dimension = vectors.shape[1]

    client.recreate_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )

    points: list[Any] = []
    for row_number, row in catalog.iterrows():
        payload = {
            "global_id": _clean(row["global_id"]),
            "content_type": _clean(row["content_type"]),
            "source": _clean(row["source"]),
            "source_id": _clean(row["source_id"]),
            "title": _clean(row["title"]),
            "description": _clean(row.get("description", "")),
            "creators": _clean(row.get("creators", "")),
            "categories": _clean(row.get("categories", "")),
            "release_date": _clean(row.get("release_date", "")),
            "popularity": _clean(row.get("popularity", "")),
            "rating": _clean(row.get("rating", "")),
        }
        # Audience metadata is queried as a Qdrant filter, not just displayed, so
        # these must be real numbers: a Range condition skips any point where the
        # field is missing or a string, which would silently hide the catalog.
        payload["domain"] = _clean(row.get("domain", ""))
        payload["maturity"] = _clean(row.get("maturity", ""))
        payload["audience_min_age"] = _as_int(row.get("audience_min_age"), default=0)
        payload["risk_tier"] = _as_int(row.get("risk_tier"), default=0)
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(row["global_id"]))),
                vector=vectors[row_number].tolist(),
                payload=payload,
            )
        )
        if len(points) >= batch_size:
            client.upsert(collection_name=settings.qdrant_collection, points=points)
            points = []

    if points:
        client.upsert(collection_name=settings.qdrant_collection, points=points)

    print("Qdrant collection built:")
    print(f"  Storage    : {settings.qdrant_path or settings.qdrant_url}")
    print(f"  Collection : {settings.qdrant_collection}")
    print(f"  Points     : {len(catalog)}")
    print(f"  Dimension  : {dimension}")


def _assert_inputs(catalog_path: Path, npy_path: Path, index_path: Path) -> None:
    missing = [path for path in [catalog_path, npy_path, index_path] if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing embedding artifact(s):\n"
            f"{formatted}\n"
            "Run: python -m preprocessing.build_content_catalog; "
            "python -m embeddings.build_embeddings"
        )


def _validate_inputs(
    catalog: pd.DataFrame,
    embedding_index: pd.DataFrame,
    vectors: np.ndarray,
) -> None:
    errors: list[str] = []
    if len(catalog) != len(embedding_index):
        errors.append("Catalog and embedding index row counts do not match.")
    if vectors.ndim != 2:
        errors.append(f"Embedding matrix must be 2D, got shape={vectors.shape}.")
    if vectors.shape[0] != len(catalog):
        errors.append("Embedding matrix row count does not match catalog.")
    missing_audience = [
        column
        for column in ("domain", "maturity", "audience_min_age", "risk_tier")
        if column not in catalog.columns
    ]
    if missing_audience:
        errors.append(
            "Catalog is missing audience columns "
            f"({', '.join(missing_audience)}); rerun preprocessing.build_content_catalog."
        )
    if "embedding_row" not in embedding_index.columns:
        errors.append("Embedding index is missing embedding_row.")
    elif list(embedding_index["embedding_row"]) != list(range(len(embedding_index))):
        errors.append("embedding_row must be contiguous and zero-based.")
    if errors:
        message = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Cannot build Qdrant collection:\n{message}")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clean(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    try:
        build_qdrant_collection()
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()


