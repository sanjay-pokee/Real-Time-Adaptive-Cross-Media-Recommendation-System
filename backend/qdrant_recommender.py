"""Qdrant-backed semantic recommendation search.

Qdrant is the production vector backend. Build the collection with:
    python -m embeddings.build_qdrant_collection
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.mysql_store import MySQLStore
from backend.settings import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "embeddings" / "content_embedding_index.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RESULT_COLUMNS = [
    "global_id",
    "content_type",
    "source",
    "source_id",
    "title",
    "description",
    "creators",
    "categories",
    "release_date",
    "popularity",
    "rating",
]


@dataclass
class QdrantRecommender:
    catalog_path: Path = CATALOG_PATH
    embedding_index_path: Path = EMBEDDING_INDEX_PATH
    embedding_model: str = EMBEDDING_MODEL
    settings: Settings = get_settings()

    def __post_init__(self) -> None:
        self._assert_artifacts_exist()
        self.catalog = pd.read_csv(self.catalog_path)
        self.embedding_index = pd.read_csv(self.embedding_index_path)
        self.client = self._load_qdrant_client()
        self.model = self._load_model()
        self._validate_artifacts()

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        content_type: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        search_k = top_k * 5 if user_id else top_k
        results = self._search_vector(query_vector, search_k, content_type)
        return self._personalize_results(results, top_k, user_id)

    def recommend_from_item(
        self,
        global_id: str,
        top_k: int = 10,
        content_type: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not global_id.strip():
            raise ValueError("global_id cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        catalog_matches = self.catalog[self.catalog["global_id"] == global_id]
        if catalog_matches.empty:
            raise ValueError(f"Unknown global_id: {global_id}")

        source_text = str(catalog_matches.iloc[0].get("embedding_text", ""))
        vector = self.model.encode(
            [source_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        search_k = (top_k * 5 if user_id else top_k) + 1
        results = self._search_vector(vector, search_k, content_type)
        results = [item for item in results if item["global_id"] != global_id]
        return self._personalize_results(results, top_k, user_id)

    def _search_vector(
        self,
        vector: np.ndarray,
        top_k: int,
        content_type: str | None,
    ) -> list[dict[str, Any]]:
        normalized_content_type = normalize_content_type(content_type)
        query_filter = self._build_qdrant_filter(normalized_content_type)

        try:
            points = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        except AttributeError:
            response = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        except Exception as exc:
            raise RuntimeError(
                "Could not query Qdrant. Start Qdrant, build the collection, and check "
                "QDRANT_PATH or QDRANT_URL/QDRANT_COLLECTION."
            ) from exc

        results: list[dict[str, Any]] = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            item = {column: _clean_value(payload.get(column, "")) for column in RESULT_COLUMNS}
            item["score"] = float(getattr(point, "score", 0.0))
            results.append(item)
        return results


    def _personalize_results(
        self,
        results: list[dict[str, Any]],
        top_k: int,
        user_id: str | None,
    ) -> list[dict[str, Any]]:
        if not user_id:
            return results[:top_k]

        try:
            profile = MySQLStore(self.settings).get_user_preference_profile(user_id)
        except Exception:
            return results[:top_k]

        category_weights = profile["category_weights"]
        content_type_weights = profile["content_type_weights"]
        positive_ids = profile["positive_ids"]
        negative_ids = profile["negative_ids"]

        reranked = []
        for item in results:
            score = float(item.get("score", 0.0))
            bonus = 0.0

            for category in _split_categories(item.get("categories")):
                bonus += min(category_weights.get(category, 0.0), 3.0) * 0.035

            content_type = str(item.get("content_type", "")).lower()
            bonus += min(content_type_weights.get(content_type, 0.0), 3.0) * 0.02

            if item.get("global_id") in positive_ids:
                bonus -= 0.15
            if item.get("global_id") in negative_ids:
                bonus -= 0.35

            personalized = dict(item)
            personalized["score"] = score + bonus
            reranked.append(personalized)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]

    def _build_qdrant_filter(self, content_type: str | None):
        if content_type is None:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            ) from exc
        return Filter(
            must=[
                FieldCondition(
                    key="content_type",
                    match=MatchValue(value=content_type),
                )
            ]
        )

    def _load_qdrant_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for the Qdrant backend.\n"
                "Install it with: pip install qdrant-client"
            ) from exc

        if self.settings.qdrant_path:
            return QdrantClient(path=self.settings.qdrant_path)

        return QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for query embeddings.\n"
                "Install project dependencies, then try again."
            ) from exc

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        print(f"Loading query embedding model on {device}: {self.embedding_model}")
        try:
            return SentenceTransformer(
                self.embedding_model,
                device=device,
                local_files_only=True,
            )
        except TypeError:
            return SentenceTransformer(self.embedding_model, device=device)
        except Exception as local_exc:
            print(
                "Local model cache load failed; trying Hugging Face download...",
                file=sys.stderr,
            )
            try:
                return SentenceTransformer(self.embedding_model, device=device)
            except Exception as remote_exc:
                raise RuntimeError(
                    "Could not load the query embedding model from local cache or Hugging Face. "
                    "Run once with internet access, or set HF_HUB_OFFLINE=1 if cached."
                ) from remote_exc

    def _assert_artifacts_exist(self) -> None:
        missing = [
            path
            for path in [self.catalog_path, self.embedding_index_path]
            if not path.exists()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Missing recommender artifact(s):\n"
                f"{formatted}\n"
                "Run: python -m preprocessing.build_content_catalog; "
                "python -m embeddings.build_embeddings; "
                "python -m embeddings.build_qdrant_collection"
            )

    def _validate_artifacts(self) -> None:
        if len(self.catalog) != len(self.embedding_index):
            raise ValueError(
                f"Catalog rows ({len(self.catalog)}) do not match "
                f"embedding index rows ({len(self.embedding_index)})."
            )


def normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None

    aliases = {
        "movie": "movie",
        "movies": "movie",
        "film": "movie",
        "films": "movie",
        "book": "book",
        "books": "book",
        "music": "music",
        "song": "music",
        "songs": "music",
        "track": "music",
        "tracks": "music",
    }
    normalized = aliases.get(content_type.strip().lower())
    if normalized is None:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown content type '{content_type}'. Use one of: {allowed}.")
    return normalized


def _clean_value(value: Any) -> Any:
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


def _split_categories(value: Any) -> list[str]:
    text = str(value or "").lower()
    return [part.strip() for part in text.split(",") if part.strip()]
