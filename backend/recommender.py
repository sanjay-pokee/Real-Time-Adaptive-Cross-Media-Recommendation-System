"""Semantic recommendation search backed by Sentence-BERT and FAISS.

Usage
-----
    python -m backend.recommender "space adventure with emotional music" --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
FAISS_INDEX_PATH = PROJECT_ROOT / "embeddings" / "content_faiss.index"
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
class SemanticRecommender:
    """Load artifacts once, then run repeated semantic searches."""

    catalog_path: Path = CATALOG_PATH
    faiss_index_path: Path = FAISS_INDEX_PATH
    embedding_index_path: Path = EMBEDDING_INDEX_PATH
    embedding_model: str = EMBEDDING_MODEL

    def __post_init__(self) -> None:
        self._assert_artifacts_exist()
        self.catalog = pd.read_csv(self.catalog_path)
        self.embedding_index = pd.read_csv(self.embedding_index_path)
        self.index = faiss.read_index(str(self.faiss_index_path))
        self.model = self._load_model()
        self._validate_artifacts()

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return top semantic matches for a natural-language query."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        content_type = _normalize_content_type(content_type)
        search_k = min(max(top_k * 5, top_k), self.index.ntotal)

        while True:
            scores, row_ids = self.index.search(query_vector, search_k)
            results = self._format_results(scores[0], row_ids[0], top_k, content_type)

            if len(results) >= top_k or search_k >= self.index.ntotal or content_type is None:
                return results

            search_k = min(search_k * 2, self.index.ntotal)

    def _format_results(
        self,
        scores: np.ndarray,
        row_ids: np.ndarray,
        top_k: int,
        content_type: str | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for score, row_id in zip(scores, row_ids):
            if row_id < 0:
                continue
            catalog_row = self.catalog.iloc[int(row_id)]
            if content_type and catalog_row["content_type"] != content_type:
                continue

            result = {
                column: _clean_value(catalog_row[column])
                for column in RESULT_COLUMNS
                if column in self.catalog.columns
            }
            result["score"] = float(score)
            results.append(result)

            if len(results) == top_k:
                break

        return results

    def _assert_artifacts_exist(self) -> None:
        missing = [
            path
            for path in [
                self.catalog_path,
                self.embedding_index_path,
                self.faiss_index_path,
            ]
            if not path.exists()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Missing recommender artifact(s):\n"
                f"{formatted}\n"
                "Run: python -m embeddings.build_faiss_index"
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
                    "Run once with internet access, or set HF_HUB_OFFLINE=1 if the model is already cached."
                ) from remote_exc

    def _validate_artifacts(self) -> None:
        errors: list[str] = []
        if len(self.catalog) != len(self.embedding_index):
            errors.append(
                f"Catalog rows ({len(self.catalog)}) do not match "
                f"embedding index rows ({len(self.embedding_index)})."
            )
        if self.index.ntotal != len(self.embedding_index):
            errors.append(
                f"FAISS rows ({self.index.ntotal}) do not match "
                f"embedding index rows ({len(self.embedding_index)})."
            )
        if "embedding_row" not in self.embedding_index.columns:
            errors.append("Embedding index is missing embedding_row.")
        elif list(self.embedding_index["embedding_row"]) != list(range(len(self.embedding_index))):
            errors.append("Embedding rows are not in contiguous catalog order.")

        if errors:
            message = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"Invalid recommender artifacts:\n{message}")


def _normalize_content_type(content_type: str | None) -> str | None:
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
    if pd.isna(value):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_results(results: list[dict[str, Any]]) -> None:
    for rank, item in enumerate(results, start=1):
        title = item.get("title", "")
        content_type = item.get("content_type", "")
        score = item.get("score", 0.0)
        categories = item.get("categories", "")
        rating = item.get("rating", "")
        print(f"{rank:>2}. [{content_type}] {title}  score={score:.4f}")
        if categories:
            print(f"    categories: {categories}")
        if rating != "":
            print(f"    rating: {rating}")


def _content_type_arg(value: str) -> str:
    """argparse type= helper that normalises and validates content-type aliases."""
    try:
        return _normalize_content_type(value)  # type: ignore[return-value]
    except ValueError as exc:
        import argparse as _ap
        raise _ap.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Search content recommendations with FAISS.")
    parser.add_argument("query", help="Natural-language recommendation query.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return.")
    parser.add_argument(
        "--content-type",
        type=_content_type_arg,
        default=None,
        metavar="{movie|movies|film, book|books, music|song|songs|track}",
        help="Optional filter by content type. Accepts common aliases.",
    )
    args = parser.parse_args()

    try:
        recommender = SemanticRecommender()
        results = recommender.recommend(
            args.query,
            top_k=args.top_k,
            content_type=args.content_type,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not results and args.content_type:
        print(
            f"No '{args.content_type}' results found for that query.\n"
            "Tip: try a broader query or omit --content-type to search all content.",
            file=sys.stderr,
        )

    _print_results(results)


if __name__ == "__main__":
    main()



