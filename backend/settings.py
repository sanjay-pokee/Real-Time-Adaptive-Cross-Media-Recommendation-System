"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    qdrant_path: str | None = os.getenv("QDRANT_PATH", "qdrant_storage") or None
    qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "content")
    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "cross_media_recs")
    tmdb_api_key: str | None = os.getenv("TMDB_API_KEY") or None
    musicbrainz_user_agent: str = os.getenv(
        "MUSICBRAINZ_USER_AGENT",
        "cross-media-recommender/0.1 (local-development)",
    )
    lightgcn_artifact_path: str = os.getenv(
        "LIGHTGCN_ARTIFACT_PATH",
        str(PROJECT_ROOT / "models" / "graph" / "artifacts" / "lightgcn_embeddings.npz"),
    )
    lightgcn_weight: float = float(os.getenv("LIGHTGCN_WEIGHT", "0.2"))
    ema_weight: float = float(os.getenv("EMA_WEIGHT", "0.15"))
    # Ranking weights for the live engine (see backend/qdrant_recommender.py).
    # Chosen against a hand-made relevance yardstick of 18 queries and 7 seed
    # titles; the paper's offline ablation used its own weights.
    kg_weight: float = float(os.getenv("KG_WEIGHT", "0.2"))
    kg_similar_weight: float = float(os.getenv("KG_SIMILAR_WEIGHT", "0.8"))
    popularity_weight: float = float(os.getenv("POPULARITY_WEIGHT", "0.3"))
    # "Find similar" narrowed to one type keeps an item only when the knowledge
    # graph links it to the seed or its text cosine reaches this. Measured on
    # BGE-small over 14 seed titles: every film's "Music" list was unrelated
    # songs at 0.47-0.62, while the right films for a book (Harry Potter, Dune)
    # scored 0.66-0.79. Re-measure if the encoder changes.
    similar_min_cosine: float = float(os.getenv("SIMILAR_MIN_COSINE", "0.63"))
    # How many vector neighbours the reranker sees. With only the final page,
    # an established title one place below the cut could never be promoted.
    candidate_pool: int = int(os.getenv("CANDIDATE_POOL", "300"))
    tmdb_movies_path: str = os.getenv(
        "TMDB_MOVIES_PATH", str(PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_movies.csv")
    )
    tmdb_shows_path: str = os.getenv(
        "TMDB_SHOWS_PATH", str(PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_shows.csv")
    )
    ema_alpha: float = float(os.getenv("EMA_ALPHA", "0.25"))
    content_embeddings_path: str = os.getenv(
        "CONTENT_EMBEDDINGS_PATH",
        str(PROJECT_ROOT / "embeddings" / "content_embeddings.npy"),
    )
    content_embedding_index_path: str = os.getenv(
        "CONTENT_EMBEDDING_INDEX_PATH",
        str(PROJECT_ROOT / "embeddings" / "content_embedding_index.csv"),
    )
    # The text encoder for item, synopsis and query vectors. Changing it means
    # re-embedding (embeddings/build_embeddings.py, build_plot_embeddings.py),
    # rebuilding Qdrant and rebuilding the EMA profiles; the recommender refuses
    # to start when the stored vectors came from a different model.
    # bge-small replaced all-MiniLM-L6-v2 after a side-by-side on the live
    # pipeline: famous plots found at #1 went from 14/30 to 22/30.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    # Prefix for queries only (never items). None: the model's own default.
    embedding_query_instruction: str | None = os.getenv("EMBEDDING_QUERY_INSTRUCTION")
    # Synopsis-only vectors for plot queries (backend/plot_index.py). Optional:
    # without the files, search runs on the main vectors alone.
    plot_embeddings_path: str = os.getenv(
        "PLOT_EMBEDDINGS_PATH",
        str(PROJECT_ROOT / "embeddings" / "plot_embeddings.npy"),
    )
    plot_embedding_index_path: str = os.getenv(
        "PLOT_EMBEDDING_INDEX_PATH",
        str(PROJECT_ROOT / "embeddings" / "plot_embedding_index.csv"),
    )
    # Accounts. The Google client id is public (it ships in the page), so it is
    # served to the frontend from GET /auth/config rather than duplicated there.
    google_client_id: str | None = os.getenv("GOOGLE_CLIENT_ID") or None
    session_days: int = int(os.getenv("SESSION_DAYS", "30"))
    # Set to 1 once the site is served over HTTPS; a Secure cookie is never
    # sent over plain http://localhost, so it must stay off for local runs.
    cookie_secure: bool = os.getenv("COOKIE_SECURE", "0") == "1"


def get_settings() -> Settings:
    return Settings()


# BGE models are trained with this prefix on short queries searched against
# passages; MiniLM was trained without one.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def query_instruction(settings: Settings) -> str:
    """The text put before a search query (not an item) when it is encoded."""
    if settings.embedding_query_instruction is not None:
        return settings.embedding_query_instruction
    return _BGE_QUERY_INSTRUCTION if "bge-" in settings.embedding_model.lower() and "-en" in settings.embedding_model.lower() else ""

