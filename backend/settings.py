"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

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


def get_settings() -> Settings:
    return Settings()
