"""Small API clients for online ingestion jobs.

These helpers fetch source data for ingestion only. Recommendation-serving
endpoints should not call external content APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from backend.settings import Settings, get_settings


@dataclass
class MovieApiClient:
    settings: Settings = get_settings()

    def search_movies(self, query: str, page: int = 1) -> dict[str, Any]:
        if not self.settings.tmdb_api_key:
            raise RuntimeError("Set TMDB_API_KEY before using TMDb ingestion.")
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": self.settings.tmdb_api_key, "query": query, "page": page},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()


@dataclass
class BookApiClient:
    def search_books(self, query: str, limit: int = 20) -> dict[str, Any]:
        response = requests.get(
            "https://openlibrary.org/search.json",
            params={"q": query, "limit": limit},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()


@dataclass
class MusicApiClient:
    settings: Settings = get_settings()

    def search_recordings(self, query: str, limit: int = 20) -> dict[str, Any]:
        response = requests.get(
            "https://musicbrainz.org/ws/2/recording",
            params={"query": query, "fmt": "json", "limit": limit},
            headers={"User-Agent": self.settings.musicbrainz_user_agent},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
