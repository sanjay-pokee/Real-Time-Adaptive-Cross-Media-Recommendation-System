"""Backfill cover art for catalogue rows whose source dataset carries none.

Books and the Amazon-sourced verticals already ship an image URL, so this only
has to cover the two that do not:

* music  - the Spotify export has no artwork column. Looked up on the iTunes
           Search API: free, no key, no OAuth. Deliberately NOT Spotify's own
           API, whose Developer Terms forbid ingesting Spotify Content into a
           machine-learning model, which is exactly what this catalogue feeds.
* movie  - TMDB's 5000-movie export has only a homepage column. Needs a TMDb
           API key in TMDB_API_KEY; skipped when that is unset.

Rows are processed in descending popularity, because a demo surfaces popular
items and a partial run should cover those first. Results are cached in
data/processed/cover_art_cache.csv so a re-run resumes instead of re-querying,
and the cache is written incrementally so an interrupted run keeps its work.

Usage:
    python -m scripts.backfill_cover_art --content-type music --limit 4000
    python -m scripts.backfill_cover_art --content-type movie --limit 4803

Then rebuild so the catalogue picks the cache up:
    python -m preprocessing.build_content_catalog
    python -m embeddings.build_qdrant_collection
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "cover_art_cache.csv"

ITUNES_SEARCH = "https://itunes.apple.com/search"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

CACHE_COLUMNS = ["global_id", "image_url"]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        # Identify the caller. Anonymous floods are what gets an IP throttled.
        "User-Agent": "nexus-recommender/1.0 (academic project; cover-art backfill)",
    })
    return session


def _itunes_artwork(session: requests.Session, title: str, artist: str) -> str:
    """Album art for one track, or "" when nothing matches."""
    term = " ".join(part for part in (artist, title) if part).strip()
    if not term:
        return ""
    try:
        response = session.get(
            ITUNES_SEARCH,
            params={"term": term, "media": "music", "entity": "song", "limit": 1},
            timeout=12,
        )
        if response.status_code == 403:
            # Being throttled. Back off rather than hammering.
            time.sleep(3)
            return ""
        response.raise_for_status()
        results = response.json().get("results") or []
    except (requests.RequestException, ValueError):
        return ""
    if not results:
        return ""
    art = str(results[0].get("artworkUrl100") or "")
    # The 100px thumbnail URL upgrades to any size by substitution.
    return art.replace("100x100bb", "400x400bb") if art else ""


def _tmdb_poster(session: requests.Session, source_id: str, api_key: str) -> str:
    try:
        response = session.get(
            f"https://api.themoviedb.org/3/movie/{source_id}",
            params={"api_key": api_key},
            timeout=12,
        )
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        path = response.json().get("poster_path")
    except (requests.RequestException, ValueError):
        return ""
    return f"{TMDB_IMAGE_BASE}{path}" if path else ""


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    frame = pd.read_csv(CACHE_PATH)
    return {
        str(row.global_id): str(row.image_url or "")
        for row in frame.itertuples(index=False)
    }


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"global_id": key, "image_url": value} for key, value in cache.items()],
        columns=CACHE_COLUMNS,
    )
    frame.to_csv(CACHE_PATH, index=False)


def backfill(content_type: str, limit: int, workers: int) -> None:
    if not CATALOG_PATH.exists():
        raise SystemExit(
            f"No catalogue at {CATALOG_PATH}.\n"
            "Run: python -m preprocessing.build_content_catalog"
        )
    catalog = pd.read_csv(CATALOG_PATH, low_memory=False)
    rows = catalog[catalog["content_type"] == content_type].copy()
    if rows.empty:
        raise SystemExit(f"No {content_type} rows in the catalogue.")

    cache = load_cache()
    # Skip anything already resolved, including previous misses: a re-lookup of
    # a track iTunes does not carry costs a request and returns nothing again.
    todo = rows[~rows["global_id"].astype(str).isin(cache)]
    todo = todo.sort_values("popularity", ascending=False, na_position="last")
    todo = todo.head(limit)

    print(f"{content_type}: {len(rows):,} rows, {len(cache):,} cached, {len(todo):,} to look up")
    if todo.empty:
        return

    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if content_type == "movie" and not api_key:
        raise SystemExit(
            "Movie posters need a TMDb API key.\n"
            "Create one at themoviedb.org, then set TMDB_API_KEY yourself."
        )

    session = _session()

    def resolve(record) -> tuple[str, str]:
        if content_type == "movie":
            return str(record.global_id), _tmdb_poster(session, str(record.source_id), api_key)
        return str(record.global_id), _itunes_artwork(
            session, str(record.title or ""), str(record.creators or "")
        )

    found = 0
    done = 0
    records = list(todo.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for global_id, url in pool.map(resolve, records):
            cache[global_id] = url
            done += 1
            if url:
                found += 1
            if done % 200 == 0:
                # Written as we go, so an interrupted run is not wasted.
                save_cache(cache)
                print(f"  {done:,}/{len(records):,} looked up, {found:,} found")

    save_cache(cache)
    print()
    print(f"wrote {CACHE_PATH}")
    print(f"  looked up {done:,}, found {found:,} ({found / max(done, 1) * 100:.0f}%)")
    print()
    print("Rebuild to apply:")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_qdrant_collection")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill cover art for the catalogue.")
    parser.add_argument("--content-type", required=True, choices=["music", "movie"])
    parser.add_argument("--limit", type=int, default=4000,
                        help="Rows to look up this run, most popular first (default 4000)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent requests. Keep modest; these are free public APIs.")
    args = parser.parse_args()

    backfill(args.content_type, args.limit, args.workers)


if __name__ == "__main__":
    main()
