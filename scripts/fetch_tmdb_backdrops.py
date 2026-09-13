"""Fetch the wide backdrop image for every movie pulled by scripts.fetch_tmdb_movies.

The catalogue carries `image_url`, which for movies is the *poster* - the tall
2:3 artwork. A detail view wants the other image: TMDb's `backdrop_path`, a wide
16:9 still from the film, which is what fills a banner behind the title without
being stretched or cropped to nothing.

/discover/movie does return backdrop_path in the same payload as poster_path, so
this could have come free at fetch time - it simply was not kept. Re-running
fetch_tmdb_movies would pick it up in ~1,100 paged requests rather than 22,116
single ones, but /discover is ordered by live popularity and seeded from whatever
TMDb holds today, so a re-run would return a *different* set of films. Every
changed row means a new global_id, a re-embed, and re-seeded interactions
pointing at items that no longer exist. Two days from a review that is not a
trade worth making, so this reads the films we already have, one id at a time,
and leaves the catalogue's composition untouched.

Written incrementally and keyed by movie id, so an interrupted run resumes and a
re-run only fetches what is missing. A film with no backdrop is cached as an
empty string - a real answer, not a gap to retry.

Usage:
    python -m scripts.fetch_tmdb_backdrops
    python -m scripts.fetch_tmdb_backdrops --limit 500
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOVIES = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_movies.csv"
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_backdrops.csv"

API = "https://api.themoviedb.org/3"
# w1280 rather than `original`: a banner is displayed a few hundred pixels tall,
# and originals run to several megabytes each.
BACKDROP_BASE = "https://image.tmdb.org/t/p/w1280"

OUTPUT_COLUMNS = ["movie_id", "title", "backdrop_url"]

THROTTLE_GIVE_UP = 12
THROTTLED = object()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "nexus-recommender/1.0 (academic project)"})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )))
    return session


def fetch_backdrop(session: requests.Session, movie_id: str, key: str):
    """Return a backdrop URL (possibly empty), or THROTTLED."""
    try:
        response = session.get(f"{API}/movie/{movie_id}",
                               params={"api_key": key}, timeout=25)
        if response.status_code == 404:
            return ""
        if response.status_code in (401, 429, 503):
            return THROTTLED
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return THROTTLED

    path = str(payload.get("backdrop_path") or "").strip()
    return f"{BACKDROP_BASE}{path}" if path else ""


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"movie_id": str})
    return {
        str(row.movie_id): {
            "movie_id": str(row.movie_id),
            "title": "" if pd.isna(row.title) else str(row.title),
            "backdrop_url": "" if pd.isna(row.backdrop_url) else str(row.backdrop_url),
        }
        for row in frame.itertuples(index=False)
    }


def save(rows: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows.values()), columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TMDb backdrop images.")
    parser.add_argument("--movies", type=Path, default=DEFAULT_MOVIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many lookups (0 = all missing)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent lookups (default 8)")
    args = parser.parse_args()

    key = os.getenv("TMDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("TMDB_API_KEY is not set in .env")
    if not args.movies.exists():
        raise SystemExit(f"No movie file at {args.movies}.")

    movies = pd.read_csv(args.movies, dtype={"id": str})
    rows = load_existing(args.out)
    todo = movies[~movies["id"].astype(str).isin(rows)]
    if args.limit:
        todo = todo.head(args.limit)

    print(f"{len(movies):,} movies, {len(rows):,} already have a backdrop, "
          f"{len(todo):,} to fetch")
    if todo.empty:
        _summarise(rows)
        return

    workers = max(1, args.workers)
    sessions = [_session() for _ in range(workers)]
    records = list(todo.itertuples(index=False))
    chunk_size = workers * 25
    fetched = found = throttled_chunks = 0

    def _lookup(task):
        position, record = task
        return record, fetch_backdrop(sessions[position % workers], str(record.id), key)

    print(f"{workers} workers; expect roughly {len(records) / (100.0 * workers):.0f} min")
    print()

    try:
        for start in range(0, len(records), chunk_size):
            chunk = list(enumerate(records[start : start + chunk_size]))
            throttled_in_chunk = 0

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for record, result in pool.map(_lookup, chunk):
                    if result is THROTTLED:
                        throttled_in_chunk += 1
                        continue
                    rows[str(record.id)] = {
                        "movie_id": str(record.id),
                        "title": str(record.title),
                        "backdrop_url": result,
                    }
                    fetched += 1
                    if result:
                        found += 1

            save(rows, args.out)
            print(f"  {min(start + chunk_size, len(records)):,}/{len(records):,} "
                  f"done, {found:,} with a backdrop", flush=True)

            if throttled_in_chunk == len(chunk):
                throttled_chunks += 1
                if throttled_chunks >= THROTTLE_GIVE_UP:
                    print(f"Stopping: {throttled_chunks} fully-refused batches.")
                    break
                time.sleep(min(2 ** throttled_chunks, 60))
            else:
                throttled_chunks = 0
    except KeyboardInterrupt:
        print("interrupted")

    save(rows, args.out)
    print()
    print(f"wrote {args.out}")
    _summarise(rows)
    print()
    print("Rebuild to apply:")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_qdrant_collection")


def _summarise(rows: dict[str, dict]) -> None:
    have = sum(1 for row in rows.values() if row["backdrop_url"])
    print(f"  {len(rows):,} cached, {have:,} with a backdrop "
          f"({have / max(len(rows), 1) * 100:.0f}%)")


if __name__ == "__main__":
    main()
