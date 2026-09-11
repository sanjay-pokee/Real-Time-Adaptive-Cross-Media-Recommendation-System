"""Fetch cast and director for the movies pulled by scripts.fetch_tmdb_movies.

/discover returns overview and poster but no credits, so the titles beyond the
original Kaggle 4,803 would otherwise have no cast or director - which matters,
because creator names are part of the embedded text and drive queries like
"Christopher Nolan".

Output deliberately mirrors the Kaggle credits CSV it sits beside:

    movie_id,title,cast,crew

with `cast` and `crew` holding list-of-dict blobs. preprocessing reads those
with ast.literal_eval, and JSON built from plain strings is valid Python
literal syntax, so no parser change is needed. Only string fields are emitted
for that reason - a JSON null or true would parse as JSON but not as a Python
literal.

Written incrementally and keyed by movie id, so an interrupted run resumes and
a re-run only fetches what is missing.

Usage:
    python -m scripts.fetch_tmdb_credits
    python -m scripts.fetch_tmdb_credits --limit 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOVIES = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_movies.csv"
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_credits.csv"

API = "https://api.themoviedb.org/3"
OUTPUT_COLUMNS = ["movie_id", "title", "cast", "crew"]

# How many refusals in a row before abandoning the run.
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


def fetch_credits(session: requests.Session, movie_id: str, key: str, top_cast: int):
    """Return (cast_blob, crew_blob) as JSON strings, or THROTTLED."""
    try:
        response = session.get(f"{API}/movie/{movie_id}/credits",
                               params={"api_key": key}, timeout=25)
        if response.status_code == 404:
            return "[]", "[]"
        if response.status_code in (401, 429, 503):
            return THROTTLED
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return THROTTLED

    # Only `name` is kept for cast, and only directors from crew, because that
    # is all the normalizer reads. Keeping the full payload would bloat the
    # file by an order of magnitude for no benefit.
    cast = [
        {"name": str(person.get("name") or "")}
        for person in (payload.get("cast") or [])[:top_cast]
        if person.get("name")
    ]
    crew = [
        {"name": str(person.get("name") or ""), "job": "Director"}
        for person in (payload.get("crew") or [])
        if person.get("job") == "Director" and person.get("name")
    ]
    return json.dumps(cast), json.dumps(crew)


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {
        str(row.movie_id): {
            "movie_id": str(row.movie_id),
            "title": "" if pd.isna(row.title) else str(row.title),
            "cast": "[]" if pd.isna(row.cast) else str(row.cast),
            "crew": "[]" if pd.isna(row.crew) else str(row.crew),
        }
        for row in frame.itertuples(index=False)
    }


def save(rows: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows.values()), columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TMDb cast and director.")
    parser.add_argument("--movies", type=Path, default=DEFAULT_MOVIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many lookups (0 = all missing)")
    parser.add_argument("--top-cast", type=int, default=8)
    parser.add_argument("--rate", type=float, default=0.1,
                        help="Seconds between requests (default 0.1, i.e. 10/sec)")
    args = parser.parse_args()

    key = os.getenv("TMDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("TMDB_API_KEY is not set in .env")
    if not args.movies.exists():
        raise SystemExit(
            f"No movie file at {args.movies}."
            "\nRun: python -m scripts.fetch_tmdb_movies"
        )

    movies = pd.read_csv(args.movies)
    rows = load_existing(args.out)
    todo = movies[~movies["id"].astype(str).isin(rows)]
    if args.limit:
        todo = todo.head(args.limit)

    print(f"{len(movies):,} movies, {len(rows):,} already have credits, "
          f"{len(todo):,} to fetch")
    if todo.empty:
        return
    print(f"pacing at {1/max(args.rate, 0.001):.0f}/sec, roughly "
          f"{len(todo) * args.rate / 60:.0f} min")
    print()

    session = _session()
    fetched = with_director = throttled_streak = 0

    try:
        for index, record in enumerate(todo.itertuples(index=False)):
            started = time.monotonic()
            result = fetch_credits(session, str(record.id), key, args.top_cast)

            if result is THROTTLED:
                throttled_streak += 1
                if throttled_streak >= THROTTLE_GIVE_UP:
                    print(f"Stopping: {throttled_streak} refusals in a row. "
                          "Nothing was cached for those, so a re-run resumes.")
                    break
                time.sleep(min(2 ** throttled_streak, 60))
                continue

            throttled_streak = 0
            cast_blob, crew_blob = result
            rows[str(record.id)] = {
                "movie_id": str(record.id),
                "title": str(record.title),
                "cast": cast_blob,
                "crew": crew_blob,
            }
            fetched += 1
            if crew_blob != "[]":
                with_director += 1

            if fetched % 250 == 0:
                save(rows, args.out)
                print(f"  {index + 1:,}/{len(todo):,} fetched, "
                      f"{with_director:,} with a director")

            elapsed = time.monotonic() - started
            if elapsed < args.rate:
                time.sleep(args.rate - elapsed)
    except KeyboardInterrupt:
        print("interrupted")

    save(rows, args.out)
    print()
    print(f"wrote {args.out}")
    print(f"  {len(rows):,} movies have credits, {with_director:,} with a director "
          f"this run")
    print()
    print("Rebuild to apply:")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_embeddings")
    print("  python -m embeddings.build_qdrant_collection")


if __name__ == "__main__":
    main()
