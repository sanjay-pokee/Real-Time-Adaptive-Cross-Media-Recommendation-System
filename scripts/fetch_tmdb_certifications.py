"""Fetch the US content rating for every movie pulled by scripts.fetch_tmdb_movies.

``config/domains.yaml`` declares the movie content type as
``maturity_source: certification``, but nothing ever fetched a certification, so
movie maturity fell through to the keyword heuristic in
``preprocessing.audience_tagging``. That heuristic reads the genre text, where
"Animation" matches the all_ages rule - and entertainment is not a regulated
domain, so nothing stopped a keyword from *relaxing* a row. The result was that
Akira, Heavy Metal, Grave of the Fireflies and Waltz with Bashir were all tagged
``all_ages`` and served to an eight-year-old.

A genre is not a rating. TMDb publishes the real one under
/movie/{id}/release_dates, so this fetches it and the normalizer uses it,
leaving the heuristic to cover only the titles TMDb has no rating for.

Why /release_dates and not a field on /discover: certification is per-country
and per-release (theatrical, digital, TV can differ), so it is only ever exposed
in that sub-resource. US is used because it is the most consistently populated
and is what the MPAA mapping in preprocessing assumes; a title with no US
release keeps an empty certification and falls back to the heuristic.

Written incrementally and keyed by movie id, so an interrupted run resumes and a
re-run only fetches what is missing. A title genuinely carrying no rating is
cached as an empty string, so it is not retried forever - same distinction the
cover-art backfill has to make between "no artwork exists" and "the API refused
to answer right now".

Usage:
    python -m scripts.fetch_tmdb_certifications
    python -m scripts.fetch_tmdb_certifications --limit 500
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
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_certifications.csv"

API = "https://api.themoviedb.org/3"
OUTPUT_COLUMNS = ["movie_id", "title", "certification"]

# Which country's rating board to read. See the module docstring.
COUNTRY = "US"

# How many refusals in a row before abandoning the run.
THROTTLE_GIVE_UP = 12
THROTTLED = object()

# TMDb release types, most authoritative first. A title can carry a different
# certification per release window; the theatrical rating is the one audiences
# know, and the TV rating is the loosest, so prefer in that order rather than
# taking whichever element happens to come back first.
RELEASE_TYPE_PRIORITY = [3, 2, 4, 5, 1, 6]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "nexus-recommender/1.0 (academic project)"})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )))
    return session


def _pick_certification(results: list[dict]) -> str:
    """Return the best US certification from a /release_dates payload."""
    for entry in results:
        if str(entry.get("iso_3166_1") or "").upper() != COUNTRY:
            continue
        dates = entry.get("release_dates") or []
        ranked = sorted(
            (d for d in dates if str(d.get("certification") or "").strip()),
            key=lambda d: (
                RELEASE_TYPE_PRIORITY.index(d.get("type"))
                if d.get("type") in RELEASE_TYPE_PRIORITY
                else len(RELEASE_TYPE_PRIORITY)
            ),
        )
        if ranked:
            return str(ranked[0]["certification"]).strip().upper()
    return ""


def fetch_certification(session: requests.Session, movie_id: str, key: str):
    """Return the US certification string (possibly empty), or THROTTLED."""
    try:
        response = session.get(f"{API}/movie/{movie_id}/release_dates",
                               params={"api_key": key}, timeout=25)
        if response.status_code == 404:
            return ""
        if response.status_code in (401, 429, 503):
            return THROTTLED
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return THROTTLED

    return _pick_certification(payload.get("results") or [])


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {
        str(row.movie_id): {
            "movie_id": str(row.movie_id),
            "title": "" if pd.isna(row.title) else str(row.title),
            # An empty certification is a cached answer ("TMDb has no US rating
            # for this title"), not a gap to retry.
            "certification": "" if pd.isna(row.certification) else str(row.certification),
        }
        for row in frame.itertuples(index=False)
    }


def save(rows: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows.values()), columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TMDb US content ratings.")
    parser.add_argument("--movies", type=Path, default=DEFAULT_MOVIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many lookups (0 = all missing)")
    parser.add_argument("--rate", type=float, default=0.1,
                        help="Seconds between requests per worker (default 0.1)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent lookups (default 8). Serial throughput is "
                             "bounded by round-trip latency, not by --rate: one "
                             "worker managed ~100/min against a 22k backlog, which "
                             "is hours. TMDb tolerates this fan-out comfortably.")
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

    print(f"{len(movies):,} movies, {len(rows):,} already have a rating, "
          f"{len(todo):,} to fetch")
    if todo.empty:
        _summarise(rows)
        return
    workers = max(1, args.workers)
    print(f"{workers} workers; a serial run managed ~100/min, so expect roughly "
          f"{len(todo) / (100.0 * workers):.0f} min")
    print()

    # One session per worker: requests.Session is not documented as thread-safe,
    # and sharing one across the pool intermittently corrupted responses into
    # spurious THROTTLED results (which are never cached, so the work was simply
    # redone on the next run - slow rather than wrong, but still wasted).
    sessions = [_session() for _ in range(workers)]
    fetched = rated = throttled_chunks = 0
    # Chunked rather than one big submit, so progress is saved as it goes and an
    # interrupted run resumes from the last chunk instead of the start.
    chunk_size = workers * 25
    records = list(todo.itertuples(index=False))

    def _lookup(task):
        position, record = task
        return record, fetch_certification(
            sessions[position % workers], str(record.id), key
        )

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
                        "certification": result,
                    }
                    fetched += 1
                    if result:
                        rated += 1

            save(rows, args.out)
            print(f"  {min(start + chunk_size, len(records)):,}/{len(records):,} "
                  f"done, {rated:,} rated", flush=True)

            # Every lookup in the chunk refused: back off rather than hammering.
            # Nothing was cached for those, so a re-run picks them up again.
            if throttled_in_chunk == len(chunk):
                throttled_chunks += 1
                if throttled_chunks >= THROTTLE_GIVE_UP:
                    print(f"Stopping: {throttled_chunks} fully-refused batches. "
                          "Nothing was cached for those, so a re-run resumes.")
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
    print("  python -m embeddings.build_embeddings")
    print("  python -m embeddings.build_qdrant_collection")
    print("  python -m scripts.load_catalog_mysql")


def _summarise(rows: dict[str, dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows.values():
        label = row["certification"] or "(none)"
        counts[label] = counts.get(label, 0) + 1
    print(f"  {len(rows):,} cached, "
          f"{sum(1 for r in rows.values() if r['certification']):,} with a rating")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {label:<10} {count:,}")


if __name__ == "__main__":
    main()
