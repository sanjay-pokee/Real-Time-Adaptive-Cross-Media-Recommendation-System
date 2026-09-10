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
items and a partial run should cover those first. Answers are cached in
data/processed/cover_art_cache.csv and written incrementally, so an interrupted
run keeps its work and a re-run resumes rather than re-querying.

Requests are serial and paced. A first version ran eight concurrent workers and
iTunes refused all but the first few dozen: 6,000 lookups produced 53 covers,
where the first 40 alone had produced 40.

Usage:
    python -m scripts.backfill_cover_art --content-type music --limit 1200
    python -m scripts.backfill_cover_art --content-type movie --limit 4803
    python -m scripts.backfill_cover_art --purge-unanswered

Then rebuild so the catalogue picks the cache up:
    python -m preprocessing.build_content_catalog
    python -m embeddings.build_qdrant_collection
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "cover_art_cache.csv"

ITUNES_SEARCH = "https://itunes.apple.com/search"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

CACHE_COLUMNS = ["global_id", "image_url"]

# How many refusals in a row before abandoning the run.
THROTTLE_GIVE_UP = 12

# A throttled or failed request is NOT the same answer as "this track has no
# artwork", and conflating them poisons the cache: an earlier run recorded
# 5,947 rate-limited lookups as permanent misses, which a re-run then skipped.
THROTTLED = object()

NEWLINE = "\n"


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        # Identify the caller. Anonymous floods are what gets an address throttled.
        "User-Agent": "nexus-recommender/1.0 (academic project; cover-art backfill)",
    })
    return session


def _itunes_artwork(session: requests.Session, title: str, artist: str):
    """Album art for one track.

    Returns a URL, "" for a genuine miss, or THROTTLED when the request was
    refused or errored, so the caller can avoid caching a non-answer.
    """
    term = " ".join(part for part in (artist, title) if part).strip()
    if not term:
        return ""
    try:
        response = session.get(
            ITUNES_SEARCH,
            params={"term": term, "media": "music", "entity": "song", "limit": 1},
            timeout=12,
        )
        if response.status_code in (403, 429, 503):
            return THROTTLED
        response.raise_for_status()
        results = response.json().get("results") or []
    except (requests.RequestException, ValueError):
        return THROTTLED
    if not results:
        return ""
    art = str(results[0].get("artworkUrl100") or "")
    # The 100px thumbnail URL upgrades to any size by substitution.
    return art.replace("100x100bb", "400x400bb") if art else ""


def _tmdb_poster(session: requests.Session, source_id: str, api_key: str):
    """Poster for one movie, "" for a genuine miss, THROTTLED on refusal."""
    try:
        response = session.get(
            f"https://api.themoviedb.org/3/movie/{source_id}",
            params={"api_key": api_key},
            timeout=12,
        )
        if response.status_code == 404:
            return ""
        if response.status_code in (401, 429, 503):
            return THROTTLED
        response.raise_for_status()
        path = response.json().get("poster_path")
    except (requests.RequestException, ValueError):
        return THROTTLED
    return f"{TMDB_IMAGE_BASE}{path}" if path else ""


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    frame = pd.read_csv(CACHE_PATH)
    return {
        str(row.global_id): ("" if pd.isna(row.image_url) else str(row.image_url))
        for row in frame.itertuples(index=False)
    }


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"global_id": key, "image_url": value} for key, value in cache.items()],
        columns=CACHE_COLUMNS,
    )
    frame.to_csv(CACHE_PATH, index=False)


def purge_unanswered() -> int:
    """Drop cached blanks, so previously throttled rows get retried.

    Needed once, because an earlier version could not tell a refusal from a
    genuine miss and wrote both as "". It also discards real misses, which only
    costs one further lookup each.
    """
    if not CACHE_PATH.exists():
        return 0
    frame = pd.read_csv(CACHE_PATH)
    empty = frame["image_url"].fillna("").astype(str).str.strip() == ""
    removed = int(empty.sum())
    frame[~empty].to_csv(CACHE_PATH, index=False)
    return removed


def backfill(content_type: str, limit: int, rate: int) -> None:
    if not CATALOG_PATH.exists():
        raise SystemExit(
            f"No catalogue at {CATALOG_PATH}." + NEWLINE
            + "Run: python -m preprocessing.build_content_catalog"
        )
    catalog = pd.read_csv(CATALOG_PATH, low_memory=False)
    rows = catalog[catalog["content_type"] == content_type].copy()
    if rows.empty:
        raise SystemExit(f"No {content_type} rows in the catalogue.")

    cache = load_cache()
    # Skip anything already answered, hit or genuine miss. Throttled lookups are
    # never written, so they come round again on the next run.
    todo = rows[~rows["global_id"].astype(str).isin(cache)]
    todo = todo.sort_values("popularity", ascending=False, na_position="last")
    todo = todo.head(limit)

    print(f"{content_type}: {len(rows):,} rows, {len(cache):,} cached, "
          f"{len(todo):,} to look up")
    if todo.empty:
        return

    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if content_type == "movie" and not api_key:
        raise SystemExit(
            "Movie posters need a TMDb API key." + NEWLINE
            + "Create one at themoviedb.org, then set TMDB_API_KEY yourself."
        )

    session = _session()
    interval = 60.0 / max(rate, 1)
    records = list(todo.itertuples(index=False))
    print(f"pacing at {rate}/min, roughly "
          f"{len(records) * interval / 60.0:.0f} min for this batch")
    print()

    found = 0
    answered = 0
    throttled_streak = 0

    try:
        for index, record in enumerate(records):
            started = time.monotonic()

            if content_type == "movie":
                result = _tmdb_poster(session, str(record.source_id), api_key)
            else:
                result = _itunes_artwork(
                    session, str(record.title or ""), str(record.creators or "")
                )

            if result is THROTTLED:
                throttled_streak += 1
                if throttled_streak >= THROTTLE_GIVE_UP:
                    print(f"Stopping: {throttled_streak} refusals in a row, so the "
                          "API is rate-limiting this address.")
                    print("None of those were cached, so a re-run resumes from here. "
                          f"Try a lower --rate than {rate}, or wait a while.")
                    break
                # Escalating back-off gives the limiter room to reset.
                time.sleep(min(2 ** throttled_streak, 60))
                continue

            throttled_streak = 0
            cache[str(record.global_id)] = result
            answered += 1
            if result:
                found += 1

            if answered % 100 == 0:
                # Written as we go, so an interrupted run is not wasted.
                save_cache(cache)
                print(f"  {index + 1:,}/{len(records):,} processed, {found:,} found")

            elapsed = time.monotonic() - started
            if elapsed < interval and index + 1 < len(records):
                time.sleep(interval - elapsed)
    except KeyboardInterrupt:
        print("interrupted")

    save_cache(cache)
    print()
    print(f"wrote {CACHE_PATH}")
    print(f"  answered {answered:,}, found {found:,} "
          f"({found / max(answered, 1) * 100:.0f}%)")
    print()
    print("Rebuild to apply:")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_qdrant_collection")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill cover art for the catalogue.")
    parser.add_argument("--content-type", choices=["music", "movie"])
    parser.add_argument("--limit", type=int, default=1200,
                        help="Rows to look up this run, most popular first (default 1200)")
    parser.add_argument("--rate", type=int, default=20,
                        help="Requests per minute. iTunes tolerates roughly 20; going "
                             "faster gets the address refused (default 20)")
    parser.add_argument("--purge-unanswered", action="store_true",
                        help="Drop cached blanks so throttled rows are retried, then exit.")
    args = parser.parse_args()

    if args.purge_unanswered:
        print(f"dropped {purge_unanswered():,} unanswered cache rows")
        return

    if not args.content_type:
        parser.error("--content-type is required unless --purge-unanswered is given")

    backfill(args.content_type, args.limit, args.rate)


if __name__ == "__main__":
    main()
