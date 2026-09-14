"""Fetch franchise and keyword metadata for the films and shows in the catalogue.

Searching "avengers" returned only titles with the word in them. Iron Man,
Black Panther and Doctor Strange were all in the catalogue and none of them
surfaced, because nothing recorded that they belong together.

Two fields fix most of that, and neither is on /discover - both need a per-title
lookup, which is why this exists as its own resumable pass:

  belongs_to_collection   TMDb's franchise grouping: "The Avengers Collection",
                          "Iron Man Collection". Films only; TMDb has no
                          equivalent for television.
  keywords                free tags: "superhero", "based on comic",
                          "marvel cinematic universe (mcu)", "time travel".

A caveat measured before building this, so nobody expects more than it gives:
the universe-level keyword is applied inconsistently. Iron Man and Black
Panther carry "marvel cinematic universe (mcu)"; Avengers: Endgame does not.
So collections reliably link a title to its own sequels, and keywords link
themes, but neither reliably links every film of a shared universe. Anything
claiming otherwise would be guessing from text.

Written incrementally and keyed by id, so an interrupted run resumes and a
re-run only fetches what is missing.

Usage:
    python -m scripts.fetch_tmdb_franchises
    python -m scripts.fetch_tmdb_franchises --top 12000 --workers 8
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
DEFAULT_SHOWS = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_shows.csv"
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_franchises.csv"

API = "https://api.themoviedb.org/3"
OUTPUT_COLUMNS = ["media_id", "media", "collection", "keywords"]

# Sentinel: a throttled lookup is never cached, so the next run retries it.
THROTTLED = object()

# Keyword tags that describe the *file* rather than the story. They add noise to
# the searchable text without ever being what someone means.
KEYWORD_NOISE = {
    "aftercreditsstinger", "duringcreditsstinger", "woman director",
    "live action remake", "3d", "imax",
}
# A popular title carries 30+ tags; past the first dozen they are long-tail
# noise, and the whole set would outweigh the synopsis in the embedding.
MAX_KEYWORDS = 12


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "nexus-recommender/1.0 (academic project)"})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, connect=4, read=4, backoff_factor=1.5,
        status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"],
    )))
    return session


def fetch_one(session: requests.Session, media: str, media_id: str, key: str):
    """Collection name and keywords for one title, or THROTTLED to retry later."""
    endpoint = "movie" if media == "movie" else "tv"
    try:
        response = session.get(
            f"{API}/{endpoint}/{media_id}",
            params={"api_key": key, "append_to_response": "keywords"},
            timeout=25,
        )
    except requests.RequestException:
        return THROTTLED

    if response.status_code == 429:
        return THROTTLED
    if response.status_code != 200:
        # 404 is a real answer - the id is gone from TMDb - so cache it as empty
        # rather than retrying it on every future run.
        return ("", "")

    payload = response.json()
    collection = str((payload.get("belongs_to_collection") or {}).get("name") or "").strip()

    # Films return keywords under "keywords"; television returns them under
    # "results". Asking for the wrong one yields an empty list rather than an
    # error, which would have looked like "television has no keywords".
    block = payload.get("keywords") or {}
    raw = block.get("keywords") if media == "movie" else block.get("results")
    names = []
    for entry in raw or []:
        name = str(entry.get("name") or "").strip().lower()
        if name and name not in KEYWORD_NOISE:
            names.append(name)
        if len(names) >= MAX_KEYWORDS:
            break

    return (collection, ", ".join(names))


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        f"{row.media}:{row.media_id}": {
            "media_id": str(row.media_id),
            "media": str(row.media),
            # Empty is a cached answer ("this title has no collection"), not a
            # gap to retry.
            "collection": str(row.collection),
            "keywords": str(row.keywords),
        }
        for row in frame.itertuples(index=False)
    }


def save(rows: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows.values()), columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def _todo(rows: dict[str, dict], top: int | None) -> list[tuple[str, str]]:
    """(media, id) pairs still missing, most popular first."""
    tasks: list[tuple[float, str, str]] = []
    for media, source in (("movie", DEFAULT_MOVIES), ("show", DEFAULT_SHOWS)):
        if not source.exists():
            continue
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        popularity = pd.to_numeric(frame.get("popularity"), errors="coerce").fillna(0.0)
        for media_id, pop in zip(frame["id"], popularity):
            if f"{media}:{media_id}" not in rows:
                tasks.append((float(pop), media, str(media_id)))

    # Most popular first, so a truncated run still covers every franchise a
    # person is likely to type.
    tasks.sort(key=lambda t: t[0], reverse=True)
    if top:
        tasks = tasks[:top]
    return [(media, media_id) for _, media, media_id in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch collection and keyword metadata for catalogue titles."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent lookups (default 8)")
    parser.add_argument("--top", type=int, default=None,
                        help="Only fetch the N most popular titles still missing. "
                             "Franchise queries are about well-known titles, so a "
                             "truncated run covers the useful head.")
    args = parser.parse_args()

    key = os.getenv("TMDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("TMDB_API_KEY is not set. Put it in .env (the v3 key).")

    rows = load_existing(args.out)
    todo = _todo(rows, args.top)
    print(f"{len(rows):,} already cached; {len(todo):,} to fetch")
    if not todo:
        print("nothing to do")
        return

    workers = max(1, args.workers)
    print(f"{workers} workers; expect roughly {len(todo) / (100.0 * workers):.0f} min")
    print()

    # One session per worker: requests.Session is not documented as thread-safe.
    sessions = [_session() for _ in range(workers)]
    fetched = with_collection = 0
    chunk_size = workers * 25

    def _lookup(task):
        position, (media, media_id) = task
        return media, media_id, fetch_one(sessions[position % workers], media, media_id, key)

    try:
        for start in range(0, len(todo), chunk_size):
            chunk = list(enumerate(todo[start : start + chunk_size]))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for media, media_id, result in pool.map(_lookup, chunk):
                    if result is THROTTLED:
                        continue
                    collection, keywords = result
                    rows[f"{media}:{media_id}"] = {
                        "media_id": media_id,
                        "media": media,
                        "collection": collection,
                        "keywords": keywords,
                    }
                    fetched += 1
                    if collection:
                        with_collection += 1
            save(rows, args.out)
            print(f"  {start + len(chunk):,}/{len(todo):,} fetched "
                  f"({with_collection:,} in a collection)", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        save(rows, args.out)
        print("\ninterrupted; progress saved - re-run to resume")
        return

    save(rows, args.out)
    print()
    print(f"wrote {args.out}")
    print(f"  {len(rows):,} titles, {with_collection:,} of this run in a collection")
    print()
    print("Rebuild to apply (stop the API server first):")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_embeddings")
    print("  python -m embeddings.build_faiss_index")
    print("  python -m embeddings.build_qdrant_collection")
    print("  python -m scripts.load_catalog_mysql")


if __name__ == "__main__":
    main()
