"""Fetch TV shows from the live TMDb API.

The catalogue had films but no television, so "Breaking Bad", "Better Call
Saul" and every Marvel series returned nothing - or worse, returned a
documentary *about* the show. El Camino was findable only because it is a film.

Output deliberately mirrors the column names of tmdb_movies.csv -
id, title, overview, genres, release_date, popularity, vote_average,
vote_count, poster_url - so `normalize_movies` consumes it unchanged and this
is a config entry rather than a new normalizer. TMDb calls the fields `name`
and `first_air_date` on /discover/tv; they are renamed on the way out.

Three details carried over from the movie fetcher, for the same reasons:

* /discover caps pagination at 500 pages, so requests are sliced by first-air
  year and each slice stays well under the cap.
* A row with no overview embeds to noise and one with no poster is a blank
  card, so both are required.
* --min-votes defaults lower than the movie fetcher's 100. Television accrues
  far fewer TMDb votes than film - at 100 the whole of TMDb holds only a few
  hundred series - and the same Western-voting-base skew that hid Tamil cinema
  applies harder here.

Usage:
    python -m scripts.fetch_tmdb_shows
    python -m scripts.fetch_tmdb_shows --min-votes 25 --from-year 1990
    python -m scripts.fetch_tmdb_shows --original-language ta --min-votes 10 --merge
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_shows.csv"

API = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# TMDb refuses page numbers above this on /discover.
MAX_PAGE = 500

# The movie fetcher's schema, which normalize_movies already reads.
OUTPUT_COLUMNS = [
    "id", "title", "overview", "genres", "release_date",
    "popularity", "vote_average", "vote_count", "poster_url",
]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "nexus-recommender/1.0 (academic project)"})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )))
    return session


def genre_map(session: requests.Session, key: str) -> dict[int, str]:
    """TMDb returns genre ids on /discover, not names.

    The television genre list is its own endpoint and does not match the film
    one - television has "Sci-Fi & Fantasy" and "Action & Adventure" where film
    has them split - so this must not reuse /genre/movie/list.
    """
    response = session.get(f"{API}/genre/tv/list", params={"api_key": key}, timeout=25)
    response.raise_for_status()
    return {int(g["id"]): str(g["name"]) for g in response.json().get("genres", [])}


def fetch_slice(
    session: requests.Session,
    key: str,
    label: str,
    slice_params: dict,
    min_votes: int,
    genres: dict[int, str],
    rate: float,
) -> list[dict]:
    """One /discover/tv sweep over a slice of the catalogue."""
    rows: list[dict] = []
    page = 1
    total_pages = 1

    while page <= min(total_pages, MAX_PAGE):
        started = time.monotonic()
        response = session.get(f"{API}/discover/tv", params={
            "api_key": key,
            "vote_count.gte": min_votes,
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "page": page,
            **slice_params,
        }, timeout=30)
        if response.status_code != 200:
            print(f"  {label}: HTTP {response.status_code}, stopping this slice")
            break

        payload = response.json()
        total_pages = int(payload.get("total_pages") or 1)

        for item in payload.get("results", []):
            overview = str(item.get("overview") or "").strip()
            poster = item.get("poster_path")
            if len(overview) < 20 or not poster:
                continue
            rows.append({
                # TMDb names these `name` and `first_air_date` for television.
                "id": item.get("id"),
                "title": str(item.get("name") or "").strip(),
                "overview": overview,
                "genres": ", ".join(
                    genres.get(int(g), "") for g in (item.get("genre_ids") or [])
                ).strip(", "),
                "release_date": item.get("first_air_date") or "",
                "popularity": item.get("popularity") or "",
                "vote_average": item.get("vote_average") or "",
                "vote_count": item.get("vote_count") or "",
                "poster_url": f"{POSTER_BASE}{poster}",
            })

        page += 1
        elapsed = time.monotonic() - started
        if elapsed < rate:
            time.sleep(rate - elapsed)

    if total_pages > MAX_PAGE:
        print(f"  {label}: {total_pages} pages available, capped at {MAX_PAGE}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TV shows from the TMDb API.")
    parser.add_argument("--min-votes", type=int, default=25,
                        help="Minimum vote_count (default 25). Lower than the "
                             "movie fetcher's 100: television accrues far fewer "
                             "TMDb votes, and 100 would leave only a few hundred "
                             "series in the whole of TMDb.")
    parser.add_argument("--from-year", type=int, default=1970)
    parser.add_argument("--to-year", type=int, default=2026)
    parser.add_argument("--original-language", action="append", dest="languages",
                        metavar="CODE",
                        help="ISO-639-1 code; repeatable. Sweeps by language "
                             "instead of by year, for a regional top-up.")
    parser.add_argument("--merge", action="store_true",
                        help="Fold results into the existing --out file instead of "
                             "replacing it. Existing rows win, so no TMDb id moves.")
    parser.add_argument("--rate", type=float, default=0.1,
                        help="Seconds between requests (default 0.1, i.e. 10/sec)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    key = os.getenv("TMDB_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "TMDB_API_KEY is not set. Put it in .env (the v3 API key, not the "
            "v4 Read Access Token)."
        )

    session = _session()
    genres = genre_map(session, key)
    print(f"tv genre map: {len(genres)} genres")

    if args.languages:
        slices = [(code, {"with_original_language": code}) for code in args.languages]
        print(f"fetching {len(slices)} languages, vote_count >= {args.min_votes}")
    else:
        slices = [(str(y), {"first_air_date_year": y})
                  for y in range(args.from_year, args.to_year + 1)]
        print(f"fetching {args.from_year}-{args.to_year}, "
              f"vote_count >= {args.min_votes}")
    print()

    all_rows: list[dict] = []
    for label, slice_params in slices:
        rows = fetch_slice(session, key, label, slice_params,
                           args.min_votes, genres, args.rate)
        all_rows.extend(rows)
        if rows:
            print(f"  {label}: {len(rows):>4} shows   (running total {len(all_rows):,})")

    frame = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    before = len(frame)
    frame = frame.drop_duplicates(subset=["id"])
    frame = frame[frame["title"].astype(str).str.strip() != ""]

    previous = 0
    if args.merge and args.out.exists():
        # dtype=str on both sides: a merge that lets pandas infer types rewrites
        # an int id column as floats the moment one value is blank, turning
        # every "1396" into "1396.0".
        existing = pd.read_csv(args.out, dtype=str, keep_default_na=False)
        previous = len(existing)
        frame = pd.concat([existing, frame.astype(str)], ignore_index=True)
        frame = frame.drop_duplicates(subset=["id"], keep="first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print()
    print(f"wrote {args.out}")
    if args.merge:
        print(f"  {previous:,} already on disk + {len(frame) - previous:,} new "
              f"(of {before:,} fetched)")
    print(f"  {len(frame):,} shows, all with an overview and a poster")
    print()
    print("Register it in config/datasets.yaml, then run the full rebuild:")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_embeddings")
    print("  python -m embeddings.build_faiss_index")
    print("  python -m embeddings.build_qdrant_collection")
    print("  python -m scripts.load_catalog_mysql")
    print("  python -m scripts.seed_demo_interactions")
    print("  python -m scripts.rebuild_ema_profiles")
    print()
    print("Stop the API server first - embedded Qdrant takes a single-process lock.")


if __name__ == "__main__":
    main()
