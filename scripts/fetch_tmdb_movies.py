"""Fetch movies from the live TMDb API, replacing the 2017 Kaggle snapshot.

The catalogue's movie rows came from "TMDB 5000 Movie Dataset", a static export
someone uploaded to Kaggle: 4,803 rows, frozen in 2017. TMDb itself carries
1.18 million titles, so the ceiling was the file, not the source.

This pulls from /discover/movie, which returns `overview` and `poster_path` in
the same response - so content and cover art arrive together and no separate
poster lookup is needed.

Two details that matter:

* /discover caps pagination at 500 pages per query, so a filter matching more
  than 10,000 titles silently truncates. Requests are therefore sliced by
  release year, keeping each slice well under the cap.
* Most of the 1.18M are shorts, unreleased entries and rows with no overview
  and two votes. --min-votes filters to titles that were actually released and
  reviewed; padding the catalogue with the rest would make search worse, not
  better.

The output keeps the same column names as the Kaggle CSV it replaces, and the
same `id` column of TMDb ids, so the existing credits file still attaches cast
and director to the titles it covers with no normalizer change.

Usage:
    python -m scripts.fetch_tmdb_movies --min-votes 100 --from-year 1970
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
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "tmdb" / "tmdb_movies.csv"

API = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# TMDb refuses page numbers above this on /discover.
MAX_PAGE = 500

OUTPUT_COLUMNS = [
    "id", "title", "overview", "genres", "release_date",
    "popularity", "vote_average", "vote_count", "poster_url",
]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "nexus-recommender/1.0 (academic project)"})
    # Transient resets do occur against this host; retrying in the adapter
    # keeps them from aborting a multi-thousand-request run.
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )))
    return session


def genre_map(session: requests.Session, key: str) -> dict[int, str]:
    """TMDb returns genre ids on /discover, not names."""
    response = session.get(f"{API}/genre/movie/list", params={"api_key": key}, timeout=25)
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
    """One /discover sweep over a slice of the catalogue.

    A slice is whatever keeps the result set under the 500-page cap: a release
    year for the general sweep, or an original language for the regional
    top-up, where the whole pool is a few hundred titles and needs no slicing.
    """
    rows: list[dict] = []
    page = 1
    total_pages = 1

    while page <= min(total_pages, MAX_PAGE):
        started = time.monotonic()
        response = session.get(f"{API}/discover/movie", params={
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
            # A row with no overview embeds to noise, and one with no poster is
            # the thing this exercise was meant to fix. Require both.
            if len(overview) < 20 or not poster:
                continue
            rows.append({
                "id": item.get("id"),
                "title": str(item.get("title") or "").strip(),
                "overview": overview,
                "genres": ", ".join(
                    genres.get(int(g), "") for g in (item.get("genre_ids") or [])
                ).strip(", "),
                "release_date": item.get("release_date") or "",
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
    parser = argparse.ArgumentParser(description="Fetch movies from the TMDb API.")
    parser.add_argument("--min-votes", type=int, default=100,
                        help="Minimum vote_count. 100 keeps released, reviewed "
                             "titles and drops the long tail (default 100)")
    parser.add_argument("--from-year", type=int, default=1970)
    parser.add_argument("--to-year", type=int, default=2026)
    parser.add_argument("--rate", type=float, default=0.1,
                        help="Seconds between requests (default 0.1, i.e. 10/sec)")
    parser.add_argument("--original-language", action="append", dest="languages",
                        metavar="CODE",
                        help="ISO-639-1 code; repeatable. Sweeps by language "
                             "instead of by year, for a regional top-up. Use with "
                             "a lower --min-votes and --merge.")
    parser.add_argument("--merge", action="store_true",
                        help="Fold results into the existing --out file instead of "
                             "replacing it. Existing rows win, so no TMDb id moves.")
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
    print(f"genre map: {len(genres)} genres")
    print(f"fetching {args.from_year}-{args.to_year}, vote_count >= {args.min_votes}")
    print()

    if args.languages:
        slices = [(code, {"with_original_language": code}) for code in args.languages]
    else:
        slices = [(str(y), {"primary_release_year": y})
                  for y in range(args.from_year, args.to_year + 1)]

    all_rows: list[dict] = []
    for label, slice_params in slices:
        rows = fetch_slice(session, key, label, slice_params,
                           args.min_votes, genres, args.rate)
        all_rows.extend(rows)
        if rows:
            print(f"  {label}: {len(rows):>4} titles   (running total {len(all_rows):,})")

    frame = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    before = len(frame)
    frame = frame.drop_duplicates(subset=["id"])
    frame = frame[frame["title"].astype(str).str.strip() != ""]

    previous = 0
    if args.merge and args.out.exists():
        # dtype=str on both sides. A merge that lets pandas infer types will
        # rewrite an int id column as floats the moment one value is blank,
        # turning every "550" into "550.0" - the same way a bare-year column
        # broke the book catalogue's release dates.
        existing = pd.read_csv(args.out, dtype=str, keep_default_na=False)
        previous = len(existing)
        fetched = frame.astype(str)
        frame = pd.concat([existing, fetched], ignore_index=True)
        # Existing rows win, so no TMDb id moves and nothing downstream is pruned.
        frame = frame.drop_duplicates(subset=["id"], keep="first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print()
    print(f"wrote {args.out}")
    if args.merge:
        print(f"  {previous:,} already on disk + {len(frame) - previous:,} new "
              f"(of {before:,} fetched; the rest were already present)")
    print(f"  {len(frame):,} titles total")
    print(f"  all have an overview and a poster URL")
    print()
    print("Point config/datasets.yaml movies.path at this file, then:")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_embeddings")
    print("  python -m embeddings.build_qdrant_collection")


if __name__ == "__main__":
    main()
