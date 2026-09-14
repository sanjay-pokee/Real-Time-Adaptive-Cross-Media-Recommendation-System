"""Backfill cover art for catalogue rows whose source dataset carries none.

Books and the Amazon-sourced verticals already ship an image URL, so this only
has to cover the two that do not:

* music  - the Spotify export has no artwork column. Two sources, --provider:

           `spotify` (fast, but see the blocker): the music rows ARE a Spotify
           export, so `source_id` is already a Spotify track id. /v1/tracks takes
           50 ids per request, making the whole catalogue ~570 requests instead
           of 28,352, with no title/artist matching to get wrong. Needs
           SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env.

           BLOCKER: Spotify now requires the app owner to hold an active Premium
           subscription. Client credentials authenticate fine (token 200), then
           every Web API endpoint - tracks, albums, search, single or batch -
           returns 403 "Active premium subscription required for the owner of the
           app". There is no free tier for this any more, so this provider is
           unusable without a paid account. Kept because the code is correct and
           the id-exact approach is the right one the moment that account exists.

           This file previously ruled that out, on the grounds that Spotify's
           Developer Terms forbid ingesting Spotify Content into a
           machine-learning model "which is exactly what this catalogue feeds".
           That is not what happens: only the image URL is read, and it is a
           display field. `embedding_text` for music is title + artist + genre +
           subgenre, all of it from the Kaggle CSV, and the URL appears in zero
           of the catalogue's embedding_text or metadata_text values - checked,
           not assumed. Nothing fetched here reaches SBERT or LightGCN. Spotify
           asks for attribution when their content is displayed; revisit this if
           the project is ever published or commercialised.

           `lastfm`: title/artist matched on Last.fm's track.getInfo. Needs a
           free LASTFM_API_KEY in .env, instant to obtain. Roughly 5 requests a
           second against iTunes' ~20 a minute, so it clears a 15k backlog in
           about an hour rather than a day - but it answers for fewer tracks:
           55% of the most popular uncovered rows against iTunes' 92%, because
           Last.fm matches on artist and title and its album database is thinner
           for obscure releases. Use it for volume, iTunes for coverage.

           `itunes` (default): title/artist matched on the iTunes Search API.
           Free, no key, no OAuth, but capped near 20 lookups/minute.
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
    python -m scripts.backfill_cover_art --content-type music --provider lastfm --limit 20000
    python -m scripts.backfill_cover_art --content-type music --provider spotify --limit 30000
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
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# The project keeps secrets in a gitignored .env that backend.settings already
# reads. Load it here too, so TMDB_API_KEY does not have to be exported into
# the shell separately just to run this script.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "cover_art_cache.csv"

ITUNES_SEARCH = "https://itunes.apple.com/search"
LASTFM_API = "https://ws.audioscrobbler.com/2.0/"
# Last.fm returns this image rather than omitting the field when it has no
# art for a track. It is a grey star, identical for every miss.
LASTFM_PLACEHOLDER = "2a96cbd8b46e442fc41c2b86b821562f"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_TRACKS_URL = "https://api.spotify.com/v1/tracks"
# /v1/tracks accepts up to 50 ids per request, which is the whole reason this
# provider exists: 28,352 tracks is 568 requests rather than 28,352.
SPOTIFY_BATCH = 50

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
    # Transient connection resets do happen over a run of several thousand
    # requests - one was observed against api.themoviedb.org while verifying
    # the key, and the very next attempt succeeded. Retrying inside the adapter
    # keeps those invisible, instead of letting them look like refusals and
    # burn the throttle back-off.
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
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


def _lastfm_artwork(session: requests.Session, title: str, artist: str, api_key: str):
    """Album art for one track, via Last.fm's track.getInfo.

    Same contract as the iTunes lookup: a URL, "" for a genuine miss, or
    THROTTLED when the request was refused, so a non-answer is never cached.

    Last.fm allows roughly 5 requests a second against one key, where iTunes
    tolerates about 20 a minute, so this is the provider that can finish the
    catalogue in an hour rather than a day.
    """
    artist = (artist or "").split(",")[0].strip()
    if not artist or not title:
        return ""
    try:
        response = session.get(
            LASTFM_API,
            params={
                "method": "track.getInfo",
                "api_key": api_key,
                "artist": artist,
                "track": title,
                "autocorrect": 1,
                "format": "json",
            },
            timeout=15,
        )
        if response.status_code in (403, 429, 503):
            return THROTTLED
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return THROTTLED

    # Last.fm reports a bad key or a rate limit as a 200 with an error body.
    if "error" in payload:
        code = payload.get("error")
        if code in (8, 11, 16, 29):      # transient / rate-limited
            return THROTTLED
        if code in (10, 26):             # invalid or suspended key: not retryable
            raise SystemExit(
                f"Last.fm rejected the API key ({payload.get('message')}). "
                "Check LASTFM_API_KEY in .env."
            )
        return ""

    images = ((payload.get("track") or {}).get("album") or {}).get("image") or []
    by_size = {img.get("size"): str(img.get("#text") or "") for img in images}
    url = by_size.get("extralarge") or by_size.get("large") or by_size.get("medium") or ""
    # Last.fm serves a grey star placeholder rather than omitting the field when
    # it has no art. Caching that would fill the UI with identical placeholders,
    # which is worse than the gradient it would be replacing.
    if not url or LASTFM_PLACEHOLDER in url:
        return ""
    return url


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


def _spotify_token(session: requests.Session) -> str:
    """Client-credentials token. Needs no user login - this reads public data."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit(
            "Spotify needs SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env."
            + NEWLINE
            + "Create an app at https://developer.spotify.com/dashboard - free, no"
            " review, takes a minute. Add the two values to .env yourself; they are"
            " secrets and belong in the gitignored file, not on a command line."
        )

    response = session.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=25,
    )
    if response.status_code != 200:
        raise SystemExit(
            f"Spotify refused the credentials ({response.status_code}). Check the"
            " client id and secret in .env."
        )
    token = response.json().get("access_token")
    if not token:
        raise SystemExit("Spotify returned no access token.")
    return str(token)


def _spotify_artwork(session: requests.Session, token: str, track_ids: list[str]):
    """Return {track_id: image_url} for up to 50 ids, or THROTTLED.

    The catalogue's music rows are a Spotify export, so `source_id` is already a
    Spotify track id - there is no title/artist matching to get wrong, and a miss
    means the track genuinely has no artwork rather than that the lookup failed.

    Only the image URL is read. It is a display field: it never reaches
    `embedding_text`, so no Spotify content enters the embedding model.
    """
    try:
        response = session.get(
            SPOTIFY_TRACKS_URL,
            params={"ids": ",".join(track_ids)},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if response.status_code == 403:
            # Not a throttle and not worth retrying: Spotify gates the whole Web
            # API on the app owner holding Premium. Fail loudly rather than
            # looping to the give-up threshold with nothing cached.
            raise SystemExit(
                "Spotify refused the request (403): "
                + (response.text or "").strip()[:160]
                + NEWLINE
                + "The Web API now requires the app owner to have an active"
                " Premium subscription. Use --provider itunes instead."
            )
        if response.status_code in (401, 429, 503):
            return THROTTLED
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return THROTTLED

    artwork: dict[str, str] = {}
    for track_id, track in zip(track_ids, payload.get("tracks") or []):
        if not track:
            # An id the catalogue has but Spotify no longer serves. A genuine
            # miss, cached as such so a re-run does not ask again.
            artwork[track_id] = ""
            continue
        images = (track.get("album") or {}).get("images") or []
        # images come widest-first; prefer a middle size over a 640px original.
        chosen = images[1] if len(images) > 1 else (images[0] if images else None)
        artwork[track_id] = str(chosen.get("url") or "") if chosen else ""
    return artwork


def _backfill_music_via_spotify(records, cache: dict[str, str]) -> None:
    """Batch path: 50 tracks per request instead of one lookup per track."""
    session = _session()
    token = _spotify_token(session)

    batches = [
        records[start : start + SPOTIFY_BATCH]
        for start in range(0, len(records), SPOTIFY_BATCH)
    ]
    print(f"{len(records):,} tracks in {len(batches):,} requests of "
          f"up to {SPOTIFY_BATCH}")
    print()

    answered = found = throttled_streak = 0
    try:
        for index, batch in enumerate(batches):
            track_ids = [str(record.source_id) for record in batch]
            result = _spotify_artwork(session, token, track_ids)

            if result is THROTTLED:
                throttled_streak += 1
                if throttled_streak >= THROTTLE_GIVE_UP:
                    print(f"Stopping: {throttled_streak} refusals in a row.")
                    print("None of those were cached, so a re-run resumes here.")
                    break
                # A 401 usually means the hour-long token expired mid-run.
                token = _spotify_token(session)
                time.sleep(min(2 ** throttled_streak, 60))
                continue

            throttled_streak = 0
            for record in batch:
                url = result.get(str(record.source_id), "")
                cache[str(record.global_id)] = url
                answered += 1
                if url:
                    found += 1

            if (index + 1) % 20 == 0:
                save_cache(cache)
                print(f"  {answered:,}/{len(records):,} processed, {found:,} found",
                      flush=True)
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


def backfill(content_type: str, limit: int, rate: int, provider: str = "itunes") -> None:
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

    if content_type == "music" and provider == "spotify":
        # Batched and id-exact, so it does not share the paced per-row loop below.
        _backfill_music_via_spotify(list(todo.itertuples(index=False)), cache)
        return

    lastfm_key = os.getenv("LASTFM_API_KEY", "").strip()
    if provider == "lastfm" and not lastfm_key:
        raise SystemExit(
            "Last.fm needs LASTFM_API_KEY in .env." + NEWLINE
            + "Create one at https://www.last.fm/api/account/create - free, instant."
        )

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
            elif provider == "lastfm":
                result = _lastfm_artwork(
                    session, str(record.title or ""), str(record.creators or ""),
                    lastfm_key,
                )
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
    parser.add_argument("--provider", choices=["itunes", "spotify", "lastfm"], default="itunes",
                        help="Music artwork source. 'spotify' matches on the track id "
                             "the catalogue already carries and fetches 50 per request, "
                             "so the whole catalogue is ~570 requests instead of 28,352 "
                             "paced lookups; needs SPOTIFY_CLIENT_ID/SECRET in .env. "
                             "'itunes' (default) needs no credentials but is title/artist "
                             "matched and capped near 20/min.")
    parser.add_argument("--purge-unanswered", action="store_true",
                        help="Drop cached blanks so throttled rows are retried, then exit.")
    args = parser.parse_args()

    if args.purge_unanswered:
        print(f"dropped {purge_unanswered():,} unanswered cache rows")
        return

    if not args.content_type:
        parser.error("--content-type is required unless --purge-unanswered is given")

    if args.provider in ("spotify", "lastfm") and args.content_type != "music":
        parser.error(f"--provider {args.provider} only applies to --content-type music")

    # Last.fm tolerates roughly 5 requests a second, against iTunes' ~20 a
    # minute, so leaving the iTunes default in place would throw that away.
    if args.provider == "lastfm" and args.rate == parser.get_default("rate"):
        args.rate = 240

    backfill(args.content_type, args.limit, args.rate, args.provider)


if __name__ == "__main__":
    main()
