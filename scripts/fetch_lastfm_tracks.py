"""Fetch current music from the live Last.fm API.

The music catalogue is a static Kaggle export of Spotify tracks that stops at
2020. Movies come from a live TMDb fetch and reach 2026, so "what came out last
month" works for film and returns nothing for music. This closes that gap.

Output deliberately mirrors the Spotify CSV's column names -
track_id, track_name, track_artist, track_album_name,
track_album_release_date, playlist_genre, playlist_subgenre, track_popularity -
so `normalize_music` consumes it unchanged and this is a config entry rather
than a new normalizer.

Where each field comes from:

  track_name / track_artist   tag.getTopTracks, then track.getInfo
  playlist_genre / subgenre   the track's top two Last.fm tags
  track_album_name            track.getInfo -> album.title
  track_popularity            listeners, rescaled - see below
  track_album_release_date    the year tag the track was fetched under

That last one is the reason this fetches by year tag at all. track.getInfo
returns no release date, so asking Last.fm for "top tracks tagged 2025" is what
makes the year knowable; the alternative was every new row having a blank year.
It is the tag's year rather than a verified release date, so a track tagged 2025
that actually shipped in late 2024 will say 2025. For "is there recent music in
here", that is accurate enough; it is not a discography.

Popularity is rescaled to 0-100 to match Spotify's own scale. Last.fm reports
raw listener counts in the millions, and both sources land in the same `music`
content type, so leaving them raw would put every Last.fm track above every
Spotify one in any popularity ordering - the same cross-source defect already
fixed once in the lexical ranker, which a within-type percentile cannot catch
when both sources *are* the same type.

Usage:
    python -m scripts.fetch_lastfm_tracks
    python -m scripts.fetch_lastfm_tracks --years 2024 2025 2026 --per-year 400
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "music" / "lastfm" / "lastfm_tracks.csv"

API = "https://ws.audioscrobbler.com/2.0/"

# The Spotify export's schema, which normalize_music already reads.
OUTPUT_COLUMNS = [
    "track_id",
    "track_name",
    "track_artist",
    "track_album_name",
    "track_album_release_date",
    "playlist_genre",
    "playlist_subgenre",
    "track_popularity",
]

# ---------------------------------------------------------------------------
# Adult-video filter
# ---------------------------------------------------------------------------
# `tag.getTopTracks` for a year tag returns whatever people scrobbled under it,
# and porn sites scrobble their scene uploads. A fetch of 2,806 tracks carried
# 25 of them into the catalogue, every one rated `teen` - music rows have no
# genre text, so the maturity rules had nothing to match and fell through to
# the content type's default. Filtering at ingestion is the fix; the keyword
# tier in preprocessing/audience_tagging.py is only a backstop and catches 14
# of the 25 on its own.
#
# Both patterns were measured against 35,615 known-good music titles (the
# Spotify export plus the cleaned Last.fm set) and the 25 known-bad listings.
# Together: 25/25 caught, 2 false positives - "Blowjob Betty" (Too $hort) and
# "Blow Job", both Spotify rows this filter never sees.

# Terms that do not appear in real track titles. Measured at 0-3 false
# positives each. "hottie" alone caught 6 of the listings.
ADULT_VIDEO_TERMS = re.compile(
    r"gangbang|porn ?star|blow ?job|cream ?pie|cum ?shot|deep ?throat|bukkake"
    r"|fisting|hentai|double penetrat|double vaginal"
    r"|(?<![a-z])dped(?![a-z])"
    r"|(?<![a-z])stepmo(?:m|ther)|(?<![a-z])stepson"
    r"|(?<![a-z])hotties?(?![a-z])|(?<![a-z])busty(?![a-z])"
    r"|\(scene \d|vol\. ?\d+ \(scene|(?<![a-z])s\d+:e\d+"   # scene/episode numbering
    r"|(?<![a-z])airtight(?![a-z])|(?<![a-z])dvp(?![a-z])|dap!"
)

# Words far too common to filter on alone - but a *long* title containing one
# is not a song. Real titles reach 83 characters only via feature credits and
# soundtrack suffixes ("- From Black Panther: Wakanda Forever"), and none of
# those carry these words: this pair scored 0 false positives in 35,615.
ADULT_VIDEO_WEAK_TERMS = re.compile(
    r"(?<![a-z])(?:ass|cocks?|bbcs?|huge|dick|tits|holes|penetrat|balls)(?![a-z])"
)
LONG_TITLE_CHARS = 70


def looks_like_adult_video(title: str) -> bool:
    """True when a 'track' is really a porn scene listing.

    Deliberately title-only. Artist is unreliable here - the performer's name
    is an ordinary personal name, and blocklisting names would age badly.
    """
    lowered = str(title).lower()
    if ADULT_VIDEO_TERMS.search(lowered):
        return True
    return len(str(title)) > LONG_TITLE_CHARS and bool(
        ADULT_VIDEO_WEAK_TERMS.search(lowered)
    )


# Errors Last.fm returns in a 200 body. 8/11/16/29 are transient.
TRANSIENT_ERRORS = {8, 11, 16, 29}
FATAL_ERRORS = {10, 26}


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "nexus-recommender/1.0 (academic project)"})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, connect=4, read=4, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )))
    return session


def _call(session: requests.Session, key: str, method: str, **params):
    """One API call. Returns the payload, or None when it should be skipped."""
    try:
        response = session.get(
            API,
            params={"method": method, "api_key": key, "format": "json", **params},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    if "error" in payload:
        code = payload.get("error")
        if code in FATAL_ERRORS:
            raise SystemExit(
                f"Last.fm rejected the API key ({payload.get('message')}). "
                "Check LASTFM_API_KEY in .env."
            )
        if code in TRANSIENT_ERRORS:
            time.sleep(2)
        return None
    return payload


def _slug(artist: str, title: str) -> str:
    """A stable id, since Last.fm's mbid is absent for most tracks."""
    raw = f"{artist}-{title}".lower()
    return "".join(ch if ch.isalnum() else "-" for ch in raw).strip("-")[:120]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch current tracks from Last.fm.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--years", nargs="+", default=["2021", "2022", "2023", "2024", "2025", "2026"],
                        help="Year tags to pull top tracks for")
    parser.add_argument("--per-year", type=int, default=500,
                        help="Tracks per year tag (Last.fm pages these at 1000 max)")
    parser.add_argument("--rate", type=float, default=0.22,
                        help="Seconds between requests (default ~4.5/sec)")
    parser.add_argument("--keep-adult-listings", action="store_true",
                        help="Skip the adult-video filter. Year tags carry porn "
                             "scene uploads, and music rows have no genre text "
                             "for the maturity rules to catch them with, so they "
                             "land in the catalogue rated `teen`. Only pass this "
                             "if you are auditing what the filter removes.")
    args = parser.parse_args()

    key = os.getenv("LASTFM_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "LASTFM_API_KEY is not set in .env.\n"
            "Create one at https://www.last.fm/api/account/create - free, instant."
        )

    session = _session()

    # 1. Collect (artist, title, year) triples from each year tag.
    seen: dict[tuple[str, str], str] = {}
    for year in args.years:
        payload = _call(session, key, "tag.getTopTracks", tag=year, limit=args.per_year)
        tracks = ((payload or {}).get("tracks") or {}).get("track") or []
        added = 0
        for entry in tracks:
            title = str(entry.get("name") or "").strip()
            artist = str((entry.get("artist") or {}).get("name") or "").strip()
            if not title or not artist:
                continue
            pair = (artist.lower(), title.lower())
            if pair in seen:
                continue
            seen[pair] = year
            added += 1
        print(f"  tag {year}: {added} new tracks ({len(seen)} total)", flush=True)
        time.sleep(args.rate)

    if not seen:
        raise SystemExit("Last.fm returned no tracks. Check the key and try again.")

    # 2. Enrich each with tags, album and listener count.
    print(f"\nenriching {len(seen):,} tracks at ~{1/args.rate:.1f}/sec "
          f"(~{len(seen) * args.rate / 60:.0f} min)")
    rows: list[dict] = []
    pairs = [(artist, title, year) for (artist, title), year in
             ((k, v) for k, v in seen.items())]
    # `seen` keys are lowercased for dedup; re-fetch gives the canonical casing.
    for index, ((artist_lc, title_lc), year) in enumerate(seen.items(), start=1):
        started = time.monotonic()
        payload = _call(session, key, "track.getInfo",
                        artist=artist_lc, track=title_lc, autocorrect=1)
        track = (payload or {}).get("track") or {}
        name = str(track.get("name") or title_lc).strip()
        artist = str((track.get("artist") or {}).get("name") or artist_lc).strip()
        tags = [str(t.get("name") or "").strip().lower()
                for t in ((track.get("toptags") or {}).get("tag") or [])]
        tags = [t for t in tags if t and not t.isdigit()]
        rows.append({
            "track_id": _slug(artist, name),
            "track_name": name,
            "track_artist": artist,
            "track_album_name": str((track.get("album") or {}).get("title") or "").strip(),
            "track_album_release_date": year,
            "playlist_genre": tags[0] if tags else "",
            "playlist_subgenre": tags[1] if len(tags) > 1 else "",
            # Raw listeners for now; rescaled below once the range is known.
            "track_popularity": float(track.get("listeners") or 0),
        })
        if index % 100 == 0:
            print(f"  {index:,}/{len(seen):,} enriched", flush=True)
        elapsed = time.monotonic() - started
        if elapsed < args.rate:
            time.sleep(args.rate - elapsed)

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame = frame[frame.track_name.str.strip() != ""]
    frame = frame.drop_duplicates(subset=["track_id"])

    if not args.keep_adult_listings:
        adult = frame.track_name.apply(looks_like_adult_video)
        if adult.any():
            print(f"  dropped {adult.sum()} adult-video listing(s):", flush=True)
            for title in frame.track_name[adult]:
                print(f"    - {title[:70]}", flush=True)
        frame = frame[~adult]

    # 3. Rescale listeners onto Spotify's 0-100 popularity scale. See the module
    #    docstring: both sources become `music` rows, so raw listener counts in
    #    the millions would outrank every Spotify track in any ordering.
    listeners = pd.to_numeric(frame.track_popularity, errors="coerce").fillna(0)
    if listeners.max() > 0:
        frame["track_popularity"] = (listeners.rank(pct=True) * 100).round().astype(int)
    else:
        frame["track_popularity"] = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print()
    print(f"wrote {args.out}")
    print(f"  {len(frame):,} tracks, {frame.track_album_release_date.nunique()} years")
    print(f"  with a genre tag: {(frame.playlist_genre.str.strip() != '').sum():,}")
    print()
    print("Rebuild to apply:")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_embeddings")
    print("  python -m embeddings.build_qdrant_collection")


if __name__ == "__main__":
    main()
