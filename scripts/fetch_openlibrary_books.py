"""Fetch books from the live Open Library API, replacing the Kaggle export.

The catalogue's book rows came from a Google Books scrape on Kaggle: 15,147
rows built by running 149 keyword searches. Two things were wrong with it.
Only 8,351 rows (55%) carried a description at all, and the top of the file is
SEO spam - the first rows are "Bestsellers", by "Ivan King, bestsellers",
published by "bestsellers", with the word repeated nine times in the subtitle.
A row with no description embeds to noise, and a keyword-stuffed one embeds to
worse than noise, because it matches every query weakly.

Open Library exposes the same shape of data for free with no key. The one
detail that makes this cheap: `search.json` will return `description` as a
Solr field, so a page of 100 books arrives in one request. The obvious
implementation - search, then GET /works/{id}.json per hit for the synopsis -
would be 100x the requests for the same data.

Two further details that matter:

* `sort=rating` is what buys the quality. Unsorted, `subject:"science fiction"`
  returns whatever Solr scores highest for the term; sorted, it returns The
  Hitchhiker's Guide and Project Hail Mary. Description coverage tracks it:
  99% on the first page of a subject, 58% by page 500. So this stays shallow
  and gets its breadth from many subjects rather than deep paging one.
* Open Library is slow and erratic - observed 0.3s to 30s for the same query
  shape, with occasional connect timeouts. Hence the retrying adapter and the
  long read timeout; without them a multi-subject run dies partway through.

The output keeps the column names of the Kaggle CSV it replaces, so the
existing `books` normalizer reads it with no change.

Usage:
    python -m scripts.fetch_openlibrary_books
    python -m scripts.fetch_openlibrary_books --per-subject 500 --rate 0.5
    python -m scripts.fetch_openlibrary_books --subject horror --subject poetry
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "books" / "openlibrary_books.csv"

API = "https://openlibrary.org/search.json"
COVER_BASE = "https://covers.openlibrary.org/b/id"

# Solr returns nothing but `key` unless the wanted fields are named.
FIELDS = ",".join([
    "key", "title", "subtitle", "author_name", "first_publish_year",
    "publisher", "subject", "cover_i", "ratings_average", "ratings_count",
    "description", "language",
])

# Open Library pages at 100; anything higher is silently clamped.
PAGE_SIZE = 100
# Ceiling on pages per subject, so a subject whose hits mostly fail the quality
# filters cannot spin for hundreds of requests chasing --per-subject.
MAX_PAGES = 10

# Column names and order are the Kaggle file's, minus the columns nothing reads.
OUTPUT_COLUMNS = [
    "book_id", "title", "subtitle", "authors", "publisher", "published_date",
    "description", "categories", "average_rating", "ratings_count", "language",
    "search_category", "thumbnail",
]

# Roughly the spread of the 149 Kaggle search categories, deduplicated down to
# subjects Open Library actually indexes. `search_category` records which one
# found a row, exactly as the Kaggle column did.
DEFAULT_SUBJECTS = [
    # Fiction
    "fiction", "fantasy", "science fiction", "mystery", "thriller", "horror",
    "romance", "historical fiction", "adventure", "crime", "short stories",
    "poetry", "drama", "graphic novels", "comics", "classic literature",
    "young adult fiction", "juvenile fiction", "children's stories",
    # Life and society
    "biography", "autobiography", "history", "philosophy", "psychology",
    "political science", "sociology", "religion", "education", "true crime",
    # Science and technology
    "science", "mathematics", "physics", "astronomy", "biology", "medicine",
    "computer science", "computer programming", "artificial intelligence",
    "machine learning", "engineering",
    # Work and money
    "business", "economics", "finance", "management", "entrepreneurship",
    # Arts and living
    "art", "music", "photography", "architecture", "design",
    "cooking", "gardening", "travel", "nature", "environment",
    "health", "fitness", "self-help", "parenting", "sports",
]

# Library and marketing metadata that Open Library files as subjects. These are
# not topics, and left in they dominate `categories`: "New York Times
# bestseller" was the 2nd most common subject in a 300-book sample.
SUBJECT_BLOCKLIST = (
    "new york times bestseller", "new york times reviewed", "large type books",
    "accessible book", "protected daisy", "in library", "overdrive",
    "internet archive wishlist", "open library staff picks", "reading level",
    "long now manual for civilization",
)

# Machine tags: "nyt:combined-print-and-e-book-fiction=2014-03-23", "lc:PZ7",
# "ddc:813/.54". 355 of them in that same 300-book sample.
MACHINE_TAG = re.compile(r"[:=]")

# Open Library descriptions often carry an editorial trailer after a horizontal
# rule - "Contained in:" plus markdown links to sibling works. It is metadata
# about the record, not about the book, so it is cut before embedding.
#
# The leading `^|` matters: some records are *only* the trailer, with no
# synopsis in front of it. Anchoring on the newline alone left those as a row
# reading "Contained in: A B", which is long enough to clear the length filter
# and says nothing. Cutting from the start empties them, so they are dropped.
DESCRIPTION_TRAILER = re.compile(
    r"(?:^|\r?\n)\s*"
    r"(?:-{3,}\s*\r?\n|(?:contained in|also contained in|see also|source)\s*:).*$",
    re.DOTALL | re.IGNORECASE,
)
MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MARKDOWN_REF = re.compile(r"^\s*\[\d+\]:\s*\S+\s*$", re.MULTILINE)
WHITESPACE = re.compile(r"\s+")


def _session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=6, connect=6, read=6, backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )))
    return session


def clean_description(value: object) -> str:
    """Flatten and de-noise one description.

    The Solr field is a plain string, but the underlying work record stores
    either a string or ``{"type": ..., "value": ...}``; handle both so this
    keeps working if the field is ever passed through unflattened.
    """
    if isinstance(value, dict):
        value = value.get("value", "")
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    text = str(value or "")

    text = DESCRIPTION_TRAILER.sub("", text)
    text = MARKDOWN_REF.sub("", text)
    text = MARKDOWN_LINK.sub(r"\1", text)
    return WHITESPACE.sub(" ", text).strip()


def clean_subjects(raw: object, cap: int) -> list[str]:
    """Topical subjects only, deduplicated case-insensitively and capped.

    Uncapped this runs to 41 subjects on a popular work, which would let the
    category text outweigh the description in the embedding.
    """
    subjects: list[str] = []
    seen: set[str] = set()

    for item in raw or []:
        text = WHITESPACE.sub(" ", str(item)).strip()
        if not text or MACHINE_TAG.search(text):
            continue
        if any(blocked in text.lower() for blocked in SUBJECT_BLOCKLIST):
            continue

        # "Imaginary wars and battles -- Fiction" is a compound heading; the
        # comma reads the same to the embedder and to the maturity keywords.
        text = text.replace(" -- ", ", ")
        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        subjects.append(text)
        if len(subjects) >= cap:
            break

    return subjects


def build_query(subject: str, from_year: int | None, to_year: int | None) -> str:
    """Solr query for one subject, optionally windowed to a publication range.

    The window exists because `sort=rating` systematically excludes new
    releases - ratings take years to accumulate, so a plain run returned 42
    books from 2026 against a median year of 2003. Sorting by `new` instead is
    the wrong trade: it surfaces records created recently rather than books
    published recently, and only 18 of 100 had a description against 86 of 100
    for the same window sorted by rating. So the window narrows the pool and
    `rating` still orders it.
    """
    query = f'subject:"{subject}"'
    if from_year or to_year:
        low = from_year if from_year else 1
        high = to_year if to_year else 9999
        query += f" AND first_publish_year:[{low} TO {high}]"
    return query


def merge_with_existing(frame: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, int]:
    """Fold a top-up run into the file already on disk.

    Existing rows win on a collision, so a book already in the catalogue keeps
    the `search_category` that first found it and its global_id does not move -
    which is what makes a top-up cheap downstream: no id changes means the
    MySQL load prunes nothing and seeded interactions survive.
    """
    if not out_path.exists():
        return frame, 0

    # dtype=str is load-bearing. `published_date` holds a bare year, and 17 of
    # 12,402 books have none; a default read types that column float64 and the
    # round-trip rewrites every "1979" as "1979.0", which then reaches the API
    # and renders as a broken year in the UI. keep_default_na keeps the blanks
    # as "" rather than NaN, matching what build_row emits.
    existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    combined = pd.concat([existing, frame], ignore_index=True)
    combined = combined.drop_duplicates(subset=["book_id"], keep="first")
    return combined, len(existing)


def _first(value: object) -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip()


def build_row(doc: dict, subject: str, subject_cap: int) -> dict | None:
    """One Solr hit to one output row, or None if it is unusable."""
    title = str(doc.get("title") or "").strip()
    work_key = str(doc.get("key") or "").rsplit("/", 1)[-1]
    cover_id = doc.get("cover_i")

    if not title or not work_key:
        return None

    authors = ", ".join(
        str(name).strip() for name in (doc.get("author_name") or []) if str(name).strip()
    )

    return {
        "book_id": work_key,
        "title": title,
        "subtitle": str(doc.get("subtitle") or "").strip(),
        "authors": authors,
        "publisher": _first(doc.get("publisher")),
        "published_date": str(doc.get("first_publish_year") or "").strip(),
        "description": clean_description(doc.get("description")),
        "categories": ", ".join(clean_subjects(doc.get("subject"), subject_cap)),
        "average_rating": doc.get("ratings_average") or "",
        "ratings_count": doc.get("ratings_count") or 0,
        "language": ", ".join(str(code) for code in (doc.get("language") or [])),
        "search_category": subject,
        "thumbnail": f"{COVER_BASE}/{cover_id}-L.jpg" if cover_id else "",
    }


def passes(row: dict, doc: dict, min_description: int, language: str) -> bool:
    """The TMDb fetcher's quality bar, in book terms.

    A cover is required for the same reason a poster is: a row without one is
    the gap this replacement exists to close. A description is required because
    it is all but the whole of a book's embedding text.
    """
    if len(row["description"]) < min_description or not row["thumbnail"]:
        return False

    # `language` is the union across every edition of the work, so an English
    # novel also published in French passes, as it should. An empty list means
    # Open Library holds no language data rather than "not English", so those
    # are kept; a work with no edition language at all is rare.
    codes = doc.get("language") or []
    return not (language and codes and language not in codes)


def fetch_subject(
    session: requests.Session,
    subject: str,
    per_subject: int,
    sort: str,
    language: str,
    min_description: int,
    subject_cap: int,
    rate: float,
    from_year: int | None = None,
    to_year: int | None = None,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    examined = 0
    query = build_query(subject, from_year, to_year)

    for page in range(1, MAX_PAGES + 1):
        if len(rows) >= per_subject:
            break

        params = {
            "q": query,
            "fields": FIELDS,
            "limit": PAGE_SIZE,
            "page": page,
        }
        if sort:
            params["sort"] = sort

        started = time.monotonic()
        try:
            response = session.get(API, params=params, timeout=(30, 180))
        except requests.RequestException as exc:
            print(f"  {subject}: {type(exc).__name__} after retries, stopping this subject")
            break

        if response.status_code != 200:
            print(f"  {subject}: HTTP {response.status_code}, stopping this subject")
            break

        docs = response.json().get("docs", [])
        if not docs:
            break

        examined += len(docs)
        for doc in docs:
            row = build_row(doc, subject, subject_cap)
            if row and passes(row, doc, min_description, language):
                rows.append(row)
                if len(rows) >= per_subject:
                    break

        # A short page means Solr has no more hits for this subject.
        if len(docs) < PAGE_SIZE:
            break

        elapsed = time.monotonic() - started
        if elapsed < rate:
            time.sleep(rate - elapsed)

    return rows, examined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch books from the Open Library API.")
    parser.add_argument("--subject", action="append", dest="subjects", metavar="SUBJECT",
                        help="Subject to fetch; repeatable. Defaults to the built-in "
                             f"list of {len(DEFAULT_SUBJECTS)} subjects.")
    parser.add_argument("--per-subject", type=int, default=300,
                        help="Target kept rows per subject (default 300). Capped at "
                             f"{MAX_PAGES} pages of {PAGE_SIZE} examined either way.")
    parser.add_argument("--min-description", type=int, default=20,
                        help="Minimum description length; shorter rows embed to noise "
                             "(default 20, matching the TMDb fetcher)")
    parser.add_argument("--subject-cap", type=int, default=12,
                        help="Maximum subjects kept per book (default 12)")
    parser.add_argument("--language", default="eng",
                        help="MARC language code to require, or '' for any (default eng)")
    parser.add_argument("--sort", default="rating",
                        help="Open Library sort key, or '' for relevance (default rating)")
    parser.add_argument("--from-year", type=int, default=None,
                        help="Only books first published in or after this year. Use with "
                             "--merge for a recency top-up; sort=rating alone under-"
                             "represents new releases, which have few ratings yet.")
    parser.add_argument("--to-year", type=int, default=None,
                        help="Only books first published in or before this year")
    parser.add_argument("--merge", action="store_true",
                        help="Fold results into the existing --out file instead of "
                             "replacing it. Existing rows win, so global_ids do not move.")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Minimum seconds between requests (default 1.0)")
    parser.add_argument("--user-agent",
                        default="cross-media-recommender/0.1 (academic project)",
                        help="Open Library asks that bulk readers identify themselves")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    subjects = args.subjects or DEFAULT_SUBJECTS
    session = _session(args.user_agent)

    print(f"fetching {len(subjects)} subjects, up to {args.per_subject} books each")
    print(f"  sort={args.sort or 'relevance'}  language={args.language or 'any'}  "
          f"min-description={args.min_description}")
    if args.from_year or args.to_year:
        print(f"  years={args.from_year or 'any'}-{args.to_year or 'any'}"
              f"{'  (merging into the existing file)' if args.merge else ''}")
    print()

    all_rows: list[dict] = []
    total_examined = 0
    empty: list[str] = []

    for subject in subjects:
        rows, examined = fetch_subject(
            session, subject, args.per_subject, args.sort, args.language,
            args.min_description, args.subject_cap, args.rate,
            args.from_year, args.to_year,
        )
        all_rows.extend(rows)
        total_examined += examined
        if rows:
            print(f"  {subject:<24} {len(rows):>4} kept of {examined:>4}   "
                  f"(running total {len(all_rows):,})")
        else:
            empty.append(subject)
            print(f"  {subject:<24}    0 kept of {examined:>4}   [no usable rows]")

    frame = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    before = len(frame)
    # A work indexed under several subjects comes back once per subject. First
    # wins, so `search_category` names the subject that ranked it highest.
    frame = frame.drop_duplicates(subset=["book_id"])
    fetched = len(frame)

    previous = 0
    if args.merge:
        frame, previous = merge_with_existing(frame, args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, encoding="utf-8")

    print()
    print(f"wrote {args.out}")
    print(f"  {len(frame):,} books ({before - fetched:,} cross-subject duplicates dropped)")
    if args.merge:
        print(f"  {previous:,} already on disk + {len(frame) - previous:,} new "
              f"(of {fetched:,} fetched; the rest were already present)")
    print(f"  {total_examined:,} examined, every kept row has a description and a cover")
    if empty:
        print(f"  [WARNING] {len(empty)} subjects returned nothing: {', '.join(empty)}")
    print()
    print("Point config/datasets.yaml books.path at this file, then run all of:")
    print("  python -m preprocessing.build_content_catalog")
    print("  python -m embeddings.build_embeddings")
    print("  python -m embeddings.build_faiss_index")
    print("  python -m embeddings.build_qdrant_collection")
    print("  python -m scripts.load_catalog_mysql")
    print("  python -m scripts.seed_demo_interactions")
    print("  python -m scripts.rebuild_ema_profiles")
    print()
    print("Run them in that order, and stop at the first failure. Changing a")
    print("dataset's `source` changes every global_id it owns, so a partial")
    print("rebuild leaves the stores disagreeing about what exists: the MySQL")
    print("load prunes the old ids and cascades away their interactions, which")
    print("is what seed/rebuild_ema restore. build_faiss_index is not optional -")
    print("backend/recommender.py serves from content_faiss.index, so skipping")
    print("it leaves search answering from the ids you just deleted.")


if __name__ == "__main__":
    main()
