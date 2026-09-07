"""Load public benchmark interaction datasets for LightGCN training/eval.

This module is deliberately independent of MySQL. It reads a raw benchmark
file (currently Amazon Reviews 2023) and returns a DataFrame with the same
column contract the rest of ``models.graph`` already expects:

    user_id, entity_id, event_type, event_value, timestamp

so it is a drop-in replacement for ``MySQLStore().get_lightgcn_interactions()``
inside ``train_lightgcn`` and ``evaluate_lightgcn``.

Supported inputs
----------------
* Amazon Reviews 2023 "rating only" CSV  (``user_id,item_id,rating,timestamp``),
  plain or gzipped, with or without a header row.
* Amazon Reviews 2023 raw review JSON lines (``*.jsonl`` / ``*.jsonl.gz``),
  from which ``user_id``, ``parent_asin``/``asin``, ``rating`` and
  ``timestamp`` are extracted.

Standard implicit-feedback protocol
-----------------------------------
Ratings ``>= min_rating`` become a single positive event (``event_type="like"``,
weight 1.0 via ``interaction_weight``); everything else is dropped. This matches
how LightGCN / NGCF papers binarise explicit ratings before training.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_USER_COLS = ("user_id", "reviewerid", "user", "userid")
_ITEM_COLS = ("parent_asin", "item_id", "asin", "item", "itemid", "product_id")
_RATING_COLS = ("rating", "overall", "stars")
_TIME_COLS = ("timestamp", "unixreviewtime", "time", "unix_time")


# ---------------------------------------------------------------------------
# Raw file readers
# ---------------------------------------------------------------------------

def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _to_unix_seconds(value: object) -> int:
    """Normalise a timestamp to unix *seconds* (2023 reviews use ms in places)."""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    if number > 100_000_000_000:  # > ~year 5138 in seconds => it's milliseconds
        number //= 1000
    return number


def _read_jsonl_reviews(path: Path) -> pd.DataFrame:
    users: list[str] = []
    items: list[str] = []
    ratings: list[float] = []
    times: list[int] = []
    for record in _iter_jsonl(path):
        user = record.get("user_id") or record.get("reviewerID")
        item = record.get("parent_asin") or record.get("asin")
        rating = record.get("rating", record.get("overall"))
        if user is None or item is None or rating is None:
            continue
        users.append(str(user))
        items.append(str(item))
        ratings.append(float(rating))
        times.append(
            _to_unix_seconds(
                record.get("timestamp")
                or record.get("sort_timestamp")
                or record.get("unixReviewTime")
                or 0
            )
        )
    return pd.DataFrame(
        {"user_id": users, "entity_id": items, "rating": ratings, "timestamp": times}
    )


def _read_rating_csv(path: Path) -> pd.DataFrame:
    # Peek at the first row to decide whether there is a header.
    with _open_text(path) as handle:
        first = handle.readline().strip()
    lower = first.lower()
    has_header = any(tag in lower for tag in ("user", "rating", "asin", "item"))

    if has_header:
        frame = pd.read_csv(path, compression="infer")
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        user_col = _first_present(frame.columns, _USER_COLS)
        item_col = _first_present(frame.columns, _ITEM_COLS)
        rating_col = _first_present(frame.columns, _RATING_COLS)
        time_col = _first_present(frame.columns, _TIME_COLS)
        if not (user_col and item_col and rating_col):
            raise ValueError(
                f"Could not identify user/item/rating columns in {path.name}. "
                f"Found columns: {list(frame.columns)}"
            )
    else:
        # Amazon "rating only" files are: user_id, item_id, rating, timestamp
        frame = pd.read_csv(
            path,
            compression="infer",
            header=None,
            names=["user_id", "entity_id", "rating", "timestamp"],
        )
        user_col, item_col, rating_col, time_col = (
            "user_id",
            "entity_id",
            "rating",
            "timestamp",
        )

    if time_col:
        timestamps = (
            pd.to_numeric(frame[time_col], errors="coerce").fillna(0).astype("int64")
        )
        # 2023 files are unix seconds, but normalise any millisecond values.
        timestamps = timestamps.where(timestamps < 100_000_000_000, timestamps // 1000)
    else:
        timestamps = 0

    out = pd.DataFrame(
        {
            "user_id": frame[user_col].astype(str),
            "entity_id": frame[item_col].astype(str),
            "rating": pd.to_numeric(frame[rating_col], errors="coerce"),
            "timestamp": timestamps,
        }
    )
    return out.dropna(subset=["rating"])


def _first_present(columns, candidates) -> str | None:
    lookup = {str(c).lower() for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return candidate
    return None


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a benchmark file into ``user_id, entity_id, rating, timestamp``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download an Amazon Reviews 2023 category (rating-only CSV or review "
            "jsonl) from https://amazon-reviews-2023.github.io/ and pass its path."
        )
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.gz", ".json.gz", ".json")):
        return _read_jsonl_reviews(path)
    if name.endswith((".csv", ".csv.gz")):
        return _read_rating_csv(path)
    raise ValueError(f"Unsupported dataset file type: {path.name} (use .csv/.csv.gz/.jsonl/.jsonl.gz)")


# ---------------------------------------------------------------------------
# Filtering / shaping
# ---------------------------------------------------------------------------

def k_core_filter(
    frame: pd.DataFrame,
    user_core: int = 5,
    item_core: int = 5,
    max_iterations: int = 20,
) -> pd.DataFrame:
    """Iteratively drop users/items below the interaction thresholds."""
    current = frame
    for _ in range(max_iterations):
        before = len(current)
        user_counts = current["user_id"].value_counts()
        keep_users = user_counts[user_counts >= user_core].index
        current = current[current["user_id"].isin(keep_users)]

        item_counts = current["entity_id"].value_counts()
        keep_items = item_counts[item_counts >= item_core].index
        current = current[current["entity_id"].isin(keep_items)]

        if len(current) == before:
            break
    return current.reset_index(drop=True)


def subsample_users(frame: pd.DataFrame, max_users: int, seed: int = 42) -> pd.DataFrame:
    """Keep a random subset of users (keeps a graph small enough to iterate on)."""
    unique_users = frame["user_id"].unique()
    if max_users <= 0 or len(unique_users) <= max_users:
        return frame
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(unique_users, size=max_users, replace=False))
    return frame[frame["user_id"].isin(chosen)].reset_index(drop=True)


def prepare_interactions(
    path: str | Path,
    min_rating: float = 4.0,
    user_core: int = 10,
    item_core: int = 10,
    max_users: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Full pipeline: read -> binarise positives -> k-core -> optional subsample.

    Returns a DataFrame with columns:
        user_id, entity_id, event_type, event_value, timestamp
    """
    raw = load_raw(path)
    positives = raw[raw["rating"] >= float(min_rating)].copy()
    if positives.empty:
        raise ValueError(
            f"No interactions with rating >= {min_rating} in {Path(path).name}."
        )

    positives = positives.drop_duplicates(subset=["user_id", "entity_id"], keep="last")
    positives = subsample_users(positives, max_users, seed=seed)
    positives = k_core_filter(positives, user_core=user_core, item_core=item_core)
    if positives.empty:
        raise ValueError(
            "k-core filter removed every interaction. Lower --user-core / --item-core "
            "or raise --max-users."
        )

    positives["event_type"] = "like"
    positives["event_value"] = 1.0
    return positives[["user_id", "entity_id", "event_type", "event_value", "timestamp"]]


def describe(frame: pd.DataFrame) -> str:
    users = frame["user_id"].nunique()
    items = frame["entity_id"].nunique()
    rows = len(frame)
    density = rows / (users * items) if users and items else 0.0
    per_user = rows / users if users else 0.0
    return (
        f"interactions={rows:,}  users={users:,}  items={items:,}  "
        f"density={density:.5%}  avg_interactions_per_user={per_user:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a benchmark dataset after filtering (no training)."
    )
    parser.add_argument("dataset", type=Path, help="Path to the raw benchmark file")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--user-core", type=int, default=10)
    parser.add_argument("--item-core", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=0, help="0 = keep all users")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_raw(args.dataset)
    print(f"raw rows read: {len(raw):,}")
    prepared = prepare_interactions(
        args.dataset,
        min_rating=args.min_rating,
        user_core=args.user_core,
        item_core=args.item_core,
        max_users=args.max_users,
        seed=args.seed,
    )
    print("after filtering:")
    print("  " + describe(prepared))


if __name__ == "__main__":
    main()
