"""Seed demo users and interactions for personalization/LightGCN experiments."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backend.mysql_store import MySQLStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
CONTENT_TYPES = ["movie", "book", "music"]
LIKE_LIMIT_PER_TYPE = 16
SKIP_LIMIT_PER_TYPE = 4
FALLBACK_LIMIT_PER_TYPE = 6

USER_PROFILES = {
    "user_scifi": {
        "likes": ["science fiction", "space", "adventure", "alien", "superhero"],
        "skips": ["romance", "drama"],
    },
    "user_fantasy": {
        "likes": ["fantasy", "magic", "adventure", "animation"],
        "skips": ["crime", "documentary"],
    },
    "user_romance": {
        "likes": ["romance", "drama", "love"],
        "skips": ["horror", "war"],
    },
    "user_action": {
        "likes": ["action", "thriller", "superhero", "crime"],
        "skips": ["family", "documentary"],
    },
    "user_music_pop": {
        "likes": ["pop", "dance", "party", "comedy"],
        "skips": ["classical", "documentary"],
    },
    "user_music_rock": {
        "likes": ["rock", "alternative", "metal", "action"],
        "skips": ["romance", "children"],
    },
    "user_books_learning": {
        "likes": ["business", "self-help", "psychology", "history", "documentary"],
        "skips": ["horror", "crime"],
    },
    "user_family": {
        "likes": ["family", "animation", "comedy", "children", "adventure"],
        "skips": ["horror", "thriller"],
    },
    "user_dark_thriller": {
        "likes": ["thriller", "horror", "crime", "mystery"],
        "skips": ["family", "children"],
    },
    "user_balanced": {
        "likes": ["adventure", "comedy", "drama", "pop", "science fiction"],
        "skips": ["horror"],
    },
}


def main() -> None:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Missing catalog: {CATALOG_PATH}. Run python -m preprocessing.build_content_catalog"
        )

    store = MySQLStore()
    store.init_schema(create_database=True)
    store.upsert_content_catalog(CATALOG_PATH)
    _clear_seeded_interactions(store)

    catalog = pd.read_csv(CATALOG_PATH).fillna("")
    interactions = []
    profiles = []
    now = datetime.now(timezone.utc)

    for user_index, (user_id, profile) in enumerate(USER_PROFILES.items()):
        profiles.append({
            "user_id": user_id,
            "preferences": profile,
            "ema_vector": [],
        })
        liked = _pick_balanced_items(catalog, profile["likes"], LIKE_LIMIT_PER_TYPE)
        skipped = _pick_balanced_items(catalog, profile["skips"], SKIP_LIMIT_PER_TYPE)
        liked = _add_popular_fallbacks(catalog, liked, FALLBACK_LIMIT_PER_TYPE)

        offset = user_index * 1000
        for item_index, global_id in enumerate(liked):
            event_type = "like" if item_index % 3 else "view"
            if item_index % 5 == 0:
                event_type = "bookmark"
            interactions.append({
                "user_id": user_id,
                "entity_id": global_id,
                "event_type": event_type,
                "event_value": 1,
                "context": {"seeded": True, "seed_version": "balanced_cross_media_v2"},
                "timestamp": now - timedelta(minutes=offset + item_index),
            })
            if item_index % 4 == 0:
                interactions.append({
                    "user_id": user_id,
                    "entity_id": global_id,
                    "event_type": "rating",
                    "event_value": 4.0 + (item_index % 2) * 0.5,
                    "context": {"seeded": True, "seed_version": "balanced_cross_media_v2"},
                    "timestamp": now - timedelta(minutes=offset + item_index + 300),
                })

        for item_index, global_id in enumerate(skipped):
            interactions.append({
                "user_id": user_id,
                "entity_id": global_id,
                "event_type": "skip",
                "event_value": 1,
                "context": {"seeded": True, "seed_version": "balanced_cross_media_v2"},
                "timestamp": now - timedelta(minutes=offset + item_index + 600),
            })

        counts = _count_by_type(catalog, liked)
        print(f"{user_id}: liked {counts}")

    profile_count = store.upsert_user_profiles(profiles)
    inserted = store.bulk_log_interactions(interactions)
    print(f"Seeded {profile_count} demo user profiles.")
    print(f"Seeded {inserted} balanced demo interactions for {len(USER_PROFILES)} users.")
    print("Try user_id values:")
    for user_id in USER_PROFILES:
        print(f"  - {user_id}")


def _clear_seeded_interactions(store: MySQLStore) -> None:
    conn = store._connect_database()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM user_interactions
                WHERE JSON_UNQUOTE(JSON_EXTRACT(context, '$.seeded')) = 'true'
                """
            )
        conn.commit()
    finally:
        conn.close()


def _pick_balanced_items(catalog: pd.DataFrame, terms: list[str], limit_per_type: int) -> list[str]:
    selected: list[str] = []
    for content_type in CONTENT_TYPES:
        matches = _pick_items(catalog, terms, limit_per_type, content_type=content_type)
        selected.extend(matches)
    return _dedupe(selected)


def _add_popular_fallbacks(catalog: pd.DataFrame, selected: list[str], limit_per_type: int) -> list[str]:
    selected_set = set(selected)
    output = list(selected)
    for content_type in CONTENT_TYPES:
        type_rows = catalog[catalog["content_type"] == content_type].copy()
        if type_rows.empty:
            continue
        type_rows["popularity_numeric"] = pd.to_numeric(type_rows["popularity"], errors="coerce").fillna(0)
        for global_id in type_rows.sort_values("popularity_numeric", ascending=False)["global_id"]:
            if global_id in selected_set:
                continue
            output.append(global_id)
            selected_set.add(global_id)
            if _count_type_ids(catalog, output, content_type) >= limit_per_type:
                break
    return output


def _pick_items(
    catalog: pd.DataFrame,
    terms: list[str],
    limit: int,
    content_type: str | None = None,
) -> list[str]:
    frame = catalog if content_type is None else catalog[catalog["content_type"] == content_type]
    text = (
        frame["title"].astype(str) + " "
        + frame["categories"].astype(str) + " "
        + frame["description"].astype(str) + " "
        + frame["metadata_text"].astype(str)
    ).str.lower()
    mask = pd.Series(False, index=frame.index)
    for term in terms:
        mask = mask | text.str.contains(term.lower(), regex=False)

    matches = frame[mask].drop_duplicates(subset=["global_id"]).copy()
    if matches.empty:
        return []

    matches["popularity_numeric"] = pd.to_numeric(matches["popularity"], errors="coerce").fillna(0)
    sampled = matches.sort_values("popularity_numeric", ascending=False)
    return sampled["global_id"].head(limit).tolist()


def _count_by_type(catalog: pd.DataFrame, global_ids: list[str]) -> dict[str, int]:
    rows = catalog[catalog["global_id"].isin(global_ids)]
    counts = rows["content_type"].value_counts().to_dict()
    return {content_type: int(counts.get(content_type, 0)) for content_type in CONTENT_TYPES}


def _count_type_ids(catalog: pd.DataFrame, global_ids: list[str], content_type: str) -> int:
    rows = catalog[catalog["global_id"].isin(global_ids)]
    return int((rows["content_type"] == content_type).sum())


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


if __name__ == "__main__":
    main()
