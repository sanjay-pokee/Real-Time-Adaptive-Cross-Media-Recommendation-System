"""Seed demo users and interactions for personalization/LightGCN experiments."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backend.mysql_store import MySQLStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"

USER_PROFILES = {
    "user_scifi": {
        "likes": ["science fiction", "space", "adventure", "superhero"],
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
        "likes": ["pop", "dance", "party"],
        "skips": ["classical", "documentary"],
    },
    "user_music_rock": {
        "likes": ["rock", "alternative", "metal"],
        "skips": ["romance", "children"],
    },
    "user_books_learning": {
        "likes": ["business", "self-help", "psychology", "history"],
        "skips": ["horror", "crime"],
    },
    "user_family": {
        "likes": ["family", "animation", "comedy", "children"],
        "skips": ["horror", "thriller"],
    },
    "user_dark_thriller": {
        "likes": ["thriller", "horror", "crime", "mystery"],
        "skips": ["family", "children"],
    },
    "user_balanced": {
        "likes": ["adventure", "comedy", "drama", "pop"],
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
        liked = _pick_items(catalog, profile["likes"], limit=18)
        skipped = _pick_items(catalog, profile["skips"], limit=5)

        offset = user_index * 100
        for item_index, global_id in enumerate(liked):
            event_type = "like" if item_index % 3 else "view"
            if item_index % 5 == 0:
                event_type = "bookmark"
            interactions.append({
                "user_id": user_id,
                "entity_id": global_id,
                "event_type": event_type,
                "event_value": 1,
                "context": {"seeded": True},
                "timestamp": now - timedelta(minutes=offset + item_index),
            })
            if item_index % 4 == 0:
                interactions.append({
                    "user_id": user_id,
                    "entity_id": global_id,
                    "event_type": "rating",
                    "event_value": 4.0 + (item_index % 2) * 0.5,
                    "context": {"seeded": True},
                    "timestamp": now - timedelta(minutes=offset + item_index + 30),
                })

        for item_index, global_id in enumerate(skipped):
            interactions.append({
                "user_id": user_id,
                "entity_id": global_id,
                "event_type": "skip",
                "event_value": 1,
                "context": {"seeded": True},
                "timestamp": now - timedelta(minutes=offset + item_index + 60),
            })

    profile_count = store.upsert_user_profiles(profiles)
    inserted = store.bulk_log_interactions(interactions)
    print(f"Seeded {profile_count} demo user profiles.")
    print(f"Seeded {inserted} demo interactions for {len(USER_PROFILES)} users.")
    print("Try user_id values:")
    for user_id in USER_PROFILES:
        print(f"  - {user_id}")


def _pick_items(catalog: pd.DataFrame, terms: list[str], limit: int) -> list[str]:
    text = (
        catalog["title"].astype(str) + " "
        + catalog["categories"].astype(str) + " "
        + catalog["description"].astype(str) + " "
        + catalog["metadata_text"].astype(str)
    ).str.lower()
    mask = pd.Series(False, index=catalog.index)
    for term in terms:
        mask = mask | text.str.contains(term.lower(), regex=False)

    matches = catalog[mask].drop_duplicates(subset=["global_id"])
    if matches.empty:
        return []

    sampled = matches.sort_values(["content_type", "popularity"], ascending=[True, False])
    return sampled["global_id"].head(limit).tolist()


if __name__ == "__main__":
    main()
