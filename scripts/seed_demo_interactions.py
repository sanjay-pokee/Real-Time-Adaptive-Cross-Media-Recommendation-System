"""Seed demo users and interactions for personalization/LightGCN experiments."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backend.mysql_store import MySQLStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"

# The entertainment trio every persona is seeded across unless it says otherwise.
# Seeding the new verticals into all of them would hand the family persona
# adult-only health products, so a persona that belongs to another domain
# declares `seed_types` instead of this being widened globally.
ENTERTAINMENT_TYPES = ["movie", "book", "music"]

# Every type any persona can be seeded from, for the summary counts.
CONTENT_TYPES = ENTERTAINMENT_TYPES + ["health", "industrial", "finance"]

LIKE_LIMIT_PER_TYPE = 16
SKIP_LIMIT_PER_TYPE = 4
FALLBACK_LIMIT_PER_TYPE = 6

USER_PROFILES = {
    "user_scifi": {
        "age_group": "young_adult",
        "profession": "software engineer",
        "skills": ["technology", "science", "programming"],
        "interests": ["space", "science fiction", "superhero"],
        "preferred_content_types": ["movie", "book"],
        "likes": ["science fiction", "space", "adventure", "alien", "superhero"],
        "skips": ["romance", "drama"],
    },
    "user_fantasy": {
        "age_group": "young_adult",
        "profession": "student",
        "skills": ["storytelling", "creativity"],
        "interests": ["fantasy", "magic", "animation"],
        "preferred_content_types": ["movie", "book"],
        "likes": ["fantasy", "magic", "adventure", "animation"],
        "skips": ["crime", "documentary"],
    },
    "user_romance": {
        "age_group": "adult",
        "profession": "designer",
        "skills": ["creativity", "art"],
        "interests": ["romance", "drama", "music"],
        "preferred_content_types": ["movie", "music", "book"],
        "likes": ["romance", "drama", "love"],
        "skips": ["horror", "war"],
    },
    "user_action": {
        "age_group": "adult",
        "profession": "software engineer",
        "skills": ["technology", "analysis"],
        "interests": ["action", "thriller", "superhero"],
        "preferred_content_types": ["movie"],
        "likes": ["action", "thriller", "superhero", "crime"],
        "skips": ["family", "documentary"],
    },
    "user_music_pop": {
        "age_group": "young_adult",
        "profession": "student",
        "skills": ["communication", "creativity"],
        "interests": ["pop", "dance", "party"],
        "preferred_content_types": ["music"],
        "likes": ["pop", "dance", "party", "comedy"],
        "skips": ["classical", "documentary"],
    },
    "user_music_rock": {
        "age_group": "young_adult",
        "profession": "designer",
        "skills": ["creativity", "media"],
        "interests": ["rock", "alternative", "metal"],
        "preferred_content_types": ["music", "movie"],
        "likes": ["rock", "alternative", "metal", "action"],
        "skips": ["romance", "children"],
    },
    "user_books_learning": {
        "age_group": "adult",
        "profession": "data analyst",
        "skills": ["data", "analytics", "business", "statistics"],
        "interests": ["business", "self-help", "psychology", "history"],
        "preferred_content_types": ["book", "course"],
        "likes": ["business", "self-help", "psychology", "history", "documentary"],
        "skips": ["horror", "crime"],
    },
    "user_family": {
        "age_group": "family",
        "profession": "student",
        "skills": ["learning", "creativity"],
        "interests": ["family", "animation", "children"],
        "preferred_content_types": ["movie", "book"],
        "avoid": ["horror", "crime", "thriller"],
        "likes": ["family", "animation", "comedy", "children", "adventure"],
        "skips": ["horror", "thriller"],
    },
    "user_dark_thriller": {
        "age_group": "adult",
        "profession": "data analyst",
        "skills": ["analysis", "investigation"],
        "interests": ["thriller", "crime", "mystery"],
        "preferred_content_types": ["movie", "book"],
        "likes": ["thriller", "horror", "crime", "mystery"],
        "skips": ["family", "children"],
    },
    "user_balanced": {
        "age_group": "young_adult",
        "profession": "student",
        "skills": ["learning", "communication"],
        "interests": ["adventure", "comedy", "drama", "pop", "science fiction"],
        "preferred_content_types": ["movie", "book", "music"],
        "likes": ["adventure", "comedy", "drama", "pop", "science fiction"],
        "skips": ["horror"],
    },
    # --- The verticals the review asked for -------------------------------
    #
    # Without these, every seeded interaction sat in entertainment: the graph
    # and EMA layers had no history at all in health, industry or finance, so
    # picking a persona and searching those verticals fell back to pure
    # semantic search. That is the one thing the panel is asking to see
    # generalised, so each vertical gets a persona whose history lives in it.
    #
    # Each pairs its product domain with `book`, because these are information
    # *and* product discovery domains - a caregiver reads about a condition and
    # buys a brace - which is also what keeps the cross-media claim true here.
    # All three are `adult`: health, industrial and finance default to adult
    # maturity, so a younger persona would have its own history filtered away.
    "user_health_caregiver": {
        "age_group": "adult",
        "profession": "nurse",
        "skills": ["care", "health", "biology"],
        "interests": ["wellness", "nutrition", "mobility", "first aid"],
        "preferred_content_types": ["health", "book"],
        "seed_types": ["health", "book"],
        "likes": ["supplement", "vitamin", "first aid", "mobility", "nutrition", "health"],
        "skips": ["horror", "gaming"],
    },
    "user_industry_engineer": {
        "age_group": "adult",
        "profession": "mechanical engineer",
        "skills": ["engineering", "safety", "measurement"],
        "interests": ["tools", "safety equipment", "lab", "measurement"],
        "preferred_content_types": ["industrial", "book"],
        "seed_types": ["industrial", "book"],
        "likes": ["safety", "tool", "measurement", "lab", "industrial", "engineering"],
        "skips": ["romance", "children"],
    },
    "user_finance_planner": {
        "age_group": "adult",
        "profession": "financial analyst",
        "skills": ["finance", "accounting", "analysis"],
        "interests": ["budgeting", "tax", "accounting", "investing basics"],
        "preferred_content_types": ["finance", "book"],
        "seed_types": ["finance", "book"],
        "likes": ["budget", "tax", "accounting", "finance", "payroll", "bookkeeping"],
        "skips": ["horror", "children"],
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
        seed_types = profile.get("seed_types", ENTERTAINMENT_TYPES)
        liked = _pick_balanced_items(catalog, profile["likes"], LIKE_LIMIT_PER_TYPE, seed_types)
        skipped = _pick_balanced_items(catalog, profile["skips"], SKIP_LIMIT_PER_TYPE, seed_types)
        liked = _add_popular_fallbacks(catalog, liked, FALLBACK_LIMIT_PER_TYPE, seed_types)

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

        counts = _count_by_type(catalog, liked, seed_types)
        print(f"{user_id}: liked {counts}")

    profile_count = store.upsert_user_profiles(profiles)
    inserted = store.bulk_log_interactions(interactions)
    print(f"Seeded {profile_count} demo user profiles.")
    print(f"Seeded {inserted} balanced demo interactions for {len(USER_PROFILES)} users.")

    # Derive each profile's EMA vector from the interactions just written.
    #
    # The profiles above are seeded with `ema_vector: []`, because the vector is
    # a function of the interaction history and that history does not exist
    # until the line above runs. Nothing then filled it in, so every demo user
    # carried an empty vector and the EMA stage contributed *nothing* to any
    # ranking - one of the four advertised signals silently absent from every
    # result, and silently re-broken by each reseed. Rebuilding here keeps the
    # two in step, since a reseed invalidates whatever vectors already existed.
    rebuilt = _rebuild_ema_vectors(store)
    print(f"Rebuilt EMA vectors for {rebuilt} users.")

    print("Try user_id values:")
    for user_id in USER_PROFILES:
        print(f"  - {user_id}")


def _rebuild_ema_vectors(store: MySQLStore) -> int:
    """Replay each user's interactions through the EMA update, in time order."""
    from backend.ema_recommender import EMAEmbeddingStore

    ema_store = EMAEmbeddingStore()
    logged = store.get_lightgcn_interactions(limit=500_000)
    if logged.empty:
        return 0

    logged = logged.sort_values(["user_id", "timestamp"])
    rebuilt = 0
    for user_id, rows in logged.groupby("user_id", sort=True):
        vector: list[float] = []
        for row in rows.itertuples(index=False):
            updated = ema_store.update_profile_vector(
                vector,
                str(row.entity_id),
                str(row.event_type),
                None if pd.isna(row.event_value) else float(row.event_value),
                alpha=store.settings.ema_alpha,
            )
            if updated is not None:
                vector = updated
        if vector:
            store.update_user_ema_vector(str(user_id), vector)
            rebuilt += 1
    return rebuilt


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


def _pick_balanced_items(
    catalog: pd.DataFrame,
    terms: list[str],
    limit_per_type: int,
    content_types: list[str],
) -> list[str]:
    selected: list[str] = []
    for content_type in content_types:
        matches = _pick_items(catalog, terms, limit_per_type, content_type=content_type)
        selected.extend(matches)
    return _dedupe(selected)


def _add_popular_fallbacks(
    catalog: pd.DataFrame,
    selected: list[str],
    limit_per_type: int,
    content_types: list[str],
) -> list[str]:
    selected_set = set(selected)
    output = list(selected)
    for content_type in content_types:
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


def _count_by_type(
    catalog: pd.DataFrame,
    global_ids: list[str],
    content_types: list[str] | None = None,
) -> dict[str, int]:
    rows = catalog[catalog["global_id"].isin(global_ids)]
    counts = rows["content_type"].value_counts().to_dict()
    return {
        content_type: int(counts.get(content_type, 0))
        for content_type in (content_types or CONTENT_TYPES)
    }


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
