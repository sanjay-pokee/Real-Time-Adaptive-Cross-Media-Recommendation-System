"""Rebuild EMA user profile vectors from stored interactions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.ema_recommender import EMAEmbeddingStore
from backend.mysql_store import MySQLStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    store = MySQLStore()
    ema_store = EMAEmbeddingStore()
    interactions = store.get_lightgcn_interactions(limit=500_000)
    if interactions.empty:
        print("No interactions found. Nothing to rebuild.")
        return

    interactions = interactions.sort_values(["user_id", "timestamp"])
    rebuilt = 0
    for user_id, rows in interactions.groupby("user_id", sort=True):
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

    print(f"Rebuilt EMA vectors for {rebuilt} users.")
    print(f"EMA dimension: {ema_store.dimension}")


if __name__ == "__main__":
    main()
