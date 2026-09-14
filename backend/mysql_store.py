"""MySQL schema initialization and interaction persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.settings import Settings, get_settings


CREATE_DATABASE_SQL = "CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"

# Rows per executemany when loading the catalogue. Sized so one statement stays
# comfortably inside a default max_allowed_packet even though each row carries a
# full description and embedding_text.
CATALOG_UPSERT_BATCH = 2000

TABLE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS content_entities (
      -- utf8mb4_bin, because these ids are case-sensitive. The database
      -- default is utf8mb4_unicode_ci, under which Google Books ids like
      -- be0XAQAAIAAJ and be0xAQAAIAAJ - two different books - collide on
      -- the primary key and one silently overwrites the other.
      global_id VARCHAR(512) COLLATE utf8mb4_bin PRIMARY KEY,
      content_type VARCHAR(64) NOT NULL,
      source VARCHAR(128) NOT NULL,
      source_id VARCHAR(256) NOT NULL,
      title TEXT NOT NULL,
      -- MEDIUMTEXT, not TEXT: Amazon product descriptions concatenate the
      -- feature bullets and run past TEXT's 65,535-byte ceiling. Two rows in a
      -- 106k catalogue do, which is enough to fail the whole load with
      -- "Data too long for column 'description'".
      description MEDIUMTEXT,
      creators TEXT,
      categories TEXT,
      release_date VARCHAR(64),
      popularity DOUBLE,
      rating DOUBLE,
      metadata JSON,
      embedding_text MEDIUMTEXT,
      text_hash VARCHAR(128),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      INDEX idx_content_type (content_type),
      INDEX idx_source (source)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
      user_id VARCHAR(191) PRIMARY KEY,
      preferences JSON,
      ema_vector JSON,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_interactions (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      user_id VARCHAR(191) NOT NULL,
      -- Must match content_entities.global_id exactly, or the foreign key
      -- below is rejected for incompatible collations.
      entity_id VARCHAR(512) COLLATE utf8mb4_bin NOT NULL,
      event_type VARCHAR(64) NOT NULL,
      event_value DOUBLE,
      context JSON,
      timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_user_timestamp (user_id, timestamp),
      INDEX idx_entity (entity_id),
      CONSTRAINT fk_interaction_entity
        FOREIGN KEY (entity_id) REFERENCES content_entities(global_id)
        ON DELETE CASCADE
    )
    """,
]


# Applied after TABLE_SQL on every init. CREATE TABLE IF NOT EXISTS silently
# leaves an existing table alone, so a schema change never reaches a database
# that was created before it. These widen columns in place and are safe to
# re-run: MODIFY to the type a column already has is a no-op.
MIGRATION_SQL = [
    "ALTER TABLE content_entities MODIFY description MEDIUMTEXT",
    # Collation change on a column a foreign key points at: the constraint has
    # to come off first and go back on after, and both sides must end up with
    # the same collation.
    ("ALTER TABLE user_interactions DROP FOREIGN KEY fk_interaction_entity", True),
    "ALTER TABLE content_entities MODIFY global_id VARCHAR(512) COLLATE utf8mb4_bin NOT NULL",
    "ALTER TABLE user_interactions MODIFY entity_id VARCHAR(512) COLLATE utf8mb4_bin NOT NULL",
    ("ALTER TABLE user_interactions ADD CONSTRAINT fk_interaction_entity "
     "FOREIGN KEY (entity_id) REFERENCES content_entities(global_id) ON DELETE CASCADE", True),
]


@dataclass
class MySQLStore:
    settings: Settings = get_settings()

    def init_schema(self, create_database: bool = True) -> None:
        connector = self._connector()

        if create_database:
            server_conn = connector.connect(
                host=self.settings.mysql_host,
                port=self.settings.mysql_port,
                user=self.settings.mysql_user,
                password=self.settings.mysql_password,
            )
            try:
                with server_conn.cursor() as cursor:
                    cursor.execute(CREATE_DATABASE_SQL.format(database=self.settings.mysql_database))
                server_conn.commit()
            finally:
                server_conn.close()

        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                for statement in TABLE_SQL:
                    cursor.execute(statement)
                for migration in MIGRATION_SQL:
                    # A tuple marks a statement that may legitimately fail:
                    # dropping a constraint that is already gone, or adding one
                    # that is already there. Everything else fails loudly,
                    # because a migration that silently no-ops is worse than one
                    # that stops.
                    statement, optional = (
                        migration if isinstance(migration, tuple) else (migration, False)
                    )
                    try:
                        cursor.execute(statement)
                    except Exception:
                        if not optional:
                            raise
            conn.commit()
        finally:
            conn.close()


    def upsert_content_catalog(self, catalog_path: Path) -> int:
        if not catalog_path.exists():
            raise FileNotFoundError(f"Content catalog not found: {catalog_path}")

        # low_memory=False is required, not cosmetic. The default reads the file in
        # chunks and types each chunk independently, so a run of rows whose
        # release_date is a bare year - the finance books are contiguous - comes
        # back as float64 and every "2010" becomes 2010.0. That reached the Qdrant
        # payload and 500'd every finance query on response validation.
        catalog = pd.read_csv(catalog_path, low_memory=False)
        rows = []
        for _, row in catalog.iterrows():
            rows.append((
                _clean(row.get("global_id")),
                _clean(row.get("content_type")),
                _clean(row.get("source")),
                _clean(row.get("source_id")),
                _clean(row.get("title")),
                _clean(row.get("description")),
                _clean(row.get("creators")),
                _clean(row.get("categories")),
                _clean(row.get("release_date")),
                _number_or_none(row.get("popularity")),
                _number_or_none(row.get("rating")),
                json.dumps({"metadata_text": _clean(row.get("metadata_text"))}),
                _clean(row.get("embedding_text")),
                _clean(row.get("text_hash")),
            ))

        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                # Chunked, because executemany builds one statement per call and
                # MySQL drops the connection when it exceeds max_allowed_packet.
                # Every row carries a full description and embedding_text, so a
                # single call for the whole catalogue is hundreds of megabytes;
                # this started failing with "Lost connection to MySQL server
                # during query" once the catalogue passed ~100k rows.
                statement = """
                    INSERT INTO content_entities
                      (global_id, content_type, source, source_id, title, description,
                       creators, categories, release_date, popularity, rating, metadata,
                       embedding_text, text_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      content_type = VALUES(content_type),
                      source = VALUES(source),
                      source_id = VALUES(source_id),
                      title = VALUES(title),
                      description = VALUES(description),
                      creators = VALUES(creators),
                      categories = VALUES(categories),
                      release_date = VALUES(release_date),
                      popularity = VALUES(popularity),
                      rating = VALUES(rating),
                      metadata = VALUES(metadata),
                      embedding_text = VALUES(embedding_text),
                      text_hash = VALUES(text_hash)
                    """
                for start in range(0, len(rows), CATALOG_UPSERT_BATCH):
                    cursor.executemany(
                        statement, rows[start : start + CATALOG_UPSERT_BATCH]
                    )

                # Drop rows the catalogue no longer contains. The upsert alone
                # is additive, so an item that changed global_id - which happens
                # whenever a dataset's `source` changes - stayed behind forever
                # beside its replacement. Re-sourcing movies from the TMDb API
                # left 4,803 dead tmdb_5000_movies rows here, and 88 seeded
                # interactions still pointing at them.
                #
                # user_interactions has ON DELETE CASCADE against this table, so
                # removing a stale entity cleans up its orphaned interactions
                # rather than leaving them referencing something unsearchable.
                keep = [row[0] for row in rows]
                # The collation must match content_entities.global_id exactly. A join
                # across different collations cannot use an index, so this DELETE
                # degrades to a row-by-row scan of the whole catalogue and takes
                # minutes instead of milliseconds.
                cursor.execute(
                    "CREATE TEMPORARY TABLE _keep ("
                    "id VARCHAR(512) COLLATE utf8mb4_bin PRIMARY KEY)"
                )
                for start in range(0, len(keep), CATALOG_UPSERT_BATCH):
                    chunk = keep[start : start + CATALOG_UPSERT_BATCH]
                    cursor.executemany(
                        "INSERT IGNORE INTO _keep (id) VALUES (%s)",
                        [(value,) for value in chunk],
                    )
                cursor.execute(
                    "DELETE ce FROM content_entities ce "
                    "LEFT JOIN _keep k ON k.id = ce.global_id "
                    "WHERE k.id IS NULL"
                )
                removed = cursor.rowcount
                cursor.execute("DROP TEMPORARY TABLE _keep")
                if removed:
                    print(f"  Removed {removed:,} catalogue rows no longer present")
            conn.commit()
        finally:
            conn.close()

        return len(rows)

    def content_exists(self, entity_id: str) -> bool:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM content_entities WHERE global_id = %s LIMIT 1",
                    (entity_id,),
                )
                return cursor.fetchone() is not None
        finally:
            conn.close()

    def log_interaction(self, payload: dict[str, Any]) -> None:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_interactions
                      (user_id, entity_id, event_type, event_value, context, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        payload["user_id"],
                        payload["entity_id"],
                        payload["event_type"],
                        payload.get("event_value"),
                        json.dumps(payload.get("context", {})),
                        payload["timestamp"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def get_user_interaction_state(
        self, user_id: str, entity_id: str
    ) -> dict[str, Any]:
        """Return the latest interaction state for a user/entity pair.

        For toggle interactions (view, like, bookmark, skip, complete) we treat
        them as *active* when the total count is **odd** (first click on,
        second click off, etc.).  For rating we return the most recent value.
        """
        conn = self._connect_database()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(
                    """
                    SELECT event_type, event_value, COUNT(*) AS cnt
                    FROM user_interactions
                    WHERE user_id = %s AND entity_id = %s
                      AND event_type IN ('view','like','bookmark','skip','complete','rating')
                    GROUP BY event_type, event_value
                    ORDER BY event_type, cnt DESC
                    """,
                    (user_id, entity_id),
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        # Aggregate: count total interactions per toggle type
        toggle_counts: dict[str, int] = {}
        latest_rating: float = 0
        for row in rows:
            etype = row["event_type"]
            if etype == "rating":
                # Take the event_value of the row with the highest count
                # (i.e. the most frequently submitted rating)
                # But we want the *most recent* rating, so we need a different
                # query for that.  For now, pick whichever row appeared first.
                if latest_rating == 0 and row["event_value"] is not None:
                    latest_rating = float(row["event_value"])
            else:
                toggle_counts[etype] = toggle_counts.get(etype, 0) + int(row["cnt"])

        # Also fetch the single most recent rating separately
        conn2 = self._connect_database()
        try:
            with conn2.cursor(dictionary=True) as cursor:
                cursor.execute(
                    """
                    SELECT event_value FROM user_interactions
                    WHERE user_id = %s AND entity_id = %s AND event_type = 'rating'
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 1
                    """,
                    (user_id, entity_id),
                )
                rating_row = cursor.fetchone()
        finally:
            conn2.close()

        if rating_row and rating_row["event_value"] is not None:
            latest_rating = float(rating_row["event_value"])

        return {
            "view": (toggle_counts.get("view", 0) % 2) == 1,
            "like": (toggle_counts.get("like", 0) % 2) == 1,
            "bookmark": (toggle_counts.get("bookmark", 0) % 2) == 1,
            "skip": (toggle_counts.get("skip", 0) % 2) == 1,
            "complete": (toggle_counts.get("complete", 0) % 2) == 1,
            "rating": latest_rating,
        }

    def upsert_user_profiles(self, profiles: list[dict[str, Any]]) -> int:
        if not profiles:
            return 0

        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO user_profiles (user_id, preferences, ema_vector)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      preferences = VALUES(preferences),
                      ema_vector = VALUES(ema_vector)
                    """,
                    [
                        (
                            profile["user_id"],
                            json.dumps(profile.get("preferences", {})),
                            json.dumps(profile.get("ema_vector", [])),
                        )
                        for profile in profiles
                    ],
                )
            conn.commit()
        finally:
            conn.close()

        return len(profiles)

    def get_user_ema_vector(self, user_id: str) -> list[float]:
        conn = self._connect_database()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(
                    "SELECT ema_vector FROM user_profiles WHERE user_id = %s",
                    (user_id,),
                )
                row = cursor.fetchone()
        finally:
            conn.close()

        if not row or not row.get("ema_vector"):
            return []
        value = row["ema_vector"]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
        else:
            parsed = value
        if not isinstance(parsed, list):
            return []
        return [float(item) for item in parsed]

    def get_user_profile_preferences(self, user_id: str) -> dict[str, Any]:
        conn = self._connect_database()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(
                    "SELECT preferences FROM user_profiles WHERE user_id = %s",
                    (user_id,),
                )
                row = cursor.fetchone()
        finally:
            conn.close()

        if not row or not row.get("preferences"):
            return {}
        value = row["preferences"]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
        else:
            parsed = value
        return parsed if isinstance(parsed, dict) else {}

    def update_user_ema_vector(self, user_id: str, ema_vector: list[float]) -> None:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_profiles (user_id, preferences, ema_vector)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE ema_vector = VALUES(ema_vector)
                    """,
                    (user_id, json.dumps({}), json.dumps(ema_vector)),
                )
            conn.commit()
        finally:
            conn.close()


    def get_user_preference_profile(self, user_id: str) -> dict[str, Any]:
        conn = self._connect_database()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(
                    """
                    SELECT i.entity_id, i.event_type, i.event_value, c.content_type,
                           c.categories, c.title
                    FROM user_interactions i
                    JOIN content_entities c ON c.global_id = i.entity_id
                    WHERE i.user_id = %s
                    ORDER BY i.timestamp DESC, i.id DESC
                    LIMIT 500
                    """,
                    (user_id,),
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        positive_ids: set[str] = set()
        negative_ids: set[str] = set()
        category_weights: dict[str, float] = {}
        content_type_weights: dict[str, float] = {}

        for row in rows:
            weight = _interaction_weight(row["event_type"], row.get("event_value"))
            entity_id = row["entity_id"]
            if weight > 0:
                positive_ids.add(entity_id)
            elif weight < 0:
                negative_ids.add(entity_id)

            content_type = _clean(row.get("content_type"))
            if content_type:
                content_type_weights[content_type] = content_type_weights.get(content_type, 0.0) + weight

            for category in _split_categories(row.get("categories")):
                category_weights[category] = category_weights.get(category, 0.0) + weight

        return {
            "positive_ids": positive_ids,
            "negative_ids": negative_ids,
            "category_weights": category_weights,
            "content_type_weights": content_type_weights,
        }

    def get_lightgcn_interactions(self, limit: int = 100_000) -> pd.DataFrame:
        conn = self._connect_database()
        try:
            query = """
                SELECT user_id, entity_id, event_type, event_value, timestamp
                FROM user_interactions
                ORDER BY timestamp DESC, id DESC
                LIMIT %s
            """
            return pd.read_sql(query, conn, params=(limit,))
        finally:
            conn.close()


    def bulk_log_interactions(self, interactions: list[dict[str, Any]]) -> int:
        if not interactions:
            return 0

        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO user_interactions
                      (user_id, entity_id, event_type, event_value, context, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            item["user_id"],
                            item["entity_id"],
                            item["event_type"],
                            item.get("event_value"),
                            json.dumps(item.get("context", {})),
                            item["timestamp"],
                        )
                        for item in interactions
                    ],
                )
            conn.commit()
        finally:
            conn.close()

        return len(interactions)

    def _connect_database(self):
        connector = self._connector()
        return connector.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            user=self.settings.mysql_user,
            password=self.settings.mysql_password,
            database=self.settings.mysql_database,
        )

    def _connector(self):
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for MySQL support.\n"
                "Install it with: pip install mysql-connector-python"
            ) from exc
        return mysql.connector


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
