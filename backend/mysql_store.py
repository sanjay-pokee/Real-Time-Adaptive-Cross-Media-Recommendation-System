"""MySQL schema initialization and interaction persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.settings import Settings, get_settings


CREATE_DATABASE_SQL = "CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"

TABLE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS content_entities (
      global_id VARCHAR(512) PRIMARY KEY,
      content_type VARCHAR(64) NOT NULL,
      source VARCHAR(128) NOT NULL,
      source_id VARCHAR(256) NOT NULL,
      title TEXT NOT NULL,
      description TEXT,
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
      entity_id VARCHAR(512) NOT NULL,
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
            conn.commit()
        finally:
            conn.close()


    def upsert_content_catalog(self, catalog_path: Path) -> int:
        if not catalog_path.exists():
            raise FileNotFoundError(f"Content catalog not found: {catalog_path}")

        catalog = pd.read_csv(catalog_path)
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
                cursor.executemany(
                    """
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
                    """,
                    rows,
                )
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
