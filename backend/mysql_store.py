"""MySQL schema initialization and interaction persistence."""

from __future__ import annotations

import json
import threading
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
    """
    CREATE TABLE IF NOT EXISTS users (
      -- Real accounts only, with acct_ ids. The demo personas (user_scifi, ...)
      -- have no row here: they are shared sandboxes, not anyone's account.
      user_id VARCHAR(191) PRIMARY KEY,
      email VARCHAR(254) NOT NULL,
      display_name VARCHAR(120) NOT NULL,
      -- NULL for an account that signs in with Google only.
      password_hash VARCHAR(255),
      -- Google's stable account id ("sub"), which survives an email change.
      google_sub VARCHAR(255),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      last_login_at TIMESTAMP NULL,
      UNIQUE KEY uq_users_email (email),
      UNIQUE KEY uq_users_google_sub (google_sub)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_sessions (
      -- SHA-256 of the cookie value, never the value itself, so a copy of
      -- this table does not hand out working sessions.
      token_hash CHAR(64) PRIMARY KEY,
      user_id VARCHAR(191) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      expires_at TIMESTAMP NOT NULL,
      INDEX idx_sessions_user (user_id),
      CONSTRAINT fk_session_user
        FOREIGN KEY (user_id) REFERENCES users(user_id)
        ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS login_failures (
      -- One row per wrong password, so a lockout survives a server restart.
      -- Keyed by the email that was typed, which need not belong to an account.
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      email VARCHAR(254) NOT NULL,
      failed_at DATETIME(6) NOT NULL,
      INDEX idx_login_failures_email (email, failed_at)
    )
    """,
]

ACCOUNT_COLUMNS = "user_id, email, display_name, password_hash, google_sub"


class AccountExistsError(Exception):
    """Raised when an email or Google account is already registered."""


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

        for row in active_interactions(rows):
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
            frame = pd.read_sql(query, conn, params=(limit,))
        finally:
            conn.close()
        # A toggled-off like must not become a LightGCN edge or an EMA step on replay.
        return pd.DataFrame(active_interactions(frame.to_dict("records")), columns=frame.columns)


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

    # ---- accounts and sessions ------------------------------------------

    def create_account(
        self,
        user_id: str,
        email: str,
        display_name: str,
        password_hash: str | None = None,
        google_sub: str | None = None,
    ) -> dict[str, Any]:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (user_id, email, display_name, password_hash, google_sub)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, email, display_name, password_hash, google_sub),
                )
            conn.commit()
        except Exception as exc:
            # 1062 is a duplicate key: the email (or Google account) was taken
            # between the caller's lookup and this insert.
            if getattr(exc, "errno", None) == 1062:
                raise AccountExistsError(email) from exc
            raise
        finally:
            conn.close()
        return self.get_account(user_id)

    def get_account(self, user_id: str) -> dict[str, Any] | None:
        return self._fetch_account("user_id = %s", user_id)

    def get_account_by_email(self, email: str) -> dict[str, Any] | None:
        return self._fetch_account("email = %s", email)

    def get_account_by_google_sub(self, google_sub: str) -> dict[str, Any] | None:
        return self._fetch_account("google_sub = %s", google_sub)

    def link_google_account(self, user_id: str, google_sub: str) -> None:
        """Attach a Google identity to an existing email account.

        The password is removed and every open session ends. Sign-up never
        verified that the person who set the password owns the address, and
        Google just did: keeping the password would let whoever registered
        someone else's email first keep a key to that person's account.
        """
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE users SET google_sub = %s, password_hash = NULL WHERE user_id = %s",
                    (google_sub, user_id),
                )
                cursor.execute("DELETE FROM user_sessions WHERE user_id = %s", (user_id,))
            conn.commit()
        finally:
            conn.close()

    def create_session(self, token_hash: str, user_id: str, days: int) -> None:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                # Expired rows are swept here rather than by a job: a user's own
                # old sessions go whenever they log in again.
                cursor.execute(
                    "DELETE FROM user_sessions WHERE user_id = %s AND expires_at <= NOW()",
                    (user_id,),
                )
                cursor.execute(
                    """
                    INSERT INTO user_sessions (token_hash, user_id, expires_at)
                    VALUES (%s, %s, DATE_ADD(NOW(), INTERVAL %s DAY))
                    """,
                    (token_hash, user_id, days),
                )
                cursor.execute(
                    "UPDATE users SET last_login_at = NOW() WHERE user_id = %s",
                    (user_id,),
                )
            conn.commit()
        finally:
            conn.close()

    def get_session_account(self, token_hash: str) -> dict[str, Any] | None:
        conn = self._connect_database()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(
                    """
                    SELECT u.user_id, u.email, u.display_name, u.password_hash, u.google_sub
                    FROM user_sessions s
                    JOIN users u ON u.user_id = s.user_id
                    WHERE s.token_hash = %s AND s.expires_at > NOW()
                    """,
                    (token_hash,),
                )
                return cursor.fetchone()
        finally:
            conn.close()

    def delete_session(self, token_hash: str) -> None:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM user_sessions WHERE token_hash = %s", (token_hash,))
            conn.commit()
        finally:
            conn.close()

    def count_interactions(self, user_id: str) -> int:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM user_interactions WHERE user_id = %s", (user_id,)
                )
                return int(cursor.fetchone()[0])
        finally:
            conn.close()

    def delete_account(self, user_id: str) -> None:
        """Remove an account and everything recorded about it, in one transaction.

        Interactions and the profile row (preferences and the EMA taste vector)
        are keyed by user_id with no foreign key to users, so they have to be
        deleted by name; sessions would cascade but are deleted explicitly too.
        """
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                try:
                    # Wrong-password records for the account's email. Optional: a
                    # database set up before this table existed has none, and a
                    # failed statement does not abort a MySQL transaction.
                    cursor.execute(
                        "DELETE lf FROM login_failures lf JOIN users u ON u.email = lf.email "
                        "WHERE u.user_id = %s",
                        (user_id,),
                    )
                except Exception:
                    pass
                for table in ("user_interactions", "user_profiles", "user_sessions", "users"):
                    cursor.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_login_failure(self, email: str) -> None:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO login_failures (email, failed_at) VALUES (%s, UTC_TIMESTAMP(6))",
                    (email,),
                )
                # A miss older than a day can never lock anyone again.
                cursor.execute(
                    "DELETE FROM login_failures WHERE email = %s AND failed_at < UTC_TIMESTAMP(6) - INTERVAL 1 DAY",
                    (email,),
                )
            conn.commit()
        finally:
            conn.close()

    def login_failures_within(self, email: str, window_seconds: int) -> tuple[int, float]:
        """(wrong passwords inside the window, seconds until the oldest of them leaves it).

        Both times come from the MySQL server clock, so app servers whose clocks
        disagree still agree on the lockout.
        """
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*), TIMESTAMPDIFF(MICROSECOND, UTC_TIMESTAMP(6), MIN(failed_at))
                    FROM login_failures
                    WHERE email = %s AND failed_at > UTC_TIMESTAMP(6) - INTERVAL %s SECOND
                    """,
                    (email, int(window_seconds)),
                )
                count, offset_us = cursor.fetchone()
        finally:
            conn.close()
        if not count:
            return 0, 0.0
        return int(count), window_seconds + float(offset_us) / 1_000_000

    def clear_login_failures(self, email: str) -> None:
        conn = self._connect_database()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM login_failures WHERE email = %s", (email,))
            conn.commit()
        finally:
            conn.close()

    def _fetch_account(self, where: str, value: str) -> dict[str, Any] | None:
        conn = self._connect_database()
        try:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(f"SELECT {ACCOUNT_COLUMNS} FROM users WHERE {where}", (value,))
                return cursor.fetchone()
        finally:
            conn.close()

    def _connect_database(self):
        """A connection from the shared pool; ``close()`` hands it back.

        Every request used to open a fresh TCP connection and log in to MySQL.
        Connections now come from one pool per database, shared by every
        MySQLStore instance (the recommender builds a new one per request). A
        burst bigger than the pool falls back to a direct connection rather
        than failing, so the pool can only make a request faster, never break it.
        """
        connector = self._connector()
        params = {
            "host": self.settings.mysql_host,
            "port": self.settings.mysql_port,
            "user": self.settings.mysql_user,
            "password": self.settings.mysql_password,
            "database": self.settings.mysql_database,
        }
        pool = _shared_pool(connector, params)
        if pool is not None:
            try:
                return pool.get_connection()
            except connector.errors.PoolError:
                pass  # every pooled connection is busy
        return connector.connect(**params)

    def _connector(self):
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for MySQL support.\n"
                "Install it with: pip install mysql-connector-python"
            ) from exc
        return mysql.connector


POOL_SIZE = 10  # FastAPI runs sync endpoints on up to 40 threads; extra callers connect directly
_POOLS: dict[tuple, Any] = {}
_POOLS_LOCK = threading.Lock()


def _shared_pool(connector: Any, params: dict[str, Any]) -> Any | None:
    """The process-wide pool for these credentials, created on first use.

    Returns None when a pool cannot be created (for example, the database does
    not exist yet). Failure is not cached, so the next call tries again.
    """
    key = (params["host"], params["port"], params["user"], params["database"])
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            try:
                pool = connector.pooling.MySQLConnectionPool(
                    pool_name=f"nexus_{abs(hash(key)) % 10**8}",
                    pool_size=POOL_SIZE,
                    pool_reset_session=True,  # a returned connection never leaks a transaction or session state
                    **params,
                )
            except Exception:
                return None
            _POOLS[key] = pool
        return pool


# The UI's interaction buttons are toggles: a second click logs the same event
# again to switch it off, and get_user_interaction_state reads the state by parity.
TOGGLE_EVENTS = frozenset({"view", "like", "bookmark", "skip", "complete"})


def active_interactions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse a raw event log into the interactions that are currently in effect.

    ``rows`` must be ordered newest first. A toggle counts once, and only if it
    was clicked an odd number of times - so un-liking an item no longer reads as
    a second like. A rating counts once, with its most recent value. Any other
    event (click) counts every time, as before.
    """
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (str(row.get("user_id", "")), str(row["entity_id"]), str(row["event_type"]).lower())
        counts[key] = counts.get(key, 0) + 1

    active: list[dict[str, Any]] = []
    emitted: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("user_id", "")), str(row["entity_id"]), str(row["event_type"]).lower())
        event = key[2]
        if event not in TOGGLE_EVENTS and event != "rating":
            active.append(row)
            continue
        if key in emitted:
            continue
        emitted.add(key)
        if event == "rating" or counts[key] % 2 == 1:
            active.append(row)
    return active


def _interaction_weight(event_type: str, event_value: Any) -> float:
    # Imported lazily so the MySQL scripts don't pay for importing torch.
    from models.graph.lightgcn import interaction_weight

    return interaction_weight(event_type, _number_or_none(event_value))


def _split_categories(value: Any) -> list[str]:
    # Must split exactly like backend.qdrant_recommender._split_categories, which
    # looks these weights up per result.
    text = str(value or "").lower()
    return [part.strip() for part in text.split(",") if part.strip()]


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
