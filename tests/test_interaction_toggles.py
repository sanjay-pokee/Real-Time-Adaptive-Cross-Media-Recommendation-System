"""Un-toggling a button must undo it, not count as the same signal twice."""

from fastapi.testclient import TestClient

from backend.app import app, get_ema_store, get_mysql_store
from backend.mysql_store import MySQLStore, active_interactions


def _event(entity, event_type, value=None, user="u1"):
    return {"user_id": user, "entity_id": entity, "event_type": event_type, "event_value": value}


def test_a_like_clicked_twice_is_no_longer_active():
    rows = [_event("movie:a", "like"), _event("movie:a", "like"), _event("movie:b", "like")]

    assert [row["entity_id"] for row in active_interactions(rows)] == ["movie:b"]


def test_a_toggle_clicked_three_times_counts_once():
    rows = [_event("movie:a", "bookmark")] * 3

    assert len(active_interactions(rows)) == 1


def test_only_the_newest_rating_counts():
    # Newest first: the 5 replaced an earlier 1.
    rows = [_event("movie:a", "rating", 5.0), _event("movie:a", "rating", 1.0)]

    assert [row["event_value"] for row in active_interactions(rows)] == [5.0]


def test_clicks_are_not_toggles_and_users_are_kept_apart():
    rows = [
        _event("movie:a", "click"),
        _event("movie:a", "click"),
        _event("movie:a", "like", user="u1"),
        _event("movie:a", "like", user="u2"),
    ]

    active = active_interactions(rows)
    assert sum(row["event_type"] == "click" for row in active) == 2
    assert sorted(row["user_id"] for row in active if row["event_type"] == "like") == ["u1", "u2"]


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_):
        pass

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self, dictionary=False):
        return _Cursor(self.rows)

    def close(self):
        pass


def test_preference_profile_runs_and_ignores_an_unliked_item(monkeypatch):
    # Regression: this method used to raise NameError on its first row, which the
    # recommender swallowed - so the profile bonus was silently never applied.
    def row(entity, event_type, categories):
        return {
            "entity_id": entity,
            "event_type": event_type,
            "event_value": None,
            "content_type": "movie",
            "categories": categories,
            "title": entity,
        }

    rows = [
        row("movie:kept", "like", "Science Fiction, Adventure"),
        row("movie:unliked", "like", "Horror"),
        row("movie:unliked", "like", "Horror"),
        row("movie:skipped", "skip", "Romance"),
    ]
    store = MySQLStore.__new__(MySQLStore)
    monkeypatch.setattr(store, "_connect_database", lambda: _Connection(rows), raising=False)

    profile = store.get_user_preference_profile("u1")

    assert profile["positive_ids"] == {"movie:kept"}
    assert profile["negative_ids"] == {"movie:skipped"}
    assert profile["category_weights"]["science fiction"] > 0
    assert "horror" not in profile["category_weights"]
    assert profile["category_weights"]["romance"] < 0


class _ToggleStore:
    """Just enough of MySQLStore for POST /interactions."""

    class settings:
        ema_alpha = 0.25

    def __init__(self):
        self.events = []
        self.saved_vectors = []

    def get_account(self, user_id):
        return None

    def content_exists(self, entity_id):
        return True

    def log_interaction(self, payload):
        self.events.append((payload["user_id"], payload["entity_id"], payload["event_type"]))

    def get_user_interaction_state(self, user_id, entity_id):
        def on(event_type):
            return self.events.count((user_id, entity_id, event_type)) % 2 == 1

        return {name: on(name) for name in ("view", "like", "bookmark", "skip", "complete")} | {"rating": 0}

    def get_user_ema_vector(self, user_id):
        return []

    def update_user_ema_vector(self, user_id, vector):
        self.saved_vectors.append(vector)


class _Ema:
    def update_profile_vector(self, current, entity_id, event_type, event_value, alpha=0.25):
        return [1.0]


def test_unliking_does_not_move_the_taste_vector_again():
    store = _ToggleStore()
    app.dependency_overrides[get_mysql_store] = lambda: store
    app.dependency_overrides[get_ema_store] = lambda: _Ema()
    try:
        client = TestClient(app)
        like = {"user_id": "user_scifi", "entity_id": "movie:test:1", "event_type": "like"}

        assert client.post("/interactions", json=like).status_code == 200
        assert len(store.saved_vectors) == 1  # the like moved the vector
        assert client.post("/interactions", json=like).status_code == 200
        assert len(store.saved_vectors) == 1  # the un-like did not
        assert client.post("/interactions", json=like).status_code == 200
        assert len(store.saved_vectors) == 2  # liking it again does
    finally:
        app.dependency_overrides.clear()
