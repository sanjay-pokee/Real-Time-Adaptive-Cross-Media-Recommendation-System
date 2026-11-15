"""The login lockout is stored with the accounts, so a server restart does not lift it."""

import time

import pytest
from fastapi.testclient import TestClient

from backend import auth
from backend.app import app, get_ema_store, get_mysql_store, get_recommender
from tests.test_auth import PASSWORD, FakeAccountStore, FakeRecommender


class StoreWithLockout(FakeAccountStore):
    """FakeAccountStore plus the three login_failures methods of MySQLStore."""

    def __init__(self):
        super().__init__()
        self.failures = {}
        self.broken = False

    def _check(self):
        if self.broken:
            raise RuntimeError("login_failures table unavailable")

    def record_login_failure(self, email):
        self._check()
        self.failures.setdefault(email, []).append(time.time())

    def login_failures_within(self, email, window_seconds):
        self._check()
        now = time.time()
        recent = [t for t in self.failures.get(email, []) if t > now - window_seconds]
        if not recent:
            return 0, 0.0
        return len(recent), min(recent) + window_seconds - now

    def clear_login_failures(self, email):
        self._check()
        self.failures.pop(email, None)


@pytest.fixture
def store(monkeypatch):
    fake = StoreWithLockout()
    app.dependency_overrides[get_mysql_store] = lambda: fake
    app.dependency_overrides[get_recommender] = lambda: FakeRecommender()
    app.dependency_overrides[get_ema_store] = lambda: None
    monkeypatch.setattr(auth, "login_throttle", auth.LoginThrottle())
    yield fake
    app.dependency_overrides.clear()


def _login(client, password):
    return client.post("/auth/login", json={"email": "asha@example.com", "password": password})


def _signup(client):
    return client.post("/auth/signup", json={"email": "asha@example.com", "password": PASSWORD, "display_name": "Asha"})


def test_lockout_survives_a_server_restart(store, monkeypatch):
    client = TestClient(app)
    _signup(client)
    for _ in range(5):
        assert _login(client, "wrong password!").status_code == 401

    # A restart wipes process memory; the stored failures are still there.
    monkeypatch.setattr(auth, "login_throttle", auth.LoginThrottle())

    response = _login(client, PASSWORD)
    assert response.status_code == 429
    assert "15 minutes" in response.json()["detail"]


def test_a_correct_password_clears_the_stored_failures(store):
    client = TestClient(app)
    _signup(client)
    for _ in range(3):
        _login(client, "wrong password!")

    assert _login(client, PASSWORD).status_code == 200
    assert store.failures.get("asha@example.com") is None


def test_a_broken_lockout_table_falls_back_to_memory_and_still_locks(store):
    client = TestClient(app)
    _signup(client)
    store.broken = True

    for _ in range(5):
        assert _login(client, "wrong password!").status_code == 401
    assert _login(client, PASSWORD).status_code == 429


def test_stores_without_the_table_methods_keep_the_in_memory_throttle():
    assert auth.throttle_for(FakeAccountStore()) is auth.login_throttle
    assert isinstance(auth.throttle_for(StoreWithLockout()), auth.StoredLoginThrottle)
