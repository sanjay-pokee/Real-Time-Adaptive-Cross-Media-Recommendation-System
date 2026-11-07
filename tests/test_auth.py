import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import auth
from backend.app import app, get_ema_store, get_mysql_store, get_recommender
from backend.mysql_store import AccountExistsError
from backend.settings import Settings

GOOGLE_CLIENT_ID = "test-client.apps.googleusercontent.com"
PASSWORD = "correct horse battery"


class FakeAccountStore:
    """In-memory stand-in for the account half of MySQLStore."""

    def __init__(self, google_client_id=GOOGLE_CLIENT_ID):
        self.settings = Settings(google_client_id=google_client_id)
        self.accounts = {}
        self.sessions = {}

    def create_account(self, user_id, email, display_name, password_hash=None, google_sub=None):
        if self.get_account_by_email(email) or (google_sub and self.get_account_by_google_sub(google_sub)):
            raise AccountExistsError(email)
        self.accounts[user_id] = {
            "user_id": user_id,
            "email": email,
            "display_name": display_name,
            "password_hash": password_hash,
            "google_sub": google_sub,
        }
        return dict(self.accounts[user_id])

    def get_account(self, user_id):
        account = self.accounts.get(user_id)
        return dict(account) if account else None

    def get_account_by_email(self, email):
        return next((dict(a) for a in self.accounts.values() if a["email"] == email), None)

    def get_account_by_google_sub(self, google_sub):
        return next((dict(a) for a in self.accounts.values() if a["google_sub"] == google_sub), None)

    def link_google_account(self, user_id, google_sub):
        self.accounts[user_id].update(google_sub=google_sub, password_hash=None)
        self.sessions = {k: v for k, v in self.sessions.items() if v != user_id}

    def create_session(self, token_hash, user_id, days):
        self.sessions[token_hash] = user_id

    def get_session_account(self, token_hash):
        user_id = self.sessions.get(token_hash)
        return self.get_account(user_id) if user_id else None

    def delete_session(self, token_hash):
        self.sessions.pop(token_hash, None)


class FakeRecommender:
    def recommend(self, query, top_k=10, content_type=None, user_id=None, audience=None):
        return []


@pytest.fixture
def store(monkeypatch):
    fake = FakeAccountStore()
    app.dependency_overrides[get_mysql_store] = lambda: fake
    app.dependency_overrides[get_recommender] = lambda: FakeRecommender()
    app.dependency_overrides[get_ema_store] = lambda: None
    # One throttle per test, or wrong passwords from one test lock the next.
    monkeypatch.setattr(auth, "login_throttle", auth.LoginThrottle())
    yield fake
    app.dependency_overrides.clear()


def _signup(client, email="asha@example.com", password=PASSWORD, name="Asha"):
    return client.post(
        "/auth/signup",
        json={"email": email, "password": password, "display_name": name},
    )


def _google_claims(monkeypatch, **overrides):
    claims = {"sub": "google-123", "email": "asha@example.com", "email_verified": True, "name": "Asha G"}
    claims.update(overrides)
    seen = {}

    def verify(credential, client_id):
        seen["client_id"] = client_id
        return claims

    monkeypatch.setattr(auth, "verify_google_credential", verify)
    return seen


def test_auth_config_serves_the_google_client_id(store):
    response = TestClient(app).get("/auth/config")

    assert response.status_code == 200
    assert response.json() == {"google_client_id": GOOGLE_CLIENT_ID}


def test_signup_starts_an_http_only_session(store):
    client = TestClient(app)

    response = _signup(client, email="  Asha@Example.COM ")

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "asha@example.com"
    assert body["user_id"].startswith("acct_")
    assert body == {**body, "has_password": True, "has_google": False}
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{auth.SESSION_COOKIE}=")
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie
    # Stored as an Argon2 hash, never as the password itself.
    stored = store.accounts[body["user_id"]]["password_hash"]
    assert stored.startswith("$argon2") and PASSWORD not in stored
    # The cookie holds the token; the store only ever sees its hash.
    token = client.cookies.get(auth.SESSION_COOKIE)
    assert token not in store.sessions and auth.hash_token(token) in store.sessions

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["user_id"] == body["user_id"]


def test_signup_rejects_a_taken_email_in_any_case(store):
    assert _signup(TestClient(app)).status_code == 201

    response = _signup(TestClient(app), email="ASHA@example.com")

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.parametrize(
    ("email", "password", "name", "message"),
    [
        ("not-an-email", PASSWORD, "Asha", "valid email"),
        ("asha@example.com", "short", "Asha", "at least 8"),
        ("asha@example.com", PASSWORD, "   ", "your name"),
    ],
)
def test_signup_explains_what_is_wrong(store, email, password, name, message):
    response = _signup(TestClient(app), email=email, password=password, name=name)

    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert store.accounts == {}


def test_login_checks_the_password(store):
    _signup(TestClient(app))
    client = TestClient(app)

    wrong = client.post("/auth/login", json={"email": "asha@example.com", "password": "nope-nope"})
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Wrong email or password."
    assert client.get("/auth/me").status_code == 401

    right = client.post("/auth/login", json={"email": "ASHA@example.com", "password": PASSWORD})
    assert right.status_code == 200
    assert client.get("/auth/me").json()["email"] == "asha@example.com"


def test_unknown_email_gets_the_same_answer_as_a_wrong_password(store):
    response = TestClient(app).post(
        "/auth/login", json={"email": "nobody@example.com", "password": PASSWORD}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Wrong email or password."


def test_login_locks_after_five_wrong_passwords(store):
    _signup(TestClient(app))
    client = TestClient(app)
    for _ in range(5):
        client.post("/auth/login", json={"email": "asha@example.com", "password": "nope-nope"})

    # Even the right password is refused until the window passes.
    response = client.post("/auth/login", json={"email": "asha@example.com", "password": PASSWORD})

    assert response.status_code == 429
    assert "Try again in 15 minutes" in response.json()["detail"]


def test_logout_ends_the_session_server_side(store):
    client = TestClient(app)
    _signup(client)
    token = client.cookies.get(auth.SESSION_COOKIE)

    response = client.post("/auth/logout")

    assert response.status_code == 200
    assert store.sessions == {}
    # Replaying the old cookie no longer works.
    replay = TestClient(app, cookies={auth.SESSION_COOKIE: token})
    assert replay.get("/auth/me").status_code == 401


def test_google_sign_in_creates_an_account(store, monkeypatch):
    seen = _google_claims(monkeypatch)
    client = TestClient(app)

    response = client.post("/auth/google", json={"credential": "id-token"})

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "Asha G"
    assert body["has_google"] and not body["has_password"]
    assert seen["client_id"] == GOOGLE_CLIENT_ID
    assert client.get("/auth/me").json()["user_id"] == body["user_id"]

    # Signing in again finds the same account rather than making a second.
    again = TestClient(app).post("/auth/google", json={"credential": "id-token"})
    assert again.json()["user_id"] == body["user_id"]
    assert len(store.accounts) == 1


def test_google_joins_an_email_account_and_retires_its_password(store, monkeypatch):
    password_client = TestClient(app)
    created = _signup(password_client).json()
    _google_claims(monkeypatch)

    response = TestClient(app).post("/auth/google", json={"credential": "id-token"})

    assert response.json()["user_id"] == created["user_id"]
    assert response.json()["has_google"] and not response.json()["has_password"]
    # Whoever set the unverified password is signed out and cannot sign back in with it.
    assert password_client.get("/auth/me").status_code == 401
    login = TestClient(app).post("/auth/login", json={"email": "asha@example.com", "password": PASSWORD})
    assert login.status_code == 400
    assert "Google" in login.json()["detail"]


@pytest.mark.parametrize(
    ("claims", "status"),
    [({"email_verified": False}, 401), ({"sub": "google-other"}, 409)],
)
def test_google_refuses_unverified_or_reassigned_emails(store, monkeypatch, claims, status):
    _google_claims(monkeypatch)
    TestClient(app).post("/auth/google", json={"credential": "id-token"})
    _google_claims(monkeypatch, **claims)

    response = TestClient(app).post("/auth/google", json={"credential": "id-token"})

    assert response.status_code == status
    assert len(store.accounts) == 1


def test_google_rejects_a_token_that_fails_verification(store, monkeypatch):
    def verify(credential, client_id):
        raise ValueError("Token used too late")

    monkeypatch.setattr(auth, "verify_google_credential", verify)

    response = TestClient(app).post("/auth/google", json={"credential": "forged"})

    assert response.status_code == 401
    assert store.accounts == {}


def test_google_sign_in_needs_a_client_id(store):
    app.dependency_overrides[get_mysql_store] = lambda: FakeAccountStore(google_client_id=None)

    response = TestClient(app).post("/auth/google", json={"credential": "id-token"})

    assert response.status_code == 503
    assert TestClient(app).get("/auth/config").json() == {"google_client_id": None}


def test_an_account_id_needs_that_accounts_session(store):
    owner = TestClient(app)
    account_id = _signup(owner).json()["user_id"]
    stranger = TestClient(app)
    _signup(stranger, email="ravi@example.com", name="Ravi")
    request = {"query": "space adventure", "user_id": account_id}

    assert TestClient(app).post("/recommend", json=request).status_code == 401
    assert stranger.post("/recommend", json=request).status_code == 403
    assert owner.post("/recommend", json=request).status_code == 200
    assert TestClient(app).get(f"/interactions/{account_id}/movie:test:1").status_code == 401
    assert stranger.post(
        "/interactions",
        json={"user_id": account_id, "entity_id": "movie:test:1", "event_type": "like"},
    ).status_code == 403


def test_demo_personas_need_no_login(store):
    response = TestClient(app).post("/recommend", json={"query": "space adventure", "user_id": "user_scifi"})

    assert response.status_code == 200
