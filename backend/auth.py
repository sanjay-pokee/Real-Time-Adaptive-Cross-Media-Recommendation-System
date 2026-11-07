"""Accounts: email + password and Sign in with Google, on server-side sessions.

Two kinds of user id reach the API. The demo personas (``user_scifi`` ...) are
shared sandboxes with seeded history and need no login, exactly as before.
Real accounts get an ``acct_`` id, and a request that names one must carry
that account's session cookie: without the check, anyone could read or write
another person's history by putting their id in the request body.

The session is an opaque random token in an HttpOnly cookie, stored hashed in
MySQL. Page scripts cannot read it, and logging out deletes it server-side,
which a self-contained token such as a JWT cannot do before it expires.
"""

from __future__ import annotations

import hashlib
import math
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import HTTPException, Request, Response

from backend.schemas import AccountResponse
from backend.settings import Settings

ACCOUNT_PREFIX = "acct_"
SESSION_COOKIE = "nexus_session"
MIN_PASSWORD_LENGTH = 8

# Deliberately loose: one @, a dot in the domain, no spaces. Whether an address
# is real is only knowable by mailing it, which this deployment does not do.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_hasher = PasswordHasher()
# Checked against when the email has no account, so a miss takes as long as a
# wrong password and response time does not reveal which emails are registered.
_DUMMY_HASH = _hasher.hash("not-a-real-password")


def new_account_id() -> str:
    return f"{ACCOUNT_PREFIX}{secrets.token_hex(8)}"


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    return email


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Use at least {MIN_PASSWORD_LENGTH} characters for the password.",
        )


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    try:
        matched = _hasher.verify(password_hash or _DUMMY_HASH, password)
    except (VerificationError, InvalidHashError):
        return False
    return matched and password_hash is not None


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_session_token() -> tuple[str, str]:
    """Return (cookie value, hash to store)."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


def set_session_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_days * 24 * 60 * 60,
        httponly=True,
        # Lax: the cookie rides along when someone follows a link to the site,
        # but not on a POST that another site's page fires at this API.
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def clear_session_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        SESSION_COOKIE,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def session_account(request: Request, store: Any) -> dict[str, Any] | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return store.get_session_account(hash_token(token))


def require_user_access(user_id: str | None, request: Request, store: Any) -> None:
    """Let a request act as ``user_id`` only if it may.

    Demo personas and anonymous requests pass untouched; an ``acct_`` id needs
    that same account's session.
    """
    if not user_id or not user_id.startswith(ACCOUNT_PREFIX):
        return
    try:
        account = session_account(request, store)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not check the session: {exc}") from exc
    if account is None:
        raise HTTPException(status_code=401, detail="Log in to use this account.")
    if account["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="This request is for a different account.")


def account_response(account: dict[str, Any]) -> AccountResponse:
    return AccountResponse(
        user_id=account["user_id"],
        email=account["email"],
        display_name=account["display_name"],
        has_password=bool(account.get("password_hash")),
        has_google=bool(account.get("google_sub")),
    )


def verify_google_credential(credential: str, client_id: str) -> dict[str, Any]:
    """Check a Sign in with Google ID token and return its claims.

    Verifies Google's signature, the expiry, the issuer, and that the token
    was minted for this client id, so a token issued to some other site cannot
    be replayed here. Raises ValueError when any of that fails.
    """
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    return id_token.verify_oauth2_token(credential, google_requests.Request(), client_id)


class LoginThrottle:
    """Lock an email for a while after repeated wrong passwords.

    In memory, so it resets on restart and is per process; enough to make
    guessing a password through the login form impractical.
    """

    def __init__(self, limit: int = 5, window_seconds: int = 15 * 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._misses: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after_minutes(self, key: str) -> int:
        """Minutes until ``key`` may try again; 0 when it may try now."""
        with self._lock:
            misses = self._prune(key)
            if len(misses) < self.limit:
                return 0
            return max(1, math.ceil((misses[0] + self.window - time.monotonic()) / 60))

    def fail(self, key: str) -> None:
        with self._lock:
            self._prune(key).append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._misses.pop(key, None)

    def _prune(self, key: str) -> deque[float]:
        misses = self._misses[key]
        cutoff = time.monotonic() - self.window
        while misses and misses[0] <= cutoff:
            misses.popleft()
        return misses


login_throttle = LoginThrottle()
