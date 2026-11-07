"""FastAPI backend for semantic cross-media recommendations.

Run locally:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend import auth
from backend.audience import AudienceContext
from backend.domains import get_registry
from backend.ema_recommender import EMAEmbeddingStore
from backend.mysql_store import AccountExistsError, MySQLStore
from backend.qdrant_recommender import QdrantRecommender
from backend.schemas import (
    AccountResponse,
    AuthConfigResponse,
    DomainSummary,
    DomainsResponse,
    GoogleLoginRequest,
    InteractionRequest,
    InteractionResponse,
    ItemRecommendRequest,
    ItemRecommendResponse,
    LoginRequest,
    RecommendRequest,
    RecommendResponse,
    SignupRequest,
    SuggestItem,
    SuggestResponse,
    UserInteractionState,
)


def _audience_for(payload: RecommendRequest | ItemRecommendRequest) -> AudienceContext:
    """Build the eligibility context for this request.

    Always builds one, including when the caller declares nothing. A request
    that says nothing about its viewer is not a request from an adult, and
    ``AudienceContext.effective_age`` already encodes that: an absent age
    resolves to ``UNKNOWN_AGE_CEILING`` (teen) rather than to unrestricted.

    Returning None here skipped that policy entirely, so an undeclared viewer
    was served 18+ items - while the UI sat above them stating "it caps at the
    teen ceiling and fails closed". The constraint layer was right; this was
    the only thing standing between it and the request.
    """
    return AudienceContext(
        age=payload.age,
        domain=payload.domain,
        safe_mode=payload.safe_mode,
    )


# ---------------------------------------------------------------------------
# Lifespan: initialize heavy singletons ONCE per worker process and clean up.
# This avoids the "qdrant_storage already locked" error that occurs when
# --reload spawns multiple processes and each tries to open the local DB.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    app.state.recommender = QdrantRecommender()
    app.state.recommender.warm_up()

    app.state.mysql_store = MySQLStore()

    try:
        app.state.ema_store = EMAEmbeddingStore()
    except (FileNotFoundError, ValueError):
        app.state.ema_store = None

    yield  # server is running

    # ---- shutdown: release Qdrant file lock cleanly ----
    try:
        app.state.recommender.client.close()
    except Exception:
        pass


app = FastAPI(
    title="Cross-Media Recommendation API",
    version="0.2.0",
    description="Semantic recommendation API backed by Sentence-BERT and Qdrant.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency helpers â€” read singletons from app.state (set in lifespan)
# ---------------------------------------------------------------------------

def get_recommender(request: Request) -> QdrantRecommender:
    return request.app.state.recommender


def get_mysql_store(request: Request) -> MySQLStore:
    # Every endpoint that takes a user id now needs the store for the account
    # check, including /recommend. A client that skipped the lifespan (a bare
    # TestClient) gets a fresh one; it opens a connection per call, so it costs
    # nothing until a request actually names an account.
    store = getattr(request.app.state, "mysql_store", None)
    return store if store is not None else MySQLStore()


def get_ema_store(request: Request) -> EMAEmbeddingStore | None:
    return request.app.state.ema_store


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Welcome to Cross-Media Recommendation API",
        "docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health",
    }


@app.get("/search/suggest", response_model=SuggestResponse)
def search_suggest(
    q: str,
    limit: int = 8,
    recommender: QdrantRecommender = Depends(get_recommender),
) -> SuggestResponse:
    """Fast prefix-based autocomplete over titles and creator names.

    Returns up to `limit` suggestions mixing:
    - content titles that contain the query substring
    - creator / cast / director names that contain the query substring
    """
    import unicodedata

    def _norm(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        return without_marks.casefold()

    q_stripped = q.strip()
    if not q_stripped or len(q_stripped) < 2:
        return SuggestResponse(q=q, suggestions=[])

    q_norm = _norm(q_stripped)
    catalog = recommender.catalog

    suggestions: list[SuggestItem] = []
    seen_labels: set[str] = set()

    max_title = min(3, limit)

    # Title matches are intentionally capped so creator/person matches always
    # have room in the dropdown.
    title_mask = catalog["title"].fillna("").apply(lambda t: q_norm in _norm(str(t)))
    for _, row in catalog[title_mask].head(max_title).iterrows():
        label = str(row.get("title", "")).strip()
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        content_type = str(row.get("content_type", "")).capitalize()
        cats = str(row.get("categories", "")).split(",")
        hint_cat = cats[0].strip() if cats else content_type
        suggestions.append(
            SuggestItem(
                label=label,
                hint=f"{content_type} - {hint_cat}" if hint_cat else content_type,
                kind="title",
                query=label,
            )
        )

    # Person / creator matches. Exploding the comma-separated creator column
    # lets pandas do most of the filtering and counting work.
    slots_left = limit - len(suggestions)
    if slots_left <= 0:
        return SuggestResponse(q=q, suggestions=suggestions[:limit])

    person_series = catalog["creators"].fillna("").str.split(",").explode().str.strip()
    person_series = person_series[person_series.ne("")]
    normalized_people = person_series.apply(_norm)
    matches = person_series[normalized_people.str.contains(q_norm, regex=False, na=False)]
    person_counts = matches.value_counts()

    def _person_rank(item: tuple[str, int]) -> tuple[int, int, str]:
        name, count = item
        name_norm = _norm(str(name).strip())
        if name_norm == q_norm:
            match_rank = 0
        elif name_norm.startswith(q_norm):
            match_rank = 1
        else:
            match_rank = 2
        return (match_rank, -int(count), name_norm)

    ranked_people = sorted(person_counts.items(), key=_person_rank)
    for name, count in ranked_people[: slots_left * 3]:
        name = str(name).strip()
        if not name or name in seen_labels:
            continue
        seen_labels.add(name)
        suggestions.append(
            SuggestItem(
                label=name,
                hint=f"Actor / Director - {count} title{'s' if count != 1 else ''}",
                kind="person",
                query=name,
            )
        )
        if len(suggestions) >= limit:
            break

    return SuggestResponse(q=q, suggestions=suggestions[:limit])

@app.get("/domains", response_model=DomainsResponse)
def domains() -> DomainsResponse:
    """Advertise the verticals this deployment serves.

    Read straight from config/domains.yaml, so a client discovers new domains
    without a code change on either side.
    """
    registry = get_registry()
    return DomainsResponse(
        domains=[
            DomainSummary(
                name=spec.name,
                label=spec.label,
                description=spec.description,
                content_types=list(spec.content_types),
                risk_tier=spec.risk_tier,
                advisory=spec.advisory,
            )
            for spec in sorted(registry.domains.values(), key=lambda d: d.name)
        ],
        content_types=list(registry.content_type_names()),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Accounts (see backend/auth.py)
# ---------------------------------------------------------------------------

@app.get("/auth/config", response_model=AuthConfigResponse)
def auth_config(store: MySQLStore = Depends(get_mysql_store)) -> AuthConfigResponse:
    return AuthConfigResponse(google_client_id=store.settings.google_client_id)


@app.post("/auth/signup", response_model=AccountResponse, status_code=201)
def signup(
    payload: SignupRequest,
    response: Response,
    store: MySQLStore = Depends(get_mysql_store),
) -> AccountResponse:
    email = auth.normalize_email(payload.email)
    display_name = payload.display_name.strip()
    if not display_name:
        raise HTTPException(status_code=400, detail="Enter your name.")
    auth.check_password(payload.password)

    try:
        existing = store.get_account_by_email(email)
        if existing is not None:
            raise _account_exists(existing)
        account = store.create_account(
            auth.new_account_id(),
            email,
            display_name,
            password_hash=auth.hash_password(payload.password),
        )
        return _start_session(response, store, account)
    except HTTPException:
        raise
    except AccountExistsError as exc:
        raise _account_exists(None) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not create the account: {exc}") from exc


@app.post("/auth/login", response_model=AccountResponse)
def login(
    payload: LoginRequest,
    response: Response,
    store: MySQLStore = Depends(get_mysql_store),
) -> AccountResponse:
    email = payload.email.strip().lower()
    wait = auth.login_throttle.retry_after_minutes(email)
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"Too many wrong passwords. Try again in {wait} minute{'s' if wait != 1 else ''}.",
        )

    try:
        account = store.get_account_by_email(email)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not check the account: {exc}") from exc

    if account is not None and not account.get("password_hash"):
        raise HTTPException(
            status_code=400,
            detail="This email signs in with Google. Use the Google button.",
        )
    if not auth.verify_password(account["password_hash"] if account else None, payload.password):
        auth.login_throttle.fail(email)
        raise HTTPException(status_code=401, detail="Wrong email or password.")
    auth.login_throttle.reset(email)

    try:
        return _start_session(response, store, account)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not start the session: {exc}") from exc


@app.post("/auth/google", response_model=AccountResponse)
def login_with_google(
    payload: GoogleLoginRequest,
    response: Response,
    store: MySQLStore = Depends(get_mysql_store),
) -> AccountResponse:
    client_id = store.settings.google_client_id
    if not client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not set up on this server.")
    try:
        claims = auth.verify_google_credential(payload.credential, client_id)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Google sign-in could not be verified.") from exc
    if not claims.get("email_verified"):
        raise HTTPException(status_code=401, detail="This Google account's email is not verified.")

    google_sub = str(claims["sub"])
    email = str(claims["email"]).strip().lower()
    display_name = (str(claims.get("name") or "").strip() or email.split("@")[0])[:120]

    try:
        account = store.get_account_by_google_sub(google_sub)
        if account is None:
            account = store.get_account_by_email(email)
            if account is None:
                account = store.create_account(
                    auth.new_account_id(), email, display_name, google_sub=google_sub
                )
            elif account.get("google_sub"):
                # Same address, different Google identity: the address was
                # reassigned, and the account belongs to its previous owner.
                raise HTTPException(
                    status_code=409,
                    detail="This email is linked to a different Google account.",
                )
            else:
                # Someone signed up with this email and a password. Google has
                # now proven who owns the address, so it becomes the same
                # account - see MySQLStore.link_google_account for why the
                # password goes.
                store.link_google_account(account["user_id"], google_sub)
                account = store.get_account(account["user_id"])
        return _start_session(response, store, account)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not sign in with Google: {exc}") from exc


@app.post("/auth/logout")
def logout(
    request: Request,
    response: Response,
    store: MySQLStore = Depends(get_mysql_store),
) -> dict[str, str]:
    token = request.cookies.get(auth.SESSION_COOKIE)
    if token:
        try:
            store.delete_session(auth.hash_token(token))
        except Exception:
            pass  # the cookie is cleared regardless; the row expires on its own
    auth.clear_session_cookie(response, store.settings)
    return {"status": "logged_out"}


@app.get("/auth/me", response_model=AccountResponse)
def me(request: Request, store: MySQLStore = Depends(get_mysql_store)) -> AccountResponse:
    try:
        account = auth.session_account(request, store)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not check the session: {exc}") from exc
    if account is None:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return auth.account_response(account)


def _start_session(response: Response, store: MySQLStore, account: dict[str, Any]) -> AccountResponse:
    token, token_hash = auth.new_session_token()
    store.create_session(token_hash, account["user_id"], store.settings.session_days)
    auth.set_session_cookie(response, token, store.settings)
    return auth.account_response(account)


def _account_exists(account: dict[str, Any] | None) -> HTTPException:
    if account is not None and not account.get("password_hash"):
        detail = "An account with this email signs in with Google. Use the Google button."
    else:
        detail = "An account with this email already exists. Log in instead."
    return HTTPException(status_code=409, detail=detail)


# ---------------------------------------------------------------------------
# Recommendations and interactions
# ---------------------------------------------------------------------------

@app.post("/recommend", response_model=RecommendResponse)
def recommend(
    payload: RecommendRequest,
    request: Request,
    recommender: QdrantRecommender = Depends(get_recommender),
    store: MySQLStore = Depends(get_mysql_store),
) -> RecommendResponse:
    auth.require_user_access(payload.user_id, request, store)
    # Deliberately no retry without ``audience``. A fallback used to catch
    # TypeError and call recommend(query, top_k, content_type) again, so any
    # TypeError raised *inside* the recommender re-ran the search with no
    # eligibility gate and served the unfiltered page with a 200. An internal
    # error has to surface as a 500, never as unfiltered results.
    try:
        audience = _audience_for(payload)
        corrected = recommender.correct_query(payload.query) if payload.spellcheck else None
        results = recommender.recommend(
            corrected or payload.query,
            top_k=payload.top_k,
            content_type=payload.content_type,
            user_id=payload.user_id,
            audience=audience,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RecommendResponse(
        query=payload.query,
        corrected_query=corrected,
        top_k=payload.top_k,
        content_type=payload.content_type,
        domain=payload.domain,
        advisory=audience.advisory() if audience else None,
        results=results,
    )


@app.post("/recommend/item", response_model=ItemRecommendResponse)
def recommend_from_item(
    payload: ItemRecommendRequest,
    request: Request,
    recommender: QdrantRecommender = Depends(get_recommender),
    store: MySQLStore = Depends(get_mysql_store),
) -> ItemRecommendResponse:
    auth.require_user_access(payload.user_id, request, store)
    # No retry without ``audience`` - see recommend() above.
    try:
        audience = _audience_for(payload)
        results = recommender.recommend_from_item(
            payload.global_id,
            top_k=payload.top_k,
            content_type=payload.content_type,
            user_id=payload.user_id,
            audience=audience,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ItemRecommendResponse(
        global_id=payload.global_id,
        top_k=payload.top_k,
        content_type=payload.content_type,
        domain=payload.domain,
        advisory=audience.advisory() if audience else None,
        results=results,
    )


@app.post("/interactions", response_model=InteractionResponse)
def log_interaction(
    payload: InteractionRequest,
    request: Request,
    store: MySQLStore = Depends(get_mysql_store),
    ema_store: EMAEmbeddingStore | None = Depends(get_ema_store),
) -> InteractionResponse:
    auth.require_user_access(payload.user_id, request, store)
    try:
        if not store.content_exists(payload.entity_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown entity_id: {payload.entity_id}. Use a real global_id "
                    "from /recommend results and make sure scripts.load_catalog_mysql ran."
                ),
            )
        store.log_interaction(payload.model_dump())
        if ema_store is not None:
            current_vector = store.get_user_ema_vector(payload.user_id)
            updated_vector = ema_store.update_profile_vector(
                current_vector,
                payload.entity_id,
                payload.event_type,
                payload.event_value,
                alpha=store.settings.ema_alpha,
            )
            if updated_vector is not None:
                store.update_user_ema_vector(payload.user_id, updated_vector)
    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not log interaction: {exc}",
        ) from exc

    return InteractionResponse(
        status="ok",
        user_id=payload.user_id,
        entity_id=payload.entity_id,
        event_type=payload.event_type,
    )


@app.get("/interactions/{user_id}/{entity_id:path}", response_model=UserInteractionState)
def get_interaction_state(
    user_id: str,
    entity_id: str,
    request: Request,
    store: MySQLStore = Depends(get_mysql_store),
) -> UserInteractionState:
    """Return the current interaction state (likes, bookmarks, rating, etc.) for
    a specific user + content entity combination."""
    auth.require_user_access(user_id, request, store)
    try:
        state = store.get_user_interaction_state(user_id, entity_id)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not fetch interaction state: {exc}",
        ) from exc

    return UserInteractionState(
        user_id=user_id,
        entity_id=entity_id,
        view=state["view"],
        like=state["like"],
        bookmark=state["bookmark"],
        skip=state["skip"],
        complete=state["complete"],
        rating=state["rating"],
    )
