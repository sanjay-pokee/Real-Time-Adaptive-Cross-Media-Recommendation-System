"""FastAPI backend for semantic cross-media recommendations.

Run locally:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.ema_recommender import EMAEmbeddingStore
from backend.mysql_store import MySQLStore
from backend.qdrant_recommender import QdrantRecommender
from backend.schemas import (
    InteractionRequest,
    InteractionResponse,
    ItemRecommendRequest,
    ItemRecommendResponse,
    RecommendRequest,
    RecommendResponse,
    SuggestItem,
    SuggestResponse,
    UserInteractionState,
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
    return request.app.state.mysql_store


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

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(
    payload: RecommendRequest,
    recommender: QdrantRecommender = Depends(get_recommender),
) -> RecommendResponse:
    try:
        try:
            results = recommender.recommend(
                payload.query,
                top_k=payload.top_k,
                content_type=payload.content_type,
                user_id=payload.user_id,
            )
        except TypeError:
            results = recommender.recommend(
                payload.query,
                top_k=payload.top_k,
                content_type=payload.content_type,
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
        top_k=payload.top_k,
        content_type=payload.content_type,
        results=results,
    )


@app.post("/recommend/item", response_model=ItemRecommendResponse)
def recommend_from_item(
    payload: ItemRecommendRequest,
    recommender: QdrantRecommender = Depends(get_recommender),
) -> ItemRecommendResponse:
    try:
        try:
            results = recommender.recommend_from_item(
                payload.global_id,
                top_k=payload.top_k,
                content_type=payload.content_type,
                user_id=payload.user_id,
            )
        except TypeError:
            results = recommender.recommend_from_item(
                payload.global_id,
                top_k=payload.top_k,
                content_type=payload.content_type,
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
        results=results,
    )


@app.post("/interactions", response_model=InteractionResponse)
def log_interaction(
    payload: InteractionRequest,
    store: MySQLStore = Depends(get_mysql_store),
    ema_store: EMAEmbeddingStore | None = Depends(get_ema_store),
) -> InteractionResponse:
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
    store: MySQLStore = Depends(get_mysql_store),
) -> UserInteractionState:
    """Return the current interaction state (likes, bookmarks, rating, etc.) for
    a specific user + content entity combination."""
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
