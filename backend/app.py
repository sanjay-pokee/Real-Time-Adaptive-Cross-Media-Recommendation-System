"""FastAPI backend for semantic cross-media recommendations.

Run locally:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.mysql_store import MySQLStore
from backend.qdrant_recommender import QdrantRecommender
from backend.schemas import (
    InteractionRequest,
    InteractionResponse,
    ItemRecommendRequest,
    ItemRecommendResponse,
    RecommendRequest,
    RecommendResponse,
)


app = FastAPI(
    title="Cross-Media Recommendation API",
    version="0.2.0",
    description="Semantic recommendation API backed by Sentence-BERT and Qdrant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_recommender() -> QdrantRecommender:
    """Load heavy recommender artifacts once per API process."""
    return QdrantRecommender()


@lru_cache(maxsize=1)
def get_mysql_store() -> MySQLStore:
    return MySQLStore()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Welcome to Cross-Media Recommendation API",
        "docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health",
    }


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
