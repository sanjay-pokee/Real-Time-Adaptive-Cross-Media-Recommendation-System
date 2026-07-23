"""FastAPI backend for semantic cross-media recommendations.

Run locally:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.recommender import SemanticRecommender


ContentType = Literal[
    "movie",
    "movies",
    "film",
    "films",
    "book",
    "books",
    "music",
    "song",
    "songs",
    "track",
    "tracks",
]


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return.")
    content_type: ContentType | None = Field(
        None,
        description="Optional filter for movie, book, or music results.",
    )


class RecommendationItem(BaseModel):
    global_id: str
    content_type: str
    source: str
    source_id: str
    title: str = ""
    description: str = ""
    creators: str = ""
    categories: str = ""
    release_date: str = ""
    popularity: float | str = ""
    rating: float | str = ""
    score: float


class RecommendResponse(BaseModel):
    query: str
    top_k: int
    content_type: str | None
    results: list[RecommendationItem]


app = FastAPI(
    title="Cross-Media Recommendation API",
    version="0.1.0",
    description="Semantic recommendation API backed by Sentence-BERT and FAISS.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_recommender() -> SemanticRecommender:
    """Load heavy recommender artifacts once per API process."""
    return SemanticRecommender()


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
    recommender: SemanticRecommender = Depends(get_recommender),
) -> RecommendResponse:
    try:
        results = recommender.recommend(
            payload.query,
            top_k=payload.top_k,
            content_type=payload.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RecommendResponse(
        query=payload.query,
        top_k=payload.top_k,
        content_type=payload.content_type,
        results=results,
    )

