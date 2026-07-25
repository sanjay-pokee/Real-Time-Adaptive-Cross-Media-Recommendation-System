"""API schemas for the recommendation backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


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

InteractionType = Literal[
    "view",
    "click",
    "like",
    "bookmark",
    "rating",
    "skip",
    "complete",
]


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    user_id: str | None = Field(None, description="Optional user id for personalization.")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return.")
    content_type: ContentType | None = Field(
        None,
        description="Optional filter for movie, book, or music results.",
    )


class ItemRecommendRequest(BaseModel):
    global_id: str = Field(..., min_length=1, description="Catalog item id to recommend from.")
    user_id: str | None = Field(None, description="Optional user id for personalization.")
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


class ItemRecommendResponse(BaseModel):
    global_id: str
    top_k: int
    content_type: str | None
    results: list[RecommendationItem]


class InteractionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    entity_id: str = Field(..., min_length=1)
    event_type: InteractionType
    event_value: float | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InteractionResponse(BaseModel):
    status: str
    user_id: str
    entity_id: str
    event_type: str
