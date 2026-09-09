"""API schemas for the recommendation backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

from backend.domains import get_registry

# Content types are declared in config/domains.yaml, not here: a Literal would
# have to be edited every time a vertical is added, which is the coupling the
# domain registry exists to remove. Validation happens in the recommender via
# ``normalize_content_type``, which raises with the list of valid aliases.
ContentType = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DomainName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

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
        description="Optional filter for a single content type (movie, book, health, ...).",
    )
    domain: DomainName | None = Field(
        None,
        description="Optional filter for a whole vertical (entertainment, health, industry).",
    )
    age: int | None = Field(
        None,
        ge=0,
        le=120,
        description="Viewer age. Drives the audience eligibility pre-filter.",
    )
    safe_mode: bool = Field(
        False,
        description="Cap results at family-appropriate content regardless of age.",
    )


class ItemRecommendRequest(BaseModel):
    global_id: str = Field(..., min_length=1, description="Catalog item id to recommend from.")
    user_id: str | None = Field(None, description="Optional user id for personalization.")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return.")
    content_type: ContentType | None = Field(
        None,
        description="Optional filter for a single content type (movie, book, health, ...).",
    )
    domain: DomainName | None = Field(
        None,
        description="Optional filter for a whole vertical (entertainment, health, industry).",
    )
    age: int | None = Field(
        None,
        ge=0,
        le=120,
        description="Viewer age. Drives the audience eligibility pre-filter.",
    )
    safe_mode: bool = Field(
        False,
        description="Cap results at family-appropriate content regardless of age.",
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
    domain: str = ""
    maturity: str = ""
    audience_min_age: int | str = ""
    risk_tier: int | str = ""
    score: float
    semantic_score: float | None = None
    graph_score: float | None = None
    ema_score: float | None = None
    kg_score: float | None = None
    profile_score: float | None = None


class RecommendResponse(BaseModel):
    query: str
    top_k: int
    content_type: str | None
    domain: str | None = None
    # Set when the request targets a regulated domain, so the UI can show it.
    advisory: str | None = None
    results: list[RecommendationItem]


class ItemRecommendResponse(BaseModel):
    global_id: str
    top_k: int
    content_type: str | None
    domain: str | None = None
    advisory: str | None = None
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


class UserInteractionState(BaseModel):
    """Current interaction state for a user+entity pair."""
    user_id: str
    entity_id: str
    # Boolean flags for toggle-able interactions
    view: bool = False
    like: bool = False
    bookmark: bool = False
    skip: bool = False
    complete: bool = False
    # Last rating (1-5), or 0 if none
    rating: float = 0


class SuggestItem(BaseModel):
    """A single autocomplete suggestion."""
    label: str
    hint: str
    kind: str  # "title" | "person" | "category"
    query: str  # text to place in the search box when chosen


class SuggestResponse(BaseModel):
    q: str
    suggestions: list[SuggestItem]


class DomainSummary(BaseModel):
    """One vertical the engine can serve, as advertised by GET /domains."""

    name: str
    label: str
    description: str
    content_types: list[str]
    risk_tier: int
    advisory: str | None = None


class DomainsResponse(BaseModel):
    domains: list[DomainSummary]
    content_types: list[str]
