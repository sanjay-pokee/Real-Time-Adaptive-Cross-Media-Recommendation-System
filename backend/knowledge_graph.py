"""Lightweight catalog knowledge graph and profile-aware reranking.

This module keeps the first knowledge-graph implementation deliberately local:
it builds useful edges from the existing processed catalog instead of requiring
Neo4j or another graph database. The graph can later be moved to a graph DB
without changing the recommender contract.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd


ROLE_TERMS: dict[str, list[str]] = {
    "student": ["education", "learning", "science", "technology", "history", "documentary"],
    "data analyst": ["data", "analytics", "business", "statistics", "technology", "science"],
    "software engineer": ["technology", "science fiction", "programming", "business", "innovation"],
    "designer": ["art", "design", "animation", "music", "creativity", "family"],
    "entrepreneur": ["business", "leadership", "finance", "innovation", "documentary"],
}

AGE_TERMS: dict[str, list[str]] = {
    "teen": ["family", "animation", "adventure", "comedy", "young adult"],
    "young_adult": ["adventure", "science fiction", "pop", "business", "self-help"],
    "adult": ["drama", "thriller", "business", "history", "documentary"],
    "family": ["family", "animation", "children", "comedy", "adventure"],
}

AGE_AVOID_TERMS: dict[str, list[str]] = {
    "teen": ["horror", "crime", "erotic"],
    "family": ["horror", "crime", "thriller", "war"],
}


@dataclass
class CatalogKnowledgeGraph:
    """In-memory graph-like scorer from catalog and user profile metadata."""

    catalog: pd.DataFrame

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        user_profile: dict[str, Any] | None = None,
        kg_weight: float = 0.08,
        profile_weight: float = 0.10,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        profile = user_profile or {}
        reranked = []
        for item in results:
            kg_score = self.score_item(query, item)
            profile_score = self.score_profile(item, profile)

            updated = dict(item)
            updated["kg_score"] = kg_score
            updated["profile_score"] = profile_score
            updated["score"] = (
                float(updated.get("score", 0.0))
                + kg_weight * kg_score
                + profile_weight * profile_score
            )
            reranked.append(updated)

        reranked.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return reranked

    def score_item(self, query: str, item: dict[str, Any]) -> float:
        """Score graph proximity between query terms and item entity metadata."""
        query_terms = _token_set(query)
        if not query_terms:
            return 0.0

        item_terms = _token_set(
            " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("description", "")),
                    str(item.get("creators", "")),
                    str(item.get("categories", "")),
                    str(item.get("content_type", "")),
                ]
            )
        )
        if not item_terms:
            return 0.0

        overlap = len(query_terms & item_terms)
        return min(1.0, overlap / max(3, len(query_terms)))

    def score_profile(self, item: dict[str, Any], profile: dict[str, Any]) -> float:
        """Score match between content metadata and age/profession profile."""
        if not profile:
            return 0.0

        item_text = _normalize_text(
            " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("description", "")),
                    str(item.get("creators", "")),
                    str(item.get("categories", "")),
                    str(item.get("content_type", "")),
                ]
            )
        )

        interests = _as_list(profile.get("interests")) + _as_list(profile.get("likes"))
        skills = _as_list(profile.get("skills"))
        profession = _normalize_text(profile.get("profession", ""))
        age_group = _normalize_text(profile.get("age_group", ""))

        positive_terms = interests + skills
        positive_terms.extend(ROLE_TERMS.get(profession, []))
        positive_terms.extend(AGE_TERMS.get(age_group, []))
        avoid_terms = AGE_AVOID_TERMS.get(age_group, []) + _as_list(profile.get("avoid"))

        score = 0.0
        for term in positive_terms:
            if _normalize_text(term) and _normalize_text(term) in item_text:
                score += 0.16
        for term in avoid_terms:
            if _normalize_text(term) and _normalize_text(term) in item_text:
                score -= 0.25

        preferred_types = {_normalize_text(value) for value in _as_list(profile.get("preferred_content_types"))}
        if preferred_types and _normalize_text(item.get("content_type", "")) in preferred_types:
            score += 0.15

        return max(-1.0, min(1.0, score))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def _normalize_text(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(without_marks.casefold().split())


def _token_set(value: Any) -> set[str]:
    text = _normalize_text(value)
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2}
