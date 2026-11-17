"""Knowledge-graph and profile-aware reranking.

The knowledge-graph signal comes from ``backend.kg_graph.TypedKnowledgeGraph``:
a typed entity graph (items, people, brands, genres, franchises, keywords)
built in memory from the catalogue and scored by query-entity links and 2-hop
paths from the items a user likes. ``scripts/export_kg_neo4j.py`` writes the
same graph out for Neo4j. The declared-profile match lives here as well.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from backend.kg_graph import TypedKnowledgeGraph


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
    """Knowledge-graph signal: typed entity graph with path scoring, plus profile matching.

    The graph itself is ``backend.kg_graph.TypedKnowledgeGraph``, built once from
    the catalogue. Items it does not know (or an empty catalogue, as in unit
    tests) fall back to the original query/item term overlap.
    """

    catalog: pd.DataFrame
    _graph: TypedKnowledgeGraph | None = field(default=None, init=False, repr=False)

    @property
    def graph(self) -> TypedKnowledgeGraph:
        if self._graph is None:
            self._graph = TypedKnowledgeGraph.from_catalog(self.catalog)
        return self._graph

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        user_profile: dict[str, Any] | None = None,
        kg_weight: float = 0.2,
        profile_weight: float = 0.10,
        seed_ids: Iterable[str] = (),
        seed_role: str = "liked",
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        profile = user_profile or {}
        graph_hits = self.graph.score(
            query, [str(item.get("global_id", "")) for item in results], seed_ids=seed_ids, seed_role=seed_role
        )
        reranked = []
        for item in results:
            hit = graph_hits.get(str(item.get("global_id", "")))
            kg_score = hit["kg_score"] if hit is not None else self.score_item(query, item)
            profile_score = self.score_profile(item, profile)

            updated = dict(item)
            if hit is not None:
                updated["kg_path"] = hit["kg_path"]
                updated["kg_explanation"] = hit["kg_explanation"]
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
