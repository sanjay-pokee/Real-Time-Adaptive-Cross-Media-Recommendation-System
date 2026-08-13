"""Generate a compact evaluation table for the implementation paper.

The report focuses on implemented behavior that can be checked from the local
catalog without requiring a running Qdrant/MySQL service:
- person/entity lookup coverage,
- knowledge-graph query term proximity,
- age/profession profile scoring,
- cross-domain spread in candidate rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.knowledge_graph import CatalogKnowledgeGraph
from backend.qdrant_recommender import QdrantRecommender
from scripts.seed_demo_interactions import USER_PROFILES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
CSV_PATH = REPORT_DIR / "recommendation_evaluation_table.csv"
MD_PATH = REPORT_DIR / "recommendation_evaluation_table.md"


SCENARIOS = [
    {
        "name": "Actor/director lookup",
        "query": "James Cameron",
        "user_id": "user_scifi",
        "expected_terms": ["james cameron"],
    },
    {
        "name": "Singer/artist lookup",
        "query": "Taylor Swift",
        "user_id": "user_music_pop",
        "expected_terms": ["taylor swift"],
    },
    {
        "name": "Profession-aware learning",
        "query": "business analytics and learning",
        "user_id": "user_books_learning",
        "expected_terms": ["business", "data", "analytics", "learning"],
    },
    {
        "name": "Family-safe profile",
        "query": "fun animation adventure",
        "user_id": "user_family",
        "expected_terms": ["family", "animation", "adventure", "children"],
    },
    {
        "name": "Cross-media semantic intent",
        "query": "space adventure with aliens",
        "user_id": "user_scifi",
        "expected_terms": ["space", "science fiction", "alien", "adventure"],
    },
]


def main() -> None:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Missing catalog: {CATALOG_PATH}. Run python -m preprocessing.build_content_catalog"
        )

    catalog = pd.read_csv(CATALOG_PATH).fillna("")
    kg = CatalogKnowledgeGraph(catalog)
    recommender = object.__new__(QdrantRecommender)
    recommender.catalog = catalog

    rows = []
    for scenario in SCENARIOS:
        candidates = _candidate_rows(catalog, scenario["query"], scenario["expected_terms"], limit=20)
        person_hits = QdrantRecommender._search_person_matches(
            recommender,
            scenario["query"],
            top_k=5,
            content_type=None,
        )
        profile = USER_PROFILES.get(scenario["user_id"], {})
        scored = kg.rerank(scenario["query"], candidates, user_profile=profile)
        top5 = scored[:5]

        rows.append(
            {
                "scenario": scenario["name"],
                "query": scenario["query"],
                "user_profile": scenario["user_id"],
                "top5_person_hits": len(person_hits),
                "top5_profile_positive": sum(1 for item in top5 if float(item.get("profile_score", 0.0)) > 0),
                "avg_kg_score_top5": round(_mean(item.get("kg_score", 0.0) for item in top5), 3),
                "avg_profile_score_top5": round(_mean(item.get("profile_score", 0.0) for item in top5), 3),
                "domains_in_top10": ",".join(sorted({str(item.get("content_type", "")) for item in scored[:10]})),
                "example_top_result": top5[0]["title"] if top5 else "",
            }
        )

    report = pd.DataFrame(rows)
    REPORT_DIR.mkdir(exist_ok=True)
    report.to_csv(CSV_PATH, index=False)
    markdown = _to_markdown(report)
    MD_PATH.write_text(markdown, encoding="utf-8")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(markdown)


def _candidate_rows(
    catalog: pd.DataFrame,
    query: str,
    expected_terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    text = (
        catalog["title"].astype(str)
        + " "
        + catalog["description"].astype(str)
        + " "
        + catalog["creators"].astype(str)
        + " "
        + catalog["categories"].astype(str)
        + " "
        + catalog["metadata_text"].astype(str)
    ).str.lower()

    terms = [query.lower(), *[term.lower() for term in expected_terms]]
    mask = pd.Series(False, index=catalog.index)
    for term in terms:
        for part in term.split():
            if len(part) > 2:
                mask = mask | text.str.contains(part, regex=False)

    rows = catalog[mask].copy()
    if rows.empty:
        rows = catalog.head(limit).copy()
    rows["popularity_numeric"] = pd.to_numeric(rows["popularity"], errors="coerce").fillna(0)
    rows = rows.sort_values("popularity_numeric", ascending=False).head(limit)
    return [
        {
            "global_id": row.get("global_id", ""),
            "content_type": row.get("content_type", ""),
            "source": row.get("source", ""),
            "source_id": row.get("source_id", ""),
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "creators": row.get("creators", ""),
            "categories": row.get("categories", ""),
            "release_date": row.get("release_date", ""),
            "popularity": row.get("popularity", ""),
            "rating": row.get("rating", ""),
            "score": 1.0,
        }
        for _, row in rows.iterrows()
    ]


def _mean(values) -> float:
    values = [float(value or 0.0) for value in values]
    return sum(values) / len(values) if values else 0.0


def _to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    lines = [
        fmt(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
