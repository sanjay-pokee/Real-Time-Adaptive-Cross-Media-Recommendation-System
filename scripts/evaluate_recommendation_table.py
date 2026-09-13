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

# Tokens carrying no retrieval signal. Substring matching over a 106k catalogue
# makes these match nearly everything, which flattened the candidate set.
STOPWORDS = frozenset({
    "and", "for", "the", "with", "from", "into", "about", "that", "this",
    "your", "you", "all", "any", "are", "was", "has", "had", "its", "out",
})
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
    # The three verticals the review asked for. Without these the measured table
    # only ever evidenced entertainment, which is the narrow reading of the
    # project the panel is asking us to widen.
    {
        "name": "Health product discovery",
        "query": "joint pain supplements and mobility support",
        "user_id": "user_health_caregiver",
        "expected_terms": ["joint", "supplement", "mobility", "pain", "support"],
    },
    {
        "name": "Industrial supply lookup",
        "query": "protective safety gloves for the lab",
        "user_id": "user_industry_engineer",
        "expected_terms": ["safety", "glove", "protective", "lab", "nitrile"],
    },
    {
        "name": "Finance literacy discovery",
        "query": "personal budgeting and bookkeeping",
        "user_id": "user_finance_planner",
        "expected_terms": ["budget", "finance", "accounting", "bookkeeping", "tax"],
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

    # Count how many distinct query tokens a row matches, rather than keeping
    # every row that matches *any* of them. Under the any-token rule a stopword
    # like "with" in "space adventure with aliens" matched most of the catalogue,
    # so the candidate set was effectively the whole corpus.
    tokens = {
        part
        for term in [query.lower(), *[t.lower() for t in expected_terms]]
        for part in term.split()
        if len(part) > 2 and part not in STOPWORDS
    }
    matches = pd.Series(0, index=catalog.index)
    for token in tokens:
        matches = matches + text.str.contains(token, regex=False).astype(int)

    rows = catalog[matches > 0].copy()
    if rows.empty:
        rows = catalog.head(limit).copy()
        rows["term_matches"] = 0
    else:
        rows["term_matches"] = matches[matches > 0]

    # Rank within content type, not on the raw number. `popularity` means a
    # different thing per source - Amazon ships rating counts (health peaks at
    # 294,761), TMDb a float that peaks near 800, Spotify a 0-100 index - so
    # sorting a mixed pool by it returned Amazon health and industrial rows for
    # every query, including "space adventure with aliens". A within-type
    # percentile is comparable across sources; the raw value is not.
    rows["popularity_numeric"] = pd.to_numeric(rows["popularity"], errors="coerce").fillna(0)
    rows["popularity_rank"] = (
        rows.groupby("content_type")["popularity_numeric"].rank(pct=True)
    )
    rows = rows.sort_values(
        ["term_matches", "popularity_rank"], ascending=[False, False]
    ).head(limit)
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
