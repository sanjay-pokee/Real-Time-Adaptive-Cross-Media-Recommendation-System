import pandas as pd

from backend.knowledge_graph import CatalogKnowledgeGraph


def test_knowledge_graph_scores_query_entity_overlap():
    kg = CatalogKnowledgeGraph(pd.DataFrame())
    item = {
        "title": "Space Adventure",
        "description": "Aliens and exploration",
        "creators": "James Cameron",
        "categories": "Science Fiction, Adventure",
        "content_type": "movie",
        "score": 1.0,
    }

    assert kg.score_item("space alien adventure", item) > 0


def test_profile_score_uses_age_profession_and_interests():
    kg = CatalogKnowledgeGraph(pd.DataFrame())
    item = {
        "title": "Business Analytics Handbook",
        "description": "Data analytics and statistics for business learning",
        "creators": "",
        "categories": "Business, Data, Education",
        "content_type": "book",
        "score": 1.0,
    }
    profile = {
        "age_group": "adult",
        "profession": "data analyst",
        "skills": ["statistics"],
        "interests": ["business"],
        "preferred_content_types": ["book"],
    }

    assert kg.score_profile(item, profile) > 0.4


def test_rerank_adds_kg_and_profile_scores():
    kg = CatalogKnowledgeGraph(pd.DataFrame())
    results = [
        {
            "global_id": "book:1",
            "title": "Business Analytics Handbook",
            "description": "Data analytics and statistics for business learning",
            "creators": "",
            "categories": "Business, Data, Education",
            "content_type": "book",
            "score": 1.0,
        }
    ]
    profile = {"profession": "data analyst", "interests": ["business"]}

    reranked = kg.rerank("business analytics", results, user_profile=profile)

    assert reranked[0]["kg_score"] > 0
    assert reranked[0]["profile_score"] > 0
    assert reranked[0]["score"] > 1.0
