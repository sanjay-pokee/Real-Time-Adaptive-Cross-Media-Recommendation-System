import sys
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import app, get_recommender
from backend.qdrant_recommender import QdrantRecommender


class FakeRecommender:
    def recommend(self, query, top_k=10, content_type=None):
        return [
            {
                "global_id": "movie:test:1",
                "content_type": "movie",
                "source": "test",
                "source_id": "1",
                "title": "Test Movie",
                "description": "A test recommendation.",
                "creators": "",
                "categories": "Science Fiction",
                "release_date": "2026",
                "popularity": 1.0,
                "rating": 8.0,
                "score": 0.99,
                "semantic_score": 0.9,
                "graph_score": 0.45,
            }
        ][:top_k]


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_returns_results():
    app.dependency_overrides.clear()
    if hasattr(get_recommender, 'cache_clear'):
        get_recommender.cache_clear()
    app.dependency_overrides[get_recommender] = lambda: FakeRecommender()
    client = TestClient(app)

    response = client.post(
        "/recommend",
        json={
            "query": "space adventure with aliens",
            "top_k": 1,
            "content_type": "movie",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "space adventure with aliens"
    assert body["top_k"] == 1
    assert body["results"][0]["title"] == "Test Movie"
    assert body["results"][0]["semantic_score"] == 0.9
    assert body["results"][0]["graph_score"] == 0.45

    app.dependency_overrides.clear()

class FakeModel:
    def encode(self, values, convert_to_numpy=True, normalize_embeddings=True):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def _catalog_row(global_id, title, creators, popularity=1.0, rating=5.0, content_type="movie"):
    return {
        "global_id": global_id,
        "content_type": content_type,
        "source": "test",
        "source_id": global_id.rsplit(":", 1)[-1],
        "title": title,
        "description": "",
        "creators": creators,
        "categories": "Drama",
        "release_date": "2026",
        "popularity": popularity,
        "rating": rating,
    }


def test_person_query_prioritizes_creator_matches_then_general_recommendations():
    recommender = object.__new__(QdrantRecommender)
    recommender.catalog = pd.DataFrame(
        [
            _catalog_row("movie:test:1", "Actor Hit 1", "Director A, Actor One", 50, 7),
            _catalog_row("movie:test:2", "Actor Hit 2", "Actor One, Co Star", 40, 8),
            _catalog_row("movie:test:3", "Actor Hit 3", "Someone Else, Actor One", 30, 9),
        ]
    )
    recommender.model = FakeModel()
    recommender.graph_store = None
    recommender.ema_store = None
    recommender._search_vector = lambda vector, top_k, content_type: [
        _catalog_row("movie:test:2", "Actor Hit 2", "Actor One, Co Star", 40, 8),
        _catalog_row("movie:test:4", "General Result 1", "Other Person", 20, 6),
        _catalog_row("movie:test:5", "General Result 2", "Other Person", 10, 6),
    ]

    results = QdrantRecommender.recommend(recommender, "Actor One", top_k=5)

    assert [item["title"] for item in results] == [
        "Actor Hit 1",
        "Actor Hit 2",
        "Actor Hit 3",
        "General Result 1",
        "General Result 2",
    ]
    assert len({item["global_id"] for item in results}) == len(results)
