import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import app, get_recommender


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

