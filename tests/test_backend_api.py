import sys
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import app, get_recommender
from backend.qdrant_recommender import QdrantRecommender


class FakeRecommender:
    def __init__(self):
        self.audiences = []

    def recommend(self, query, top_k=10, content_type=None, user_id=None, audience=None):
        self.audiences.append(audience)
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
    fake = FakeRecommender()
    app.dependency_overrides[get_recommender] = lambda: fake
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
    # A request that declares no age still reaches the recommender gated, at
    # the teen ceiling, rather than as an unrestricted adult.
    assert len(fake.audiences) == 1
    assert fake.audiences[0] is not None
    assert fake.audiences[0].age is None
    assert fake.audiences[0].effective_age == 13

    app.dependency_overrides.clear()


ADULT_ROW = {
    "global_id": "movie:test:18",
    "content_type": "movie",
    "source": "test",
    "source_id": "18",
    "title": "Adults Only",
    "maturity": "adult",
    "audience_min_age": 18,
    "risk_tier": 0,
    "score": 0.9,
}


class GatedPathBugRecommender:
    """Fails inside the gated search; would serve an 18+ row if called ungated.

    Stands in for a real bug in the recommender: the old endpoints caught that
    TypeError and retried without ``audience``, which returned ADULT_ROW to an
    eight-year-old with a 200.
    """

    def __init__(self):
        self.audiences = []

    def recommend(self, query, top_k=10, content_type=None, user_id=None, audience=None):
        return self._respond(audience)

    def recommend_from_item(self, global_id, top_k=10, content_type=None, user_id=None, audience=None):
        return self._respond(audience)

    def _respond(self, audience):
        self.audiences.append(audience)
        if audience is not None:
            raise TypeError("simulated bug inside the gated search path")
        return [dict(ADULT_ROW)]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/recommend", {"query": "dark violent thriller", "top_k": 5, "age": 8}),
        ("/recommend/item", {"global_id": "movie:test:1", "top_k": 5, "age": 8}),
    ],
)
def test_internal_error_is_not_retried_without_the_eligibility_gate(path, body):
    fake = GatedPathBugRecommender()
    app.dependency_overrides[get_recommender] = lambda: fake
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post(path, json=body)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    # Exactly one call, and it carried the viewer's gate.
    assert [audience.age for audience in fake.audiences] == [8]

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
    recommender._search_vector = lambda vector, top_k, content_type, audience=None: [
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
