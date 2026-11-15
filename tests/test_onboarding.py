import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import app, get_recommender
from backend.qdrant_recommender import QdrantRecommender


def _row(n, content_type, title, categories, popularity, rating=8.0, image="https://img/x.jpg", min_age=0):
    return {
        "global_id": f"{content_type}:test:{n}",
        "content_type": content_type,
        "source": "test",
        "source_id": str(n),
        "title": title,
        "description": "",
        "creators": "Someone",
        "categories": categories,
        "release_date": "2020",
        "popularity": popularity,
        "rating": rating,
        "image_url": image,
        "domain": "entertainment",
        "maturity": "all_ages" if min_age == 0 else "adult",
        "audience_min_age": min_age,
        "risk_tier": 0,
    }


def _recommender(rows):
    recommender = QdrantRecommender.__new__(QdrantRecommender)
    recommender.catalog = pd.DataFrame(rows)
    return recommender


def test_each_genre_gets_its_best_known_eligible_item():
    recommender = _recommender([
        _row(1, "movie", "Big Comedy", "Comedy", 90),
        _row(2, "movie", "Small Comedy", "Comedy", 10),
        _row(3, "movie", "Adult Horror", "Horror", 99, min_age=18),
        _row(4, "movie", "Family Horror", "Horror", 40),
        _row(5, "movie", "Coverless Action", "Action", 99, image=""),
        _row(6, "movie", "Awful Action", "Action", 98, rating=3.0),
        _row(7, "movie", "Good Action", "Action", 50),
    ])

    picks = {item["genre"]: item["title"] for item in recommender.starter_items()}

    assert picks["comedy"] == "Big Comedy"
    # The unknown-age default never sees 18+ items, however popular.
    assert picks["horror"] == "Family Horror"
    # No cover art, or a poor rating, rules an item out.
    assert picks["action"] == "Good Action"


def test_an_item_stands_for_one_genre_only():
    recommender = _recommender([
        _row(1, "movie", "Funny Scary", "Comedy, Horror", 90),
        _row(2, "movie", "Plain Horror", "Horror", 10),
    ])

    picks = recommender.starter_items()

    assert [item["title"] for item in picks] == ["Funny Scary", "Plain Horror"]


def test_the_onboarding_endpoint_serves_the_picks():
    class Fake:
        def starter_items(self):
            return [{**_row(1, "book", "1984", "Fiction", 5), "genre": "fiction", "title": 1984}]

    app.dependency_overrides[get_recommender] = lambda: Fake()
    try:
        response = TestClient(app).get("/onboarding/items")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    item = response.json()["items"][0]
    # A numeric-looking title still arrives as text.
    assert item == {
        "global_id": "book:test:1",
        "content_type": "book",
        "title": "1984",
        "creators": "Someone",
        "image_url": "https://img/x.jpg",
        "genre": "fiction",
    }
