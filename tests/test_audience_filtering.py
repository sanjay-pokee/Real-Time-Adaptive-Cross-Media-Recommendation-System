"""The constraint layer as the retrieval path actually uses it.

The point of these tests is that the age gate is enforced *before* ranking. It is
easy to write a filter that only tidies up the final list; that version is
defeated the moment a reranker promotes something. So these assert on the query
sent to the vector store, and on the one retrieval path that bypasses it.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.audience import AudienceContext, eligibility_conditions, is_eligible
from backend.qdrant_recommender import QdrantRecommender

qdrant_models = pytest.importorskip(
    "qdrant_client.models", reason="qdrant-client is not installed in this environment"
)


def _conditions_by_key(conditions):
    return {condition.key: condition for condition in conditions}


# ---------------------------------------------------------------------------
# The pre-filter sent to Qdrant
# ---------------------------------------------------------------------------

def test_age_becomes_a_range_condition_on_the_query():
    conditions = _conditions_by_key(eligibility_conditions(AudienceContext(age=10)))

    assert conditions["audience_min_age"].range.lte == 10.0
    assert conditions["risk_tier"].range.lte == 1.0


def test_safe_mode_lowers_the_range_below_the_real_age():
    conditions = _conditions_by_key(
        eligibility_conditions(AudienceContext(age=40, safe_mode=True))
    )

    assert conditions["audience_min_age"].range.lte == 7.0


def test_a_domain_scopes_the_query_to_its_content_types():
    conditions = _conditions_by_key(
        eligibility_conditions(AudienceContext(age=30, domain="entertainment"))
    )

    assert set(conditions["content_type"].match.any) == {"movie", "book", "music"}


def test_an_explicit_content_type_narrows_further_than_its_domain():
    conditions = _conditions_by_key(
        eligibility_conditions(AudienceContext(age=30, domain="entertainment"), "films")
    )

    assert conditions["content_type"].match.value == "movie"


def test_recommender_builds_an_age_filter_when_given_an_audience():
    query_filter = QdrantRecommender._build_qdrant_filter(
        object(), "movie", AudienceContext(age=12)
    )

    keys = {condition.key for condition in query_filter.must}
    assert "audience_min_age" in keys


def test_recommender_keeps_the_plain_content_type_filter_without_an_audience():
    """Requests that declare no viewer must behave exactly as before."""
    query_filter = QdrantRecommender._build_qdrant_filter(object(), "movie", None)

    assert [condition.key for condition in query_filter.must] == ["content_type"]
    assert QdrantRecommender._build_qdrant_filter(object(), None, None) is None


# ---------------------------------------------------------------------------
# The person-name path reads the catalog directly, bypassing Qdrant
# ---------------------------------------------------------------------------

def _catalog_row(global_id, title, creators, maturity, min_age):
    return {
        "global_id": global_id,
        "content_type": "movie",
        "source": "test",
        "source_id": global_id.split(":")[-1],
        "title": title,
        "description": "",
        "creators": creators,
        "categories": "",
        "release_date": "",
        "popularity": 10,
        "rating": 7,
        "domain": "entertainment",
        "maturity": maturity,
        "audience_min_age": min_age,
        "risk_tier": 0,
    }


def _recommender_with(rows):
    recommender = QdrantRecommender.__new__(QdrantRecommender)
    recommender.catalog = pd.DataFrame(rows)
    return recommender


def test_person_lookup_withholds_content_above_the_viewers_age():
    recommender = _recommender_with(
        [
            _catalog_row("movie:test:1", "Family Film", "Jane Doe", "all_ages", 0),
            _catalog_row("movie:test:2", "Adult Film", "Jane Doe", "adult", 18),
        ]
    )

    for_child = QdrantRecommender._search_person_matches(
        recommender, "Jane Doe", top_k=10, content_type=None, audience=AudienceContext(age=9)
    )
    for_adult = QdrantRecommender._search_person_matches(
        recommender, "Jane Doe", top_k=10, content_type=None, audience=AudienceContext(age=30)
    )

    assert [item["title"] for item in for_child] == ["Family Film"]
    assert {item["title"] for item in for_adult} == {"Family Film", "Adult Film"}


def test_person_lookup_is_unfiltered_when_no_audience_is_declared():
    recommender = _recommender_with(
        [
            _catalog_row("movie:test:1", "Family Film", "Jane Doe", "all_ages", 0),
            _catalog_row("movie:test:2", "Adult Film", "Jane Doe", "adult", 18),
        ]
    )

    results = QdrantRecommender._search_person_matches(
        recommender, "Jane Doe", top_k=10, content_type=None
    )

    assert len(results) == 2


# ---------------------------------------------------------------------------
# The Python predicate and the compiled filter must agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age", [5, 10, 13, 16, 18, 25, 40])
@pytest.mark.parametrize(
    "item_min_age_value", [0, 7, 13, 16, 18, 21]
)
def test_predicate_agrees_with_the_compiled_range(age, item_min_age_value):
    """is_eligible and the Qdrant Range must encode the same rule.

    The violation-rate metric compares one against the other, so a divergence
    here would make that metric measure nothing.
    """
    context = AudienceContext(age=age)
    item = {"content_type": "movie", "audience_min_age": item_min_age_value, "risk_tier": 0}

    conditions = _conditions_by_key(eligibility_conditions(context))
    range_allows = item_min_age_value <= conditions["audience_min_age"].range.lte

    assert is_eligible(item, context) == range_allows
