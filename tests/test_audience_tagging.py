import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.audience import AudienceContext, is_eligible
from preprocessing.audience_tagging import (
    annotate_audience,
    audience_summary,
    maturity_for_row,
)
from preprocessing.content_schema import AUDIENCE_COLUMNS, CATALOG_COLUMNS, CONTENT_COLUMNS


def test_catalog_schema_extends_rather_than_replaces_the_base():
    assert CATALOG_COLUMNS[: len(CONTENT_COLUMNS)] == CONTENT_COLUMNS
    assert CATALOG_COLUMNS[len(CONTENT_COLUMNS) :] == AUDIENCE_COLUMNS


@pytest.mark.parametrize(
    "content_type, categories, expected",
    [
        ("movie", "Family, Animation, Comedy", "all_ages"),
        ("movie", "Horror, Thriller", "adult"),
        ("book", "Juvenile Fiction", "all_ages"),
        ("book", "Young Adult Fiction", "teen"),
        ("health", "Children Vitamins", "all_ages"),
    ],
)
def test_category_keywords_drive_the_maturity_label(content_type, categories, expected):
    assert maturity_for_row(content_type, categories) == expected


def test_no_category_signal_falls_back_to_the_content_type_default():
    # movie -> teen, industrial -> adult, per config/domains.yaml.
    assert maturity_for_row("movie", "Drama") == "teen"
    assert maturity_for_row("industrial", "Lab Equipment") == "adult"


def test_the_most_restrictive_matching_rule_wins():
    # Both "young adult" (teen) and "horror" (adult) are present.
    assert maturity_for_row("book", "Young Adult Horror") == "adult"


def test_a_fixed_maturity_source_ignores_category_text():
    # industrial declares maturity_source: fixed, so no keyword can lower it.
    assert maturity_for_row("industrial", "Children Toys Family Animation") == "adult"


def test_an_unrecognised_content_type_is_gated_not_waved_through():
    assert maturity_for_row("podcast", "Comedy") == "adult"


def test_annotate_adds_every_audience_column():
    catalog = pd.DataFrame(
        [{"global_id": "movie:t:1", "content_type": "movie", "title": "X", "categories": "Drama"}]
    )

    annotated = annotate_audience(catalog)

    for column in AUDIENCE_COLUMNS:
        assert column in annotated.columns
    row = annotated.iloc[0]
    assert row["domain"] == "entertainment"
    assert row["maturity"] == "teen"
    assert row["audience_min_age"] == 13
    assert row["risk_tier"] == 0


def test_a_real_rating_on_the_row_is_preserved():
    """A dataset that ships a genuine rating must not be overwritten by the heuristic."""
    catalog = pd.DataFrame(
        [
            {
                "global_id": "movie:t:1",
                "content_type": "movie",
                "title": "Certified Adult Film",
                "categories": "Family, Animation",  # would otherwise derive all_ages
                "maturity": "adult",
            }
        ]
    )

    annotated = annotate_audience(catalog)

    assert annotated.iloc[0]["maturity"] == "adult"
    assert annotated.iloc[0]["audience_min_age"] == 18


def test_regulated_domains_inherit_their_risk_tier():
    catalog = pd.DataFrame(
        [
            {"global_id": "h:1", "content_type": "health", "title": "Vitamin D", "categories": "Vitamins"},
            {"global_id": "m:1", "content_type": "movie", "title": "X", "categories": "Drama"},
        ]
    )

    annotated = annotate_audience(catalog).set_index("global_id")

    assert annotated.loc["h:1", "risk_tier"] == 1
    assert annotated.loc["m:1", "risk_tier"] == 0


def test_tagged_rows_are_directly_usable_by_the_constraint_layer():
    """The build output and the filter must speak the same vocabulary."""
    catalog = pd.DataFrame(
        [
            {"global_id": "m:1", "content_type": "movie", "title": "Minions", "categories": "Family, Animation"},
            {"global_id": "m:2", "content_type": "movie", "title": "The Conjuring", "categories": "Horror"},
        ]
    )

    rows = annotate_audience(catalog).to_dict("records")
    child = AudienceContext(age=8)

    assert [row["title"] for row in rows if is_eligible(row, child)] == ["Minions"]


def test_summary_counts_by_domain_and_maturity():
    catalog = pd.DataFrame(
        [
            {"global_id": "m:1", "content_type": "movie", "title": "A", "categories": "Family"},
            {"global_id": "m:2", "content_type": "movie", "title": "B", "categories": "Family"},
            {"global_id": "i:1", "content_type": "industrial", "title": "C", "categories": "Lab"},
        ]
    )

    summary = audience_summary(annotate_audience(catalog))

    entertainment = summary[summary["domain"] == "entertainment"]
    assert int(entertainment["items"].sum()) == 2
    assert set(summary["domain"]) == {"entertainment", "industry"}


def test_summary_rejects_an_untagged_catalog():
    with pytest.raises(ValueError, match="missing audience columns"):
        audience_summary(pd.DataFrame([{"content_type": "movie"}]))
