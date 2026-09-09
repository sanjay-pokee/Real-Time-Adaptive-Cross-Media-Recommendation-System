import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.audience_tagging import annotate_audience
from scripts.build_benchmark_table import (
    build_rows,
    group_reports,
    load_reports,
    to_latex,
    to_markdown,
)
from scripts.evaluate_constraints import evaluate_profile, load_catalog, sample_withheld


# ---------------------------------------------------------------------------
# Cross-domain benchmark table
# ---------------------------------------------------------------------------

def _report(dataset, model, recall, ndcg=0.02, mrr=0.01, users=1000, items=500):
    return {
        "model": model,
        "dataset": dataset,
        "k": 20,
        "metrics": {
            "users_evaluated": users,
            "hit_rate": recall * 1.4,
            "recall": recall,
            "precision": recall / 20,
            "ndcg": ndcg,
            "mrr": mrr,
        },
        "config": {"epochs": 60} if model == "lightgcn" else None,
        "graph": {"users": users, "items": items, "interactions": users * 12},
    }


def _write_reports(directory: Path, reports):
    directory.mkdir(parents=True, exist_ok=True)
    for index, report in enumerate(reports):
        (directory / f"r{index}.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


def test_reports_group_by_dataset_regardless_of_the_cache_filename(tmp_path):
    """The notebook feeds prepared_<category>_<max_users>.parquet, not the raw CSV."""
    reports = [
        _report("prepared_movies_and_tv_40000.parquet", "lightgcn", 0.06),
        _report("prepared_movies_and_tv_40000.parquet", "popularity", 0.02),
    ]

    grouped = group_reports(reports)

    assert set(grouped) == {"movies_and_tv"}
    assert set(grouped["movies_and_tv"]) == {"lightgcn", "popularity"}


def test_lift_is_lightgcn_recall_over_the_baseline():
    rows = build_rows(
        group_reports(
            [
                _report("prepared_books_40000.parquet", "lightgcn", 0.06),
                _report("prepared_books_40000.parquet", "popularity", 0.02),
            ]
        ),
        k=20,
    )

    assert len(rows) == 1
    assert rows[0]["recall_lift"] == pytest.approx(3.0)
    assert rows[0]["domain"] == "Entertainment"


def test_a_domain_without_a_baseline_still_renders():
    rows = build_rows(
        group_reports([_report("prepared_health_and_household_40000.parquet", "lightgcn", 0.07)]),
        k=20,
    )

    assert rows[0]["recall_lift"] is None
    assert "--" in to_markdown(rows, 20)


def test_a_baseline_with_no_lightgcn_run_is_skipped():
    rows = build_rows(group_reports([_report("prepared_books_40000.parquet", "popularity", 0.02)]), 20)

    assert rows == []


def test_latex_escapes_ampersands_in_dataset_labels():
    """'Movies & TV' unescaped becomes a column separator and mangles the table."""
    rows = build_rows(
        group_reports([_report("prepared_movies_and_tv_40000.parquet", "lightgcn", 0.06)]), 20
    )

    latex = to_latex(rows, 20)

    assert r"Movies \& TV" in latex
    assert "Movies & TV" not in latex
    # Every body row must have exactly the 6 columns the tabular declares. Count
    # only *unescaped* ampersands -- an escaped one is content, not a separator.
    body = [line for line in latex.splitlines() if line.endswith(r"\\") and "textbf" not in line]
    assert body and all(line.replace(r"\&", "").count("&") == 5 for line in body)


def test_load_reports_explains_an_empty_directory(tmp_path):
    (tmp_path / "benchmarks").mkdir()

    with pytest.raises(FileNotFoundError, match="no .json reports"):
        load_reports(tmp_path / "benchmarks")


def test_load_reports_names_the_file_that_is_bad_json(tmp_path):
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    (directory / "broken.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="broken.json"):
        load_reports(directory)


# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------

def _tagged_catalog():
    catalog = pd.DataFrame(
        [
            {"global_id": "m:1", "content_type": "movie", "title": "Minions",
             "categories": "Family, Animation", "popularity": 100},
            {"global_id": "m:2", "content_type": "movie", "title": "The Conjuring",
             "categories": "Horror", "popularity": 90},
            {"global_id": "m:3", "content_type": "movie", "title": "Interstellar",
             "categories": "Adventure", "popularity": 80},
            {"global_id": "i:1", "content_type": "industrial", "title": "Centrifuge",
             "categories": "Lab", "popularity": 70},
        ]
    )
    return annotate_audience(catalog)


def test_the_filter_reports_zero_violations_and_the_absence_of_it_does_not():
    catalog = _tagged_catalog()

    child = evaluate_profile(catalog, "Child", {"age": 8}, k=4)

    assert child["violations_filter_on"] == 0
    assert child["violation_rate_on"] == 0.0
    # Without the layer the same child would have been shown the horror film,
    # Interstellar (teen) and the lab equipment.
    assert child["violations_filter_off"] == 3
    assert child["violation_rate_off"] == pytest.approx(0.75)


def test_an_adult_loses_nothing_to_the_constraint():
    adult = evaluate_profile(_tagged_catalog(), "Adult", {"age": 30}, k=4)

    assert adult["coverage"] == 1.0
    assert adult["overlap_with_unconstrained"] == 1.0
    assert adult["violations_filter_on"] == 0


def test_coverage_shrinks_as_the_viewer_gets_younger():
    catalog = _tagged_catalog()

    coverages = [
        evaluate_profile(catalog, str(age), {"age": age}, k=4)["coverage"]
        for age in (8, 15, 30)
    ]

    assert coverages == sorted(coverages)
    assert coverages[0] < coverages[-1]


def test_domain_scoping_is_reported_as_its_own_constraint():
    result = evaluate_profile(
        _tagged_catalog(), "Industry only", {"age": 30, "domain": "industry"}, k=4
    )

    assert result["domain"] == "industry"
    assert result["eligible_items"] == 1


def test_withheld_items_come_with_a_reason():
    reasons = sample_withheld(_tagged_catalog(), {"age": 8}, limit=3)

    assert reasons
    assert all("requires age" in reason for reason in reasons)


def test_load_catalog_refuses_an_untagged_file(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame([{"global_id": "m:1", "content_type": "movie", "title": "X"}]).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="missing audience columns"):
        load_catalog(path)
