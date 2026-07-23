"""Unit tests for the preprocessing pipeline.

All tests use small synthetic DataFrames — no file I/O, no real datasets.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make sure the package is importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.cleaners import clean_text, join_non_empty, make_text_hash, parse_name_list
from preprocessing.content_schema import CONTENT_COLUMNS, REQUIRED_NON_EMPTY, make_global_id
from preprocessing.validate_content_catalog import validate_catalog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_valid_row(
    global_id: str = "movie:tmdb_5000_movies:123",
    content_type: str = "movie",
    source: str = "tmdb_5000_movies",
    source_id: str = "123",
    title: str = "Test Movie",
    description: str = "A test movie description.",
    creators: str = "",
    categories: str = "Action, Drama",
    release_date: str = "2020-01-01",
    popularity: float = 42.0,
    rating: float = 7.5,
    metadata_text: str = "Test Movie Action, Drama A test movie description.",
    embedding_text: str = "Test Movie Action, Drama A test movie description.",
) -> dict:
    text_hash = make_text_hash(embedding_text)
    return {
        "global_id": global_id,
        "content_type": content_type,
        "source": source,
        "source_id": source_id,
        "title": title,
        "description": description,
        "creators": creators,
        "categories": categories,
        "release_date": release_date,
        "popularity": popularity,
        "rating": rating,
        "metadata_text": metadata_text,
        "embedding_text": embedding_text,
        "text_hash": text_hash,
    }


def _valid_catalog(n: int = 3) -> pd.DataFrame:
    """Build a small valid catalog DataFrame with n rows across media types."""
    media = [
        ("movie", "tmdb_5000_movies", "101"),
        ("book", "google_books_dataset", "b202"),
        ("music", "spotify_songs", "s303"),
    ]
    rows = []
    for i in range(n):
        ctype, source, sid = media[i % len(media)]
        gid = make_global_id(ctype, source, sid + str(i))
        embed_text = f"{ctype} title {i} creator cat description {i}"
        rows.append(
            _make_valid_row(
                global_id=gid,
                content_type=ctype,
                source=source,
                source_id=sid + str(i),
                title=f"{ctype.title()} Title {i}",
                embedding_text=embed_text,
                metadata_text=embed_text + " extra",
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# content_schema tests
# ---------------------------------------------------------------------------

class TestContentSchema:
    def test_content_columns_count(self):
        assert len(CONTENT_COLUMNS) == 14

    def test_content_columns_includes_required(self):
        required = [
            "global_id", "content_type", "source", "source_id",
            "title", "description", "creators", "categories",
            "release_date", "popularity", "rating",
            "metadata_text", "embedding_text", "text_hash",
        ]
        for col in required:
            assert col in CONTENT_COLUMNS, f"Missing column: {col}"

    def test_make_global_id_format(self):
        gid = make_global_id("movie", "tmdb_5000_movies", "19995")
        assert gid == "movie:tmdb_5000_movies:19995"

    def test_make_global_id_all_parts(self):
        gid = make_global_id("book", "google_books_dataset", "abc123")
        parts = gid.split(":")
        assert len(parts) == 3
        assert parts[0] == "book"
        assert parts[1] == "google_books_dataset"
        assert parts[2] == "abc123"


# ---------------------------------------------------------------------------
# cleaners tests
# ---------------------------------------------------------------------------

class TestCleaners:
    def test_clean_text_strips_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_clean_text_collapses_internal_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_clean_text_nan_returns_empty(self):
        import math
        assert clean_text(math.nan) == ""
        assert clean_text(None) == ""

    def test_parse_name_list_json_dicts(self):
        result = parse_name_list("[{'name': 'Action'}, {'name': 'Drama'}]")
        assert result == ["Action", "Drama"]

    def test_parse_name_list_plain_string(self):
        result = parse_name_list("Action, Drama, Comedy")
        assert result == ["Action", "Drama", "Comedy"]

    def test_join_non_empty_skips_blanks(self):
        assert join_non_empty(["Hello", "", "World"]) == "Hello World"

    def test_join_non_empty_custom_separator(self):
        assert join_non_empty(["A", "B"], separator=", ") == "A, B"

    def test_make_text_hash_is_sha256(self):
        text = "hello world"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert make_text_hash(text) == expected

    def test_make_text_hash_length_64(self):
        assert len(make_text_hash("any text")) == 64

    def test_make_text_hash_hex_chars(self):
        h = make_text_hash("test")
        assert all(c in "0123456789abcdef" for c in h)

    def test_make_text_hash_deterministic(self):
        assert make_text_hash("same") == make_text_hash("same")

    def test_make_text_hash_different_inputs(self):
        assert make_text_hash("a") != make_text_hash("b")


# ---------------------------------------------------------------------------
# validate_catalog tests
# ---------------------------------------------------------------------------

class TestValidateCatalog:
    def test_valid_catalog_passes(self):
        """A well-formed catalog should not raise."""
        validate_catalog(_valid_catalog())

    def test_missing_column_raises(self):
        df = _valid_catalog()
        df = df.drop(columns=["embedding_text"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_catalog(df)

    def test_duplicate_global_id_raises(self):
        df = _valid_catalog()
        df2 = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="global_id is not unique"):
            validate_catalog(df2)

    def test_bad_global_id_format_raises(self):
        df = _valid_catalog()
        df.loc[0, "global_id"] = "bad-format-no-colons"
        with pytest.raises(ValueError, match="global_id format invalid"):
            validate_catalog(df)

    def test_empty_title_raises(self):
        df = _valid_catalog()
        df.loc[0, "title"] = ""
        with pytest.raises(ValueError, match="Empty 'title'"):
            validate_catalog(df)

    def test_empty_embedding_text_raises(self):
        df = _valid_catalog()
        df.loc[0, "embedding_text"] = ""
        with pytest.raises(ValueError, match="Empty 'embedding_text'"):
            validate_catalog(df)

    def test_empty_text_hash_raises(self):
        df = _valid_catalog()
        df.loc[0, "text_hash"] = ""
        with pytest.raises(ValueError, match="Empty 'text_hash'"):
            validate_catalog(df)

    def test_invalid_text_hash_format_raises(self):
        df = _valid_catalog()
        df.loc[0, "text_hash"] = "not-a-sha256"
        with pytest.raises(ValueError, match="Invalid text_hash"):
            validate_catalog(df)

    def test_empty_metadata_text_raises(self):
        df = _valid_catalog()
        df.loc[0, "metadata_text"] = ""
        with pytest.raises(ValueError, match="Empty 'metadata_text'"):
            validate_catalog(df)

    def test_global_id_format_movie(self):
        df = _valid_catalog(1)
        assert df["global_id"].iloc[0].startswith("movie:tmdb_5000_movies:")

    def test_all_content_types_valid(self):
        """Catalog with all three content types should pass validation."""
        validate_catalog(_valid_catalog(3))
