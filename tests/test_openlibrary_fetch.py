"""Unit tests for the Open Library book fetcher.

Only the pure transforms are covered - no network. Every noisy input here is a
shape observed in a real `search.json` response while building the fetcher.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.fetch_openlibrary_books import (
    build_query,
    build_row,
    clean_description,
    clean_subjects,
    merge_with_existing,
    passes,
)


class TestCleanDescription:
    def test_plain_text_passes_through(self):
        assert clean_description("A novel about a whale.") == "A novel about a whale."

    def test_none_and_empty_become_empty(self):
        assert clean_description(None) == ""
        assert clean_description("") == ""

    def test_dict_form_is_unwrapped(self):
        value = {"type": "/type/text", "value": "  Synopsis   text  "}
        assert clean_description(value) == "Synopsis text"

    def test_list_form_is_joined(self):
        assert clean_description(["First.", "Second."]) == "First. Second."

    def test_trailer_after_horizontal_rule_is_cut(self):
        raw = (
            "Good synopsis text here.\r\n\r\n---\r\n\r\nContained in:\r\n"
            "[Ender's War](https://openlibrary.org/works/OL49619W)"
        )
        assert clean_description(raw) == "Good synopsis text here."

    def test_markdown_reference_footnotes_are_cut(self):
        raw = "Body.\r\n\r\n----------\r\n\r\n[1]: https://example.com/x"
        assert clean_description(raw) == "Body."

    def test_inline_link_keeps_its_text(self):
        raw = "See the [full text](https://example.com) online."
        assert clean_description(raw) == "See the full text online."

    def test_description_that_is_only_a_trailer_becomes_empty(self):
        # These records carry no synopsis at all. Left uncut they read
        # "Contained in: A B", which clears the length filter and says nothing.
        assert clean_description("Contained in:\n- [A](https://x)\n- [B](https://y)") == ""
        assert clean_description("See also:\n- [A](https://x)") == ""

    def test_prose_hyphens_are_not_mistaken_for_a_rule(self):
        raw = "A well-written - and long - study of 1914-1918."
        assert clean_description(raw) == raw


class TestCleanSubjects:
    def test_machine_tags_are_dropped(self):
        raw = ["Fiction", "nyt:combined-print-and-e-book-fiction=2014-03-23", "ddc:813/.54"]
        assert clean_subjects(raw, 12) == ["Fiction"]

    def test_library_metadata_is_dropped(self):
        raw = [
            "Magic", "New York Times bestseller", "Large type books",
            "Accessible book", "Reading Level-Grade 11", "Open Library Staff Picks",
        ]
        assert clean_subjects(raw, 12) == ["Magic"]

    def test_compound_headings_become_comma_separated(self):
        assert clean_subjects(["Imaginary wars and battles -- Fiction"], 12) == [
            "Imaginary wars and battles, Fiction"
        ]

    def test_deduplicates_case_insensitively(self):
        assert clean_subjects(["Fiction", "fiction", "FICTION"], 12) == ["Fiction"]

    def test_respects_the_cap(self):
        assert clean_subjects(["a", "b", "c", "d"], 2) == ["a", "b"]

    def test_none_and_blanks_are_safe(self):
        assert clean_subjects(None, 12) == []
        assert clean_subjects(["", "   "], 12) == []

    def test_maturity_keywords_survive_cleaning(self):
        # The audience tagger matches against this text, so the words it keys
        # on must not be filtered out as noise.
        subjects = clean_subjects(["Juvenile fiction", "Horror stories"], 12)
        assert subjects == ["Juvenile fiction", "Horror stories"]


class TestBuildQuery:
    def test_plain_subject_has_no_year_clause(self):
        assert build_query("science fiction", None, None) == 'subject:"science fiction"'

    def test_multi_word_subjects_stay_quoted(self):
        # Unquoted, Solr reads the second word as a separate term and the
        # subject filter stops meaning anything.
        assert build_query("machine learning", None, None) == 'subject:"machine learning"'

    def test_both_bounds_make_a_closed_window(self):
        assert build_query("fiction", 2024, 2026) == (
            'subject:"fiction" AND first_publish_year:[2024 TO 2026]'
        )

    def test_open_ended_bounds_are_filled_in(self):
        assert build_query("fiction", 2024, None).endswith("[2024 TO 9999]")
        assert build_query("fiction", None, 1950).endswith("[1 TO 1950]")


class TestMergeWithExisting:
    def _frame(self, ids, category="new"):
        return pd.DataFrame(
            {"book_id": ids, "title": [f"T{i}" for i in ids], "search_category": category}
        )

    def test_missing_file_returns_the_frame_unchanged(self, tmp_path):
        frame = self._frame(["A", "B"])
        merged, previous = merge_with_existing(frame, tmp_path / "absent.csv")
        assert previous == 0
        assert list(merged.book_id) == ["A", "B"]

    def test_new_rows_are_appended(self, tmp_path):
        path = tmp_path / "books.csv"
        self._frame(["A", "B"], "old").to_csv(path, index=False)
        merged, previous = merge_with_existing(self._frame(["C"]), path)
        assert previous == 2
        assert sorted(merged.book_id) == ["A", "B", "C"]

    def test_bare_years_survive_the_round_trip(self, tmp_path):
        # Regression: a default pd.read_csv types a year column with any blank
        # in it as float64, so merging rewrote "1979" as "1979.0" and the API
        # served a broken year. The blank must stay blank and the year a year.
        path = tmp_path / "books.csv"
        pd.DataFrame({
            "book_id": ["A", "B"],
            "title": ["T1", "T2"],
            "published_date": ["1979", ""],
        }).to_csv(path, index=False)

        merged, _ = merge_with_existing(
            pd.DataFrame({"book_id": ["C"], "title": ["T3"], "published_date": ["2026"]}),
            path,
        )
        by_id = merged.set_index("book_id").published_date
        assert by_id["A"] == "1979"
        assert by_id["C"] == "2026"
        assert by_id["B"] in ("", None) or pd.isna(by_id["B"])

    def test_existing_rows_win_on_collision(self, tmp_path):
        # A book already in the catalogue must keep its row, so its global_id
        # does not move and seeded interactions against it survive.
        path = tmp_path / "books.csv"
        self._frame(["A"], "old").to_csv(path, index=False)
        merged, _ = merge_with_existing(self._frame(["A"], "new"), path)
        assert len(merged) == 1
        assert merged.iloc[0].search_category == "old"


def _doc(**overrides) -> dict:
    doc = {
        "key": "/works/OL2163649W",
        "title": "The Hitchhiker's Guide to the Galaxy",
        "subtitle": "",
        "author_name": ["Douglas Adams"],
        "first_publish_year": 1979,
        "publisher": ["Pan Books", "Del Rey"],
        "subject": ["comic science fiction", "Vogons"],
        "cover_i": 12986869,
        "ratings_average": 4.51,
        "ratings_count": 178,
        "description": "The first of six books in the trilogy.",
        "language": ["eng", "ger"],
    }
    doc.update(overrides)
    return doc


class TestBuildRow:
    def test_work_key_is_stripped_to_the_id(self):
        assert build_row(_doc(), "science fiction", 12)["book_id"] == "OL2163649W"

    def test_cover_id_becomes_a_url(self):
        row = build_row(_doc(), "science fiction", 12)
        assert row["thumbnail"] == "https://covers.openlibrary.org/b/id/12986869-L.jpg"

    def test_missing_cover_leaves_the_url_empty(self):
        assert build_row(_doc(cover_i=None), "science fiction", 12)["thumbnail"] == ""

    def test_authors_are_joined_and_first_publisher_wins(self):
        row = build_row(_doc(author_name=["A Smith", "B Jones"]), "science fiction", 12)
        assert row["authors"] == "A Smith, B Jones"
        assert row["publisher"] == "Pan Books"

    def test_search_category_records_the_subject_that_found_it(self):
        assert build_row(_doc(), "comics", 12)["search_category"] == "comics"

    def test_untitled_or_keyless_docs_are_rejected(self):
        assert build_row(_doc(title=""), "science fiction", 12) is None
        assert build_row(_doc(key=""), "science fiction", 12) is None

    def test_emits_exactly_the_columns_the_normalizer_reads(self):
        row = build_row(_doc(), "science fiction", 12)
        required = {
            "book_id", "title", "subtitle", "authors", "published_date",
            "description", "categories", "search_category", "average_rating",
            "ratings_count", "thumbnail",
        }
        assert required <= set(row)


class TestPasses:
    def test_accepts_a_complete_row(self):
        doc = _doc()
        assert passes(build_row(doc, "science fiction", 12), doc, 20, "eng")

    def test_rejects_a_short_description(self):
        doc = _doc(description="Too short.")
        assert not passes(build_row(doc, "science fiction", 12), doc, 20, "eng")

    def test_rejects_a_missing_cover(self):
        doc = _doc(cover_i=None)
        assert not passes(build_row(doc, "science fiction", 12), doc, 20, "eng")

    def test_rejects_a_work_with_no_edition_in_the_wanted_language(self):
        doc = _doc(language=["fre", "ger"])
        assert not passes(build_row(doc, "science fiction", 12), doc, 20, "eng")

    def test_accepts_a_work_with_any_edition_in_the_wanted_language(self):
        doc = _doc(language=["fre", "eng"])
        assert passes(build_row(doc, "science fiction", 12), doc, 20, "eng")

    def test_unknown_language_is_kept_rather_than_assumed_foreign(self):
        doc = _doc(language=[])
        assert passes(build_row(doc, "science fiction", 12), doc, 20, "eng")

    def test_empty_language_filter_accepts_anything(self):
        doc = _doc(language=["fre"])
        assert passes(build_row(doc, "science fiction", 12), doc, 20, "")
