import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.audience import AudienceContext, is_eligible
from preprocessing import build_content_catalog
from preprocessing.audience_tagging import (
    annotate_audience,
    audience_summary,
    maturity_for_row,
)
from preprocessing.content_schema import (
    AUDIENCE_COLUMNS,
    CATALOG_COLUMNS,
    CONTENT_COLUMNS,
    MEDIA_COLUMNS,
)


def test_catalog_schema_extends_rather_than_replaces_the_base():
    """The base schema stays intact and the derived passes append to it.

    Order matters beyond tidiness: build_content_catalog returns
    `catalog[CATALOG_COLUMNS]`, so a column appended here must be produced by a
    pass that has already run by that point.
    """
    assert CATALOG_COLUMNS[: len(CONTENT_COLUMNS)] == CONTENT_COLUMNS
    assert (
        CATALOG_COLUMNS[len(CONTENT_COLUMNS) :] == AUDIENCE_COLUMNS + MEDIA_COLUMNS
    )
    # No column is declared twice across the three groups.
    assert len(set(CATALOG_COLUMNS)) == len(CATALOG_COLUMNS)


@pytest.mark.parametrize(
    "content_type, categories, expected",
    [
        # Movies declare maturity_source: certification, so a genre may restrict
        # but never relax - this holds at the `teen` default rather than reading
        # "Animation" as an audience. A G rating from
        # scripts.fetch_tmdb_certifications is what makes it all_ages.
        ("movie", "Family, Animation, Comedy", "teen"),
        ("movie", "Horror, Thriller", "adult"),
        ("book", "Juvenile Fiction", "all_ages"),
        ("book", "Young Adult Fiction", "teen"),
        # Health is a regulated domain (risk_tier 1), so a keyword may restrict
        # an item but never relax it below the content type's `adult` default.
        # This case previously expected all_ages; children's vitamins are a
        # leading cause of paediatric iron poisoning, so "safe for any age" was
        # the wrong answer for a product-discovery result.
        ("health", "Children Vitamins", "adult"),
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
            # Carries the maturity a G certification resolves to, the way a
            # catalogue row does once apply_certification_ratings has run.
            {"global_id": "m:1", "content_type": "movie", "title": "Minions",
             "categories": "Family, Animation", "maturity": "all_ages"},
            {"global_id": "m:2", "content_type": "movie", "title": "The Conjuring",
             "categories": "Horror", "maturity": ""},
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


@pytest.mark.parametrize(
    "content_type, category_text, expected",
    [
        # A violence signal must outrank a child-friendly one. "Animation,
        # Science Fiction, Thriller" previously matched only "animation" and
        # was served to an 8-year-old.
        ("movie", "Animation, Science Fiction, Thriller", "teen"),
        ("book", "Child care, mystery thriller", "teen"),
        ("music", "rap, gangster rap", "adult"),
        ("music", "rap, southern hip hop Murder After Midnight", "adult"),
        # A board-rated type holds at its default instead of being relaxed by a
        # genre; a type whose ratings only ever come from category text still is.
        ("movie", "Animation, Family", "teen"),
        ("book", "Juvenile Fiction, picture book", "all_ages"),
    ],
)
def test_violence_signals_outrank_child_friendly_ones(content_type, category_text, expected):
    assert maturity_for_row(content_type, category_text) == expected


@pytest.mark.parametrize(
    "category_text",
    ["Award-winning history", "Notes from the canteen", "Sixteen candles"],
)
def test_keywords_only_match_at_a_left_word_boundary(category_text):
    """"teen" must not fire inside "canteen"/"sixteen", nor "war" inside "award"."""
    assert maturity_for_row("book", category_text) == "teen"  # the book default


def test_music_without_an_explicit_flag_is_capped_at_teen():
    """The Spotify source ships no explicit-lyrics column.

    Defaulting music to all_ages failed *open* and rated 96% of the catalogue
    safe for children. Unknown music is capped at teen instead.
    """
    assert maturity_for_row("music", "pop, dance pop") == "teen"
    assert maturity_for_row("music", "") == "teen"


@pytest.mark.parametrize(
    "category_text",
    [
        # Every one of these was tagged all_ages by the keyword rules alone,
        # because its category text happened to contain an all-ages word.
        "Health & Household, Vitamins, Minerals & Supplements, Iron",
        "Health & Household, Sexual Wellness, Bondage Gear & Accessories",
        "Health & Household, Household Supplies, Indoor Insect & Pest Control",
        "Health & Household, Family Planning Tests, Pregnancy",
        "Health & Household, Medical Supplies & Equipment, Kids Knee Braces",
    ],
)
def test_a_regulated_domain_is_never_relaxed_below_its_default(category_text):
    """Health is risk_tier 1, so a keyword may restrict it but not relax it."""
    assert maturity_for_row("health", category_text) == "adult"


def test_a_regulated_domain_can_still_be_escalated_by_a_keyword():
    """Restrict-only means *only* relaxation is blocked, not escalation."""
    assert maturity_for_row("health", "Health & Household, adults only") == "restricted"


def test_a_type_without_a_rating_board_still_relaxes_on_keywords():
    """Books have no certification source, so category text is the best signal.

    Entertainment is risk_tier 0, so nothing else blocks the relaxation.
    """
    assert maturity_for_row("book", "Juvenile Fiction, picture book") == "all_ages"


def test_a_board_rated_type_is_never_relaxed_by_its_genre():
    """A genre is a production technique, not an audience.

    "Animation" matched the all-ages rule and relaxed Akira, Heavy Metal, Grave
    of the Fireflies and Waltz with Bashir - adult war and science-fiction films
    - from the movie default of `teen` to `all_ages`, where an 8-year-old was
    served them. Movies declare maturity_source: certification, so the rating
    from scripts.fetch_tmdb_certifications is the only thing that may relax one.
    """
    for genre_text in (
        "Animation, Science Fiction, Action",           # Akira
        "Animation, Science Fiction, Adventure, Music",  # Heavy Metal
        "Animation, Drama, War",                         # Grave of the Fireflies
        "Animation, Documentary, Drama, War",            # Waltz with Bashir
        "Family, Animation, Adventure, Comedy, War",     # Valiant
    ):
        assert maturity_for_row("movie", genre_text) == "teen"


def test_a_board_rated_type_can_still_be_escalated_by_its_genre():
    """Restrict-only blocks relaxation, not escalation - an unrated slasher
    must still be kept out of a child's results."""
    assert maturity_for_row("movie", "Animation, Horror") == "adult"
    assert maturity_for_row("movie", "Animation, Thriller") == "teen"


# ---------------------------------------------------------------------------
# Board certifications
# ---------------------------------------------------------------------------

def _certification_catalog(tmp_path, monkeypatch, rows):
    """Point apply_certification_ratings at a synthetic certification file."""
    path = tmp_path / "tmdb_certifications.csv"
    pd.DataFrame(rows, columns=["movie_id", "title", "certification"]).to_csv(
        path, index=False
    )
    monkeypatch.setattr(build_content_catalog, "CERTIFICATION_PATH", path)

    catalog = pd.DataFrame(
        [
            # Would derive all_ages from "Animation" under the old rules.
            {"global_id": "movie:tmdb_api:1", "content_type": "movie", "source": "tmdb_api",
             "source_id": "1", "title": "Rated G Cartoon", "categories": "Animation, Family"},
            {"global_id": "movie:tmdb_api:2", "content_type": "movie", "source": "tmdb_api",
             "source_id": "2", "title": "Akira", "categories": "Animation, Science Fiction, Action"},
            {"global_id": "movie:tmdb_api:3", "content_type": "movie", "source": "tmdb_api",
             "source_id": "3", "title": "Unrated Cartoon", "categories": "Animation, Family"},
            {"global_id": "book:g:4", "content_type": "book", "source": "g",
             "source_id": "4", "title": "A Picture Book", "categories": "Juvenile Fiction"},
        ]
    )
    return build_content_catalog.apply_certification_ratings(catalog)


@pytest.mark.parametrize(
    "certification, expected_maturity, expected_min_age",
    [
        ("G", "all_ages", 0),
        ("PG", "child", 7),
        ("PG-13", "teen", 13),
        ("R", "adult", 18),
        ("NC-17", "restricted", 21),
    ],
)
def test_a_board_rating_sets_the_maturity(
    tmp_path, monkeypatch, certification, expected_maturity, expected_min_age
):
    rated = _certification_catalog(
        tmp_path, monkeypatch, [{"movie_id": "1", "title": "x", "certification": certification}]
    )
    annotated = annotate_audience(rated).set_index("global_id")

    assert annotated.loc["movie:tmdb_api:1", "maturity"] == expected_maturity
    assert annotated.loc["movie:tmdb_api:1", "audience_min_age"] == expected_min_age


@pytest.mark.parametrize("certification", ["NR", "Unrated", "", "??"])
def test_an_absent_or_meaningless_rating_holds_at_the_default(
    tmp_path, monkeypatch, certification
):
    """NR says no board rated the title, not that it is harmless.

    Mapping it to anything permissive would reopen the hole the rating closes, so
    it falls through to the keyword heuristic - which, for a board-rated type, may
    only restrict.
    """
    rated = _certification_catalog(
        tmp_path, monkeypatch, [{"movie_id": "1", "title": "x", "certification": certification}]
    )
    annotated = annotate_audience(rated).set_index("global_id")

    assert annotated.loc["movie:tmdb_api:1", "maturity"] == "teen"
    assert annotated.loc["movie:tmdb_api:1", "audience_min_age"] == 13


def test_an_r_rating_keeps_an_animated_film_away_from_a_child(tmp_path, monkeypatch):
    """The case that started this: Akira is animated, and rated R."""
    rated = _certification_catalog(
        tmp_path, monkeypatch, [{"movie_id": "2", "title": "Akira", "certification": "R"}]
    )
    rows = annotate_audience(rated).to_dict("records")

    child = AudienceContext(age=8)
    eligible = [row["title"] for row in rows if is_eligible(row, child)]

    assert "Akira" not in eligible
    # The genuinely child-oriented book in the same catalogue is unaffected.
    assert "A Picture Book" in eligible


def test_only_movies_are_matched_against_the_certification_file(tmp_path, monkeypatch):
    """source_id collides across datasets, so a book must not take a movie rating."""
    rated = _certification_catalog(
        tmp_path, monkeypatch, [{"movie_id": "4", "title": "x", "certification": "R"}]
    )
    annotated = annotate_audience(rated).set_index("global_id")

    # The book keeps its own keyword-derived label, not the movie id 4 rating.
    assert annotated.loc["book:g:4", "maturity"] == "all_ages"


def test_a_missing_certification_file_is_a_warning_not_a_failure(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        build_content_catalog, "CERTIFICATION_PATH", tmp_path / "absent.csv"
    )
    catalog = pd.DataFrame(
        [{"global_id": "movie:tmdb_api:1", "content_type": "movie", "source": "tmdb_api",
          "source_id": "1", "title": "X", "categories": "Animation, Family"}]
    )

    result = build_content_catalog.apply_certification_ratings(catalog)

    assert "WARNING" in capsys.readouterr().out
    # Still fails closed: the genre cannot relax a board-rated type.
    assert annotate_audience(result).iloc[0]["maturity"] == "teen"


class TestHardcoreKeywordHardening:
    """Porn scene listings reached the music catalogue rated `teen`.

    Music rows carry no genre text, so the rules had nothing to match and fell
    through to the content type's default. These keywords are the backstop.
    """

    @pytest.mark.parametrize("title", [
        "Private Gangbangs Vol. 5 (Scene 3)",
        "7on1 MEGA GANGBANG! Sweet Slut Sofia Smith",
        "Colombian pornstars Madison Stil getting used",
        "Beautiful teen undergoes hardcore double penetration",
        "Double Vaginal Fantasy Of a Busty Beauty",
        "Anal-Crazy Hottie Gets DPed",
    ])
    def test_explicit_titles_are_restricted(self, title):
        assert maturity_for_row("music", title) == "restricted"

    @pytest.mark.parametrize("title", [
        # Each of these was measured against the full catalogue; a stem match
        # or a looser word list would wrongly restrict every one.
        "Final Analysis",
        "Analyze This",
        "El analfabeto",
        "xXx: State of the Union",
        "Kick-Ass",
        "A Pain in the Ass",
        "DAP 18130 2 Pack 10.1 oz. Alex Plus All Purpose Sealant",
        "Take Me Home - BBC Children In Need Single 2011",
        "A Cock and Bull Story",
    ])
    def test_ordinary_titles_are_not_restricted(self, title):
        assert maturity_for_row("movie", title) != "restricted"

    def test_pornograph_still_matches_as_a_stem(self):
        # The whole-word family must not have broken stem matching.
        assert maturity_for_row("movie", "A history of pornography") == "restricted"

    def test_whole_word_family_matches_its_own_terms(self):
        assert maturity_for_row("movie", "MILF Comedy") == "restricted"

    def test_keywords_alone_cannot_catch_every_case(self):
        # Documented limitation, asserted so it is not mistaken for coverage:
        # 10 of the 24 real listings carried no explicit word at all. Ingestion
        # filtering is the real defence; see scripts/fetch_lastfm_tracks.py.
        assert maturity_for_row("music", "Stepmoms Protein Supplements") != "restricted"


class TestPornWholeWordTier:
    """"porn" belongs in `adult`, not `restricted`: these are films about the
    industry, not hardcore material. Before this, 7 of 10 were rated `teen`."""

    @pytest.mark.parametrize("title", [
        "After Porn Ends 2", "Bikini Porn", "Zack and Miri Make a Porno",
        "Porn in the Hood", "Android Porn",
    ])
    def test_porn_titles_are_at_least_adult(self, title):
        assert maturity_for_row("movie", title) == "adult"

    def test_pornography_is_still_restricted_not_merely_adult(self):
        # The whole-word "porn" must not shadow the stem "pornograph".
        assert maturity_for_row("movie", "A history of pornography") == "restricted"

    @pytest.mark.parametrize("title", [
        "xXx: Return of Xander Cage",       # the film
        "XXX. FEAT. U2.",                   # a song
        "Elastic Knee Support Beige XXX-Large 24\" - 26\"",  # a clothing size
    ])
    def test_xxx_is_not_used_as_a_signal(self, title):
        assert maturity_for_row("movie", title) not in ("restricted",)


class TestMusicKeywordsMayOnlyRestrict:
    """Music declares `maturity_source: explicit_flag`, but the source ships no
    explicit-lyrics column, so the flag never fires and a genre is all there is.
    A genre says nothing about audience, so a keyword must never relax music
    below its `teen` default - it may only push it up."""

    @pytest.mark.parametrize("title", [
        "Kids in America", "Sour Patch Kids", "Cool Kids", "Children",
        "Family Matters", "family ties (with Kendrick Lamar)",
    ])
    def test_a_title_word_cannot_relax_music_below_teen(self, title):
        assert maturity_for_row("music", f"pop, dance pop {title}") == "teen"

    def test_music_can_still_be_escalated(self):
        # Escalation is the half that must keep working.
        assert maturity_for_row("music", "gangster rap, horrorcore") == "adult"
        assert maturity_for_row("music", "pornographic industrial") == "restricted"

    def test_books_are_unaffected_and_may_still_relax(self):
        # Books rate from `category`, where "Juvenile fiction" is a real
        # audience signal rather than an accident of the title.
        assert maturity_for_row("book", "Juvenile fiction, picture book") == "all_ages"

    def test_movies_keep_their_existing_guard(self):
        # Movies were already covered: "animation" is a production technique,
        # not an audience, and must not relax a film below the teen default.
        assert maturity_for_row("movie", "Animation, Science Fiction") == "teen"


class TestShowsAreGuardedLikeFilms:
    """Television declares `maturity_source: certification` even though no TV
    board rating is fetched. The label is what activates _restrict_only: under
    it a genre may only restrict a show, never relax one. Without it
    "Animation" would drop adult series to all_ages, the same way it once
    relaxed Akira and Grave of the Fireflies on the film side."""

    def test_animation_cannot_relax_a_show_below_teen(self):
        assert maturity_for_row("show", "Animation, Family") == "teen"

    def test_kids_genres_cannot_relax_a_show(self):
        assert maturity_for_row("show", "Kids, Animation") == "teen"

    def test_a_show_can_still_escalate(self):
        assert maturity_for_row("show", "Horror, Thriller") == "adult"

    def test_an_unremarkable_show_takes_the_teen_default(self):
        assert maturity_for_row("show", "Drama, Crime") == "teen"

    def test_show_is_registered_in_entertainment(self):
        from backend.domains import get_registry
        registry = get_registry()
        assert "show" in registry.content_types_for_domain("entertainment")
        assert registry.content_type("show").domain == "entertainment"
