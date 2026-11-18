"""Ranking quality: popularity prior, graph candidates, query intent, and the guards against junk matches."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend import qdrant_recommender as qr
from backend.audience import AudienceContext
from backend.graph_recommender import GraphEmbeddingStore
from backend.plot_index import PlotIndex
from backend.prominence import compute_prominence
from backend.qdrant_recommender import (
    QdrantRecommender,
    drop_duplicate_works,
    embedding_text,
    parse_query_intent,
    parse_year_range,
)


def _row(gid, title, ctype="movie", creators="", categories="Drama", popularity=1.0, source="test",
         domain="entertainment", description="", year="2020"):
    return {
        "global_id": gid, "content_type": ctype, "source": source, "source_id": gid.rsplit(":", 1)[-1],
        "title": title, "description": description, "creators": creators, "categories": categories,
        "release_date": year, "popularity": popularity, "rating": 7.0, "domain": domain,
        "maturity": "all_ages", "audience_min_age": 0, "risk_tier": 0, "embedding_text": title,
    }


class Vectors:
    """Stands in for the stored item embeddings (EMAEmbeddingStore.get_item_vector)."""

    def __init__(self, vectors):
        self.vectors = {gid: np.asarray(v, dtype=np.float32) / np.linalg.norm(v) for gid, v in vectors.items()}

    def get_item_vector(self, gid):
        return self.vectors.get(gid)


class Model:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32) / np.linalg.norm(vector)

    def encode(self, texts, **kwargs):
        return np.tile(self.vector, (len(texts), 1))


def _recommender(rows, vectors, query_vector, monkeypatch=None, plot=None):
    rec = QdrantRecommender.__new__(QdrantRecommender)
    rec.catalog = pd.DataFrame(rows)
    rec.graph_store = None
    rec.ema_store = Vectors(vectors)
    rec.model = Model(query_vector)
    rec._plot_index_cache = plot  # never the locally built synopsis index

    def search(vector, top_k, content_type, audience=None):
        scored = []
        for row in rows:
            if content_type and row["content_type"] != content_type:
                continue
            item = dict(row)
            item["score"] = float(rec.ema_store.get_item_vector(row["global_id"]) @ vector)
            scored.append(item)
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    rec._search_vector = search
    return rec


# --------------------------------------------------------------- query intent
@pytest.mark.parametrize("query, expected", [
    ("books about artificial intelligence", ("book", None)),
    ("movies like inception", ("movie", "inception")),
    ("similar to the dark knight", (None, "the dark knight")),
    ("songs like shape of you", ("music", "shape of you")),
    ("something like breaking bad", (None, "breaking bad")),
    ("i like dogs", (None, None)),
    ("movie or book about space", (None, None)),
    ("space adventure", (None, None)),
])
def test_query_intent(query, expected):
    assert parse_query_intent(query) == expected


# ----------------------------------------------------------------- prominence
def test_prominence_scales_each_source_on_its_own(tmp_path):
    exports = tmp_path / "tmdb_movies.csv"
    pd.DataFrame({"id": [10, 11], "vote_count": [40000, 50]}).to_csv(exports, index=False)
    catalog = pd.DataFrame([
        _row("movie:tmdb_api:10", "Famous", popularity=3.0, source="tmdb_api"),
        _row("movie:tmdb_api:11", "Obscure", popularity=90.0, source="tmdb_api"),   # trending, few votes
        _row("music:x:1", "Hit", ctype="music", popularity=80.0),
        _row("music:x:2", "Deep cut", ctype="music", popularity=20.0),
    ])
    prominence = compute_prominence(catalog, {"movies": exports})
    # Vote counts, not the trending figure, decide for TMDb rows.
    assert prominence[0] > prominence[1]
    # Music is a 0-100 index already: used linearly.
    assert prominence[2] == pytest.approx(0.8) and prominence[3] == pytest.approx(0.2)
    assert ((prominence >= 0) & (prominence <= 1)).all()


def test_prominence_without_exports_falls_back_to_catalogue_popularity():
    catalog = pd.DataFrame([_row("book:x:1", "A", ctype="book", popularity=500.0),
                            _row("book:x:2", "B", ctype="book", popularity=2.0)])
    prominence = compute_prominence(catalog, {})
    assert prominence[0] > prominence[1] > 0


# ------------------------------------------------------------- ranking order
def test_an_established_relevant_title_outranks_an_obscure_namesake():
    rows = [
        _row("movie:t:1", "Dark Water Two", popularity=2.0, categories="Thriller, dark"),
        _row("movie:t:2", "Prisoners", popularity=5000.0, categories="Thriller, dark"),
        _row("movie:t:3", "Unrelated", popularity=1.0, categories="Comedy"),
    ]
    rec = _recommender(rows, {"movie:t:1": [1, 0.30], "movie:t:2": [1, 0.42], "movie:t:3": [0, 1]},
                       query_vector=[1, 0.28])
    results = QdrantRecommender.recommend(rec, "dark thriller", top_k=3, audience=AudienceContext(age=30))
    assert [item["title"] for item in results][:2] == ["Prisoners", "Dark Water Two"]
    top = results[0]
    assert top["semantic_score"] is not None and top["popularity_score"] > 0.5


def test_only_an_exact_title_is_navigational():
    rows = [
        _row("movie:t:1", "Time Travel Mater", popularity=1.0, categories="Animation, time travel"),
        _row("movie:t:2", "Looper", popularity=900.0, categories="Science Fiction, time travel"),
        _row("movie:t:3", "Primer", popularity=300.0, categories="Science Fiction, time travel"),
    ]
    rec = _recommender(rows, {"movie:t:1": [1, 0.1], "movie:t:2": [1, 0.1], "movie:t:3": [1, 0.1]},
                       query_vector=[1, 0.1])
    results = QdrantRecommender.recommend(rec, "time travel", top_k=3, audience=AudienceContext(age=30))
    mater = next(item for item in results if item["title"] == "Time Travel Mater")
    assert mater.get("match_score") is None          # a prefix title competes on relevance
    assert results[0]["title"] == "Looper"


def test_movies_like_a_title_answers_with_similar_items(monkeypatch):
    rows = [_row("movie:t:1", "Inception", popularity=900.0), _row("music:t:2", "Inception", ctype="music")]
    rec = _recommender(rows, {"movie:t:1": [1, 0], "music:t:2": [0, 1]}, query_vector=[1, 0])
    calls = []

    def fake_similar(self, global_id, top_k=10, content_type=None, user_id=None, audience=None, years=None):
        calls.append((global_id, content_type))
        return [{"global_id": "movie:x", "title": "Tenet", "score": 1.0}]

    monkeypatch.setattr(QdrantRecommender, "recommend_from_item", fake_similar)
    results = QdrantRecommender.recommend(rec, "movies like inception", top_k=5)
    assert calls == [("movie:t:1", "movie")]          # the film, not the song of the same name
    assert results[0]["similar_to"] == "Inception"


# ----------------------------------------------------------- junk creators
def test_a_brand_matches_only_by_its_full_name():
    rows = [_row("health:a:1", "Cookies", ctype="health", creators="LOVE GOOD FATS", source="amazon_reviews_2023",
                 domain="health")]
    rec = QdrantRecommender.__new__(QdrantRecommender)
    rec.catalog = pd.DataFrame(rows)
    assert QdrantRecommender._search_person_matches(rec, "love", top_k=5, content_type=None) == []
    assert len(QdrantRecommender._search_person_matches(rec, "love good fats", top_k=5, content_type=None)) == 1


def test_a_common_single_word_is_not_taken_for_a_name(monkeypatch):
    rows = [
        _row("music:s:1", "So Alive", ctype="music", creators="Love and Rockets", popularity=60.0),
        _row("music:s:2", "Love", ctype="music", popularity=70.0),
        _row("music:s:3", "Love", ctype="music", popularity=50.0),
    ]
    rec = _recommender(rows, {"music:s:1": [0, 1], "music:s:2": [1, 0], "music:s:3": [1, 0]}, query_vector=[1, 0])
    common = SimpleNamespace(rows={"love": qr.COMMON_WORD_ROWS + 1}, has_known_word=lambda words: True)
    monkeypatch.setattr(QdrantRecommender, "_query_vocabulary", lambda self: common)
    results = QdrantRecommender.recommend(rec, "love", top_k=5, audience=AudienceContext(age=30))
    assert all(item.get("match_kind") != "creator" for item in results)


def test_partial_creator_matches_are_ordered_by_prominence():
    rows = [
        _row("movie:n:1", "Riddick", creators="David Twohy, Nolan Gerard Funk", popularity=10.0),
        _row("movie:n:2", "Inception", creators="Christopher Nolan, Leonardo DiCaprio", popularity=5000.0),
    ]
    rec = QdrantRecommender.__new__(QdrantRecommender)
    rec.catalog = pd.DataFrame(rows)
    results = QdrantRecommender._search_person_matches(rec, "nolan", top_k=5, content_type=None)
    assert [item["title"] for item in results] == ["Inception", "Riddick"]


# ------------------------------------------------------------- preferences
def test_taste_reorders_close_calls_but_cannot_dominate(monkeypatch):
    many = {f"tag{i}": 3.0 for i in range(20)}

    class Store:
        def __init__(self, *args):
            pass

        def get_user_preference_profile(self, user_id):
            return {"category_weights": many, "content_type_weights": {"movie": 3.0},
                    "positive_ids": set(), "negative_ids": set()}

    monkeypatch.setattr(qr, "MySQLStore", Store)
    rec = QdrantRecommender.__new__(QdrantRecommender)
    item = {"global_id": "movie:t:1", "content_type": "movie", "score": 0.5,
            "categories": ", ".join(many)}
    [out] = QdrantRecommender._personalize_results(rec, [item], 5, "user_x")
    assert out["preference_bonus"] == pytest.approx(qr.PREFERENCE_CAP)


def test_retrieval_semantic_score_is_not_overwritten_by_lightgcn(tmp_path):
    artifact = tmp_path / "lightgcn.npz"
    np.savez_compressed(artifact, user_ids=np.array(["u1"], dtype=object), item_ids=np.array(["i1"], dtype=object),
                        user_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
                        item_embeddings=np.array([[1.0, 0.0]], dtype=np.float32))
    [item] = GraphEmbeddingStore(artifact).rerank("u1", [{"global_id": "i1", "score": 1.4, "semantic_score": 0.6}])
    assert item["semantic_score"] == 0.6


# ----------------------------------------------------------- more like this
def test_more_like_this_uses_graph_neighbours_and_stays_in_the_medium():
    rows = [
        _row("movie:m:1", "Inception", creators="Christopher Nolan, Leonardo DiCaprio", categories="Science Fiction, dream",
             popularity=900.0),
        _row("movie:m:2", "Tenet", creators="Christopher Nolan, John David Washington", categories="Science Fiction, time",
             popularity=800.0),
        _row("movie:m:3", "Dream House", creators="Jim Sheridan", categories="Thriller, dream", popularity=50.0),
        _row("music:m:4", "Inception", ctype="music", creators="Some Band", categories="pop", popularity=90.0),
    ]
    # Text alone puts the namesake song (0.74) and "Dream House" (0.64) above Tenet (0.53);
    # the graph knows Tenet shares the director.
    vectors = {"movie:m:1": [1, 0, 0], "movie:m:2": [1, 1.6, 0], "movie:m:3": [1, 1.2, 0], "music:m:4": [1, 0.9, 0]}
    rec = _recommender(rows, vectors, query_vector=[1, 0, 0])
    results = QdrantRecommender.recommend_from_item(rec, "movie:m:1", top_k=3, audience=AudienceContext(age=30))
    titles = [item["title"] for item in results]
    assert titles[0] == "Tenet"
    assert "Inception" not in titles                       # neither the seed nor the namesake song
    assert results[0]["signal_weights"] == {"kg": pytest.approx(0.8)}
    assert results[0]["kg_explanation"] == "Directed by Christopher Nolan, like Inception"


def _frozen_and_songs(without=(), moved=None):
    rows = [
        _row("movie:f:1", "Frozen", creators="Chris Buck, Idina Menzel", categories="Animation, princess",
             popularity=900.0),
        _row("music:f:2", "Let It Go", ctype="music", creators="Idina Menzel", categories="soundtrack",
             popularity=80.0),
        _row("music:f:3", "Cinderella", ctype="music", creators="Some Rapper", categories="hip hop",
             popularity=95.0),
        _row("music:f:4", "Winter Song", ctype="music", creators="Other Band", categories="folk", popularity=40.0),
    ]
    # Cosines to Frozen: Let It Go 0.55 (but shares Idina Menzel), Cinderella 0.61
    # (nothing shared), Winter Song 0.89 (nothing shared, reads close).
    vectors = {"movie:f:1": [1, 0, 0], "music:f:2": [1, 1.5, 0], "music:f:3": [1, 1.3, 0], "music:f:4": [1, 0.5, 0]}
    vectors.update(moved or {})
    rows = [row for row in rows if row["global_id"] not in without]
    return _recommender(rows, {gid: v for gid, v in vectors.items() if gid not in without}, query_vector=[1, 0, 0])


def test_more_like_this_of_one_type_keeps_only_linked_or_close_items():
    rec = _frozen_and_songs()
    audience = AudienceContext(age=30)
    padded = QdrantRecommender.recommend_from_item(rec, "movie:f:1", top_k=5, content_type="music", audience=audience)
    assert "Cinderella" in [item["title"] for item in padded]          # without the cutoff: nearest songs, related or not

    close = QdrantRecommender.recommend_from_item(rec, "movie:f:1", top_k=5, content_type="music", audience=audience,
                                                  close_only=True)
    assert sorted(item["title"] for item in close) == ["Let It Go", "Winter Song"]
    linked = next(item for item in close if item["title"] == "Let It Go")
    assert "Idina Menzel" in linked["kg_explanation"]


def test_more_like_this_of_one_type_can_honestly_be_empty():
    # No Let It Go, and Winter Song reads no closer than Cinderella (0.61).
    rec = _frozen_and_songs(without=("music:f:2",), moved={"music:f:4": [1, 1.3, 0]})
    results = QdrantRecommender.recommend_from_item(rec, "movie:f:1", top_k=5, content_type="music",
                                                    audience=AudienceContext(age=30), close_only=True)
    assert results == []


# ------------------------------------------------------------------ years
@pytest.mark.parametrize("query, years, rest, bare", [
    ("90s movies", (1990, 1999), "movies", False),
    ("80's songs", (1980, 1989), "songs", False),
    ("sci-fi from the 1980s", (1980, 1989), "sci-fi", False),
    ("best movies of the nineties", (1990, 1999), "best movies", False),
    ("songs from 2010", (2010, 2010), "songs", False),
    ("movies after 2015", (2016, 2100), "movies", False),
    ("books before 1950", (0, 1949), "books", False),
    ("1984 book", (1984, 1984), "book", True),
    ("2012", None, "2012", False),
    ("30 seconds to mars", None, "30 seconds to mars", False),
])
def test_year_phrases(query, years, rest, bare):
    assert parse_year_range(query) == (years, rest, bare)


@pytest.mark.parametrize("query, setting", [
    ("90s movies", False),
    ("sci-fi movies from the 80s", False),
    ("horror movies with zombies from the 80s", False),
    ("a movie set in the 80s", True),
    ("group of kids in the 80s search for their missing friend and meet a girl with powers", True),
])
def test_a_period_inside_a_story_is_its_setting(query, setting):
    from backend.qdrant_recommender import is_period_setting

    assert is_period_setting(query, parse_year_range(query)[1]) is setting


def _decades():
    return [
        _row("movie:y:1", "Pulp Fiction", popularity=5000.0, year="1994-10-14", categories="Crime"),
        _row("movie:y:2", "The Matrix", popularity=3000.0, year="1999-03-30", categories="Science Fiction, space"),
        _row("movie:y:3", "Inception", popularity=9000.0, year="2010-07-15", categories="Science Fiction, dream"),
        _row("movie:y:4", "Star Voyage", popularity=40.0, year="1982-06-01", categories="Science Fiction, space"),
        _row("movie:y:5", "Interstellar", popularity=8000.0, year="2014-11-05", categories="Science Fiction, space"),
        _row("music:y:6", "Wannabe", ctype="music", popularity=80.0, year="1996-07-08", categories="pop"),
    ]


def test_a_decade_with_no_topic_browses_its_best_known_titles():
    rows = _decades()
    rec = _recommender(rows, {row["global_id"]: [1, 0] for row in rows}, query_vector=[1, 0])
    results = QdrantRecommender.recommend(rec, "90s movies", top_k=5, audience=AudienceContext(age=30))
    assert [item["title"] for item in results] == ["Pulp Fiction", "The Matrix"]


def test_a_decade_filters_a_topical_search():
    rows = _decades()
    vectors = {row["global_id"]: [1, 0.2] for row in rows}
    rec = _recommender(rows, vectors, query_vector=[1, 0.2])
    results = QdrantRecommender.recommend(rec, "space movies from the 80s", top_k=5, audience=AudienceContext(age=30))
    assert [item["title"] for item in results] == ["Star Voyage"]


@pytest.mark.parametrize("query, title, year", [
    ("that 70s show", "That '70s Show", "1998-08-23"),     # the decade is part of the title
    ("1984 book", "1984", "1949-06-08"),                   # Orwell, not books from 1984
])
def test_a_year_that_belongs_to_a_title_is_not_a_filter(query, title, year):
    ctype = "book" if "book" in query else "show"
    rows = [_row("x:t:1", title, ctype=ctype, popularity=500.0, year=year),
            _row("x:t:2", "Other", ctype=ctype, popularity=10.0, year="1975")]
    rec = _recommender(rows, {"x:t:1": [1, 0], "x:t:2": [1, 0]}, query_vector=[1, 0])
    results = QdrantRecommender.recommend(rec, query, top_k=3, audience=AudienceContext(age=30))
    assert results[0]["title"] == title


# -------------------------------------------------------- query rewrites
def test_slang_is_spelled_out_for_the_encoder():
    assert embedding_text("rom-com movies") == "romantic comedy movies"
    assert embedding_text("best sci fi") == "best science fiction"
    assert embedding_text("romance") == "romance"


def test_like_an_unknown_title_searches_the_description(monkeypatch):
    rows = [_row("movie:d:1", "Inception", categories="Science Fiction, dream", popularity=900.0),
            _row("book:d:2", "Dream Journal", ctype="book", categories="dream")]
    rec = _recommender(rows, {"movie:d:1": [1, 0], "book:d:2": [1, 0]}, query_vector=[1, 0])
    monkeypatch.setattr(QdrantRecommender, "recommend_from_item",
                        lambda *a, **k: pytest.fail("no title named 'dream manipulation' exists"))
    results = QdrantRecommender.recommend(rec, "movies like dream manipulation", top_k=5,
                                          audience=AudienceContext(age=30))
    assert [item["title"] for item in results] == ["Inception"]       # movies only, found by topic
    assert results[0].get("similar_to") is None


# --------------------------------------------------------------- creators
def test_one_word_prefers_the_established_partial_name_over_an_obscure_whole_name():
    rows = [
        _row("music:n:1", "Conclusion", ctype="music", creators="NOLAN", popularity=39.0),
        _row("movie:n:2", "Inception", creators="Christopher Nolan, Leonardo DiCaprio", popularity=5000.0),
    ]
    rec = QdrantRecommender.__new__(QdrantRecommender)
    rec.catalog = pd.DataFrame(rows)
    results = QdrantRecommender._search_person_matches(rec, "nolan", top_k=5, content_type=None)
    assert [item["title"] for item in results] == ["Inception", "Conclusion"]
    assert results[1]["match_tier"] == 0                 # still recorded as a whole-name match


# ------------------------------------------------------------- duplicates
def test_copies_of_one_work_are_dropped_but_covers_and_remakes_kept():
    def item(gid, ctype, title, creators, date):
        return {"global_id": gid, "content_type": ctype, "title": title, "creators": creators, "release_date": date}

    items = [
        item("a", "music", "Shape of You", "Ed Sheeran", "2017-01-06"),
        item("b", "music", "Shape of You", "Ed Sheeran", "2017-03-03"),
        item("c", "music", "Shape of You", "EZA", "2017"),
        item("d", "movie", "The Man Who Knew Too Much", "Alfred Hitchcock", "1934"),
        item("e", "movie", "The Man Who Knew Too Much", "Alfred Hitchcock", "1956"),
    ]
    assert [x["global_id"] for x in drop_duplicate_works(items)] == ["a", "c", "d", "e"]
    assert [x["global_id"] for x in drop_duplicate_works(items, seed=items[0])] == ["c", "d", "e"]


def test_more_like_a_song_never_returns_another_copy_of_it():
    rows = [
        _row("music:c:1", "Shape of You", ctype="music", creators="Ed Sheeran", categories="pop", popularity=90.0),
        _row("music:c:2", "Shape of You", ctype="music", creators="Ed Sheeran", categories="pop", popularity=85.0),
        _row("music:c:3", "Photograph", ctype="music", creators="Ed Sheeran", categories="pop", popularity=80.0),
    ]
    rec = _recommender(rows, {"music:c:1": [1, 0], "music:c:2": [1, 0], "music:c:3": [1, 0.3]}, query_vector=[1, 0])
    results = QdrantRecommender.recommend_from_item(rec, "music:c:1", top_k=5, audience=AudienceContext(age=30))
    assert [item["title"] for item in results] == ["Photograph"]


# ------------------------------------------------------- titles kept from a viewer
def _godfather_rows():
    film = _row("movie:g:1", "The Godfather", creators="Francis Ford Coppola, Marlon Brando",
                categories="Crime, Drama, mafia", popularity=9000.0, year="1972")
    film.update(maturity="adult", audience_min_age=18)
    brasco = _row("movie:g:2", "Mafia Family Story", creators="Mike Newell, Al Pacino",
                  categories="Crime, Drama, mafia", popularity=3000.0, year="1997")
    brasco.update(maturity="teen", audience_min_age=13)
    hit = _row("movie:g:3", "Popular Space Film", categories="Science Fiction", popularity=9500.0)
    hit.update(maturity="teen", audience_min_age=13)
    return [film, brasco, hit]


def test_a_title_the_viewer_cannot_see_is_answered_with_what_is_like_it():
    rows = _godfather_rows()
    vectors = {"movie:g:1": [1, 0], "movie:g:2": [1, 0.3], "movie:g:3": [0, 1]}
    rec = _recommender(rows, vectors, query_vector=[0.6, 0.8])      # a bare title says little: closer to the hit
    no_age = AudienceContext()                                        # unknown age stops at 13+
    results = QdrantRecommender.recommend(rec, "The Godfather", top_k=3, content_type="movie", audience=no_age)
    assert [item["title"] for item in results][0] == "Mafia Family Story"
    assert "The Godfather" not in [item["title"] for item in results]
    assert all(item["similar_to"] == "The Godfather" for item in results)
    notice = QdrantRecommender.hidden_title_notice(rec, "The Godfather", "movie", no_age, results)
    assert notice.startswith("“The Godfather” (movie, 18+) is hidden: no age is set")
    assert notice.endswith("Showing similar titles you can see.")


def test_an_adult_sees_the_title_and_no_notice():
    rows = _godfather_rows()
    rec = _recommender(rows, {"movie:g:1": [1, 0], "movie:g:2": [1, 0.3], "movie:g:3": [0, 1]}, query_vector=[0.6, 0.8])
    adult = AudienceContext(age=30)
    results = QdrantRecommender.recommend(rec, "The Godfather", top_k=3, audience=adult)
    assert results[0]["title"] == "The Godfather"
    assert QdrantRecommender.hidden_title_notice(rec, "The Godfather", None, adult, results) is None


def test_a_visible_namesake_keeps_the_search_but_the_notice_still_explains():
    rows = _godfather_rows()
    book = _row("book:g:4", "The Godfather", ctype="book", creators="Mario Puzo", categories="Crime", popularity=800.0)
    book.update(maturity="teen", audience_min_age=13)
    rows.append(book)
    vectors = {"movie:g:1": [1, 0], "movie:g:2": [1, 0.3], "movie:g:3": [0, 1], "book:g:4": [1, 0.1]}
    rec = _recommender(rows, vectors, query_vector=[0.6, 0.8])
    teen = AudienceContext(age=15)
    results = QdrantRecommender.recommend(rec, "The Godfather", top_k=3, audience=teen)
    assert results[0]["global_id"] == "book:g:4" and results[0].get("similar_to") is None
    notice = QdrantRecommender.hidden_title_notice(rec, "The Godfather", None, teen, results)
    assert notice == "“The Godfather” (movie, 18+) is hidden: this profile's age is 15."


# ---------------------------------------------------------------- encoder
@pytest.mark.parametrize("model, expected", [
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
    ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ("BAAI/bge-m3", ""),
])
def test_only_bge_english_models_get_the_query_instruction(model, expected):
    import dataclasses

    from backend.settings import get_settings, query_instruction

    settings = dataclasses.replace(get_settings(), embedding_model=model, embedding_query_instruction=None)
    assert query_instruction(settings) == expected


def test_queries_are_encoded_with_the_instruction_and_items_without(monkeypatch):
    import dataclasses

    from backend.settings import get_settings

    # A word counts as known (and the query is encoded) once two rows use it.
    rows = [_row("movie:q:1", "Inception", categories="dream"), _row("movie:q:2", "Dreamscape", categories="dream")]
    rec = _recommender(rows, {"movie:q:1": [1, 0], "movie:q:2": [1, 0]}, query_vector=[1, 0])
    rec.settings = dataclasses.replace(get_settings(), embedding_model="BAAI/bge-small-en-v1.5",
                                       embedding_query_instruction=None)
    seen = []
    original = rec.model.encode
    rec.model.encode = lambda texts, **kwargs: seen.extend(texts) or original(texts, **kwargs)
    QdrantRecommender.recommend(rec, "inception dream", top_k=1, audience=AudienceContext(age=30))
    assert seen == ["Represent this sentence for searching relevant passages: inception dream"]


def test_vectors_from_another_model_stop_startup():
    rec = QdrantRecommender.__new__(QdrantRecommender)
    rec.catalog = pd.DataFrame({"global_id": ["a"]})
    rec.embedding_index = pd.DataFrame({"global_id": ["a"], "embedding_model": ["sentence-transformers/all-MiniLM-L6-v2"]})
    rec.embedding_model = "BAAI/bge-small-en-v1.5"
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        QdrantRecommender._validate_artifacts(rec)
    rec.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    QdrantRecommender._validate_artifacts(rec)


# ------------------------------------------------------- synopsis vectors
def test_plot_index_search_and_cosines():
    index = PlotIndex(np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32), np.array(["a", "b", "c"], dtype=object))
    query = np.array([1, 0], dtype=np.float32)
    assert [gid for gid, _ in index.search(query, 2)] == ["a", "c"]
    assert [gid for gid, _ in index.search(query, 2, allowed=np.array([False, True, True]))] == ["c", "b"]
    assert index.cosines(np.array([0, 1], dtype=np.float32), ["b", "zzz"]) == {"b": pytest.approx(1.0)}


def test_a_synopsis_that_reads_like_the_query_lifts_the_film():
    rows = [
        _row("movie:g:1", "Every Day", popularity=300.0, categories="Romance"),
        _row("movie:g:2", "Groundhog Day", popularity=300.0, categories="Comedy, deja vu",
             description="A weatherman finds himself repeating the same day over and over."),
    ]
    # The full-text vector (title, genres, keywords, cast) favours the title echo...
    vectors = {"movie:g:1": [1, 0.5, 0], "movie:g:2": [0.2, 1, 0]}
    # ...the synopsis alone says what the query says.
    plot = PlotIndex(np.array([[1, 0, 0]], dtype=np.float32), np.array(["movie:g:2"], dtype=object))
    rec = _recommender(rows, vectors, query_vector=[1, 0, 0], plot=plot)
    results = QdrantRecommender.recommend(rec, "man relives the same day over and over", top_k=2,
                                          audience=AudienceContext(age=30))
    assert results[0]["title"] == "Groundhog Day"
    assert results[0]["semantic_score"] == pytest.approx(1.0)
