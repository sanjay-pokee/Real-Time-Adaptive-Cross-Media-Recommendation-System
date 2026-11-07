import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.qdrant_recommender import QdrantRecommender


def _row(n, title, creators="", description="", categories="Drama"):
    return {
        "global_id": f"movie:test:{n}",
        "content_type": "movie",
        "source": "test",
        "source_id": str(n),
        "title": title,
        "description": description,
        "creators": creators,
        "categories": categories,
        "release_date": "2026",
        "popularity": 10,
        "rating": 7,
    }


@pytest.fixture
def recommender():
    rows = [
        _row(1, "Interstellar", "Christopher Nolan", "Explorers travel through a wormhole in space."),
        _row(2, "Harry Potter and the Philosopher's Stone", "J. K. Rowling", "A boy learns he is a wizard."),
        _row(3, "The Shawshank Redemption", "Frank Darabont", "Two prisoners bond over years."),
        _row(4, "The Dark Knight", "Christopher Nolan", "Batman faces the Joker."),
        _row(5, "Blxt", "Some Band", "A song."),
        _row(6, "Una película de terror", "", "Una película de terror con fantasmas."),
    ]
    # Commoner neighbours of every typo above, so frequency alone would pick
    # the wrong word and the tests show what else decides.
    filler = [
        ("hard", "hard shell case"), ("power", "power drill"), ("relieve", "relieve sore muscles"),
        ("knit", "knit sweater"), ("blot", "blot the stain"),
    ]
    n = 100
    for _, text in filler:
        for _ in range(15):
            n += 1
            rows.append(_row(n, f"Product {n}", "Acme", text, categories="Home"))
    for _ in range(3):
        n += 1
        rows.append(_row(n, f"Parcel {n}", "Acme", "receive it tomorrow", categories="Home"))

    rec = QdrantRecommender.__new__(QdrantRecommender)
    rec.catalog = pd.DataFrame(rows)
    rec.graph_store = None
    rec.ema_store = None
    return rec


def test_a_misspelt_title_word_is_corrected(recommender):
    assert recommender.correct_query("intersteller") == "interstellar"
    assert recommender.correct_query("The Shawshenk Redemtion") == "the shawshank redemption"


def test_typos_resolve_to_the_title_they_form_together(recommender):
    # "knit" is fifteen times commoner than "knight", and both are one edit
    # from "knigt"; only "dark knight" is a phrase some title contains.
    assert recommender.correct_query("the dark knigt") == "the dark knight"


def test_doubled_and_swapped_letters_beat_a_commoner_word(recommender):
    assert recommender.correct_query("hary poter") == "harry potter"
    assert recommender.correct_query("recieve") == "receive"


def test_known_words_are_left_alone(recommender):
    assert recommender.correct_query("christopher nolan") is None
    assert recommender.correct_query("wizard in space") is None


def test_a_rare_word_that_is_a_title_is_not_corrected(recommender):
    # "blot" is common and one edit away, but "Blxt" is a song and the query
    # finds it exactly as typed.
    assert recommender.correct_query("blxt") is None


def test_keyboard_mash_skips_the_vector_search(recommender):
    def no_vector_search(*args, **kwargs):
        raise AssertionError("vector search ran for a query with no real word in it")

    recommender._search_vector = no_vector_search

    assert QdrantRecommender.recommend(recommender, "qzxv blorf wumple") == []
    # A function word does not vouch for the rest of the query.
    assert QdrantRecommender.recommend(recommender, "the qzxv") == []


def test_a_query_with_a_real_word_still_reaches_the_vector_search(recommender):
    calls = []

    class Model:
        def encode(self, texts, **kwargs):
            import numpy as np

            return np.ones((len(texts), 4), dtype=np.float32)

    recommender.model = Model()
    recommender._search_vector = lambda vector, top_k, content_type, audience=None: calls.append(top_k) or []

    # "nolan" is in two rows. One would not do: listings carry typos too, so
    # a word seen once is not yet taken as real.
    QdrantRecommender.recommend(recommender, "nolan qzxv")

    assert calls


def test_accented_descriptions_index_as_whole_words(recommender):
    # NFKD splits "í" into "i" and a combining mark; the mark used to become a
    # space, indexing "película" as "peli cula".
    assert recommender._fulltext_haystack().str.contains(" pelicula de terror ", regex=False).any()
