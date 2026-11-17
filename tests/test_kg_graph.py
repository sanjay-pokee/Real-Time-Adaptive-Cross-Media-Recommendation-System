"""The typed knowledge graph: construction, entity linking, path scoring, explanations."""

import numpy as np
import pandas as pd
import pytest

from backend.kg_graph import ENTITY_TYPES, TypedKnowledgeGraph
from backend.knowledge_graph import CatalogKnowledgeGraph


def _catalog() -> pd.DataFrame:
    rows = [
        ("movie:1", "movie", "tmdb_api", "Inception", "Christopher Nolan, Leonardo DiCaprio, Elliot Page",
         "Action, Science Fiction, dream, heist", "entertainment"),
        ("movie:2", "movie", "tmdb_api", "Interstellar", "Christopher Nolan, Matthew McConaughey",
         "Adventure, Science Fiction, space, wormhole", "entertainment"),
        ("movie:3", "movie", "tmdb_api", "Titanic", "James Cameron, Leonardo DiCaprio",
         "Drama, Romance, ship", "entertainment"),
        ("movie:4", "movie", "tmdb_api", "Toy Story", "John Lasseter, Tom Hanks",
         "Animation, Family, Toy Story Collection, toy", "entertainment"),
        ("movie:5", "movie", "tmdb_api", "Toy Story 2", "John Lasseter, Tom Hanks",
         "Animation, Family, Toy Story Collection, toy", "entertainment"),
        ("book:1", "book", "openlibrary_api", "Mistborn", "Brandon Sanderson", "Fantasy, Magic", "entertainment"),
        ("music:1", "music", "spotify_songs", "Shape of You", "Ed Sheeran", "pop, dance pop", "entertainment"),
        ("health:1", "health", "amazon_reviews_2023", "Lighter", "MEETOO", "Health & Household, Lighters", "health"),
    ]
    return pd.DataFrame(rows, columns=["global_id", "content_type", "source", "title", "creators", "categories", "domain"])


@pytest.fixture(scope="module")
def graph() -> TypedKnowledgeGraph:
    return TypedKnowledgeGraph.from_catalog(_catalog())


def _entity(graph, name):
    hits = [i for i, n in enumerate(graph.entity_names) if n == name]
    assert hits, name
    return hits[0]


def test_edges_are_typed_by_content_type_and_source(graph):
    stats = graph.stats()
    assert stats["items"] == 8
    assert stats["edges"]["DIRECTED_BY"] == 5          # first credit of each movie
    assert stats["edges"]["STARS"] == 6                # the remaining movie credits
    assert stats["edges"]["WRITTEN_BY"] == 1
    assert stats["edges"]["PERFORMED_BY"] == 1
    assert stats["edges"]["MADE_BY"] == 1              # Amazon creator is a brand
    assert stats["edges"]["PART_OF"] == 2              # "... Collection" is a franchise
    assert stats["edges"]["TAGGED"] == 7               # lower-case TMDb keywords on movies
    assert ENTITY_TYPES[graph.entity_types[_entity(graph, "pop")]] == "genre"   # music tags are genres
    assert ENTITY_TYPES[graph.entity_types[_entity(graph, "MEETOO")]] == "brand"


def test_query_links_the_longest_entity_name(graph):
    linked = graph.link_query("films by christopher nolan")
    assert [graph.entity_names[e] for e in linked] == ["Christopher Nolan"]


def test_query_entity_scores_connected_items(graph):
    scores = graph.score("christopher nolan", ["movie:1", "movie:2", "movie:3"])
    assert scores["movie:1"]["kg_score"] == pytest.approx(1.0)
    assert scores["movie:2"]["kg_score"] == pytest.approx(1.0)
    assert scores["movie:3"]["kg_score"] == 0.0
    assert scores["movie:1"]["kg_explanation"] == "Directed by Christopher Nolan (matches your search)"


def test_liked_item_reaches_neighbours_through_two_hop_paths(graph):
    scores = graph.score("", ["movie:2", "movie:3", "book:1"], seed_ids=["movie:1"])
    # Interstellar shares Nolan and Science Fiction with Inception; Titanic only DiCaprio; Mistborn nothing.
    assert scores["movie:2"]["kg_score"] > scores["movie:3"]["kg_score"] > scores["book:1"]["kg_score"] == 0.0
    path = scores["movie:2"]["kg_path"]
    assert [step.get("edge") for step in path[1::2]] == ["DIRECTED_BY", "DIRECTED_BY"]
    assert path[2]["node"] == "Christopher Nolan" and path[-1]["node"] == "Inception"
    assert scores["movie:2"]["kg_explanation"] == "Directed by Christopher Nolan, like Inception, which you liked"


def test_franchise_paths_beat_generic_genres(graph):
    scores = graph.score("", ["movie:5", "movie:3"], seed_ids=["movie:4"])
    assert scores["movie:5"]["kg_score"] > 0.5 > scores["movie:3"]["kg_score"]


def test_structure_nodes_are_in_the_graph_but_not_scored(graph):
    stats = graph.stats()
    assert stats["entities"]["content_type"] == 4 and stats["entities"]["domain"] == 2
    # Sharing only "movie" / "entertainment" must not count as a path.
    assert graph.score("", ["movie:3"], seed_ids=["book:1"])["movie:3"]["kg_score"] == 0.0


def test_unknown_items_and_empty_catalogue_fall_back_to_term_overlap():
    kg = CatalogKnowledgeGraph(pd.DataFrame())
    [item] = kg.rerank("space adventure", [{"global_id": "x", "title": "Space Adventure", "score": 1.0}])
    assert item["kg_score"] > 0 and "kg_explanation" not in item


def test_rerank_attaches_the_path_and_explanation():
    kg = CatalogKnowledgeGraph(_catalog())
    results = [{"global_id": "movie:2", "title": "Interstellar", "score": 1.0},
               {"global_id": "movie:3", "title": "Titanic", "score": 1.0}]
    reranked = kg.rerank("", results, seed_ids=["movie:1"])
    assert reranked[0]["global_id"] == "movie:2"
    assert reranked[0]["kg_explanation"].startswith("Directed by Christopher Nolan")


def test_export_rows_cover_every_node_and_edge(graph):
    nodes, edges = graph.export_rows()
    assert len(nodes) == graph.stats()["nodes"]
    assert len(edges) == graph.stats()["edge_total"]
    assert set(edges["type"]) >= {"DIRECTED_BY", "STARS", "PART_OF", "IN_GENRE"}


def test_entity_candidates_reach_every_item_carrying_a_named_entity(graph):
    found = dict(graph.entity_candidates("films by christopher nolan", limit=10))
    assert set(found) == {"movie:1", "movie:2"}
    assert all(share == pytest.approx(1.0) for share in found.values())


def test_entity_candidates_break_ties_with_the_bonus(graph):
    bonus = np.zeros(len(graph.item_ids))
    bonus[graph.item_index["movie:2"]] = 0.5
    assert [gid for gid, _ in graph.entity_candidates("christopher nolan", limit=2, bonus=bonus)] == ["movie:2", "movie:1"]


def test_similar_items_follow_shared_entities(graph):
    ranked = [gid for gid, _ in graph.similar_items("movie:4", limit=3)]
    assert ranked[0] == "movie:5"                       # same franchise, director and cast
    assert "movie:4" not in ranked


def test_query_share_matches_score_without_explanations(graph):
    shares = graph.query_share("christopher nolan", ["movie:1", "movie:3"])
    assert shares == {"movie:1": pytest.approx(1.0)}


def test_everyday_words_reach_formal_genres():
    catalog = pd.DataFrame(
        [("movie:a", "movie", "tmdb_api", "A", "Director A", "Science Fiction", "entertainment"),
         ("movie:b", "movie", "tmdb_api", "B", "Director B", "Romance, Comedy", "entertainment"),
         ("movie:c", "movie", "tmdb_api", "C", "Director C", "Drama", "entertainment")],
        columns=["global_id", "content_type", "source", "title", "creators", "categories", "domain"],
    )
    g = TypedKnowledgeGraph.from_catalog(catalog)
    assert [g.entity_names[e] for e in g.link_query("a sci-fi film")] == ["Science Fiction"]
    assert sorted(g.entity_names[e] for e in g.link_query("rom com")) == ["Comedy", "Romance"]


def test_plural_query_words_reach_singular_keywords(graph):
    # TMDb tags are singular; "old man flies his house with balloons" never reached Up ("balloon").
    assert [graph.entity_names[e] for e in graph.link_query("films about toys")] == ["toy"]
    assert [graph.entity_names[e] for e in graph.link_query("dreams")] == ["dream"]
    assert graph.link_query("glass") == []          # "-ss" is not a plural


def test_entity_candidates_respect_an_allowed_mask(graph):
    allowed = np.zeros(len(graph.item_ids), dtype=bool)
    allowed[graph.item_index["movie:2"]] = True
    assert [gid for gid, _ in graph.entity_candidates("christopher nolan", limit=5, allowed=allowed)] == ["movie:2"]


def test_a_franchise_is_found_without_the_word_collection(graph):
    assert [graph.entity_names[e] for e in graph.link_query("toy story")] == ["Toy Story Collection"]


def test_a_chosen_seed_is_explained_without_claiming_a_like(graph):
    hit = graph.score("", ["movie:2"], seed_ids=["movie:1"], seed_role="selected")["movie:2"]
    assert hit["kg_explanation"] == "Directed by Christopher Nolan, like Inception"
    assert hit["kg_path"][-1].get("selected") is True and "liked" not in hit["kg_path"][-1]
