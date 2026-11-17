"""Qdrant-backed semantic recommendation search.

Qdrant is the production vector backend. Build the collection with:
    python -m embeddings.build_qdrant_collection
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.audience import (
    AudienceContext,
    build_eligibility_filter,
    explain_ineligibility,
    is_eligible,
    item_min_age,
    item_risk_tier,
)
from backend.domains import normalize_content_type
from backend.ema_recommender import EMAEmbeddingStore
from backend.graph_recommender import GraphEmbeddingStore
from backend.kg_graph import COVERAGE_IGNORE
from backend.knowledge_graph import CatalogKnowledgeGraph
from backend.mysql_store import MySQLStore
from backend.plot_index import PlotIndex
from backend.prominence import compute_prominence
from backend.query_vocabulary import QueryVocabulary
from backend.settings import Settings, get_settings, query_instruction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "embeddings" / "content_embedding_index.csv"
EMBEDDING_MODEL = get_settings().embedding_model

RESULT_COLUMNS = [
    "global_id",
    "content_type",
    "source",
    "source_id",
    "title",
    "description",
    "creators",
    "categories",
    "release_date",
    "popularity",
    "rating",
    # Cover art where the source dataset carries one. Empty for movies and
    # music, whose datasets ship no artwork column.
    "image_url",
    # The wide 16:9 still, for the detail view's banner. Movies only.
    "backdrop_url",
    # Audience metadata: carried through so a client can show why an item
    # qualified, and so the constraint evaluation can re-check the filter.
    "domain",
    "maturity",
    "audience_min_age",
    "risk_tier",
]


# The "pick a few things you like" step offers one item per (type, genre). A
# spread of tastes teaches the profile more than the most popular items would:
# by TMDb popularity those are mostly this year's releases. Genres whose top
# item is mislabelled are left out - Spotify files a Brazilian funk remix
# under "edm", and the top "biography" is a German edition.
STARTER_GENRES = {
    "movie": ["action", "comedy", "horror", "animation", "science fiction", "romance", "thriller", "documentary"],
    "show": ["drama", "comedy", "crime", "anime", "sci-fi & fantasy"],
    "book": ["fantasy", "romance", "mystery", "science fiction", "self-help"],
    "music": ["pop", "rap", "rock", "r&b"],
}
# Ratings are on a different scale per source - TMDb 0-10, books 1-5 - and
# music has none, so each type gets its own floor or no floor at all.
STARTER_MIN_RATING = {"movie": 6.5, "show": 7.0, "book": 3.5}
# Where popularity is ranked together with rating. TMDb popularity is
# "trending now", so alone it offers this year's releases over Jaws and The
# Dark Knight; book ratings come from a handful of votes, so blending them in
# surfaces obscure titles instead.
STARTER_BLEND_RATING = {"movie", "show"}

# Knowledge-graph candidates added to the vector pool per request.
ENTITY_CANDIDATES = 150

# Synopsis-only neighbours added to the pool per request (backend/plot_index.py).
PLOT_CANDIDATES = 150

# Most a user's category and content-type affinity may move an item, either way.
PREFERENCE_CAP = 0.15

# A one-word query used in more catalogue rows than this is a common word, not a name.
COMMON_WORD_ROWS = 500

# For a one-word creator query, what matching a whole name ("NOLAN") is worth over
# matching one word of a name ("Christopher Nolan"), in prominence units.
WHOLE_NAME_BONUS = 0.15

# What "more like this" may return when no content type is asked for. Films and
# shows answer each other and books are often their source; music and products
# stay within their own type.
MEDIUM_FAMILY = {
    "movie": {"movie", "show", "book"},
    "show": {"show", "movie", "book"},
    "book": {"book", "movie", "show"},
    "music": {"music"},
    "health": {"health"},
    "industrial": {"industrial"},
    "finance": {"finance"},
}

# A word in the query that names a kind of item. One such word ("books about
# AI") restricts the search to that type; two different ones leave it open.
TYPE_WORDS = {
    "movie": "movie", "movies": "movie", "film": "movie", "films": "movie",
    "show": "show", "shows": "show", "series": "show", "tv": "show",
    "book": "book", "books": "book", "novel": "book", "novels": "book",
    "song": "music", "songs": "music", "music": "music", "track": "music", "tracks": "music",
}
# Words that may stand before "like" in "something like X" without changing it.
LIKE_FILLERS = {"something", "anything", "more", "stuff", "things", "other", "some", "any", "good", "great", "best"}
# Words that ask for quality or recency, not a topic: "best 90s movies" is a browse.
BROWSE_WORDS = {"the", "top", "popular", "famous", "classic", "classics", "a", "an",
                "old", "new", "released", "made", "hit", "hits"}
_LIKE_PATTERN = re.compile(r"^(?:(?P<before>.*?)\s+)?(?:like|similar to)\s+(?P<target>.+)$")


def parse_query_intent(query: str) -> tuple[str | None, str | None]:
    """(content type the query asks for, title it asks for more of), either may be None.

    "books about artificial intelligence" -> ("book", None)
    "movies like inception"               -> ("movie", "inception")
    "similar to the dark knight"          -> (None, "the dark knight")
    "i like dogs"                         -> (None, None): "i" is not a type word.
    """
    text = _tokenize_lookup_text(query)
    words = text.split()
    types = {TYPE_WORDS[word] for word in words if word in TYPE_WORDS}
    intent = next(iter(types)) if len(types) == 1 else None

    target = None
    match = _LIKE_PATTERN.match(text)
    if match:
        before = (match.group("before") or "").split()
        if all(word in TYPE_WORDS or word in LIKE_FILLERS for word in before):
            target = match.group("target").strip() or None
    return intent, target


# Release-year phrases. A decade or a year with a preposition is unambiguous;
# a bare year before a type word ("1984 book") may be a title and is checked
# against the catalogue first.
_PREPOSITION = r"(?:(?:from|in|of|during)\s+)?(?:the\s+)?"
_DECADE_NUMBER = re.compile(rf"\b{_PREPOSITION}(?P<year>(?:19|20)?\d0)\s*['’]?\s*s\b")
_DECADE_WORD = re.compile(
    rf"\b{_PREPOSITION}(?P<word>twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b"
)
DECADE_WORDS = {"twenties": 1920, "thirties": 1930, "forties": 1940, "fifties": 1950,
                "sixties": 1960, "seventies": 1970, "eighties": 1980, "nineties": 1990}
_YEAR = r"(?P<year>19\d{2}|20[0-2]\d)"
_YEAR_AFTER = re.compile(rf"\b(?P<op>since|after)\s+{_YEAR}\b")
_YEAR_BEFORE = re.compile(rf"\bbefore\s+{_YEAR}\b")
_YEAR_IN = re.compile(rf"\b(?:from|in|of|released in|made in)\s+{_YEAR}\b")
_YEAR_TYPED = re.compile(rf"\b{_YEAR}\s+(?=(?:{'|'.join(sorted(TYPE_WORDS, key=len, reverse=True))})\b)")
LATEST_YEAR = 2100


def parse_year_range(query: str) -> tuple[tuple[int, int] | None, str, bool]:
    """(first year, last year) the query asks for, the query without that phrase, and
    whether the phrase was a bare year that might be a title instead.

    "90s movies"               -> ((1990, 1999), "movies", False)
    "sci-fi from the 1980s"    -> ((1980, 1989), "sci-fi", False)
    "songs from 2010"          -> ((2010, 2010), "songs", False)
    "1984 book"                -> ((1984, 1984), "book", True)  - may be Orwell's 1984
    "2012"                     -> (None, "2012", False)         - a bare year is a title
    """
    text = str(query or "")
    lowered = text.lower()

    def cut(match: re.Match) -> str:
        return " ".join((lowered[: match.start()] + " " + lowered[match.end():]).split())

    match = _DECADE_NUMBER.search(lowered)
    if match:
        digits = match.group("year")
        start = int(digits) if len(digits) == 4 else (2000 + int(digits) if int(digits) <= 20 else 1900 + int(digits))
        return (start, start + 9), cut(match), False
    match = _DECADE_WORD.search(lowered)
    if match:
        start = DECADE_WORDS[match.group("word")]
        return (start, start + 9), cut(match), False
    match = _YEAR_AFTER.search(lowered)
    if match:
        year = int(match.group("year"))
        return (year + (match.group("op") == "after"), LATEST_YEAR), cut(match), False
    match = _YEAR_BEFORE.search(lowered)
    if match:
        return (0, int(match.group("year")) - 1), cut(match), False
    match = _YEAR_IN.search(lowered)
    if match:
        year = int(match.group("year"))
        return (year, year), cut(match), False
    match = _YEAR_TYPED.search(lowered)
    if match:
        year = int(match.group("year"))
        return (year, year), cut(match), True
    return None, text, False


_SETTING_CUE = re.compile(r"\b(?:set|takes? place|taking place|growing up|grew up|living|lives|life)\s+(?:in|during)\b")
# A query with more topic words than this around a decade is describing a story.
SETTING_DESCRIPTION_WORDS = 4


def is_period_setting(query: str, without_years: str) -> bool:
    """Whether a decade or year in the query is when a story is set, not when it came out.

    "a movie set in the 80s" says so; "group of kids in the 80s search for their
    missing friend" is a plot description, and a period inside one is its setting.
    "sci-fi movies from the 80s" and "90s movies" are release filters.
    """
    if _SETTING_CUE.search(str(query or "").lower()):
        return True
    topic = [
        word for word in _tokenize_lookup_text(without_years).split()
        if word not in COVERAGE_IGNORE and word not in LIKE_FILLERS and word not in BROWSE_WORDS and len(word) > 1
    ]
    return len(topic) > SETTING_DESCRIPTION_WORDS


# Words the sentence encoder does not know, rewritten before encoding. "rom com"
# embedded close to Latin pop songs and budgeting software; "romantic comedy"
# lands on The Proposal and Love Actually.
EMBEDDING_REWRITES = (
    (re.compile(r"\brom[\s-]?coms?\b", re.IGNORECASE), "romantic comedy"),
    (re.compile(r"\bsci[\s-]?fi\b", re.IGNORECASE), "science fiction"),
    (re.compile(r"\bzom[\s-]?coms?\b", re.IGNORECASE), "zombie comedy"),
    (re.compile(r"\bdocos?\b", re.IGNORECASE), "documentary"),
    (re.compile(r"\banimes?\b", re.IGNORECASE), "anime japanese animation"),
)


def embedding_text(query: str) -> str:
    """The query as the sentence encoder should see it."""
    text = str(query or "")
    for pattern, replacement in EMBEDDING_REWRITES:
        text = pattern.sub(replacement, text)
    return text


def _is_product_brand(item: dict) -> bool:
    """A creator hit on an Amazon listing names a brand, not a person."""
    return str(item.get("source", "")).startswith("amazon")


def _released_in(item: Any, years: tuple[int, int]) -> bool:
    match = re.match(r"\s*(\d{4})", str(item.get("release_date", "") or ""))
    return bool(match) and years[0] <= int(match.group(1)) <= years[1]


def _same_work_key(item: Any) -> tuple[str, str, str, str]:
    """(type, title, first creator, year): two rows with the same key are copies of one work.

    Music ignores the year, since a single and its album release differ by
    months; a director's remake under the same title does not.
    """
    creators = _split_creators(item.get("creators", ""))
    content_type = str(item.get("content_type", ""))
    year = "" if content_type == "music" else str(item.get("release_date", "") or "")[:4]
    return (
        content_type,
        _tokenize_lookup_text(item.get("title", "")),
        _tokenize_lookup_text(creators[0]) if creators else "",
        year,
    )


def drop_duplicate_works(results: list[dict[str, Any]], seed: Any = None) -> list[dict[str, Any]]:
    """Keep the first of several copies of one work, and none of the seed's copies.

    Spotify lists a song once per release: "Shape of You" by Ed Sheeran is in
    the catalogue twice, so "more like Shape of You" returned Shape of You.
    A cover by another artist, or a film and its book, is a different work.
    """
    seen = {_same_work_key(seed)} if seed is not None else set()
    kept = []
    for item in results:
        key = _same_work_key(item)
        if not key[1] or key not in seen:
            kept.append(item)
            seen.add(key)
    return kept


@dataclass
class QdrantRecommender:
    catalog_path: Path = CATALOG_PATH
    embedding_index_path: Path = EMBEDDING_INDEX_PATH
    embedding_model: str = EMBEDDING_MODEL
    settings: Settings = get_settings()

    def __post_init__(self) -> None:
        self._assert_artifacts_exist()
        # low_memory=False is required, not cosmetic. The default reads the file in
        # chunks and types each chunk independently, so a run of rows whose
        # release_date is a bare year - the finance books are contiguous - comes
        # back as float64 and every "2010" becomes 2010.0. That reached the Qdrant
        # payload and 500'd every finance query on response validation.
        self.catalog = pd.read_csv(self.catalog_path, low_memory=False)
        self.embedding_index = pd.read_csv(self.embedding_index_path)
        self.client = self._load_qdrant_client()
        self.model = self._load_model()
        self.graph_store = self._load_graph_store()
        self.ema_store = self._load_ema_store()
        self.knowledge_graph = CatalogKnowledgeGraph(self.catalog)
        self._validate_artifacts()

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        content_type: str | None = None,
        user_id: str | None = None,
        audience: AudienceContext | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # "90s movies", "sci-fi from the 1980s": the years filter the results
        # and leave the query. Not when the query is a title ("That '70s Show")
        # or the bare year is one ("1984 book" is Orwell).
        years, without_years, bare_year = parse_year_range(query)
        if years is not None and is_period_setting(query, without_years):
            # "kids in the 80s search for their missing friend" is Stranger
            # Things (2016): the decade is when the story happens. Searched as
            # a release filter, the show could never be found.
            years = None
        if years is not None:
            year_type = parse_query_intent(without_years)[0]
            if self._has_exact_title(query):
                years = None
            elif bare_year and self._has_exact_title(str(years[0]), content_type or year_type):
                # "1984 book" names Orwell's novel: search that title, among books.
                query, content_type, years = str(years[0]), content_type or year_type, None
            else:
                query = without_years
        intent_type, like_target = parse_query_intent(query)

        # "movies like Inception" asks for what resembles Inception, not for
        # titles containing the word. Searched as text it returned Inception
        # itself, then Blow-Up, Waist Deep and Attack of the Killer Tomatoes;
        # answered through the item-similarity path it returns Tenet,
        # Interstellar, Memento and The Prestige.
        if like_target:
            seed = self._find_title(like_target, preferred_type=content_type or intent_type)
            if seed is not None:
                results = self.recommend_from_item(
                    seed["global_id"],
                    top_k=top_k,
                    content_type=content_type or intent_type,
                    user_id=user_id,
                    audience=audience,
                    years=years,
                )
                for item in results:
                    item["similar_to"] = seed["title"]
                return results
            # No such title: "movies like dream manipulation" describes what
            # is wanted. Search that, in the type asked for; searching the whole
            # sentence served any popular film for "movies like asdfqwer".
            query = like_target
        else:
            # The query names a title this viewer may not see ("The Godfather"
            # is 18+; a profile with no age stops at 13+). Searched as text, a
            # bare title carries no topic, so the page filled with popular
            # PG-13 films the profile liked: Inception, Michael, Batman Begins.
            # What the viewer can see that is most like the title answers better,
            # and the API says why the title itself is missing.
            hidden, visible = self.hidden_titles(query, content_type, audience)
            if hidden and not visible:
                results = self.recommend_from_item(
                    str(hidden[0]["global_id"]),
                    top_k=top_k,
                    content_type=content_type,
                    user_id=user_id,
                    audience=audience,
                    years=years,
                )
                for item in results:
                    item["similar_to"] = str(hidden[0]["title"])
                return results

        if years is not None and not self._content_words(query):
            # "90s movies" names no topic: the decade's most established titles.
            results = self._browse(content_type or intent_type, audience, years, max(top_k * 3, top_k))
            results = self._personalize_results(results, len(results), user_id)
            results = self._graph_rerank(results, user_id)
            results = self._ema_rerank(results, user_id)
            results = self._knowledge_graph_rerank(query, results, user_id)
            return drop_duplicate_works(results)[:top_k]

        # Retrieval is hybrid. Names and titles are found directly - a pure
        # vector search cannot find a title: "F1" embeds to a vector with
        # essentially no intent in it, so its nearest neighbours were a Punjabi
        # song and an aromatherapy necklace while the film sat in the catalogue.
        # A query that is a name or a title is navigational, so those hits sit
        # above everything else; topical matches compete on relevance below.
        person_results = self._search_person_matches(
            query,
            top_k=min(5, top_k),
            content_type=content_type,
            audience=audience,
        )
        lexical_results = self._search_lexical_matches(
            query,
            top_k=max(3, top_k // 2),
            content_type=content_type,
            audience=audience,
        )
        # A creator hit and a title hit can be the same row; the creator match is
        # the more specific claim, so it wins.
        person_ids = {item["global_id"] for item in person_results}
        lexical_results = [
            item for item in lexical_results if item["global_id"] not in person_ids
        ]

        names_a_title = any(int(item.get("match_tier", 9)) <= 1 for item in lexical_results)
        if names_a_title:
            # The query is a title. "interstellar" is a film; an electronics brand
            # of that name is not what was asked for, and neither is a stranger
            # whose name merely contains "batman".
            person_results = [
                item
                for item in person_results
                if not _is_product_brand(item) and int(item.get("match_tier", 0)) == 0
            ]
        query_words = _tokenize_lookup_text(query).split()
        if len(query_words) == 1 and self._query_vocabulary().rows.get(query_words[0], 0) > COMMON_WORD_ROWS:
            # A single common word is not a name. "love" is in 6,670 rows, so
            # "Love and Rockets" and "Love Harder" matched it as creators and
            # took the page; "nolan" is in 14 and still finds Christopher Nolan.
            person_results = [item for item in person_results if int(item.get("match_tier", 0)) == 0]
        if content_type is None and intent_type and not names_a_title:
            # "books about artificial intelligence" wants books. A query that is
            # itself a title ("Scary Movie", "The Book Thief") keeps every type.
            content_type = intent_type
            person_results = [item for item in person_results if item.get("content_type") == intent_type]
            lexical_results = [item for item in lexical_results if item.get("content_type") == intent_type]
        if years is not None:
            person_results = [item for item in person_results if _released_in(item, years)]
            lexical_results = [item for item in lexical_results if _released_in(item, years)]

        weights = self._weights()
        prominence = self._prominence_by_id()
        # Only an exact title is navigational. A title that merely starts with or
        # contains the query competes on relevance: "time travel" put the Cars
        # short "Time Travel Mater" first, ahead of every time-travel film.
        navigational = person_results + [
            item for item in lexical_results if int(item.get("match_tier", 0)) == 0
        ]
        topical = [item for item in lexical_results if int(item.get("match_tier", 0)) >= 1]
        for item in navigational:
            popularity = prominence.get(str(item["global_id"]), 0.0)
            item["match_score"] = float(item["score"])
            item["semantic_score"] = None
            item["popularity_score"] = round(popularity, 6)
            item["score"] = float(item["score"]) + weights["popularity"] * popularity
        navigational.sort(key=lambda item: item["score"], reverse=True)
        nav_ids = {str(item["global_id"]) for item in navigational}

        search_k = max(top_k * 3, top_k + len(nav_ids))
        pool: list[dict[str, Any]] = []
        query_vector = None
        if self._query_vocabulary().has_known_word(_tokenize_lookup_text(query).split()):
            query_vector = self.model.encode(
                [self._query_instruction() + embedding_text(query)],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)[0]
            # A wide pool, not just the page: Inception sits 3,151st by raw
            # cosine for "mind bending sci-fi thriller", behind titles whose
            # names echo the words. Only a pool this deep lets the reranker see it.
            pool_size = max(weights["pool"], search_k)
            if years is None:
                pool = self._search_vector(query_vector, pool_size, content_type, audience)
            else:
                pool = self._search_vector_in_years(query_vector, pool_size, content_type, audience, years)
            pool = [item for item in pool if str(item["global_id"]) not in nav_ids]
        # else: not one word of the query appears anywhere in the catalogue, so
        # its nearest vectors are arbitrary and would be served as if they
        # answered it - see backend/query_vocabulary.py. Creator and title
        # matches found above still stand.

        if query_vector is not None:
            seen = nav_ids | {str(item["global_id"]) for item in pool}
            extra = [item for item in topical if str(item["global_id"]) not in seen]
            seen |= {str(item["global_id"]) for item in extra}
            # Items carrying an entity the query names ("time travel", "heist",
            # "zombie apocalypse") however their synopsis is worded.
            extra += [
                item
                for item in self._entity_candidate_items(
                    query, content_type, audience, weights["entity_pool"], years=years
                )
                if str(item["global_id"]) not in seen
            ]
            seen |= {str(item["global_id"]) for item in extra}
            # Films, shows and books whose synopsis alone reads like the query
            # ("man relives the same day over and over" -> Groundhog Day).
            extra += [
                item
                for item in self._plot_candidate_items(query_vector, content_type, audience, weights["plot_pool"], years)
                if str(item["global_id"]) not in seen
            ]
            cosines = self._cosines(query_vector, [str(item["global_id"]) for item in extra])
            for item in extra:
                item["score"] = cosines.get(str(item["global_id"]), 0.0)
            pool += extra
            # Relevance is the closer of the two views of an item: the full
            # text, or the synopsis alone where there is one.
            plot = self._plot_index()
            if plot is not None:
                synopsis = plot.cosines(query_vector, [str(item["global_id"]) for item in pool])
                for item in pool:
                    plot_cosine = synopsis.get(str(item["global_id"]))
                    if plot_cosine is not None and plot_cosine > float(item.get("score", 0.0) or 0.0):
                        item["score"] = plot_cosine
        else:
            pool = [item for item in topical if str(item["global_id"]) not in nav_ids]
            for item in pool:
                item["score"] = 0.0

        graph = self._knowledge_graph().graph
        shares = graph.query_share(query, [str(item["global_id"]) for item in pool])
        ranked = self._rank_candidates(
            pool,
            keep=search_k,
            prominence=prominence,
            extra={gid: weights["kg"] * share for gid, share in shares.items()},
        )
        results = navigational + ranked
        results = self._personalize_results(results, search_k, user_id)
        results = self._graph_rerank(results, user_id)
        results = self._ema_rerank(results, user_id)
        results = self._knowledge_graph_rerank(query, results, user_id)
        return drop_duplicate_works(results)[:top_k]

    def recommend_from_item(
        self,
        global_id: str,
        top_k: int = 10,
        content_type: str | None = None,
        user_id: str | None = None,
        audience: AudienceContext | None = None,
        years: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        if not global_id.strip():
            raise ValueError("global_id cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        catalog_matches = self.catalog[self.catalog["global_id"] == global_id]
        if catalog_matches.empty:
            raise ValueError(f"Unknown global_id: {global_id}")
        seed = catalog_matches.iloc[0]

        weights = self._weights()
        prominence = self._prominence_by_id()
        vector = self._item_vector(global_id, str(seed.get("embedding_text", "")))
        search_k = max(top_k * 3, top_k + 1)

        # Two candidate sources. Text neighbours alone are dominated by the
        # title: "Titanic" returned the Titanic miniseries, Titanic II and
        # Titanic 666. Knowledge-graph neighbours share the director, cast,
        # franchise and keywords, which is what "more like this" means.
        pool = self._search_vector(vector, max(weights["pool"], search_k), content_type, audience)
        pool = [item for item in pool if item["global_id"] != global_id]
        seen = {global_id} | {str(item["global_id"]) for item in pool}
        neighbours = [
            item
            for item in self._similar_candidate_items(global_id, content_type, audience, weights["entity_pool"])
            if str(item["global_id"]) not in seen
        ]
        cosines = self._cosines(vector, [str(item["global_id"]) for item in neighbours])
        for item in neighbours:
            item["score"] = cosines.get(str(item["global_id"]), 0.0)
        pool += neighbours

        if content_type is None:
            # Films and shows answer each other, and books are often their
            # source; a song or a lighter is not "more like" a film.
            family = MEDIUM_FAMILY.get(str(seed.get("content_type", "")))
            if family:
                pool = [item for item in pool if str(item.get("content_type", "")) in family]
        if years is not None:
            pool = [item for item in pool if _released_in(item, years)]
        # Another copy of the seed (a song's single and album release) is not "more like" it.
        pool = drop_duplicate_works(pool, seed=seed)

        graph = self._knowledge_graph().graph
        paths = graph.score("", [str(item["global_id"]) for item in pool], seed_ids=[global_id], seed_role="selected")
        ranked = self._rank_candidates(
            pool,
            keep=search_k,
            prominence=prominence,
            extra={gid: weights["kg_similar"] * hit["kg_score"] for gid, hit in paths.items()},
        )
        results = self._personalize_results(ranked, search_k, user_id)
        results = self._graph_rerank(results, user_id)
        results = self._ema_rerank(results, user_id)
        results = self._knowledge_graph_rerank(
            "",
            results,
            user_id,
            seed_ids=[global_id],
            seed_role="selected",
            kg_weight=weights["kg_similar"],
        )
        for item in results:
            # The client multiplies kg_score by its default weight; this path weighs it more.
            item["signal_weights"] = {"kg": weights["kg_similar"]}
        return results[:top_k]

    # ------------------------------------------------- titles the viewer can't see
    def eligible_mask(self, audience: AudienceContext) -> np.ndarray:
        """Per catalogue row, whether this viewer may see it: ``is_eligible`` for the whole catalogue at once.

        Autocomplete runs on every keystroke, so the per-row ages and risk tiers
        are worked out once and each request is three array comparisons.
        """
        cached = getattr(self, "_eligibility_columns", None)
        if cached is None or len(cached[0]) != len(self.catalog):
            rows = self.catalog.to_dict("records")
            registry = audience.registry
            cached = (
                np.array([item_min_age(row, registry) for row in rows], dtype=np.int32),
                np.array([item_risk_tier(row, registry) for row in rows], dtype=np.int32),
                self.catalog["content_type"].fillna("").astype(str).str.strip().str.lower().to_numpy(),
            )
            self._eligibility_columns = cached
        min_age, risk_tier, content_types = cached
        mask = (min_age <= audience.effective_age) & (risk_tier <= audience.max_risk_tier)
        if audience.domain is not None:
            mask &= np.isin(content_types, list(audience.allowed_content_types))
        return mask

    def hidden_titles(
        self,
        query: str,
        content_type: str | None,
        audience: AudienceContext | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Items titled exactly ``query`` that the audience filter withholds (most established first),
        and whether any item with that exact title is still visible."""
        if audience is None or "title" not in getattr(self, "catalog", pd.DataFrame()).columns:
            return [], True
        tokens = _tokenize_lookup_text(query)
        if not tokens:
            return [], True
        mask = self._title_haystack().eq(f" {tokens} ")
        normalized_type = normalize_content_type(content_type)
        if normalized_type is not None:
            mask &= self.catalog["content_type"] == normalized_type
        if not mask.any():
            return [], True
        items = [
            {column: _clean_value(row.get(column, "")) for column in RESULT_COLUMNS}
            for row in self.catalog[mask].to_dict("records")
        ]
        hidden = [item for item in items if not is_eligible(item, audience)]
        prominence = self._prominence_by_id()
        hidden.sort(key=lambda item: -prominence.get(str(item["global_id"]), 0.0))
        return hidden, len(hidden) < len(items)

    def hidden_title_notice(
        self,
        query: str,
        content_type: str | None,
        audience: AudienceContext | None,
        results: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """One sentence for the UI when the title searched for exists but this viewer may not see it."""
        hidden, _ = self.hidden_titles(query, content_type, audience)
        if not hidden or audience is None:
            return None
        first = hidden[0]
        minimum = item_min_age(first, audience.registry)
        kinds = {str(item.get("content_type", "")) for item in hidden}
        label = str(first.get("content_type", "")) if len(hidden) == 1 else (
            f"{len(hidden)} {next(iter(kinds))}s" if len(kinds) == 1 else f"{len(hidden)} items"
        )
        reason = explain_ineligibility(first, audience) or ""
        if reason.startswith("content type"):
            why = "it is outside the selected domain"
        elif reason.startswith("risk tier"):
            why = "it is above the risk level allowed here"
        elif audience.safe_mode:
            why = f"safe mode is on, so results stop at {audience.effective_age}+"
        elif audience.age is None:
            why = f"no age is set for this profile, so results stop at {audience.effective_age}+. Set an age in Refine to see it"
        else:
            why = f"this profile's age is {audience.age}"
        similar = bool(results) and str(results[0].get("similar_to") or "") == str(first.get("title", ""))
        tail = " Showing similar titles you can see." if similar else ""
        return f"“{first.get('title')}” ({label}, {minimum}+) is hidden: {why}.{tail}"

    # ------------------------------------------------------- ranking helpers
    def _query_instruction(self) -> str:
        """Prefix for a search query (BGE's retrieval instruction); items are encoded without one."""
        settings = getattr(self, "settings", None)
        return query_instruction(settings) if settings is not None else ""

    def _weights(self) -> dict[str, float]:
        settings = getattr(self, "settings", None)
        return {
            "kg": float(getattr(settings, "kg_weight", 0.2)),
            "kg_similar": float(getattr(settings, "kg_similar_weight", 0.8)),
            "popularity": float(getattr(settings, "popularity_weight", 0.3)),
            "pool": int(getattr(settings, "candidate_pool", 300)),
            "entity_pool": ENTITY_CANDIDATES,
            "plot_pool": PLOT_CANDIDATES,
        }

    def _rank_candidates(
        self,
        pool: list[dict[str, Any]],
        keep: int,
        prominence: dict[str, float],
        extra: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Score each candidate cosine + popularity, order by that plus ``extra``, keep the best.

        ``extra`` is the knowledge-graph term the final rerank will add, so the
        pool is cut in the same order the page will be ranked in.
        """
        popularity_weight = self._weights()["popularity"]
        for item in pool:
            cosine = float(item.get("score", 0.0) or 0.0)
            popularity = prominence.get(str(item["global_id"]), 0.0)
            item["semantic_score"] = round(cosine, 6)
            item["popularity_score"] = round(popularity, 6)
            item["score"] = cosine + popularity_weight * popularity
        pool.sort(
            key=lambda item: item["score"] + extra.get(str(item["global_id"]), 0.0),
            reverse=True,
        )
        return pool[:keep]

    def _prominence_array(self) -> np.ndarray:
        """Prominence per catalogue row (see backend/prominence.py), built once."""
        cached = getattr(self, "_prominence_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached
        settings = getattr(self, "settings", None)
        exports = {}
        if settings is not None:
            exports = {
                "movies": Path(getattr(settings, "tmdb_movies_path", "")),
                "shows": Path(getattr(settings, "tmdb_shows_path", "")),
            }
        values = compute_prominence(self.catalog.reset_index(drop=True), exports)
        self._prominence_cache = values
        return values

    def _prominence_by_id(self) -> dict[str, float]:
        cached = getattr(self, "_prominence_ids_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached
        values = self._prominence_array()
        cached = dict(zip(self.catalog["global_id"].astype(str), values.tolist())) if len(self.catalog) else {}
        self._prominence_ids_cache = cached
        return cached

    def _catalog_position(self) -> dict[str, int]:
        cached = getattr(self, "_catalog_position_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached
        cached = {gid: pos for pos, gid in enumerate(self.catalog["global_id"].astype(str))} if len(self.catalog) else {}
        self._catalog_position_cache = cached
        return cached

    def _knowledge_graph(self) -> CatalogKnowledgeGraph:
        if not hasattr(self, "knowledge_graph"):
            self.knowledge_graph = CatalogKnowledgeGraph(self.catalog)
        return self.knowledge_graph

    def _catalog_items(
        self,
        global_ids: list[str],
        content_type: str | None,
        audience: AudienceContext | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Catalogue rows as result items, through the same eligibility gate as Qdrant."""
        normalized_content_type = normalize_content_type(content_type)
        positions = self._catalog_position()
        items: list[dict[str, Any]] = []
        for gid in global_ids:
            position = positions.get(gid)
            if position is None:
                continue
            row = self.catalog.iloc[position]
            item = {column: _clean_value(row.get(column, "")) for column in RESULT_COLUMNS}
            if normalized_content_type is not None and item.get("content_type") != normalized_content_type:
                continue
            # These rows bypass Qdrant, so the eligibility rule is applied here.
            if audience is not None and not is_eligible(item, audience):
                continue
            items.append(item)
            if len(items) >= limit:
                break
        return items

    def _entity_candidate_items(
        self,
        query: str,
        content_type: str | None,
        audience: AudienceContext | None,
        limit: int,
        years: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        graph = self._knowledge_graph().graph
        bonus = self._prominence_array() * self._weights()["popularity"]
        allowed = self._year_mask(years) if years is not None else None
        if allowed is not None and len(allowed) != len(graph.item_ids):
            allowed = None
        pairs = graph.entity_candidates(query, limit * 4, bonus=bonus, allowed=allowed)
        return self._catalog_items([gid for gid, _ in pairs], content_type, audience, limit)

    def _plot_index(self) -> PlotIndex | None:
        """Synopsis vectors, loaded on first use; None when they have not been built."""
        if not hasattr(self, "_plot_index_cache"):
            settings = getattr(self, "settings", None)
            index = None
            if settings is not None and getattr(settings, "plot_embeddings_path", None):
                try:
                    index = PlotIndex.load(
                        Path(settings.plot_embeddings_path), Path(settings.plot_embedding_index_path)
                    )
                except (OSError, ValueError, KeyError) as exc:
                    print(f"Plot index not loaded: {exc}", file=sys.stderr)
                    index = None
            self._plot_index_cache = index
        return self._plot_index_cache

    def _plot_candidate_items(
        self,
        query_vector: np.ndarray,
        content_type: str | None,
        audience: AudienceContext | None,
        limit: int,
        years: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        plot = self._plot_index()
        if plot is None or limit <= 0:
            return []
        allowed = None
        if years is not None or normalize_content_type(content_type) is not None:
            allowed = self._row_filter(self._catalog_rows_of("plot", plot.global_ids), content_type, years)
        pairs = plot.search(query_vector, limit * 2, allowed=allowed)
        return self._catalog_items([gid for gid, _ in pairs], content_type, audience, limit)

    def _release_years(self) -> np.ndarray:
        """Release year per catalogue row, 0 when unknown, built once."""
        cached = getattr(self, "_release_years_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached
        dates = self.catalog.get("release_date", pd.Series("", index=self.catalog.index)).astype(str)
        years = pd.to_numeric(dates.str.extract(r"^\s*(\d{4})", expand=False), errors="coerce").fillna(0)
        cached = years.astype(np.int32).to_numpy()
        self._release_years_cache = cached
        return cached

    def _year_mask(self, years: tuple[int, int]) -> np.ndarray:
        released = self._release_years()
        return (released >= years[0]) & (released <= years[1])

    def _catalog_rows_of(self, name: str, global_ids: np.ndarray) -> np.ndarray:
        """Catalogue row of each id in another index (-1 when absent), cached per index."""
        caches = self.__dict__.setdefault("_catalog_rows_cache", {})
        cached = caches.get(name)
        if cached is not None and len(cached) == len(global_ids):
            return cached
        positions = self._catalog_position()
        cached = np.array([positions.get(str(gid), -1) for gid in global_ids], dtype=np.int64)
        caches[name] = cached
        return cached

    def _row_filter(
        self,
        catalog_rows: np.ndarray,
        content_type: str | None,
        years: tuple[int, int] | None,
    ) -> np.ndarray:
        """Per entry of another index: in the catalogue, of ``content_type`` and released in ``years``."""
        known = catalog_rows >= 0
        allowed = known.copy()
        if years is not None:
            allowed[known] &= self._year_mask(years)[catalog_rows[known]]
        normalized_type = normalize_content_type(content_type)
        if normalized_type is not None:
            types = self.catalog["content_type"].astype(str).to_numpy()
            allowed[known] &= types[catalog_rows[known]] == normalized_type
        return allowed

    def _has_exact_title(self, text: str, content_type: str | None = None) -> bool:
        tokens = _tokenize_lookup_text(text)
        if not tokens or "title" not in self.catalog.columns:
            return False
        exact = self._title_haystack().eq(f" {tokens} ")
        normalized_type = normalize_content_type(content_type)
        if normalized_type is not None:
            exact &= self.catalog["content_type"] == normalized_type
        return bool(exact.any())

    @staticmethod
    def _content_words(query: str) -> list[str]:
        """The words of a query that say what it is about, not what kind or how good."""
        return [
            word
            for word in _tokenize_lookup_text(query).split()
            if word not in TYPE_WORDS and word not in LIKE_FILLERS and word not in BROWSE_WORDS
        ]

    def _browse(
        self,
        content_type: str | None,
        audience: AudienceContext | None,
        years: tuple[int, int],
        limit: int,
    ) -> list[dict[str, Any]]:
        """The most established items released in ``years``, of one type when given."""
        mask = self._year_mask(years)
        normalized_type = normalize_content_type(content_type)
        if normalized_type is not None:
            mask &= self.catalog["content_type"].astype(str).to_numpy() == normalized_type
        rows = np.flatnonzero(mask)
        if not rows.size:
            return []
        prominence = self._prominence_array()
        rows = rows[np.argsort(-prominence[rows], kind="stable")][: limit * 4]
        ids = self.catalog["global_id"].astype(str).to_numpy()[rows].tolist()
        items = self._catalog_items(ids, content_type, audience, limit)
        weight = self._weights()["popularity"]
        by_id = self._prominence_by_id()
        for item in items:
            popularity = by_id.get(str(item["global_id"]), 0.0)
            item["semantic_score"] = None
            item["popularity_score"] = round(popularity, 6)
            item["score"] = weight * popularity
        return items

    def _search_vector_in_years(
        self,
        vector: np.ndarray,
        limit: int,
        content_type: str | None,
        audience: AudienceContext | None,
        years: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Nearest items released in ``years``.

        The Qdrant payload has no numeric year, and filtering its top 300
        afterwards leaves a handful for "sci-fi from the 80s". The stored item
        vectors are in memory, so the decade's rows are searched exactly.
        """
        store = getattr(self, "ema_store", None)
        embeddings = getattr(store, "embeddings", None)
        index = getattr(store, "index", None)
        if embeddings is None or index is None:
            pool = self._search_vector(vector, limit * 10, content_type, audience)
            return [item for item in pool if _released_in(item, years)][:limit]
        store_ids = index["global_id"].astype(str).to_numpy()
        store_rows = index["embedding_row"].to_numpy(dtype=np.int64)
        allowed = self._row_filter(self._catalog_rows_of("store", store_ids), content_type, years)
        picked = np.flatnonzero(allowed)
        if not picked.size:
            return []
        scores = embeddings[store_rows[picked]] @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[: limit * 2]
        ids = [str(store_ids[picked[i]]) for i in order]
        cosine = {str(store_ids[picked[i]]): float(scores[i]) for i in order}
        items = self._catalog_items(ids, content_type, audience, limit)
        for item in items:
            item["score"] = cosine.get(str(item["global_id"]), 0.0)
        return items

    def _similar_candidate_items(
        self,
        global_id: str,
        content_type: str | None,
        audience: AudienceContext | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        graph = self._knowledge_graph().graph
        bonus = self._prominence_array() * self._weights()["popularity"]
        pairs = graph.similar_items(global_id, limit * 4, bonus=bonus)
        return self._catalog_items([gid for gid, _ in pairs], content_type, audience, limit)

    def _cosines(self, vector: np.ndarray, global_ids: list[str]) -> dict[str, float]:
        """Cosine between ``vector`` and stored item embeddings (normalised), where known."""
        store = getattr(self, "ema_store", None)
        if store is None or vector is None:
            return {}
        out: dict[str, float] = {}
        for gid in global_ids:
            item_vector = store.get_item_vector(gid)
            if item_vector is not None:
                out[gid] = float(np.dot(item_vector, vector))
        return out

    def _item_vector(self, global_id: str, fallback_text: str) -> np.ndarray:
        """The stored embedding of an item, or its text encoded now when none is loaded."""
        store = getattr(self, "ema_store", None)
        if store is not None:
            stored = store.get_item_vector(global_id)
            if stored is not None:
                return np.asarray(stored, dtype=np.float32)
        return self.model.encode(
            [fallback_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]

    def _find_title(self, text: str, preferred_type: str | None = None) -> dict[str, Any] | None:
        """The catalogue item a phrase like "inception" names: exact title, else title prefix.

        Among several ("Inception" is a film and a song), the requested type
        wins, then the most established item.
        """
        tokens = _tokenize_lookup_text(text)
        if not tokens:
            return None
        titles = self._title_haystack()
        exact = titles.eq(f" {tokens} ")
        candidates = self.catalog[exact]
        tier = 0
        if candidates.empty:
            candidates = self.catalog[titles.str.startswith(f" {tokens} ", na=False)]
            tier = 1
        if candidates.empty:
            return None
        prominence = self._prominence_array()
        positions = self._catalog_position()
        best = max(
            candidates.to_dict("records"),
            key=lambda row: (
                preferred_type is not None and row.get("content_type") == preferred_type,
                prominence[positions[str(row["global_id"])]],
            ),
        )
        return {
            "global_id": str(best["global_id"]),
            "title": str(best.get("title", "")),
            "content_type": str(best.get("content_type", "")),
            "match_tier": tier,
        }

    def correct_query(self, query: str) -> str | None:
        """The query with near-miss typos fixed, or None when it needs none.

        Kept apart from recommend() so the caller can tell the user what was
        searched instead and offer the query exactly as typed.
        """
        words = _tokenize_lookup_text(query).split()
        vocabulary = self._query_vocabulary()
        corrected = vocabulary.correct(words)
        if corrected is None:
            return None

        # A word seen in a single row may be that row's whole title or name -
        # a song called "Blxt" - so a query that already finds one as typed is
        # left alone. Unseen words cannot be, so the scan is skipped for them.
        replaced = [word for word, fixed in zip(words, corrected) if word != fixed]
        if any(vocabulary.rows.get(word, 0) for word in replaced):
            phrase = f" {' '.join(words)} "
            if (
                self._title_haystack().str.contains(phrase, regex=False).any()
                or self._creator_haystack().str.contains(phrase, regex=False).any()
            ):
                return None
        return " ".join(corrected)

    def starter_items(self) -> list[dict[str, Any]]:
        """Well-known items across genres, for a new account to pick from.

        Per (type, genre), the best-known item with cover art and a decent
        rating that the unknown-age default may see - a new account has not
        said how old it is. The catalogue does not change while the process
        runs, so this is worked out once.
        """
        cached = getattr(self, "_starter_items_cache", None)
        if cached is not None:
            return cached

        catalog = self.catalog
        audience = AudienceContext()
        images = catalog.get("image_url", pd.Series("", index=catalog.index)).fillna("").astype(str)
        genres_per_row = (
            catalog["categories"].fillna("").astype(str).str.lower().str.split(",")
            .map(lambda parts: {part.strip() for part in parts})
        )
        popularity = pd.to_numeric(catalog.get("popularity", 0), errors="coerce").fillna(0)
        rating = pd.to_numeric(
            catalog.get("rating", pd.Series(float("nan"), index=catalog.index)), errors="coerce"
        )

        picks: list[dict[str, Any]] = []
        chosen: set[Any] = set()
        for content_type, genres in STARTER_GENRES.items():
            candidates = (catalog["content_type"] == content_type) & images.ne("")
            if content_type in STARTER_MIN_RATING:
                candidates &= rating.ge(STARTER_MIN_RATING[content_type])
            standing = popularity[candidates].rank(pct=True)
            if content_type in STARTER_BLEND_RATING:
                standing = standing + rating[candidates].rank(pct=True)
            ranked = standing.sort_values(ascending=False, kind="stable").index
            for genre in genres:
                for index in ranked:
                    if index in chosen or genre not in genres_per_row[index]:
                        continue
                    row = catalog.loc[index]
                    item = {column: _clean_value(row.get(column, "")) for column in RESULT_COLUMNS}
                    if not is_eligible(item, audience):
                        continue
                    chosen.add(index)
                    picks.append({**item, "genre": genre})
                    break

        self._starter_items_cache = picks
        return picks

    def warm_up(self) -> None:
        """Build the lookup tables now rather than inside the first search.

        Each is built on first use and reused after. Together they take several
        seconds, which the first search used to pay on top of its own work -
        close to the frontend's 15 s timeout once the vocabulary was added.
        """
        self._query_vocabulary()
        self._keyword_haystack()
        self._popularity_rank()
        self._prominence_by_id()
        self._catalog_position()
        self._release_years()
        self._title_haystack()
        self.eligible_mask(AudienceContext())  # per-row ages for autocomplete
        plot = self._plot_index()
        if plot is not None:
            self._catalog_rows_of("plot", plot.global_ids)
        store_index = getattr(getattr(self, "ema_store", None), "index", None)
        if store_index is not None:
            self._catalog_rows_of("store", store_index["global_id"].astype(str).to_numpy())
        graph = self.knowledge_graph.graph  # typed entity graph, about 3 s
        graph._relations_by_entity()
        graph._row_norms()

    def _query_vocabulary(self) -> QueryVocabulary:
        """Word counts over the catalogue, for typo and noise detection; built once."""
        cached = getattr(self, "_query_vocabulary_cache", None)
        if cached is not None and cached.catalog_rows == len(self.catalog):
            return cached

        vocabulary = QueryVocabulary.build(
            self._fulltext_haystack(),
            self._title_haystack(),
            self._creator_haystack(),
        )
        self._query_vocabulary_cache = vocabulary
        return vocabulary

    def _search_vector(
        self,
        vector: np.ndarray,
        top_k: int,
        content_type: str | None,
        audience: AudienceContext | None = None,
    ) -> list[dict[str, Any]]:
        normalized_content_type = normalize_content_type(content_type)
        query_filter = self._build_qdrant_filter(normalized_content_type, audience)

        try:
            points = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        except AttributeError:
            response = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        except Exception as exc:
            raise RuntimeError(
                "Could not query Qdrant. Start Qdrant, build the collection, and check "
                "QDRANT_PATH or QDRANT_URL/QDRANT_COLLECTION."
            ) from exc

        results: list[dict[str, Any]] = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            item = {column: _clean_value(payload.get(column, "")) for column in RESULT_COLUMNS}
            item["score"] = float(getattr(point, "score", 0.0))
            results.append(item)
        return results

    def _search_person_matches(
        self,
        query: str,
        top_k: int,
        content_type: str | None,
        audience: AudienceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return catalog rows whose creator/cast/director list matches query."""
        normalized_content_type = normalize_content_type(content_type)
        q_norm = _normalize_lookup_text(query)
        if len(q_norm) < 2 or "creators" not in self.catalog.columns:
            return []

        haystack = self._creator_haystack()
        catalog = self.catalog
        if normalized_content_type is not None:
            type_mask = catalog["content_type"] == normalized_content_type
            catalog = catalog[type_mask]
            haystack = haystack[type_mask]

        # Vectorised, because this runs on every /recommend call before the
        # vector search does. Row-by-row iteration over the catalogue cost 15-49 s
        # per request at 106,332 rows - past the frontend's 15 s timeout, so a live
        # search failed outright. It was merely slow at the old 48,294.
        #
        # Each row's names are pre-joined as "\nname1\nname2\n", so the three
        # match tiers below are substring tests on one string. The delimiter is
        # safe: _normalize_lookup_text collapses all whitespace via
        # " ".join(str.split()), so a normalized name cannot contain a newline.
        # Matched against the padded haystack above, so every tier lands on a
        # whole word: the full name, a name beginning with the query, or the
        # query appearing as a complete word within a name ("nolan" in
        # "christopher nolan", but not "inception" inside "skinception").
        q_tokens = _tokenize_lookup_text(q_norm)
        if not q_tokens:
            return []
        exact = haystack.str.contains(f"\n {q_tokens} \n", regex=False, na=False)
        prefix = haystack.str.contains(f"\n {q_tokens} ", regex=False, na=False)
        substring = haystack.str.contains(f" {q_tokens} ", regex=False, na=False)

        # Two tiers: the whole name, or a word of it. A name that starts with the
        # query is no stronger a match than one that ends with it: "nolan" means
        # Christopher Nolan at least as often as Nolan North, and prominence
        # orders the partial matches below.
        ranks = pd.Series(3, index=catalog.index)
        ranks = ranks.mask(substring | prefix, 1)
        ranks = ranks.mask(exact, 0)
        # An Amazon "creator" is a brand. Brands reuse ordinary words - "LOVE
        # GOOD FATS", "Safe Space" - so only the brand's full name may match;
        # otherwise searching "love" listed cookies and orthotics as creator hits.
        if "source" in catalog.columns:
            brand = catalog["source"].astype(str).str.startswith("amazon")
            ranks = ranks.mask(brand & (ranks > 0), 3)
        matched = ranks < 3
        if not matched.any():
            return []

        # Within a tier, the most established work first. TMDb's "popularity" is
        # a trending figure, so ordering by it put Riddick and Truth or Dare
        # (an actor named Nolan) ahead of Inception for the query "nolan".
        prominence = self._prominence_by_id()
        # One word is a surname as often as a whole name. For "nolan" an R&B
        # artist called NOLAN matched the whole name and outranked every
        # Christopher Nolan film; between one-word matches the whole name only
        # earns a bonus, and how established the work is decides.
        one_word = len(q_tokens.split()) == 1
        rows: list[tuple[int, float, str, dict[str, Any]]] = []
        for row, match_rank in zip(
            catalog[matched].to_dict("records"), ranks[matched]
        ):
            item = {column: _clean_value(row.get(column, "")) for column in RESULT_COLUMNS}
            # This path reads the catalog directly instead of going through
            # Qdrant, so the eligibility gate has to be applied here too.
            if audience is not None and not is_eligible(item, audience):
                continue
            popularity = _safe_float(row.get("popularity"), 0.0)
            rating = _safe_float(row.get("rating"), 0.0)
            item["score"] = 1.2 - (match_rank * 0.1) + min(popularity, 250.0) / 10000.0
            item["semantic_score"] = None
            item["match_kind"] = "creator"
            item["match_tier"] = int(match_rank)
            established = prominence.get(str(item.get("global_id", "")), 0.0) + rating / 1000.0
            if one_word:
                established += WHOLE_NAME_BONUS * (match_rank == 0)
            rows.append((
                0 if one_word else int(match_rank),
                -established,
                str(item.get("title", "")).casefold(),
                item,
            ))

        rows.sort(key=lambda row: (row[0], row[1], row[2]))
        return [item for _, _, _, item in rows[:top_k]]

    def _creator_haystack(self) -> pd.Series:
        """Normalized creator names per row, newline-delimited, built once.

        The catalogue is loaded at startup and not mutated, so this is computed on
        first use and reused for the life of the process rather than being
        recomputed per request.
        """
        cached = getattr(self, "_creator_haystack_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached

        # Each name is newline-delimited *and* space-padded: "\n john sturges \n".
        # The newlines anchor a whole-name match, the spaces anchor a whole-word
        # one. Without the spaces a bare substring test matched "inception"
        # inside the brand "Skinception", which then ranked a scar cream as a
        # creator hit for the film Inception.
        haystack = self.catalog["creators"].map(
            lambda value: (
                "\n"
                + "\n".join(
                    f" {_tokenize_lookup_text(name)} " for name in _split_creators(value)
                )
                + "\n"
            )
            if _split_creators(value)
            else ""
        )
        self._creator_haystack_cache = haystack
        return haystack

    def _title_haystack(self) -> pd.Series:
        """Normalized title per row, newline-delimited, built once.

        Same shape and the same reasoning as :meth:`_creator_haystack`: one
        pass over the catalogue at first use, reused for the life of the process.
        """
        cached = getattr(self, "_title_haystack_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached

        haystack = self.catalog["title"].map(
            lambda value: " " + _tokenize_lookup_text(value) + " "
        )
        self._title_haystack_cache = haystack
        return haystack

    def _keyword_haystack(self) -> pd.Series:
        """Title plus categories, for token matching on a descriptive query.

        Punctuation becomes whitespace and the whole string is space-padded, so
        a token can be matched as `" token "` and only ever hits a whole word.
        Without that, a bare substring test matches inside anything: "f1" found
        an ULAB Erlenmeyer flask because the sequence appears in its product
        code, and ranked it second for the query "F1".
        """
        cached = getattr(self, "_keyword_haystack_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached

        titles = self.catalog["title"].fillna("").astype(str)
        categories = self.catalog.get(
            "categories", pd.Series("", index=self.catalog.index)
        ).fillna("").astype(str)
        haystack = (titles + " " + categories).map(
            lambda value: " " + _tokenize_lookup_text(value) + " "
        )
        self._keyword_haystack_cache = haystack
        return haystack

    def _fulltext_haystack(self) -> pd.Series:
        """Title, categories and description, for topical keyword queries.

        Title and categories alone cannot answer "racing". It is not a TMDb
        genre, so it appears in no category, and the racing films whose titles
        omit the word - F1, Rush, Ford v Ferrari, Gran Turismo - were
        unreachable by keyword. Their descriptions say it plainly: F1's opens
        "Racing legend Sonny Hayes is coaxed out of retirement". Indexing the
        description is what turns a topic word into a usable query.

        Built with vectorised pandas string operations rather than a per-row
        Python function: descriptions total 58 million characters and one
        Amazon row runs to 83,711, so a per-character loop is not viable. This
        way it is ~2 s at startup and 57 MB resident.

        Descriptions are truncated to 1,000 characters. The topic of an item is
        established in its opening lines; the tail of an Amazon listing is
        specifications and boilerplate, and indexing all 58 million characters
        would cost memory to make matches *worse*.
        """
        cached = getattr(self, "_fulltext_haystack_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached

        def column(name: str) -> pd.Series:
            return (
                self.catalog.get(name, pd.Series("", index=self.catalog.index))
                .fillna("")
                .astype(str)
            )

        combined = (
            column("title") + " " + column("categories") + " "
            + column("description").str.slice(0, 1000)
        ).str.lower()
        # Accents are dropped before punctuation is. NFKD splits "í" into "i"
        # plus a combining mark, and the mark alone matched [^a-z0-9], so
        # "película" was indexed as two words, "peli cula", that no query - which
        # strips the mark - could ever match.
        combined = (
            combined.str.normalize("NFKD")
            .str.replace(r"[̀-ͯ]", "", regex=True)
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
        )
        haystack = " " + combined.str.strip() + " "
        self._fulltext_haystack_cache = haystack
        return haystack

    def _popularity_rank(self) -> pd.Series:
        """Within-content-type popularity percentile, built once.

        `popularity` means a different quantity per source: Amazon ships review
        counts that peak near 294,000, TMDb a float that peaks near 800, Spotify
        a 0-100 index. Ordering a mixed set of lexical hits by the raw number
        therefore returns Amazon rows for every query regardless of relevance.
        A within-type percentile is comparable across sources; the raw value is
        not.
        """
        cached = getattr(self, "_popularity_rank_cache", None)
        if cached is not None and len(cached) == len(self.catalog):
            return cached

        numeric = pd.to_numeric(
            self.catalog.get("popularity", 0), errors="coerce"
        ).fillna(0)
        ranked = numeric.groupby(
            self.catalog["content_type"].astype(str)
        ).rank(pct=True).fillna(0.0)
        self._popularity_rank_cache = ranked
        return ranked

    def _search_lexical_matches(
        self,
        query: str,
        top_k: int,
        content_type: str | None,
        audience: AudienceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Title and keyword matches, which semantic retrieval alone cannot find.

        Retrieval used to be purely a cosine search over SBERT embeddings, with
        one lexical escape hatch that matched the `creators` column only. Titles
        were never matched at all, so "Christopher Nolan" worked and "F1" did
        not: a two-character query carries almost no semantic signal, and its
        nearest vectors are effectively arbitrary. Searching "F1" returned a
        Punjabi song, a 3D-printer pad and an aromatherapy necklace, while the
        2025 film *F1* sat in the catalogue the whole time.

        Four tiers, most exact first, so an exact title always outranks a title
        that merely contains the words:

          0  the whole query is the title
          1  the title starts with the query
          2  the title contains the query
          3  every token of the query appears in the title or its categories

        Tier 3 is what makes a descriptive keyword query work - "space
        adventure" matches an item whose categories carry both words - without
        letting a single common token drag in the whole catalogue.
        """
        normalized_content_type = normalize_content_type(content_type)
        q_norm = _normalize_lookup_text(query)
        if len(q_norm) < 1:
            return []

        titles = self._title_haystack()
        keywords = self._keyword_haystack()
        fulltext = self._fulltext_haystack()
        popularity_rank = self._popularity_rank()
        catalog = self.catalog

        if normalized_content_type is not None:
            type_mask = catalog["content_type"] == normalized_content_type
            catalog = catalog[type_mask]
            titles = titles[type_mask]
            keywords = keywords[type_mask]
            fulltext = fulltext[type_mask]
            popularity_rank = popularity_rank[type_mask]

        # Whole words throughout, via the space-padded tokenized haystack. A bare
        # substring test is far too loose on a short query: "F1" matched
        # FORMUFIT F1144WT, a Tacwise nail gun pack and a PVC ball valve, all of
        # which carry the sequence inside a product code, and they took the
        # three places behind the film.
        q_tokens = _tokenize_lookup_text(q_norm)
        if not q_tokens:
            return []
        exact = titles.eq(f" {q_tokens} ")
        prefix = titles.str.startswith(f" {q_tokens} ", na=False)
        contains = titles.str.contains(f" {q_tokens} ", regex=False, na=False)

        # Tier 3: every meaningful token present. Tokens of one character are
        # dropped - they match almost everything and carry no intent - but the
        # whole query is still matched verbatim by tiers 0-2, so a genuinely
        # short title like "1" or "F1" is reachable there.
        tokens = [token for token in _tokenize_lookup_text(q_norm).split() if len(token) > 1]
        if tokens:
            # Space-padded, so a token matches a whole word rather than any
            # occurrence of the characters inside a longer one.
            in_keywords = pd.Series(True, index=catalog.index)
            for token in tokens:
                in_keywords &= keywords.str.contains(f" {token} ", regex=False, na=False)

            # Prose needs the whole phrase, not the tokens scattered through it.
            # Requiring each token independently anywhere in 1,000 characters of
            # description matched a CPAP hose for "time travel", because its
            # listing happens to contain both "travel" and "time". A title or a
            # genre is a handful of words, so independent tokens are safe there;
            # a synopsis is not.
            in_fulltext = fulltext.str.contains(
                f" {q_tokens} ", regex=False, na=False
            )
        else:
            in_keywords = pd.Series(False, index=catalog.index)
            in_fulltext = pd.Series(False, index=catalog.index)

        ranks = pd.Series(9, index=catalog.index)
        # Tier 4 before tier 3, so the stronger signal overwrites it: a word in
        # the title or the genre says more about an item than the same word
        # buried in its synopsis.
        ranks = ranks.mask(in_fulltext, 4)
        ranks = ranks.mask(in_keywords, 3)
        ranks = ranks.mask(contains, 2)
        ranks = ranks.mask(prefix, 1)
        ranks = ranks.mask(exact, 0)
        matched = ranks < 9
        if not matched.any():
            return []

        # Cap the scan: a common token can match tens of thousands of rows, and
        # only the strongest handful ever reach the caller.
        candidates = catalog[matched]
        candidate_ranks = ranks[matched]
        candidate_pop = popularity_rank[matched]

        # Tiers 2-4 order together, by popularity, rather than strictly by tier.
        #
        # Strict tier order let any title containing the word beat every
        # description match, however weak: searching "racing" returned Racing
        # Extinction - a documentary about species loss - above F1, Rush and
        # Ford v Ferrari, because those three do not have the word in the
        # title. Exact and prefix hits stay ahead of everything, since someone
        # typing a full title wants that title; below them, "the word appears
        # somewhere" is one band and prominence decides the order.
        order = pd.DataFrame(
            {
                "band": candidate_ranks.clip(upper=2),
                "rank": candidate_ranks,
                "pop": candidate_pop,
            }
        ).sort_values(["band", "pop"], ascending=[True, False])
        order = order.head(max(top_k * 6, 60))

        rows: list[dict[str, Any]] = []
        for index in order.index:
            row = candidates.loc[index]
            item = {column: _clean_value(row.get(column, "")) for column in RESULT_COLUMNS}
            # Read straight from the catalogue rather than through Qdrant, so the
            # eligibility gate has to be applied here the same way the creator
            # path applies it.
            if audience is not None and not is_eligible(item, audience):
                continue
            match_rank = int(order.loc[index, "rank"])
            # Above the semantic band (cosine tops out near 1.0) so an exact
            # title wins, but tiered so a weaker lexical match does not outrank
            # a strong one.
            item["score"] = 1.30 - (match_rank * 0.06)
            item["semantic_score"] = None
            item["match_kind"] = "title" if match_rank <= 2 else "keyword"
            item["match_tier"] = match_rank
            rows.append(item)
            if len(rows) >= top_k:
                break
        return rows

    def _personalize_results(
        self,
        results: list[dict[str, Any]],
        top_k: int,
        user_id: str | None,
    ) -> list[dict[str, Any]]:
        if not user_id:
            return results[:top_k]

        try:
            profile = MySQLStore(self.settings).get_user_preference_profile(user_id)
        except Exception:
            return results[:top_k]

        category_weights = profile["category_weights"]
        content_type_weights = profile["content_type_weights"]
        positive_ids = profile["positive_ids"]
        negative_ids = profile["negative_ids"]

        reranked = []
        for item in results:
            score = float(item.get("score", 0.0))
            affinity = 0.0
            for category in _split_categories(item.get("categories")):
                affinity += min(category_weights.get(category, 0.0), 3.0) * 0.035

            content_type = str(item.get("content_type", "")).lower()
            affinity += min(content_type_weights.get(content_type, 0.0), 3.0) * 0.02

            # Capped. A film's categories now carry its TMDb keywords too, so an
            # item could collect fifteen matching categories and +0.7 - more than
            # its whole relevance - and taste alone put Lightyear on a page for
            # "christopher nolan". Taste reorders close calls; it does not decide.
            bonus = max(-PREFERENCE_CAP, min(PREFERENCE_CAP, affinity))

            if item.get("global_id") in positive_ids:
                bonus -= 0.15
            if item.get("global_id") in negative_ids:
                bonus -= 0.35

            personalized = dict(item)
            personalized["score"] = score + bonus
            personalized["preference_bonus"] = bonus
            reranked.append(personalized)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]
    def _graph_rerank(
        self,
        results: list[dict[str, Any]],
        user_id: str | None,
    ) -> list[dict[str, Any]]:
        if not user_id or self.graph_store is None:
            return results
        return self.graph_store.rerank(
            user_id,
            results,
            graph_weight=self.settings.lightgcn_weight,
        )

    def _ema_rerank(
        self,
        results: list[dict[str, Any]],
        user_id: str | None,
    ) -> list[dict[str, Any]]:
        if not user_id or self.ema_store is None:
            return results
        try:
            user_vector = MySQLStore(self.settings).get_user_ema_vector(user_id)
        except Exception:
            return results
        return self.ema_store.rerank(
            user_vector,
            results,
            ema_weight=self.settings.ema_weight,
        )

    def _knowledge_graph_rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        user_id: str | None,
        *,
        seed_ids: list[str] | None = None,
        seed_role: str = "liked",
        kg_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        knowledge_graph = self._knowledge_graph()

        user_profile: dict[str, Any] = {}
        seeds: Any = set(seed_ids) if seed_ids is not None else set()
        if user_id:
            store = MySQLStore(self.settings)
            try:
                user_profile = store.get_user_profile_preferences(user_id)
            except Exception:
                user_profile = {}
            if seed_ids is None:
                try:
                    # Items the user currently likes are the seeds of the graph paths.
                    seeds = store.get_user_preference_profile(user_id)["positive_ids"]
                except Exception:
                    seeds = set()
        return knowledge_graph.rerank(
            query,
            results,
            user_profile=user_profile,
            kg_weight=self._weights()["kg"] if kg_weight is None else kg_weight,
            seed_ids=seeds,
            seed_role=seed_role,
        )

    def _build_qdrant_filter(
        self,
        content_type: str | None,
        audience: AudienceContext | None = None,
    ):
        """Compile the request's constraints into one pre-search filter.

        With an audience, the age and risk gates go into the vector query itself,
        so an ineligible item is never retrieved and no downstream reranker can
        promote it back. Without one, behaviour is unchanged: a plain
        content_type match, or no filter at all.
        """
        if audience is not None:
            return build_eligibility_filter(audience, content_type)

        if content_type is None:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            ) from exc
        return Filter(
            must=[
                FieldCondition(
                    key="content_type",
                    match=MatchValue(value=content_type),
                )
            ]
        )

    def _load_qdrant_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for the Qdrant backend.\n"
                "Install it with: pip install qdrant-client"
            ) from exc

        if self.settings.qdrant_path:
            return QdrantClient(path=self.settings.qdrant_path)

        return QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
    def _load_graph_store(self) -> GraphEmbeddingStore | None:
        try:
            return GraphEmbeddingStore(Path(self.settings.lightgcn_artifact_path))
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _load_ema_store(self) -> EMAEmbeddingStore | None:
        try:
            return EMAEmbeddingStore(
                Path(self.settings.content_embeddings_path),
                Path(self.settings.content_embedding_index_path),
            )
        except FileNotFoundError:
            return None
        except ValueError:
            return None
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for query embeddings.\n"
                "Install project dependencies, then try again."
            ) from exc

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        print(f"Loading query embedding model on {device}: {self.embedding_model}")
        try:
            return SentenceTransformer(
                self.embedding_model,
                device=device,
                local_files_only=True,
            )
        except TypeError:
            return SentenceTransformer(self.embedding_model, device=device)
        except Exception as local_exc:
            print(
                "Local model cache load failed; trying Hugging Face download...",
                file=sys.stderr,
            )
            try:
                return SentenceTransformer(self.embedding_model, device=device)
            except Exception as remote_exc:
                raise RuntimeError(
                    "Could not load the query embedding model from local cache or Hugging Face. "
                    "Run once with internet access, or set HF_HUB_OFFLINE=1 if cached."
                ) from remote_exc

    def _assert_artifacts_exist(self) -> None:
        missing = [
            path
            for path in [self.catalog_path, self.embedding_index_path]
            if not path.exists()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Missing recommender artifact(s):\n"
                f"{formatted}\n"
                "Run: python -m preprocessing.build_content_catalog; "
                "python -m embeddings.build_embeddings; "
                "python -m embeddings.build_qdrant_collection"
            )

    def _validate_artifacts(self) -> None:
        if len(self.catalog) != len(self.embedding_index):
            raise ValueError(
                f"Catalog rows ({len(self.catalog)}) do not match "
                f"embedding index rows ({len(self.embedding_index)})."
            )
        # Queries encoded by one model searched against items encoded by
        # another return confident nonsense, so a mismatch stops startup.
        if "embedding_model" in self.embedding_index.columns:
            stored = set(self.embedding_index["embedding_model"].dropna().astype(str))
            if stored and stored != {self.embedding_model}:
                raise ValueError(
                    f"Item vectors were built with {sorted(stored)} but EMBEDDING_MODEL is "
                    f"{self.embedding_model}. Rebuild: python -m embeddings.build_embeddings; "
                    "python -m embeddings.build_plot_embeddings; python -m embeddings.build_qdrant_collection; "
                    "python -m scripts.rebuild_ema_profiles"
                )


def _clean_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _split_categories(value: Any) -> list[str]:
    text = str(value or "").lower()
    return [part.strip() for part in text.split(",") if part.strip()]

def _split_creators(value: Any) -> list[str]:
    text = str(_clean_value(value) or "")
    return [part.strip() for part in text.split(",") if part.strip()]


def _tokenize_lookup_text(text: Any) -> str:
    """Normalized text with punctuation reduced to single spaces.

    Token matching pads a token with spaces to force a whole-word hit, which
    only works if punctuation is not glued to the word: "Action, Drama" has to
    become "action drama" or the token "action" never matches " action ".
    """
    normalized = _normalize_lookup_text(text)
    return " ".join(
        "".join(char if char.isalnum() else " " for char in normalized).split()
    )


def _normalize_lookup_text(text: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(_clean_value(text) or ""))
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(without_marks.casefold().split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
