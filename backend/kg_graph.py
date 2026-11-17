"""Typed knowledge graph over the catalogue, scored by entity links and 2-hop paths.

Nodes
    Item (every catalogue row) and the entities it links to:
    person, brand, genre, franchise, keyword, content_type, domain.

Edges (item -> entity, typed)
    DIRECTED_BY  first creator of a movie
    STARS        further credited cast of a movie (up to five)
    WRITTEN_BY   book author (Open Library, including finance books)
    PERFORMED_BY music artist
    MADE_BY      product brand (Amazon health / industrial / finance)
    PART_OF      franchise ("... Collection" from TMDb)
    TAGGED       TMDb plot keyword (lower-case tokens on movies and shows)
    IN_GENRE     genre, subject or product category
    IS_A         content type          (structure only, not scored)
    IN_DOMAIN    domain                (structure only, not scored)

Scoring
    * Query links: n-grams of the query are matched to entity names, so
      "christopher nolan" becomes the person node rather than two words. An item
      scores the share of the linked entities' weight it is connected to.
    * User paths: the items a user currently likes are seeds; a candidate scores
      by the meta-path  seed -> entity <- candidate,  weighted by relation and by
      how rare the entity is (IDF), as a cosine between the two entity vectors.
    * Each scored item also gets the best path as a human-readable explanation.

The graph lives in one sparse item x entity matrix, so a request only touches the
rows of its ~50-100 candidates. ``export_rows`` feeds the Neo4j export script.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

ENTITY_TYPES = ("person", "brand", "genre", "franchise", "keyword", "content_type", "domain")

# relation -> (entity type, weight in scoring, phrase used in explanations)
RELATIONS: dict[str, tuple[str, float, str]] = {
    "DIRECTED_BY": ("person", 1.0, "Directed by"),
    "STARS": ("person", 0.7, "Stars"),
    "WRITTEN_BY": ("person", 1.0, "Written by"),
    "PERFORMED_BY": ("person", 1.0, "By"),
    "MADE_BY": ("brand", 0.6, "Made by"),
    "PART_OF": ("franchise", 1.0, "Part of"),
    "TAGGED": ("keyword", 0.4, "About"),
    "IN_GENRE": ("genre", 0.5, "Genre:"),
    "IS_A": ("content_type", 0.0, "Is a"),
    "IN_DOMAIN": ("domain", 0.0, "In"),
}
RELATION_NAMES = tuple(RELATIONS)
_REL_CODE = {name: code for code, name in enumerate(RELATION_NAMES)}

MAX_CAST = 5
MIN_EXPLAINED_PATH = 0.05  # weaker user paths are scored but not put into words
MAX_QUERY_NGRAM = 4
# One word of a longer query links a person, brand, franchise or genre only if
# that entity has at least this many items, and a person, brand or franchise
# only if the word is in at most COMMON_TITLE_WORD titles ("man": 717, "day":
# 769, "house": 426; "drake": 27, "nolan": 3).
MIN_LINK_DEGREE = 3
COMMON_TITLE_WORD = 200
# Words naming a kind of item; within a longer query they are not entities.
KIND_WORDS = frozenset(
    "movie movies film films show shows series tv book books novel novels song songs music track tracks".split()
)
# Entity types a plural query word may reach through its singular.
_TOPIC_TYPES = frozenset(ENTITY_TYPES.index(t) for t in ("genre", "keyword"))
STOP_WORDS = frozenset(
    "a an and the of for to in on with by from at as is are be or not no my me i you your "
    "it its this that these those about into over under like some any all more most best new".split()
)
# Words that carry no topic, ignored when measuring how much of a query the
# linked entities cover.
COVERAGE_IGNORE = STOP_WORDS | KIND_WORDS | frozenset(
    "he she they we his her hers their them him who what which when where why how there here was were "
    "has have had do does did can could will would just also very really want watch read listen find "
    "something anything someone good great".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")

# Everyday words for the genres TMDb names formally. Without them "sci-fi" never
# reached Science Fiction and "romantic comedy" only reached a book subject,
# because TMDb files a rom-com under Romance and Comedy separately.
QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "sci fi": ("science fiction",),
    "scifi": ("science fiction",),
    "rom com": ("romance", "comedy"),
    "romcom": ("romance", "comedy"),
    "romantic comedy": ("romance", "comedy"),
    "romantic": ("romance",),
    "funny": ("comedy",),
    "comedies": ("comedy",),
    "scary": ("horror",),
    "animated": ("animation",),
    "cartoon": ("animation",),
    "cartoons": ("animation",),
    "kids": ("family",),
    "children": ("family",),
    "documentaries": ("documentary",),
    "thrillers": ("thriller",),
    "superheroes": ("superhero",),
    # TMDb tags the "same day again" plot two ways: Happy Death Day "time loop",
    # Groundhog Day and Edge of Tomorrow "deja vu".
    "relive": ("time loop", "deja vu"),
    "relives": ("time loop", "deja vu"),
    "reliving": ("time loop", "deja vu"),
    "relived": ("time loop", "deja vu"),
}


def _norm(text: Any) -> str:
    return " ".join(_TOKEN.findall(str(text or "").lower()))


def _singulars(phrase: str) -> list[str]:
    """Singular forms of a phrase's last word, most likely first: "zombies" -> "zombie"."""
    head, _, last = phrase.rpartition(" ")
    prefix = f"{head} " if head else ""
    if len(last) < 4 or not last.endswith("s") or last.endswith("ss"):
        return []
    forms = []
    if last.endswith("ies"):
        forms.append(last[:-3] + "y")          # "comedies" -> "comedy"
    if last.endswith("es"):
        forms.append(last[:-2])                # "witches" -> "witch"
    forms.append(last[:-1])                    # "balloons" -> "balloon", "houses" -> "house"
    return [prefix + form for form in forms]


@dataclass
class TypedKnowledgeGraph:
    item_ids: np.ndarray                 # global_id per item row
    item_titles: np.ndarray              # display title per item row
    entity_names: np.ndarray             # display name per entity
    entity_types: np.ndarray             # index into ENTITY_TYPES per entity
    weights: sparse.csr_matrix           # item x entity, relation weight x IDF
    relations: sparse.csr_matrix         # item x entity, relation code + 1
    name_index: dict[str, list[int]]     # normalised entity name -> entity ids (scored types only)
    item_index: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray = field(default_factory=lambda: np.zeros(0))         # rarity weight per entity
    degree: np.ndarray = field(default_factory=lambda: np.zeros(0))      # items per entity
    common_words: frozenset = field(default_factory=frozenset)           # words in many titles
    build_seconds: float = 0.0

    # ------------------------------------------------------------------ build
    @classmethod
    def from_catalog(cls, catalog: pd.DataFrame) -> "TypedKnowledgeGraph":
        started = time.perf_counter()
        frame = catalog.reset_index(drop=True)
        n_items = len(frame)
        content_type = frame.get("content_type", pd.Series("", index=frame.index)).fillna("").astype(str)
        source = frame.get("source", pd.Series("", index=frame.index)).fillna("").astype(str)

        parts = [
            _creator_edges(frame, content_type, source),
            _category_edges(frame, content_type),
            _structural_edges(frame, content_type),
        ]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return cls._empty(frame)
        edges = pd.concat(parts, ignore_index=True)
        if edges.empty:
            return cls._empty(frame)

        edges["key"] = edges["etype"] + ":" + edges["norm"]
        edges = edges[edges["norm"] != ""].drop_duplicates(subset=["row", "key"], keep="first")
        codes, keys = pd.factorize(edges["key"], sort=False)
        edges["entity"] = codes
        first = edges.drop_duplicates(subset=["entity"], keep="first").sort_values("entity")
        entity_names = first["name"].to_numpy(dtype=object)
        entity_types = first["etype"].map({t: i for i, t in enumerate(ENTITY_TYPES)}).to_numpy(dtype=np.int8)

        rows = edges["row"].to_numpy(dtype=np.int64)
        cols = edges["entity"].to_numpy(dtype=np.int64)
        rel_codes = edges["relation"].map(_REL_CODE).to_numpy(dtype=np.int64)
        rel_weight = np.array([RELATIONS[name][1] for name in RELATION_NAMES])[rel_codes]

        n_entities = len(keys)
        degree = np.bincount(cols, minlength=n_entities)
        idf = np.log1p(n_items / np.maximum(degree, 1))
        data = (rel_weight * idf[cols]).astype(np.float32)

        weights = sparse.csr_matrix((data, (rows, cols)), shape=(n_items, n_entities))
        relations = sparse.csr_matrix(((rel_codes + 1).astype(np.int8), (rows, cols)), shape=(n_items, n_entities))
        weights.eliminate_zeros()

        scored_types = {i for i, t in enumerate(ENTITY_TYPES) if t not in ("content_type", "domain")}
        name_index: dict[str, list[int]] = {}
        franchise = ENTITY_TYPES.index("franchise")
        for entity, (etype, norm) in enumerate(zip(entity_types, first["norm"].to_numpy())):
            if etype in scored_types and norm not in STOP_WORDS and len(norm) >= 2:
                name_index.setdefault(norm, []).append(entity)
                # TMDb names a franchise "Harry Potter Collection"; people search
                # "harry potter", which otherwise only reached a book subject.
                if etype == franchise and norm.endswith(" collection") and len(norm) > len(" collection") + 1:
                    name_index.setdefault(norm[: -len(" collection")], []).append(entity)

        graph = cls(
            item_ids=frame["global_id"].astype(str).to_numpy(dtype=object),
            item_titles=frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str).to_numpy(dtype=object),
            entity_names=entity_names,
            entity_types=entity_types,
            weights=weights,
            relations=relations,
            name_index=name_index,
        )
        graph.item_index = {gid: i for i, gid in enumerate(graph.item_ids)}
        graph.idf = idf
        graph.degree = degree
        title_words = Counter(
            word for title in graph.item_titles for word in set(_TOKEN.findall(str(title).lower()))
        )
        graph.common_words = frozenset(word for word, count in title_words.items() if count > COMMON_TITLE_WORD)
        graph.build_seconds = time.perf_counter() - started
        return graph

    @classmethod
    def _empty(cls, frame: pd.DataFrame) -> "TypedKnowledgeGraph":
        n = len(frame)
        empty = sparse.csr_matrix((n, 0), dtype=np.float32)
        ids = frame["global_id"].astype(str).to_numpy(dtype=object) if "global_id" in frame else np.array([], dtype=object)
        return cls(ids, np.array([""] * n, dtype=object), np.array([], dtype=object), np.array([], dtype=np.int8),
                   empty, empty.astype(np.int8), {}, {gid: i for i, gid in enumerate(ids)})

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        coo = self.relations.tocoo()
        rel_counts = np.bincount(coo.data.astype(np.int64) - 1, minlength=len(RELATION_NAMES))
        type_counts = np.bincount(self.entity_types.astype(np.int64), minlength=len(ENTITY_TYPES))
        return {
            "items": int(len(self.item_ids)),
            "entities": {t: int(c) for t, c in zip(ENTITY_TYPES, type_counts)},
            "nodes": int(len(self.item_ids) + len(self.entity_names)),
            "edges": {r: int(c) for r, c in zip(RELATION_NAMES, rel_counts)},
            "edge_total": int(coo.nnz),
            "build_seconds": round(self.build_seconds, 2),
        }

    # ------------------------------------------------------------- query links
    def link_query(self, query: str) -> list[int]:
        """Entity ids named in the query, longest match first (no overlapping tokens)."""
        return self._link(query)[0]

    def query_coverage(self, query: str) -> float:
        """Share of the query's meaningful words that the linked entities account for.

        "boy discovers he is a wizard" links only "wizard". Taking that as a full
        match gave every film tagged "wizard" the whole graph bonus and put
        Shazam! and Black Clover above Harry Potter, which TMDb never tagged
        "wizard". One word of three is a third of a match.
        """
        return self._link(query)[1]

    def _link(self, query: str) -> tuple[list[int], float]:
        tokens = _TOKEN.findall(str(query or "").lower())
        used = [False] * len(tokens)
        linked: list[int] = []
        for size in range(min(MAX_QUERY_NGRAM, len(tokens)), 0, -1):
            for start in range(0, len(tokens) - size + 1):
                if any(used[start:start + size]):
                    continue
                phrase = " ".join(tokens[start:start + size])
                if size == 1 and (phrase in STOP_WORDS or len(phrase) < 3):
                    continue
                if size == 1 and len(tokens) > 1 and phrase in KIND_WORDS:
                    # "nolan movies", "songs by drake": the kind of item asked
                    # for, not the book subject "Movies" or the genre "Songs".
                    continue
                hits = list(self.name_index.get(phrase, []))
                for target in QUERY_ALIASES.get(phrase, ()):
                    hits.extend(self.name_index.get(target, []))
                # TMDb keywords are singular ("balloon", "zombie", "robot");
                # people search in plurals. "balloons" only reached a book
                # subject of three children's books, never Up, tagged "balloon".
                for singular in _singulars(phrase):
                    found = [e for e in self.name_index.get(singular, []) if int(self.entity_types[e]) in _TOPIC_TYPES]
                    if found:
                        hits.extend(found)
                        break
                if size == 1 and len(tokens) > 1:
                    hits = [e for e in hits if self._links_as_word(e, phrase)]
                if hits:
                    linked.extend(hits)
                    for k in range(start, start + size):
                        used[k] = True
        meaningful = [k for k, token in enumerate(tokens) if token not in COVERAGE_IGNORE and len(token) >= 2]
        coverage = sum(used[k] for k in meaningful) / len(meaningful) if meaningful else 1.0
        return list(dict.fromkeys(linked)), coverage

    def _links_as_word(self, entity: int, word: str) -> bool:
        """Whether one word inside a longer query may stand for ``entity``.

        In "man relives the same day over and over", "man" and "day" linked an
        artist called Man and one called DAY; in "old man flies his house with
        balloons", "house" linked the one-film House Collection and "flies" a
        two-book subject. A common word is not a name, and an entity with one or
        two items is too thin to steer a sentence. TMDb keywords are curated
        topics and always link. A query that is only the word links as before.
        """
        etype = ENTITY_TYPES[self.entity_types[entity]]
        if etype == "keyword":
            return True
        degree = int(self.degree[entity]) if entity < len(self.degree) else MIN_LINK_DEGREE
        if degree < MIN_LINK_DEGREE:
            return False
        return not (etype in ("person", "brand", "franchise") and word in self.common_words)

    # ----------------------------------------------------------------- scoring
    # ------------------------------------------------------ candidate sources
    def _relations_by_entity(self) -> sparse.csc_matrix:
        cached = getattr(self, "_relations_csc", None)
        if cached is None:
            cached = self._relations_csc = self.relations.tocsc()
        return cached

    def _row_norms(self) -> np.ndarray:
        cached = getattr(self, "_weight_norms", None)
        if cached is None:
            cached = self._weight_norms = np.sqrt(np.asarray(self.weights.multiply(self.weights).sum(axis=1)).ravel())
        return cached

    def entity_candidates(
        self,
        query: str,
        limit: int,
        bonus: np.ndarray | None = None,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """Items connected to the entities a query names, strongest first.

        A query naming a keyword ("time travel", "heist") or a person reaches
        every item carrying that entity, however its synopsis is worded; the
        vector search only sees items whose text happens to read like the query.
        Ranked by the IDF-weighted share of the named entities, plus ``bonus``
        (per item row, e.g. prominence) to choose among a large tie. ``allowed``
        (per item row) restricts the candidates, e.g. to one decade.
        """
        linked = self.link_query(query)
        if not linked or limit <= 0:
            return []
        linked_arr = np.array(linked, dtype=np.int64)
        connected = (self._relations_by_entity()[:, linked_arr] != 0).astype(np.float64)
        total = float(self.idf[linked_arr].sum())
        if total <= 0:
            return []
        share = np.asarray(connected @ self.idf[linked_arr]).ravel() / total
        keep = share > 0
        if allowed is not None:
            keep &= allowed
        rows = np.flatnonzero(keep)
        if not rows.size:
            return []
        rank = share[rows] + (bonus[rows] if bonus is not None else 0.0)
        top = rows[np.argsort(-rank)[:limit]]
        return [(str(self.item_ids[r]), float(share[r])) for r in top]

    def query_share(self, query: str, candidate_ids: Iterable[str]) -> dict[str, float]:
        """The query-entity half of ``score`` alone, without explanations: cheap enough for a 450-item pool."""
        linked, coverage = self._link(query)
        if not linked:
            return {}
        rows = {gid: self.item_index[gid] for gid in candidate_ids if gid in self.item_index}
        if not rows:
            return {}
        linked_arr = np.array(linked, dtype=np.int64)
        total = float(self.idf[linked_arr].sum())
        if total <= 0:
            return {}
        block = self.relations[np.fromiter(rows.values(), dtype=np.int64)][:, linked_arr]
        share = coverage * np.asarray((block != 0).astype(np.float64) @ self.idf[linked_arr]).ravel() / total
        return {gid: float(value) for gid, value in zip(rows, share) if value > 0}

    def similar_items(self, seed_id: str, limit: int, bonus: np.ndarray | None = None) -> list[tuple[str, float]]:
        """Items sharing the most weighted entities with ``seed_id`` (cosine), strongest first.

        This is the 2-hop path seed -> entity <- candidate over the whole
        catalogue: same director, same cast, same franchise, same keywords.
        """
        row = self.item_index.get(seed_id)
        if row is None or limit <= 0 or self.weights.shape[1] == 0:
            return []
        norms = self._row_norms()
        if norms[row] == 0:
            return []
        dots = np.asarray((self.weights @ self.weights[row].T).todense()).ravel()
        cosine = dots / np.maximum(norms * norms[row], 1e-12)
        cosine[row] = 0.0
        rows = np.flatnonzero(cosine > 0)
        if not rows.size:
            return []
        rank = cosine[rows] + (bonus[rows] if bonus is not None else 0.0)
        top = rows[np.argsort(-rank)[:limit]]
        return [(str(self.item_ids[r]), float(cosine[r])) for r in top]

    def score(
        self,
        query: str,
        candidate_ids: Iterable[str],
        seed_ids: Iterable[str] = (),
        seed_role: str = "liked",
    ) -> dict[str, dict[str, Any]]:
        """kg_score in [0, 1] plus the best explaining path for every known candidate.

        ``seed_role`` words the explanation: "liked" for a user's liked items,
        "selected" when the seed is the item the user asked for more like.
        """
        rows = {gid: self.item_index[gid] for gid in candidate_ids if gid in self.item_index}
        if not rows or self.weights.shape[1] == 0:
            return {}
        candidate_rows = np.fromiter(rows.values(), dtype=np.int64)
        block = self.weights[candidate_rows]                       # candidates x entities
        rel_block = self.relations[candidate_rows]

        linked, coverage = self._link(query)
        query_scores = np.zeros(len(candidate_rows))
        if linked:
            linked_arr = np.array(linked, dtype=np.int64)
            # Share of the named entities (weighted by rarity) the item is connected to,
            # times the share of the query those entities account for.
            # Any credited role counts: a query naming a person matches all their work.
            connected = (rel_block[:, linked_arr] != 0).astype(np.float64)
            total = float(self.idf[linked_arr].sum())
            if total > 0:
                query_scores = coverage * np.asarray(connected @ self.idf[linked_arr]).ravel() / total

        seed_rows = np.array([self.item_index[s] for s in dict.fromkeys(seed_ids) if s in self.item_index], dtype=np.int64)
        path_scores = np.zeros(len(candidate_rows))
        profile = None
        seed_columns = None
        if seed_rows.size:
            seed_columns = self.weights[seed_rows].tocsc()  # seed x entity, for "which liked item shares it" 
            profile = np.asarray(self.weights[seed_rows].sum(axis=0)).ravel()
            profile_norm = float(np.linalg.norm(profile))
            if profile_norm > 0:
                dots = block @ profile
                cand_norms = np.sqrt(np.asarray(block.multiply(block).sum(axis=1)).ravel())
                path_scores = dots / np.maximum(cand_norms * profile_norm, 1e-12)

        out: dict[str, dict[str, Any]] = {}
        for pos, (gid, row) in enumerate(rows.items()):
            q, p = float(query_scores[pos]), float(path_scores[pos])
            # Average only the halves that apply: a query with no entity in it, or a
            # user with no likes, must not halve the other half.
            if profile is None:
                score = q
            elif not linked:
                score = p
            else:
                score = 0.5 * q + 0.5 * p
            path, text = self._explain(row, block[pos], rel_block[pos], linked, profile, seed_rows, seed_columns, q, p, seed_role)
            out[gid] = {"kg_score": round(min(1.0, score), 6), "kg_path": path, "kg_explanation": text}
        return out

    def _explain(self, row, weights_row, rel_row, linked, profile, seed_rows, seed_columns, q, p, seed_role="liked"):
        item_title = self.item_titles[row]
        cols = weights_row.indices
        if not len(cols):
            return None, None
        rel_of = dict(zip(rel_row.indices, rel_row.data))

        def phrase(entity: int) -> tuple[str, str]:
            rel_name = RELATION_NAMES[int(rel_of[entity]) - 1]
            return rel_name, f"{RELATIONS[rel_name][2]} {self.entity_names[entity]}"

        # Prefer the user path when it is the stronger half of the score.
        if profile is not None and p >= MIN_EXPLAINED_PATH and p >= q:
            contributions = weights_row.data * profile[cols]
            best = int(cols[int(np.argmax(contributions))])
            if profile[best] > 0:
                sharing = seed_rows[seed_columns[:, best].indices]
                seed = next((int(s) for s in sharing if int(s) != row), None)
                rel_name, text = phrase(best)
                path = [
                    {"node": item_title, "type": "item"},
                    {"edge": rel_name},
                    {"node": str(self.entity_names[best]), "type": ENTITY_TYPES[self.entity_types[best]]},
                ]
                if seed is not None:
                    seed_rel = RELATION_NAMES[int(self.relations[seed, best]) - 1]
                    if seed_role == "selected":
                        path += [{"edge": seed_rel}, {"node": self.item_titles[seed], "type": "item", "selected": True}]
                        text += f", like {self.item_titles[seed]}"
                    else:
                        path += [{"edge": seed_rel}, {"node": self.item_titles[seed], "type": "item", "liked": True}]
                        text += f", like {self.item_titles[seed]}, which you liked"
                return path, text
        hits = [e for e in linked if e in rel_of]
        if hits:
            best = max(hits, key=lambda e: float(weights_row[0, e]))
            rel_name, text = phrase(best)
            path = [
                {"node": item_title, "type": "item"},
                {"edge": rel_name},
                {"node": str(self.entity_names[best]), "type": ENTITY_TYPES[self.entity_types[best]], "in_query": True},
            ]
            return path, text + " (matches your search)"
        return None, None

    # ----------------------------------------------------------------- export
    def export_rows(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(nodes, edges) tables for loading the graph into a graph database."""
        n_items = len(self.item_ids)
        nodes = pd.concat(
            [
                pd.DataFrame({"id": self.item_ids, "label": "Item", "name": self.item_titles}),
                pd.DataFrame({
                    "id": [f"e{i}" for i in range(len(self.entity_names))],
                    "label": [ENTITY_TYPES[t].title().replace("_", "") for t in self.entity_types],
                    "name": self.entity_names,
                }),
            ],
            ignore_index=True,
        )
        coo = self.relations.tocoo()
        edges = pd.DataFrame({
            "source": self.item_ids[coo.row],
            "target": [f"e{c}" for c in coo.col],
            "type": [RELATION_NAMES[int(r) - 1] for r in coo.data],
        })
        assert len(nodes) == n_items + len(self.entity_names)
        return nodes, edges


# ---------------------------------------------------------------------- edges
def _split(series: pd.Series) -> pd.DataFrame:
    """Explode a comma-separated column into (row, position, name) rows."""
    exploded = series.fillna("").astype(str).str.split(",").explode()
    exploded = exploded.str.strip()
    out = pd.DataFrame({"row": exploded.index.to_numpy(), "name": exploded.to_numpy()})
    out = out[out["name"] != ""]
    out["pos"] = out.groupby("row").cumcount()
    out["norm"] = out["name"].map(_norm)
    return out


def _creator_edges(frame: pd.DataFrame, content_type: pd.Series, source: pd.Series) -> pd.DataFrame:
    if "creators" not in frame:
        return pd.DataFrame()
    edges = _split(frame["creators"])
    ctype = content_type.to_numpy()[edges["row"].to_numpy()]
    src = source.to_numpy()[edges["row"].to_numpy()]
    is_amazon = np.char.startswith(src.astype(str), "amazon")
    relation = np.select(
        [
            (ctype == "movie") & (edges["pos"].to_numpy() == 0),
            (ctype == "movie") & (edges["pos"].to_numpy() <= MAX_CAST),
            ctype == "music",
            is_amazon,
            ctype == "book",
            (ctype == "finance") & ~is_amazon,
        ],
        ["DIRECTED_BY", "STARS", "PERFORMED_BY", "MADE_BY", "WRITTEN_BY", "WRITTEN_BY"],
        default="",
    )
    edges["relation"] = relation
    edges = edges[edges["relation"] != ""]
    edges["etype"] = edges["relation"].map(lambda r: RELATIONS[r][0])
    return edges[["row", "name", "norm", "relation", "etype"]]


def _category_edges(frame: pd.DataFrame, content_type: pd.Series) -> pd.DataFrame:
    if "categories" not in frame:
        return pd.DataFrame()
    edges = _split(frame["categories"])
    ctype = content_type.to_numpy()[edges["row"].to_numpy()]
    names = edges["name"].astype(str)
    is_franchise = names.str.endswith(" Collection").to_numpy()
    # TMDb appends its plot keywords in lower case after the title-case genres.
    is_keyword = np.isin(ctype, ["movie", "show"]) & (names == names.str.lower()).to_numpy() & ~is_franchise
    edges["relation"] = np.where(is_franchise, "PART_OF", np.where(is_keyword, "TAGGED", "IN_GENRE"))
    edges["etype"] = edges["relation"].map(lambda r: RELATIONS[r][0])
    return edges[["row", "name", "norm", "relation", "etype"]]


def _structural_edges(frame: pd.DataFrame, content_type: pd.Series) -> pd.DataFrame:
    rows = np.arange(len(frame))
    parts = [pd.DataFrame({"row": rows, "name": content_type.to_numpy(), "relation": "IS_A", "etype": "content_type"})]
    if "domain" in frame:
        parts.append(pd.DataFrame({"row": rows, "name": frame["domain"].fillna("").astype(str).to_numpy(),
                                   "relation": "IN_DOMAIN", "etype": "domain"}))
    out = pd.concat(parts, ignore_index=True)
    out["norm"] = out["name"].map(_norm)
    return out[out["norm"] != ""][["row", "name", "norm", "relation", "etype"]]


def idf_of(degree: int, n_items: int) -> float:
    """The entity weight used above, exposed for tests and documentation."""
    return math.log1p(n_items / max(degree, 1))
