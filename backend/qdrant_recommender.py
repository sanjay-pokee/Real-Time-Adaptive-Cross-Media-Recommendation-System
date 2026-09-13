"""Qdrant-backed semantic recommendation search.

Qdrant is the production vector backend. Build the collection with:
    python -m embeddings.build_qdrant_collection
"""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.audience import AudienceContext, build_eligibility_filter, is_eligible
from backend.domains import normalize_content_type
from backend.ema_recommender import EMAEmbeddingStore
from backend.graph_recommender import GraphEmbeddingStore
from backend.knowledge_graph import CatalogKnowledgeGraph
from backend.mysql_store import MySQLStore
from backend.settings import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
EMBEDDING_INDEX_PATH = PROJECT_ROOT / "embeddings" / "content_embedding_index.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


@dataclass
class QdrantRecommender:
    catalog_path: Path = CATALOG_PATH
    embedding_index_path: Path = EMBEDDING_INDEX_PATH
    embedding_model: str = EMBEDDING_MODEL
    settings: Settings = get_settings()

    def __post_init__(self) -> None:
        self._assert_artifacts_exist()
        self.catalog = pd.read_csv(self.catalog_path)
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

        # Retrieval is hybrid: lexical first, then semantic fills the rest.
        #
        # A pure vector search cannot find a title. "F1" embeds to a vector with
        # essentially no intent in it, so its nearest neighbours were a Punjabi
        # song and an aromatherapy necklace while the film sat in the catalogue.
        # Exact and keyword matches are found directly and placed above the
        # semantic band, which is also what a user typing a title expects: the
        # thing they named, not something that reads like it.
        person_results = self._search_person_matches(
            query,
            top_k=min(5, top_k),
            content_type=content_type,
            audience=audience,
        )
        lexical_results = self._search_lexical_matches(
            query,
            top_k=min(8, top_k),
            content_type=content_type,
            audience=audience,
        )
        # A creator hit and a title hit can be the same row; the creator match is
        # the more specific claim, so it wins.
        person_ids = {item["global_id"] for item in person_results}
        lexical_results = [
            item for item in lexical_results if item["global_id"] not in person_ids
        ]
        exact_ids = person_ids | {item["global_id"] for item in lexical_results}

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        search_k = max(top_k * 5 if user_id else top_k, top_k + len(exact_ids))
        results = self._search_vector(query_vector, search_k, content_type, audience)
        if exact_ids:
            results = [item for item in results if item["global_id"] not in exact_ids]
            results = person_results + lexical_results + results
        results = self._personalize_results(results, search_k, user_id)
        results = self._graph_rerank(results, user_id)
        results = self._ema_rerank(results, user_id)
        results = self._knowledge_graph_rerank(query, results, user_id)
        return results[:top_k]

    def recommend_from_item(
        self,
        global_id: str,
        top_k: int = 10,
        content_type: str | None = None,
        user_id: str | None = None,
        audience: AudienceContext | None = None,
    ) -> list[dict[str, Any]]:
        if not global_id.strip():
            raise ValueError("global_id cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        catalog_matches = self.catalog[self.catalog["global_id"] == global_id]
        if catalog_matches.empty:
            raise ValueError(f"Unknown global_id: {global_id}")

        source_text = str(catalog_matches.iloc[0].get("embedding_text", ""))
        vector = self.model.encode(
            [source_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        search_k = (top_k * 5 if user_id else top_k) + 1
        results = self._search_vector(vector, search_k, content_type, audience)
        results = [item for item in results if item["global_id"] != global_id]
        results = self._personalize_results(results, search_k, user_id)
        results = self._graph_rerank(results, user_id)
        results = self._ema_rerank(results, user_id)
        results = self._knowledge_graph_rerank(source_text, results, user_id)
        return results[:top_k]

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
        exact = haystack.str.contains(f"\n{q_norm}\n", regex=False, na=False)
        prefix = haystack.str.contains(f"\n{q_norm}", regex=False, na=False)
        substring = haystack.str.contains(q_norm, regex=False, na=False)

        ranks = pd.Series(3, index=catalog.index)
        ranks = ranks.mask(substring, 2)
        ranks = ranks.mask(prefix, 1)
        ranks = ranks.mask(exact, 0)
        matched = ranks < 3
        if not matched.any():
            return []

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
            rows.append((
                int(match_rank),
                -(popularity + rating),
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

        haystack = self.catalog["creators"].map(
            lambda value: (
                "\n" + "\n".join(
                    _normalize_lookup_text(name) for name in _split_creators(value)
                ) + "\n"
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
        popularity_rank = self._popularity_rank()
        catalog = self.catalog

        if normalized_content_type is not None:
            type_mask = catalog["content_type"] == normalized_content_type
            catalog = catalog[type_mask]
            titles = titles[type_mask]
            keywords = keywords[type_mask]
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
            all_tokens = pd.Series(True, index=catalog.index)
            for token in tokens:
                # Space-padded, so this matches a whole word rather than any
                # occurrence of the characters inside a longer one.
                all_tokens &= keywords.str.contains(f" {token} ", regex=False, na=False)
        else:
            all_tokens = pd.Series(False, index=catalog.index)

        ranks = pd.Series(9, index=catalog.index)
        ranks = ranks.mask(all_tokens, 3)
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
        order = pd.DataFrame(
            {"rank": candidate_ranks, "pop": candidate_pop}
        ).sort_values(["rank", "pop"], ascending=[True, False])
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
            bonus = 0.0

            for category in _split_categories(item.get("categories")):
                bonus += min(category_weights.get(category, 0.0), 3.0) * 0.035

            content_type = str(item.get("content_type", "")).lower()
            bonus += min(content_type_weights.get(content_type, 0.0), 3.0) * 0.02

            if item.get("global_id") in positive_ids:
                bonus -= 0.15
            if item.get("global_id") in negative_ids:
                bonus -= 0.35

            personalized = dict(item)
            personalized["score"] = score + bonus
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
    ) -> list[dict[str, Any]]:
        if not hasattr(self, "knowledge_graph"):
            self.knowledge_graph = CatalogKnowledgeGraph(self.catalog)

        user_profile: dict[str, Any] = {}
        if user_id:
            try:
                user_profile = MySQLStore(self.settings).get_user_profile_preferences(user_id)
            except Exception:
                user_profile = {}
        return self.knowledge_graph.rerank(query, results, user_profile=user_profile)

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
