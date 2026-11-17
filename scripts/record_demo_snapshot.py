"""Record the hosted demo's snapshot from the current ranking engine.

    python -m scripts.record_demo_snapshot

The static demo (``npm run build:demo``, published on GitHub Pages) has no API
server: it answers from ``frontend/public/demo-snapshot.json``. This replays
every scope the previous snapshot recorded (each demo profile, query chip, domain
and age) plus the SHOWCASE searches through the real ranking code, and records
"more like this" for the top results so "Find similar" works on what is shown.

Qdrant is the one part swapped out: the embedded store allows a single process,
so vectors are searched exactly over the in-memory item matrix, behind the same
eligibility mask the Qdrant filter compiles to. Personalisation reads MySQL as
the API does, so the MySQL80 service must be running.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.audience import AudienceContext
from backend.domains import normalize_content_type
from backend.knowledge_graph import CatalogKnowledgeGraph
from backend.qdrant_recommender import QdrantRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT_ROOT / "frontend" / "public" / "demo-snapshot.json"

# Free-text searches worth trying on the hosted demo, recorded for every profile
# (as unpersonalised results) at no age and at 25.
SHOWCASE = [
    "man relives the same day over and over",
    "old man flies his house with balloons",
    "a rat who wants to be a chef in paris",
    "gangster in england with brothers who wears caps",
    "banker wrongly imprisoned escapes from prison",
    "a chemistry teacher with cancer starts making meth",
    "people in debt play deadly children's games for money",
    "a girl trapped in a spirit world bathhouse",
    "small time boxer gets a shot at the heavyweight title",
    "boy befriends an alien stranded on earth",
    "a lonely man falls in love with his computer operating system",
    "a ship hits an iceberg and sinks",
    "a man with memory loss uses tattoos to find his wife's killer",
    "a clownfish dad crosses the ocean to find his son",
    "dream manipulation",
    "stuck in a time loop",
    "90s movies",
    "sci-fi movies from the 80s",
    "songs from 2010",
    "horror movies after 2020",
    "books before 1950",
    "rom com",
    "zom com",
    "movies like inception",
    "movies like interstellar",
    "shows like breaking bad",
    "songs like shape of you",
    "books like the hunger games",
    "movies like dream manipulation",
    "the godfather",
    "christopher nolan",
    "nolan",
    "tom hanks",
    "stephen king",
    "inceptoin",
    "harry poter",
    "time travel",
    "zombies",
    "heist",
    "toy story",
]
SHOWCASE_AGES = (None, 25)
RECOMMEND_TOP_K = 20
SIMILAR_TOP_K = 10
SIMILAR_FOR_TOP = 3           # record "more like this" for this many results of every search
DESCRIPTION_CHARS = 700        # cards show a few lines; keeps the snapshot small to download

RESULT_FIELDS = (
    "global_id", "score", "match_kind", "semantic_score", "match_score", "popularity_score",
    "preference_bonus", "graph_score", "ema_score", "kg_score", "profile_score", "signal_weights",
    "similar_to", "kg_explanation", "kg_path",
)
ITEM_FIELDS = (
    "global_id", "content_type", "source", "source_id", "title", "description", "creators", "categories",
    "release_date", "popularity", "backdrop_url", "rating", "image_url", "domain", "maturity",
    "audience_min_age", "risk_tier",
)


class SnapshotRecommender(QdrantRecommender):
    """The live engine without the Qdrant client: exact search behind the same eligibility mask."""

    def __post_init__(self) -> None:
        self._assert_artifacts_exist()
        self.catalog = pd.read_csv(self.catalog_path, low_memory=False)
        self.embedding_index = pd.read_csv(self.embedding_index_path)
        self.client = None
        self.model = self._load_model()
        self.graph_store = self._load_graph_store()
        self.ema_store = self._load_ema_store()
        self.knowledge_graph = CatalogKnowledgeGraph(self.catalog)
        self._validate_artifacts()

    def _search_vector(self, vector, top_k, content_type, audience=None):
        store = self.ema_store
        store_ids = store.index["global_id"].astype(str).to_numpy()
        store_rows = store.index["embedding_row"].to_numpy(dtype=np.int64)
        catalog_rows = self._catalog_rows_of("store", store_ids)
        allowed = catalog_rows >= 0
        if audience is not None:
            allowed[allowed] &= self.eligible_mask(audience)[catalog_rows[allowed]]
        normalized = normalize_content_type(content_type)
        if normalized is not None:
            types = self.catalog["content_type"].astype(str).to_numpy()
            allowed[allowed] &= types[catalog_rows[allowed]] == normalized
        picked = np.flatnonzero(allowed)
        if not picked.size:
            return []
        scores = store.embeddings[store_rows[picked]] @ np.asarray(vector, dtype=np.float32)
        take = min(top_k, len(scores))
        order = np.argpartition(-scores, take - 1)[:take]
        order = order[np.argsort(-scores[order])]
        ids = [str(store_ids[picked[i]]) for i in order]
        cosine = {str(store_ids[picked[i]]): float(scores[i]) for i in order}
        items = self._catalog_items(ids, content_type, audience, top_k)
        for item in items:
            item["score"] = cosine[str(item["global_id"])]
        return items


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return None if not np.isfinite(value) else round(value, 4)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _clean(float(value))
    return value


def _slim(result: dict[str, Any]) -> dict[str, Any]:
    return {field: _clean(result[field]) for field in RESULT_FIELDS if result.get(field) is not None}


def _age(text: str) -> int | None:
    return int(text) if text not in ("", None) else None


def main() -> None:
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    started = time.perf_counter()
    rec = SnapshotRecommender()
    rec.warm_up()
    print(f"engine ready in {time.perf_counter() - started:.0f}s")

    recommend_keys = list(old["recommend"])
    for query in SHOWCASE:
        for age in SHOWCASE_AGES:
            key = f"*|{query}||{'' if age is None else age}"
            if key not in old["recommend"]:
                recommend_keys.append(key)

    recommend: dict[str, Any] = {}
    similar_wanted: dict[str, None] = dict.fromkeys(old["similar"])
    shown: set[str] = set()
    for number, key in enumerate(recommend_keys, 1):
        user, query, domain, age = key.split("|")
        audience = AudienceContext(age=_age(age), domain=domain or None)
        corrected = rec.correct_query(query)
        results = rec.recommend(
            corrected or query,
            top_k=RECOMMEND_TOP_K,
            user_id=None if user == "*" else user,
            audience=audience,
        )
        entry: dict[str, Any] = {"advisory": audience.advisory(), "results": [_slim(r) for r in results]}
        if corrected:
            entry["corrected_query"] = corrected
        notice = rec.hidden_title_notice(corrected or query, None, audience, results)
        if notice:
            entry["notice"] = notice
        recommend[key] = entry
        shown.update(str(r["global_id"]) for r in results)
        for r in results[:SIMILAR_FOR_TOP]:
            similar_wanted.setdefault(f"{r['global_id']}|{age}", None)
        if number % 25 == 0:
            print(f"  searches {number}/{len(recommend_keys)}")

    similar: dict[str, Any] = {}
    keys = list(similar_wanted)
    for number, key in enumerate(keys, 1):
        global_id, age = key.rsplit("|", 1)
        if global_id not in rec._catalog_position():
            continue
        audience = AudienceContext(age=_age(age))
        results = rec.recommend_from_item(global_id, top_k=SIMILAR_TOP_K, audience=audience)
        similar[key] = {"advisory": None, "results": [_slim(r) for r in results]}
        shown.update(str(r["global_id"]) for r in results)
        shown.add(global_id)
        if number % 50 == 0:
            print(f"  more-like-this {number}/{len(keys)}")

    positions = rec._catalog_position()
    items = {}
    for global_id in sorted(shown):
        position = positions.get(global_id)
        if position is None:
            continue
        row = rec.catalog.iloc[position]
        item = {field: _clean(row.get(field)) for field in ITEM_FIELDS}
        item = {k: (None if isinstance(v, float) and v != v else v) for k, v in item.items()}
        if isinstance(item.get("description"), str) and len(item["description"]) > DESCRIPTION_CHARS:
            item["description"] = item["description"][: DESCRIPTION_CHARS - 1].rstrip() + "…"
        items[global_id] = item

    snapshot = {
        "health": old.get("health", {"status": "ok"}),
        "domains": old.get("domains", {}),
        "recommend": recommend,
        "similar": similar,
        "items": items,
    }
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size = SNAPSHOT.stat().st_size / 1e6
    print(
        f"wrote {SNAPSHOT.relative_to(PROJECT_ROOT)}: {len(recommend)} searches, {len(similar)} more-like-this, "
        f"{len(items)} items, {size:.1f} MB in {time.perf_counter() - started:.0f}s"
    )


if __name__ == "__main__":
    main()
