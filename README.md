# Real-Time-Adaptive-Cross-Media-Recommendation-System
This project aims to build a cross media recommendation system which adapts in real time according to user's taste of media selection , using semantic embeddings and graph based collaborative learning to understand the user's taste and provide precise recommendations in an unified platform (music + movies + songs)
Takes the description from the data set ---> Sentence Bert ----> Match

## Backend API

Start the recommendation API:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.app:app --reload
```

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/recommend `
  -ContentType "application/json" `
  -Body '{"query":"space adventure with aliens","top_k":5,"content_type":"movie"}'
```

## Qdrant + MySQL Backend Setup

The backend now uses Qdrant for vector search instead of FAISS.

Install the new Python dependency:

```powershell
pip install -r requirements.txt
```

Run Qdrant locally with Docker:

```powershell
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Set optional environment variables if you do not want the defaults:

```powershell
$env:QDRANT_URL="http://127.0.0.1:6333"
$env:QDRANT_COLLECTION="content"
```

Build the recommendation artifacts and upload vectors to Qdrant:

```powershell
python -m preprocessing.build_content_catalog
python -m embeddings.build_embeddings
python -m embeddings.build_qdrant_collection
```

For MySQL interaction logging, set:

```powershell
$env:MYSQL_HOST="127.0.0.1"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_DATABASE="cross_media_recs"
```

Create the database and tables automatically:

```powershell
python -m scripts.init_mysql
```

You do not need to manually create tables. You only need a running MySQL server and a user with permission to create the configured database.

Movies are fetched from the live TMDb API rather than a static dump. Set `TMDB_API_KEY`
in `.env` (the v3 API key, not the v4 Read Access Token), then:

```powershell
python -m scripts.fetch_tmdb_movies --min-votes 100 --from-year 1960
python -m scripts.fetch_tmdb_credits
python -m scripts.fetch_tmdb_certifications
```

`fetch_tmdb_movies` pulls from `/discover/movie`, which returns the overview and the poster
path together. It slices requests by release year because `/discover` caps pagination at 500
pages, and `--min-votes` drops the long tail of shorts and unreleased entries, which embed to
noise. `fetch_tmdb_credits` then adds cast and director, which `/discover` does not return.

`fetch_tmdb_certifications` reads `/movie/{id}/release_dates` for the real US board rating
(G / PG / PG-13 / R / NC-17), which is what drives movie maturity — see
[Audience eligibility](#audience-eligibility). Certification is only exposed per country and
per release window, so it needs its own lookup. All three scripts cache incrementally by movie
id, so an interrupted run resumes and a re-run only fetches what is missing; the certification
fetch is concurrent (`--workers`, default 8) because serial throughput is bounded by round-trip
latency and a 22k backlog takes hours at one request in flight.

`backend/api_ingestion.py` contains unused TMDb, Open Library and MusicBrainz clients. Nothing
imports it; it is scaffolding from an earlier approach and is not part of the pipeline.

Load the processed catalog into MySQL after building it:

```powershell
python -m scripts.load_catalog_mysql
```

This command creates the database/tables if needed, then upserts all rows from `data/processed/content_catalog.csv` into `content_entities`.

Seed demo users for personalization and future LightGCN experiments:

```powershell
python -m scripts.seed_demo_interactions
```

Then call `/recommend` with a `user_id` to rerank results using that user's interaction history:

```json
{
  "query": "space adventure",
  "user_id": "user_scifi",
  "top_k": 5,
  "content_type": null
}
```

Demo users include `user_scifi`, `user_fantasy`, `user_romance`, `user_action`, `user_music_pop`, `user_music_rock`, `user_books_learning`, `user_family`, `user_dark_thriller`, and `user_balanced`.

## LightGCN Collaborative Filtering

LightGCN is implemented in PyTorch under `models/graph`. It trains from the MySQL `user_interactions` table and exports user/item embeddings that the backend can use for hybrid reranking.

Train LightGCN after MySQL has catalog rows and interactions:

```powershell
python -m scripts.load_catalog_mysql
python -m scripts.seed_demo_interactions
python -m models.graph.train_lightgcn --epochs 50
```

The trainer writes:

```text
models/graph/artifacts/lightgcn_embeddings.npz
models/graph/artifacts/lightgcn_embeddings.json
```

After that, restart `uvicorn`. Calls to `/recommend` with a `user_id` will use semantic Qdrant search, the existing interaction-profile reranker, and the LightGCN graph score when the user and item exist in the trained artifact.

Evaluate LightGCN with holdout metrics:

```powershell
python -m models.graph.evaluate_lightgcn --epochs 20 --k 10
```

The evaluator hides recent positive interactions per user, trains on the remaining interactions, then reports `HitRate@K`, `Recall@K`, `Precision@K`, `NDCG@K`, and `MRR@K`.

## EMA Real-Time Personalization

The backend updates a user EMA vector whenever `/interactions` logs a meaningful event. Positive events such as `like`, `bookmark`, `complete`, and high `rating` pull the user vector toward the content embedding. Negative events such as `skip` push it away.

Useful tuning variables:

```powershell
$env:EMA_ALPHA="0.25"
$env:EMA_WEIGHT="0.15"
```

Backfill EMA vectors from existing interactions:

```powershell
python -m scripts.rebuild_ema_profiles
```

After an interaction is logged, later `/recommend` calls with the same `user_id` include `ema_score` and use it for instant reranking. This works immediately without retraining LightGCN.



## Multi-Domain Engine & Audience Constraints

The retrieval stack (SBERT -> Qdrant -> rerank) is domain-agnostic: it only ever sees text and a `content_type` payload field. Everything vertical-specific lives in `config/domains.yaml`, so adding a domain is a config change, not a code change.

Shipped domains:

| Domain | Content types | Items | Risk tier |
| --- | --- | --- | --- |
| `entertainment` | `movie`, `book`, `music` | 65,607 | 0 (informational) |
| `health` | `health` | 20,000 | 1 (advisory shown) |
| `industry` | `industrial` | 20,000 | 0 |
| `finance` | `finance` | 725 | 1 (advisory shown) |

106,332 items in total. Finance is smaller because it is not one of Amazon's 28 top-level
categories: it is carved out of the Software category's "Accounting & Finance" subtree, and
725 is what is genuinely there rather than what padding with office software would give.

Health and finance are scoped to **information and product discovery**. The engine does not
give medical, diagnostic or investment advice, and `risk_tier: 2` items (those needing a
licensed professional) are withheld by default.

List what a deployment serves:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/domains
```

### Audience eligibility

Recommendation runs in two stages: eligibility, then relevance. Age, safe mode, domain scope and risk tier are compiled into the Qdrant query itself, so an ineligible item is never retrieved, never scored, and cannot be promoted back by the graph, EMA or KG rerankers.

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/recommend `
  -ContentType "application/json" `
  -Body '{"query":"fun adventure","top_k":5,"age":8}'
```

Policy, all of which fails closed:

- A request with **no age** is not treated as an adult; it is capped at the teen ceiling.
- `safe_mode: true` caps below a real adult age, so an adult can ask for family-appropriate results.
- A catalog row with no audience metadata falls back to its content type's default, not to unrestricted.
- A maturity label that is **present but unrecognised** — a `NaN` read back from a CSV stringifies to `"nan"` — resolves to the *most* restrictive level, not to 0. Corrupt data fails closed like everything else here.

Requests that declare no audience (no `age`, `safe_mode` or `domain`) behave exactly as before.

Audience metadata (`domain`, `maturity`, `audience_min_age`, `risk_tier`) is derived in `preprocessing/audience_tagging.py` after normalization.

**Where the maturity comes from depends on the content type.** Movies declare
`maturity_source: certification` in `config/domains.yaml` and take the real TMDb US board rating
fetched by `scripts.fetch_tmdb_certifications`, mapped G → `all_ages`, PG → `child`,
PG-13 → `teen`, R → `adult`, NC-17 → `restricted`. `NR`/`Unrated` means *no board rated the
title*, not that it is harmless, so those fall through to the keyword heuristic.

For a content type whose ratings come from a board, a category keyword may **restrict** a row but
never **relax** one. Genre-only inference rated *Akira*, *Heavy Metal*, *Grave of the Fireflies*
and *Waltz with Bashir* as `all_ages`, because `Animation` matched the child-friendly rule — a
production technique read as an audience. Books and music have no certification source, so their
category text is still the best available signal and may relax a row.

Everything outside a real certification remains a **heuristic, not a certified rating**; an
explicit rating already present on a row is always preserved. Rebuild the catalog and collection
after pulling this:

```powershell
python -m preprocessing.build_content_catalog
python -m embeddings.build_embeddings
python -m embeddings.build_qdrant_collection
```

### Evaluation

Constraint enforcement and what it costs:

```powershell
python -m scripts.evaluate_constraints --k 20
```

Reports violation rate@K with the filter on (should be exactly 0) against the same viewer with it off, plus catalog coverage and overlap@K. Exits non-zero if any ineligible item reaches a viewer. Writes `reports/constraint_evaluation.{md,csv}`.

Cross-domain benchmark table:

```powershell
python -m scripts.build_benchmark_table
```

Each Kaggle run writes a JSON via `evaluate_lightgcn --report`. Drop them in `reports/benchmarks/` and this emits `reports/domain_benchmark_table.md` and `.tex`, the latter for direct `\input` into the deck.
