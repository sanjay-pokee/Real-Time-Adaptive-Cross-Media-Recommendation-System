# Real-Time Adaptive Cross-Media Recommendation System

A domain-agnostic recommendation engine over **120,747 items**:
- **Entertainment:** movies, TV shows, books, music
- **Products:** health, industrial and finance

The user types what they want, and the engine:
1. retrieves candidates by meaning (Sentence-BERT + Qdrant) and by keyword;
2. removes anything the viewer may not see (age, safe mode, domain, risk tier) **inside the vector query**, before anything is scored;
3. reranks what survives with collaborative filtering (LightGCN), a live taste vector (EMA) and profile signals.

Liking an item changes the next search immediately, with no retraining.

**Stack:** Python 3.11 · FastAPI · PyTorch · Sentence-BERT · Qdrant · MySQL · React 19 + Vite + Tailwind.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Data and catalog](#data-and-catalog)
- [Retrieval](#retrieval)
- [Audience eligibility](#audience-eligibility)
- [Personalization and ranking](#personalization-and-ranking)
- [Evaluation](#evaluation)
- [API](#api)
- [Frontend](#frontend)
- [Setup](#setup)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Known limitations and roadmap](#known-limitations-and-roadmap)

---

## Highlights

| | |
| --- | --- |
| Catalog | 120,747 items × 20 columns, 7 content types, 4 domains |
| Embeddings | `all-MiniLM-L6-v2`, 384-d, L2-normalized (~185 MB for the whole catalog) |
| Vector store | Qdrant, cosine distance, eligibility filter applied inside the search |
| Collaborative filtering | LightGCN in PyTorch (d=64, 3 layers, BPR loss) |
| LightGCN vs popularity, Recall@20 | **1.59x** on Movies & TV, **1.24x** on Industrial & Scientific, 0.55x on Health & Household (Amazon Reviews 2023) |
| Real-time adaptation | EMA taste vector (α = 0.25), updated on every interaction, no retraining |
| Explainability | Per-signal score breakdown plus counterfactual re-ranking in the UI |
| Tests | 328 passing (`pytest`) |

---

## Architecture

```
                 ┌─────────────────────────── offline ────────────────────────────┐
 TMDb · Open Library · Last.fm · Spotify CSV · Amazon Reviews 2023
        │  scripts/fetch_*.py
        ▼
 preprocessing/build_content_catalog.py ──► content_catalog.csv (120,747 × 20)
        │  normalize · audience tagging · cover art
        ├──► embeddings/build_embeddings.py ──► 384-d SBERT matrix (incremental, hash-cached)
        │        └──► embeddings/build_qdrant_collection.py ──► Qdrant collection + payload
        └──► scripts/load_catalog_mysql.py ──► MySQL content_entities
 Amazon Reviews 2023 interactions ──► models/graph (LightGCN, Kaggle GPU) ──► user/item embeddings
                 └────────────────────────────────────────────────────────────────┘

 POST /recommend {query, user_id, age, safe_mode, domain, content_type, top_k}
   1. AudienceContext   always built; no age → teen ceiling, safe_mode → child ceiling
   2. Creator match     cast / director / author / artist names        (eligibility re-checked)
   3. Lexical match     title and keyword tiers, ≤ half the page       (eligibility re-checked)
   4. Vector search     SBERT query → Qdrant search WITH the eligibility filter
   5. Rerank            profile bonus → LightGCN → EMA → KG overlap + profile match
   6. top_k results (+ advisory for regulated domains)

 POST /interactions ──► MySQL event log ──► EMA vector updated in the same request
```

The pieces:
- **Backend:** FastAPI.
- **Model and catalog loading:** heavy objects (the SBERT model, the catalog and the Qdrant client) are created once in the app's `lifespan` hook.
- **Qdrant:** runs embedded by default (`QDRANT_PATH`), or as a server (`QDRANT_URL`).
- **MySQL:** stores the catalog mirror, user profiles and the interaction log.

---

## Data and catalog

### Sources

| Content type | Domain | Source | Items | Notes |
| --- | --- | --- | --- | --- |
| `movie` | entertainment | TMDb `/discover/movie` (live API) | 23,492 | ≥ 100 votes, from 1960. Credits, US certifications, backdrops and franchise keywords are fetched separately. |
| `show` | entertainment | TMDb `/discover/tv` (live API) | 10,617 | ≥ 25 votes (TV accrues far fewer votes) |
| `book` | entertainment | Open Library search API | 12,402 | 60 subjects, `sort=rating`, description and cover required |
| `music` | entertainment | Spotify songs CSV + Last.fm API | 31,133 | Spotify stops at 2020; Last.fm adds current music; adult-listing filter on ingest |
| `health` | health | Amazon Reviews 2023 metadata | 20,000 | Health & Household |
| `industrial` | industry | Amazon Reviews 2023 metadata | 20,000 | Industrial & Scientific |
| `finance` | finance | Amazon Software › Accounting & Finance + Open Library finance books | 3,103 | Finance is not an Amazon top-level category |

Why these sources:
- **Movies:** the live TMDb API replaced a frozen 2017 Kaggle dump of 4,803 movies.
- **Books:** Open Library replaced a Google Books scrape where only 55% of rows had a description and the top-ranked rows were SEO spam. A row with no description embeds to noise.
- **Recent books:** `sort=rating` under-ranks recent releases, because ratings take years to accumulate. Recent years are topped up with a year window and `--merge`, which keeps existing IDs stable.

### Unified schema

`preprocessing/build_content_catalog.py` sends each dataset in `config/datasets.yaml` to a normalizer named in that config. The three Amazon verticals share one normalizer, so adding another Amazon vertical is a config entry, not code.

| Column group | Columns |
| --- | --- |
| Identity | `global_id` (`content_type:source:source_id`), `content_type`, `source`, `source_id` |
| Content | `title`, `description`, `creators`, `categories`, `release_date`, `popularity`, `rating`, `image_url`, `backdrop_url` |
| Text | `metadata_text` (everything), `embedding_text` (curated per type), `text_hash` (SHA-256 of `embedding_text`) |
| Audience | `domain`, `maturity`, `audience_min_age`, `risk_tier` |

`embedding_text` is curated per type:
- **Movie:** title + categories + description + director + top-5 cast.
- **Book:** title + authors + categories + description.
- **Music:** title + artist + genre + subgenre.
- **Amazon product:** title + brand + categories + description.

TMDb franchise names and keywords are appended to `categories`, so a keyword search for "avengers" reaches *Iron Man* and *Black Panther*.

### Embeddings and vector store

- **Encoding:** `embeddings/build_embeddings.py` encodes `embedding_text` in batches of 256, on GPU when available. The embedding index is keyed on (`global_id`, model, version, `text_hash`), so a rebuild only re-encodes rows whose text changed.
- **Upload:** `embeddings/build_qdrant_collection.py` uploads every vector with a deterministic `uuid5(global_id)` point ID.
- **Payload:** each point carries the display fields plus the audience fields, stored as integers so Range filters work on them.
- **Rebuilds:** in embedded mode, a rebuild deletes the collection *and* purges its directory. Otherwise the old points would be re-attached.

### Database

`backend/mysql_store.py` creates three tables and runs idempotent migrations on every init.

| Table | Purpose | Notes |
| --- | --- | --- |
| `content_entities` | Catalog mirror | Primary key `global_id`, `utf8mb4_bin` (IDs are case-sensitive); `MEDIUMTEXT` descriptions |
| `user_profiles` | Preferences JSON and EMA vector JSON | Upserted by the EMA update and the seed script |
| `user_interactions` | Append-only event log | Index on `(user_id, timestamp)`; foreign key to the catalog with `ON DELETE CASCADE` |

The catalog loader:
- upserts in chunks of 2,000 rows, to stay under `max_allowed_packet`;
- then prunes rows that are no longer in the catalog.

Interaction toggles (like, bookmark, …) are read back by parity: an odd number of events means the toggle is active.

---

## Retrieval

Retrieval is hybrid, because each method fails where the other works.

- **Semantic:** the query is encoded with the same SBERT model and searched in Qdrant. This answers intent ("space adventure with aliens").
- **Creator:** cast, director, author and artist names are matched as whole words, in three tiers (whole name, prefix, word). This answers "Christopher Nolan".
- **Lexical:** five tiers:
  1. exact title
  2. title prefix
  3. title contains the query
  4. every token appears in the title or categories
  5. the phrase appears in the first 1,000 characters of the description

  This answers titles that carry no semantic signal: a pure vector search for "F1" returned a Punjabi song while the film sat in the catalog.

How the results are merged:
- **Score band:** creator and lexical hits get fixed scores just above the cosine range, so an exact title outranks a semantic neighbour.
- **Page share:** lexical hits are capped at half the page, so a common word can't bury the semantic results.
- **Ordering:** below the exact and prefix tiers, matches are ordered by popularity percentile *within their content type*. Raw popularity isn't comparable across sources: Amazon review counts reach ~294k, TMDb popularity ~800, Spotify 0–100.

Every match is whole-word: haystacks are space-padded and tokenized, so "inception" does not match the brand "Skinception". The haystacks are built once with vectorized pandas operations. A per-row version took 15–49 s per request at this catalog size.

---

## Audience eligibility

Recommendation runs in two stages: **eligibility, then relevance**.

The constraints (age, safe mode, domain, risk tier) are compiled into the Qdrant query as a `Filter(must=[...])`. So an ineligible item is never retrieved and never scored, and the graph, EMA or profile rerankers cannot promote it back.

The creator and lexical paths read the catalog directly, so they apply the same policy through `backend.audience.is_eligible`.

The domain and maturity scheme comes from `config/domains.yaml`.

**Maturity levels**

| Level | Minimum age |
| --- | --- |
| all_ages | 0 |
| child | 7 |
| teen | 13 |
| young_adult | 16 |
| adult | 18 |
| restricted | 21 |

**Domains**

| Domain | Content types | Risk tier | Advisory |
| --- | --- | --- | --- |
| entertainment | movie, show, book, music | 0 | — |
| industry | industrial | 0 | — |
| health | health | 1 | "Not medical advice" |
| finance | finance | 1 | "Not financial advice" |

Health and finance are scoped to information and product discovery. The engine gives no medical, diagnostic or investment advice, and risk-tier-2 items are withheld by default.

**Policy — every unknown fails closed:**
- A request with **no age** is capped at the teen ceiling, not treated as an adult.
- `safe_mode` caps at the child ceiling, even for an adult viewer.
- A row with no maturity falls back to its content type's default, not to "unrestricted".
- An unrecognized label (for example, a CSV `NaN` read back as `"nan"`) resolves to the **most** restrictive level.
- An unknown content type is tagged `adult`, and an unknown risk tier is treated as 1.

**Where the labels come from** (`preprocessing/audience_tagging.py`):
- **Movies** use the real TMDb US certification:

  | Certification | Maturity |
  | --- | --- |
  | G | all_ages |
  | PG | child |
  | PG-13 | teen |
  | R | adult |
  | NC-17 | restricted |

  TV ratings are mapped the same way. `NR`/Unrated means no board rated the title, so it falls through to the heuristic.
- **Everything else** uses keyword rules over categories and title:
  - ordered most-restrictive first;
  - matched at a left word boundary;
  - every candidate keyword was measured against all titles for false positives before it was added.
- **For board-rated types and regulated domains**, a keyword may *restrict* an item but never *relax* it. Genre-only inference once rated *Akira* and *Grave of the Fireflies* `all_ages`, because "Animation" matched the child-friendly rule.

Book and music labels remain heuristics, because no certification source exists for them.

---

## Personalization and ranking

Scores are fused additively, in this order:

| Signal | Computation | Weight |
| --- | --- | --- |
| Retrieval | Cosine for vector hits; fixed tier scores for creator/lexical hits | 1.0 |
| Profile bonus | From the user's last 500 events: category and content-type affinity; already-liked items −0.15, skipped −0.35 | added directly |
| LightGCN | cosine(user embedding, item embedding) | 0.20 (`LIGHTGCN_WEIGHT`) |
| EMA | cosine(user taste vector, item embedding) | 0.15 (`EMA_WEIGHT`) |
| KG overlap | Shared query/item terms over title, description, creators and categories | 0.08 |
| Profile match | Interests, skills, profession and age-group terms from `user_profiles.preferences` | 0.10 |

### LightGCN (`models/graph/`)

- **Structure:** users and items are nodes and positive interactions are edges. Propagation uses the symmetric normalized adjacency D^-½AD^-½.
- **Model:**
  - The only parameters are the embedding tables (d = 64).
  - There are no feature transforms or nonlinearities.
  - There are 3 propagation layers, and the final embedding is the mean of layers 0–3.
- **Training:** BPR loss with uniform negative sampling, vectorized as sorted int64 keys + `searchsorted`. Adam, lr 1e-3, L2 1e-4.
- **Interaction weights:** view 0.4, click 0.6, like/bookmark 1.0, complete 1.2, rating (r − 3)/2, skip −1. Only positive weights become edges.
- **Artifact:** training exports `models/graph/artifacts/lightgcn_embeddings.{npz,json}`, which the backend loads at startup.
- **Benchmarks:** `notebooks/lightgcn_cloud.ipynb` runs them on a Kaggle GPU. It clones `main`, so launch it as a committed run.

### EMA taste vector (`backend/ema_recommender.py`)

Each logged interaction moves the user's 384-d taste vector, which lives in the same space as queries and items:

```
step = α · min(|w|, 1)          α = EMA_ALPHA = 0.25
new  = normalize((1 − step) · old + step · (±item_vector))
```

- A like moves the vector 25% of the way toward the item; a skip moves it toward the *negated* item vector.
- The update is one read and one write, so the next search reflects it with nothing retrained. LightGCN captures long-term taste; EMA covers the session.

---

## Evaluation

### LightGCN vs popularity (Amazon Reviews 2023)

Protocol:
- Ratings ≥ 4 are positives.
- Iterative 10-core filter (5-core for Industrial, which is an order of magnitude smaller), then a sample of up to 50k users.
- Leave-last-one-out per user, by timestamp.
- Full-catalog ranking, with training items masked.
- K = 20, seed 42, 60 epochs.

The popularity baseline ranks by training-set counts and also skips seen items.

| Domain | Users | Items | Interactions | Pop. R@20 | LightGCN R@20 | NDCG@20 | MRR@20 | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Movies & TV | 50,000 | 52,263 | 914,346 | 0.0182 | 0.0288 | 0.0125 | 0.0080 | **1.59x** |
| Industrial & Scientific | 33,959 | 19,024 | 238,752 | 0.0264 | 0.0326 | 0.0138 | 0.0087 | **1.24x** |
| Health & Household | 50,000 | 42,640 | 809,212 | 0.0221 | 0.0121 | 0.0046 | 0.0025 | 0.55x |

- **Health:** household goods are commodity purchases where most users buy the same best-sellers, so popularity is a strong baseline there. Collaborative filtering helps where taste varies between people. This is the case for a hybrid with per-domain weights.
- **Small numbers:** the absolute values are low because the interaction matrices are ~0.04% dense.
- **Hit rate:** it equals recall because exactly one item per user is held out.

To reproduce, run each `evaluate_lightgcn --report` into `reports/benchmarks/`, then:

```powershell
python -m scripts.build_benchmark_table   # writes reports/domain_benchmark_table.{md,tex}
```

### Constraint cost

```powershell
python -m scripts.evaluate_constraints --k 20   # writes reports/constraint_evaluation.{md,csv}
```

For each viewer profile, the report shows:
- **coverage:** how much of the catalog the viewer may see. A child sees 4.1%, a teen or an undeclared viewer 54.5%, an adult 100%.
- **leakage:** how often an *unfiltered* popularity ranking would show them something ineligible. For a child, that is 100% of the top 20.

The "filter ON" column re-checks the same Python predicate, so it is a consistency check, not independent evidence. The Qdrant filter is tested against that predicate in `tests/test_audience_filtering.py`.

### Scenario checks

```powershell
python -m scripts.evaluate_recommendation_table   # writes reports/recommendation_evaluation_table.{md,csv}
```

These are eight hand-picked queries (creator lookup, cross-media intent, family profile, health, industrial and finance discovery). They are qualitative sanity checks, not a statistical evaluation.

---

## API

Interactive docs are at `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/domains` | Domains and content types, read from `config/domains.yaml` |
| GET | `/search/suggest?q=` | Autocomplete over titles and creator names |
| POST | `/recommend` | Query-based recommendations |
| POST | `/recommend/item` | "More like this" for a `global_id`, through the same eligibility gate |
| POST | `/interactions` | Log view / click / like / bookmark / rating / skip / complete; updates the EMA vector |
| GET | `/interactions/{user_id}/{entity_id}` | Current toggle state and rating for a user–item pair |
| GET | `/` | Index |

Error codes:
- **400:** bad input — unknown domain or content type, an empty query, or an unknown `entity_id`.
- **503:** Qdrant, the model or the artifacts are unavailable.
- **500:** an internal error. The endpoints never fall back to an unfiltered search.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommend -ContentType "application/json" `
  -Body '{"query":"space adventure with aliens","user_id":"user_scifi","top_k":5,"age":15}'
```

Each result carries:
- **scores:** `score`, plus the per-signal `semantic_score`, `graph_score`, `ema_score`, `kg_score` and `profile_score`;
- `match_kind` (creator / title / keyword) for lexical hits;
- the audience fields.

---

## Frontend

- **Stack:** React 19, Vite, Tailwind CSS, framer-motion and axios.
- **Size:** 19 components and 2 pages — a demo login with personas, and the main search page.
- **Requests:** go through Vite's `/api` proxy, with a 15 s timeout.

Features:
- **Search:** autocomplete, query chips, a domain selector built from `/domains`, age and safe-mode controls, a content-type filter, and an advisory banner for regulated domains.
- **Result cards:** cover art, a maturity badge, a "matched by" badge, a score-breakdown bar, interaction buttons and a "more like this" drawer.
- **Explainability:**
  - Because fusion is additive, the UI splits each score into per-signal points.
  - It also re-ranks the whole page with one signal removed. That shows whether a signal *changed* the outcome, not just how much it added.
- **Robustness:**
  - Each search carries a sequence number, so stale responses are dropped.
  - Interaction toggles update optimistically and roll back if the request fails.
  - The UI shows a clean offline state when the backend is down.

---

## Setup

### Prerequisites

- Python 3.11 and Node.js 20+
- MySQL 8 (for personalization and interaction logging)
- A TMDb v3 API key, and optionally a Last.fm API key

Qdrant runs embedded by default, so no Qdrant server is needed.

### 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend; npm install; cd ..
copy .env.example .env   # then fill in MySQL credentials and API keys
```

Key variables:

| Variable | Default |
| --- | --- |
| `QDRANT_PATH` | `qdrant_storage` — embedded mode; unset it and set `QDRANT_URL` to use a Qdrant server |
| `MYSQL_*` | — |
| `TMDB_API_KEY` | — |
| `LASTFM_API_KEY` | — |
| `LIGHTGCN_WEIGHT` | 0.2 |
| `EMA_WEIGHT` | 0.15 |
| `EMA_ALPHA` | 0.25 |

### 2. Fetch data (optional if `datasets/` is already populated)

```powershell
python -m scripts.fetch_tmdb_movies --min-votes 100 --from-year 1960
python -m scripts.fetch_tmdb_credits
python -m scripts.fetch_tmdb_certifications        # real US ratings, 8 workers
python -m scripts.fetch_tmdb_shows
python -m scripts.fetch_tmdb_backdrops
python -m scripts.fetch_tmdb_franchises
python -m scripts.fetch_openlibrary_books
python -m scripts.fetch_openlibrary_books --from-year 2024 --to-year 2026 --per-subject 120 --merge
python -m scripts.fetch_lastfm_tracks
python -m scripts.fetch_amazon_metadata --category Health_and_Household --limit 20000
python -m scripts.fetch_amazon_metadata --category Industrial_and_Scientific --limit 20000
```

- **Resumable:** the fetchers retry with backoff and cache incrementally, so an interrupted run resumes where it stopped.
- **Finance:** the finance sources are carved out with the commands documented in `config/datasets.yaml`.

### 3. Build the catalog, embeddings and vector collection

```powershell
python -m preprocessing.build_content_catalog
python -m embeddings.build_embeddings
python -m embeddings.build_qdrant_collection
```

**Stop the API server before rebuilding the Qdrant collection.** In embedded mode the storage folder takes a single-process lock. A running server makes the rebuild fail with "already accessed by another instance" or `WinError 32`. On Windows, wait a few seconds after stopping the server for the file handle to clear.

### 4. Load MySQL and seed demo users

```powershell
python -m scripts.init_mysql
python -m scripts.load_catalog_mysql
python -m scripts.seed_demo_interactions
python -m scripts.rebuild_ema_profiles
```

There are 13 demo personas:

| Group | Personas |
| --- | --- |
| Entertainment | `user_scifi`, `user_fantasy`, `user_romance`, `user_action`, `user_music_pop`, `user_music_rock`, `user_books_learning`, `user_family`, `user_dark_thriller`, `user_balanced` |
| Verticals | `user_health_caregiver`, `user_industry_engineer`, `user_finance_planner` |

### 5. Train LightGCN (optional)

```powershell
python -m models.graph.train_lightgcn --epochs 50                       # from MySQL interactions
python -m models.graph.evaluate_lightgcn --epochs 20 --k 10             # holdout metrics
python -m scripts.fetch_amazon_dataset --category Movies_and_TV         # benchmark data
python -m models.graph.evaluate_lightgcn --dataset <path> --max-users 50000 --epochs 60 --k 20 --report reports/benchmarks/movies_and_tv_lightgcn.json
```

### 6. Run

```powershell
uvicorn backend.app:app            # ~40 s to load the model, catalog and Qdrant
cd frontend; npm run dev           # http://localhost:5173
```

---

## Repository layout

```
backend/            FastAPI app, Qdrant recommender, audience policy, domain registry,
                    MySQL store, LightGCN/EMA rerankers, KG overlap scorer
preprocessing/      Normalizers, audience tagging, catalog build and validation
embeddings/         SBERT embedding build, Qdrant collection build (FAISS builder kept for reference)
models/graph/       LightGCN model, training, evaluation, benchmark dataset preparation
scripts/            Data fetchers, MySQL init/load/seed, evaluation reports, cover-art backfill
config/             datasets.yaml (sources), domains.yaml (domains, maturity, risk tiers)
frontend/           React 19 + Vite + Tailwind UI
notebooks/          Kaggle GPU benchmark notebook
reports/            Benchmark, constraint and scenario evaluation outputs
tests/              pytest suite
```

`backend/recommender.py` (the original FAISS path) and `backend/api_ingestion.py` are earlier scaffolding. They are not used by the running app.

---

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q     # 328 passed
```

The suite covers:
- audience tagging rules and their false-positive guards
- the eligibility policy and the agreement between the compiled Qdrant filter and the Python predicate
- the domain registry
- preprocessing and catalog validation
- embedding caching
- LightGCN math and negative sampling
- the fetchers (Open Library, the Last.fm adult filter)
- evaluation-report consistency
- the API contract, including a regression test that an internal error is never retried without the eligibility gate

No server is required.

---

## Known limitations and roadmap

- **LightGCN artifact in the app:** the artifact shipped for the running app is trained on the 13 seeded demo personas, because the platform has no real user base. The Amazon benchmark is the evidence for the model.
- **Knowledge graph:** the KG signal is term overlap between query and item metadata, not a graph traversal. A typed graph with path-based scoring is planned.
- **Fusion and evaluation:**
  - Fusion weights are hand-set, and the full hybrid ranker has not been evaluated end to end; learning-to-rank needs real interaction logs.
  - The benchmark uses leave-one-out with a single seed. A global time split and confidence intervals would be stricter.
- **Hypothesis to test (Health result):** LightGCN is trained with a dot product but served (and evaluated) with cosine. Normalizing removes the popularity information carried in an item vector's length. That may explain the Health result.
- **Book and music maturity:** these labels are keyword heuristics; only movies carry board certifications.
- **Interaction toggles:** un-toggling a like posts a second `like` event. The state endpoint reads it correctly by parity, but the EMA update and the profile bonus count it as another positive.
- **EMA update:**
  - It is a read-modify-write with no locking.
  - It has no time decay or session boundary.
- **Database connections:** MySQL connections are opened per call, with no pooling.
- **Score breakdown:** once a user is selected, the UI's "semantic" segment includes the profile bonus.
- **Login:** the login screen is a demo with personas, not real authentication.
- **Search mode:** embedded Qdrant is used for development. A production deployment would run Qdrant as a server with HNSW and payload indexes on the filter fields.
