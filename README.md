# Real-Time Adaptive Cross-Media Recommendation System

A domain-agnostic recommendation engine over **120,747 items**:
- **Entertainment:** movies, TV shows, books, music
- **Products:** health, industrial and finance

The user types what they want, and the engine:
1. retrieves candidates by meaning (BGE-small text embeddings + Qdrant) and by keyword;
2. removes anything the viewer may not see (age, safe mode, domain, risk tier) **inside the vector query**, before anything is scored;
3. reranks what survives with collaborative filtering (LightGCN), a live taste vector (EMA) and profile signals.

Liking an item changes the next search immediately, with no retraining.

**Stack:** Python 3.11 · FastAPI · PyTorch · sentence-transformers (BGE-small) · Qdrant · MySQL · React 19 + Vite + Tailwind.

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
| Embeddings | `BAAI/bge-small-en-v1.5` (replaced `all-MiniLM-L6-v2` on 17 Nov 2026), 384-d, L2-normalized: 120,747 item vectors (~185 MB) plus 46,354 synopsis vectors |
| Vector store | Qdrant, cosine distance, eligibility filter applied inside the search |
| Collaborative filtering | LightGCN in PyTorch (d=64, 3 layers, BPR loss) |
| LightGCN vs popularity, Recall@20 | **1.59x** on Movies & TV, **1.24x** on Industrial & Scientific, 0.55x on Health & Household (Amazon Reviews 2023) |
| Real-time adaptation | EMA taste vector (α = 0.25), updated on every interaction, no retraining |
| Knowledge graph | 260,109 nodes, 1,015,044 typed edges; query-entity links and 2-hop paths, each result explained in words |
| Search quality | Recall@10 on an 18-query yardstick: **0.08 → 0.42** (popularity prior, knowledge-graph candidates, synopsis vectors, query intent, BGE-small); "more like this" **0.20 → 0.33**. 22 of 30 famous plots described without their title return the film at #1 |
| Explainability | Per-signal score breakdown plus counterfactual re-ranking in the UI |
| NCF baseline (NeuMF), Recall@20 | Beats cosine-scored LightGCN (1.19x / 1.27x), but LightGCN scored by dot product beats NeuMF: **1.32x** on Movies & TV, **1.42x** on Industrial & Scientific |
| Tests | 459 passing (`pytest`), run by GitHub Actions on every push |

---

## Architecture

```
                 ┌─────────────────────────── offline ────────────────────────────┐
 TMDb · Open Library · Last.fm · Spotify CSV · Amazon Reviews 2023
        │  scripts/fetch_*.py
        ▼
 preprocessing/build_content_catalog.py ──► content_catalog.csv (120,747 × 20)
        │  normalize · audience tagging · cover art
        ├──► embeddings/build_embeddings.py ──► 384-d BGE-small matrix (incremental, hash-cached)
        │        └──► embeddings/build_qdrant_collection.py ──► Qdrant collection + payload
        └──► scripts/load_catalog_mysql.py ──► MySQL content_entities
 Amazon Reviews 2023 interactions ──► models/graph (LightGCN, Kaggle GPU) ──► user/item embeddings
                 └────────────────────────────────────────────────────────────────┘

 POST /recommend {query, user_id, age, safe_mode, domain, content_type, top_k}
   1. AudienceContext   always built; no age → teen ceiling, safe_mode → child ceiling
   2. Creator match     cast / director / author / artist names        (eligibility re-checked)
   3. Lexical match     title and keyword tiers, ≤ half the page       (eligibility re-checked)
   4. Vector search     BGE-small query → Qdrant search WITH the eligibility filter
   5. Rerank            profile bonus → LightGCN → EMA → knowledge-graph paths + profile match
   6. top_k results (+ advisory for regulated domains)

 POST /interactions ──► MySQL event log ──► EMA vector updated in the same request
```

The pieces:
- **Backend:** FastAPI.
- **Model and catalog loading:** heavy objects (the text encoder, the catalog and the Qdrant client) are created once in the app's `lifespan` hook.
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
- **Encoder:** `BAAI/bge-small-en-v1.5` (`EMBEDDING_MODEL`), a 12-layer BERT-style encoder (~33M parameters) trained for retrieval: short queries against long passages. Queries, and only queries, get its instruction prefix "Represent this sentence for searching relevant passages: " (`EMBEDDING_QUERY_INSTRUCTION` overrides it). The embedding index records the model per row, and the API refuses to start when the stored vectors came from a different model than `EMBEDDING_MODEL`.

**Why BGE-small replaced all-MiniLM-L6-v2.** Both produce 384-d vectors, so Qdrant and the rest of the pipeline were unchanged. MiniLM was trained for sentence similarity, so a short query like "man relives the same day" matched short titles like *Every Day* more than Groundhog Day's synopsis, which says exactly that. Measured through the full live pipeline (popularity, knowledge graph, synopsis vectors, same weights):

| | all-MiniLM-L6-v2 | BGE-small |
|---|---|---|
| 18-query yardstick, recall@10 | 0.33 | **0.42** |
| 7-seed "more like this", recall@10 | **0.36** | 0.33 |
| 30 famous plots without their title: found at #1 | 14 | **22** |
| … in the top 3 / top 10 | 21 / **27** | **25** / 26 |
| Query encoding (RTX 3050) | 12 ms | 24 ms |

The 30 plot descriptions ("a rat who wants to be a chef in paris", "old man flies his house with balloons") were written after the weights were set, so neither encoder was tuned on them. Halving the popularity weight for BGE-small made every number worse (0.29, 19 of 30), so the weights stayed.

**Switching the encoder back (or to another one):** set `EMBEDDING_MODEL`, then run `build_embeddings`, `build_plot_embeddings`, `build_qdrant_collection` (API stopped) and `python -m scripts.rebuild_ema_profiles`, which replays every user's interactions into a taste vector in the new space. The MiniLM vectors from before the switch are kept in `embeddings/minilm/`.
- **Payload:** each point carries the display fields plus the audience fields, stored as integers so Range filters work on them.
- **Rebuilds:** in embedded mode, a rebuild deletes the collection *and* purges its directory. Otherwise the old points would be re-attached.

### Database

`backend/mysql_store.py` creates six tables and runs idempotent migrations on every init (`python -m scripts.init_mysql`). Connections come from one shared pool of 10 (`mysql.connector.pooling`); a burst beyond that connects directly instead of failing. Pooled, a profile-plus-state read takes 3.9 ms against 18 ms when every call opened its own connection.

| Table | Purpose | Notes |
| --- | --- | --- |
| `users` | Email / Google accounts | Argon2id password hash, Google `sub` |
| `user_sessions` | Login sessions | SHA-256 of the cookie token, never the token |
| `login_failures` | Wrong-password log | Makes the login lockout survive a restart; misses older than a day are pruned |
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

- **Semantic:** the query is encoded with the same encoder (with BGE's search instruction in front) and searched in Qdrant. This answers intent ("space adventure with aliens").
- **Creator:** cast, director, author and artist names are matched as whole words, in two tiers (the whole name, or a word of it), most established work first. This answers "Christopher Nolan" and "nolan". Guards against junk: an Amazon brand matches only by its full name; a one-word query used in more than 500 catalogue rows ("love") only matches a full name; and a query that is itself a title drops partial-name hits.
- **Lexical:** five tiers:
  1. exact title
  2. title prefix
  3. title contains the query
  4. every token appears in the title or categories
  5. the phrase appears in the first 1,000 characters of the description

  This answers titles that carry no semantic signal: a pure vector search for "F1" returned a Punjabi song while the film sat in the catalog.

How the results are merged:
- **Navigational band:** creator hits and *exact* title hits get fixed scores just above the cosine range, so a query that names something gets that thing first. A title that merely starts with or contains the query competes on relevance instead ("time travel" used to put the Cars short *Time Travel Mater* first).
- **Candidate pool:** everything else competes in one pool:
  - the top 300 vector neighbours (`CANDIDATE_POOL`);
  - up to 150 items carrying an entity the query names in the knowledge graph ("time travel", "heist", "zombie apocalypse");
  - up to 150 films, shows and books whose *synopsis alone* is closest to the query (below);
  - the partial-title and keyword hits.

  A pool this deep matters: for "mind bending sci-fi thriller", *Inception* is 3,151st by raw cosine.
- **Synopsis vectors (`backend/plot_index.py`):** the main item vector embeds title, genres, TMDb keywords, synopsis and cast in one string, and the lists drown the synopsis. Groundhog Day's synopsis says "repeating the same day over and over", yet for "man relives the same day over and over" it was 1,421st by the main vector; by its synopsis alone it is 110th (both measured with all-MiniLM-L6-v2). Plot descriptions are scored by whichever is closer, the full text or the synopsis. The 46,354 synopsis vectors are searched exactly with one matrix product. Build them with `python -m embeddings.build_plot_embeddings` (about a minute); without the files the search runs on the main vectors alone.
- **Ranking the pool:** `relevance + 0.30 × popularity + 0.20 × knowledge-graph entity match`, then the personalisation rerankers below.
  - **Relevance** is the cosine to the full text or to the synopsis, whichever is higher.
  - **Popularity** (`backend/prominence.py`) is how established an item is: TMDb vote counts for films and shows, the 0–100 index for music, log-scaled rating or review counts elsewhere, each scaled within its content type.
  - **Entity match** is the share of the named entities an item carries, times the share of the query's words those entities account for. "boy discovers he is a wizard" links only "wizard", a third of a match; at full credit every film tagged "wizard" outranked Harry Potter, which TMDb never tagged "wizard".
- **Query intent:**
  - One type word in the query ("books about artificial intelligence", "songs for running") restricts the search to that type, unless the query is itself a title ("Scary Movie").
  - "movies like X", "shows like X", "something like X" and "similar to X" find the title X and answer with "more like this" for it. When no title X exists ("movies like dream manipulation"), X is searched as a description in the type asked for, and gibberish returns nothing rather than any popular film.
  - **Years:** "90s movies", "sci-fi from the 1980s", "songs from 2010", "movies after 2015" and "books before 1950" filter by release year. A decade with no topic ("90s movies") lists its best-known titles: The Shawshank Redemption, Pulp Fiction, Forrest Gump, Titanic, The Matrix. With a topic, that decade's vectors are searched exactly from memory, since the Qdrant payload has no numeric year. A year that is part of a title is left alone ("That '70s Show", or "1984 book" when a book is titled "1984").
  - **Slang:** words the encoder does not know are spelled out before encoding: "rom com" embedded near Latin pop songs and budgeting software, "romantic comedy" lands on The Proposal and Love Actually. Likewise "sci-fi" and "zom-com".
- **Duplicates:** copies of one work (same type, title, first creator and, except for music, year) are shown once. Spotify lists a song once per release, so "more like Shape of You" returned Shape of You. A cover by another artist and a director's remake are different works.

**Why the pool and the popularity prior.** Meaning-based search alone returns the obscure titles whose *names* echo the query: "dark psychological thriller" gave *Through the Darkness*, *Dark Water*, *Dark House*; "feel good comedy for the whole family" gave *No Good Nick* and *My Family*. On a hand-made yardstick of 18 realistic queries with well-known expected titles, recall@10 went from **0.08** (cosine only) to **0.31** through the live engine, to **0.33** with the synopsis vectors and the query-linking rules above, and to **0.42** with the BGE-small encoder (below); on 7 "more like this" seeds from **0.20** to **0.36** (0.33 with BGE-small). The yardstick favours famous titles by construction, so the popularity weight was kept moderate (0.30): at 0.50, popular but irrelevant songs crowded into "cooking" and "sad love story".

Every match is whole-word: haystacks are space-padded and tokenized, so "inception" does not match the brand "Skinception". The haystacks are built once with vectorized pandas operations. A per-row version took 15–49 s per request at this catalog size.

### Typos and nonsense (`backend/query_vocabulary.py`)

A sentence encoder embeds a misspelling or a keyboard mash like any other text, and its nearest neighbours score like real matches. Measured with all-MiniLM-L6-v2, "qwertyuiop" scored 0.53, above the real searches "dog" (0.46) and "f1" (0.48). Over 53 real and 25 mashed queries the best-match scores overlap from 0.45 to 0.54, so no cosine cut-off can separate them. The catalogue's own words can: the vocabulary counts how many of the 120,747 rows use each word.

- **Typo correction:** a word in fewer than 2 rows that is one edit from a known word (two edits from 8 letters) is corrected before searching. Ties go to the usual slips (a doubled letter typed once, two letters swapped), then to the pair that forms a title phrase. "hary poter" becomes "harry potter", not the commoner "hard power". The response's `corrected_query` says what was searched, the UI shows "Showing results for … / Search instead for …", and `"spellcheck": false` searches the query as typed. A query that already names a title or creator is never corrected.
- **Nonsense gate:** if no word of the query (stop words aside) is in the catalogue, the vector search is skipped, and only creator and title matches can appear. "qzxv blorf wumple" returns nothing instead of ten arbitrary items.

The vocabulary takes about 10 s to build with the other lookup tables, so `warm_up()` builds them at startup instead of in the first search.

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
| Retrieval | Cosine for vector hits (`semantic_score`); fixed tier scores for exact creator/title hits (`match_score`) | 1.0 |
| Popularity | Prominence in [0, 1] (`popularity_score`, see Retrieval) | 0.30 (`POPULARITY_WEIGHT`) |
| Profile bonus | From the user's last 500 events: category and content-type affinity, **capped at ±0.15**; already-liked items −0.15, skipped −0.35 | added directly |
| LightGCN | cosine(user embedding, item embedding) | 0.20 (`LIGHTGCN_WEIGHT`) |
| EMA | cosine(user taste vector, item embedding) | 0.15 (`EMA_WEIGHT`) |
| Knowledge graph | Typed entity graph: query-entity links and 2-hop paths from liked items (below) | 0.20 (`KG_WEIGHT`); 0.80 on "more like this" (`KG_SIMILAR_WEIGHT`) |
| Profile match | Interests, skills, profession and age-group terms from `user_profiles.preferences` | 0.10 |

The cap on the profile bonus exists because a film's categories now carry its TMDb keywords: an item could collect fifteen matching categories and +0.7, more than its whole relevance, and taste alone put *Lightyear* on a page for "christopher nolan".

These are the live engine's weights. The paper's 16-combination ablation used its own weights offline (KG 0.08); the live weights were chosen against the yardstick above.

### More like this (`POST /recommend/item`)

- **Candidates:** the item's 300 nearest vectors plus its 150 strongest knowledge-graph neighbours (shared director, cast, franchise, keywords, genres).
- **Ranking:** `cosine + 0.30 × popularity + 0.80 × graph similarity`, then the personalisation rerankers. Text neighbours alone are dominated by the title: "Titanic" returned the Titanic miniseries, *Titanic II* and *Titanic 666*; "Inception" now returns *Tenet*, *Interstellar*, *Memento*, *The Prestige*.
- **Medium:** films and shows answer each other, and books are often their source; music and products stay within their own type (unless a content type is asked for).
- **Explanation:** "Directed by Christopher Nolan, like Inception". Each result carries `signal_weights: {"kg": 0.8}` so the score bar adds up.

### Knowledge graph (`backend/kg_graph.py`)

A typed graph built from the catalogue at start-up (about 3 s), held as one sparse item × entity matrix.

| Node type | Count | Edge (item → node) |
| --- | --- | --- |
| Item | 120,747 | — |
| Person | 72,921 | `DIRECTED_BY` (first movie credit), `STARS` (next five), `WRITTEN_BY` (books), `PERFORMED_BY` (music) |
| Brand | 20,500 | `MADE_BY` (Amazon products) |
| Genre / category | 34,225 | `IN_GENRE` |
| Franchise | 987 | `PART_OF` (TMDb "… Collection") |
| Keyword | 10,718 | `TAGGED` (lower-case TMDb plot keywords) |
| Content type, domain | 11 | `IS_A`, `IN_DOMAIN` (structure only, not scored) |

260,109 nodes and 1,015,044 typed edges in total (`GET /kg/stats`).

- **Query links:** n-grams of the query are matched to entity names, longest first, so "christopher nolan" is one Person node. An item scores the share of the linked entities, weighted by rarity (IDF), it is connected to. Any credited role counts here.
- **User paths:** the items a user currently likes are seeds. A candidate scores by the meta-path *liked item → entity ← candidate*: the cosine between the two entity vectors, weighted by relation (director 1.0, star 0.7, genre 0.5, keyword 0.4, …) and IDF.
- **Blend:** the average of whichever halves apply (query entities present, user has likes).
- **Explanation:** each result carries `kg_path` and `kg_explanation`, its strongest path in words, e.g. "Directed by Christopher Nolan, like Inception, which you liked". The result card shows it as **Why**. Paths weaker than 0.05 are scored but not explained.
- **Cost:** under 10 ms for 100 candidates and 60 liked items.
- **Neo4j:** `python -m scripts.export_kg_neo4j` writes `nodes.csv`, `edges.csv` and `import.cypher` to `data/processed/kg/` for loading the same graph into Neo4j. The export has not been loaded into a running Neo4j yet.

Items the graph does not know fall back to the original query/item term overlap.

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

### NeuMF baseline (Neural Collaborative Filtering)

`models/graph/neumf.py` implements NeuMF (GMF + MLP towers, pointwise log loss, 4 sampled negatives per positive). `models.graph.evaluate_neumf` runs it through the same preparation, split and metrics as LightGCN, on graphs of exactly the same size.

| Domain (Recall@20) | Popularity | LightGCN, cosine | NeuMF | LightGCN, dot product |
| --- | --- | --- | --- | --- |
| Movies & TV | 0.0182 | 0.0288 | 0.0344 | **0.0454** |
| Industrial & Scientific | 0.0264 | 0.0326 | 0.0416 | **0.0589** |

- **Cosine vs dot product:** the backend and the paper rank LightGCN by cosine (normalised embeddings). Against that, NeuMF is 1.19x / 1.27x better. Ranked by the raw dot product BPR was trained with, LightGCN beats NeuMF by 1.32x / 1.42x: normalising discards the item popularity carried in vector length. `python -m models.graph.evaluate_lightgcn --scoring dot` reproduces it; the cosine rerun matches the Kaggle numbers exactly.
- **Caveats:** one seed per model, no tuning for either, and Health & Household was not rerun (its rating file is not on the training machine).

Full table (NDCG, MRR), configuration and commands: `reports/neumf_comparison.md` (`python -m scripts.build_neumf_table`).

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
| GET | `/kg/stats` | Knowledge-graph node and edge counts by type |
| GET | `/domains` | Domains and content types, read from `config/domains.yaml` |
| GET | `/search/suggest?q=&age=&safe_mode=&domain=` | Autocomplete over titles and creator names, limited to what this viewer may open (same audience policy as `/recommend`; no age means the teen ceiling) |
| POST | `/recommend` | Query-based recommendations |
| POST | `/recommend/item` | "More like this" for a `global_id`, through the same eligibility gate |
| POST | `/interactions` | Log view / click / like / bookmark / rating / skip / complete; updates the EMA vector |
| GET | `/interactions/{user_id}/{entity_id}` | Current toggle state and rating for a user–item pair |
| GET | `/auth/config` | The Google client id, or null when Google sign-in is off |
| POST | `/auth/signup` | Create an email + password account and start a session |
| POST | `/auth/login` | Email + password login; five wrong passwords lock the email for 15 minutes |
| POST | `/auth/google` | Sign in with a Google ID token, verified server-side |
| POST | `/auth/logout` | End the session |
| GET | `/auth/me` | The logged-in account (with its `interaction_count`), or 401 |
| DELETE | `/auth/me` | Delete the logged-in account and everything recorded about it |
| GET | `/onboarding/items` | The starter picks a new account chooses from |
| GET | `/` | Index |

Accounts (`backend/auth.py`):
- **Ids:** real accounts get `acct_` ids. A request that names one, in `/recommend`, `/recommend/item` or `/interactions`, needs that account's session, or it gets 401 (no session) or 403 (someone else's). The demo personas need no login.
- **Sessions:** a random token in an HttpOnly, SameSite=Lax cookie (`nexus_session`), stored as a SHA-256 hash in `user_sessions` and deleted on logout.
- **Passwords:** Argon2id hashes in `users`. There is no email verification or password reset, since the deployment sends no email.
- **Lockout:** each wrong password is a row in `login_failures`, timed by the MySQL server clock, so the 15-minute lock survives a restart and holds across server processes. If that table cannot be reached, an in-memory throttle takes over rather than blocking logins.
- **Google:** when a Google account's verified email matches an email account, the two become one and the old password is removed, because sign-up never proved who owned that address.
- **First visit:** an account with no interactions is offered 22 starter picks (one per genre across movies, shows, books and music) before its first search. Each pick is saved as an ordinary `like`, so the EMA taste vector and the LightGCN training log start from the person's own taste rather than from nothing. Movies and shows are ranked by popularity and rating together, because TMDb popularity alone means "trending now"; books and music by popularity, because book ratings rest on a handful of votes.
- **Deleting:** `DELETE /auth/me` removes the account's interactions, profile (preferences and EMA vector), sessions and user row in one transaction. It only ever acts on the session's own account. The UI asks for confirmation in a native modal `<dialog>`, with focus starting on Cancel.
- **Privacy note:** the login and sign-up forms say what is saved (name, email, and what you like, rate, skip or open) and that it can all be deleted, before anyone hands over an email.

Error codes:
- **400:** bad input — unknown domain or content type, an empty query, or an unknown `entity_id`.
- **401 / 403:** a missing or wrong session for an account id.
- **503:** Qdrant, the model or the artifacts are unavailable.
- **500:** an internal error. The endpoints never fall back to an unfiltered search.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommend -ContentType "application/json" `
  -Body '{"query":"space adventure with aliens","user_id":"user_scifi","top_k":5,"age":15}'
```

The response's `corrected_query` is set when a typo was fixed before searching (send `"spellcheck": false` to turn that off).

The response's `notice` is set when the exact title searched for exists but the audience filter withholds it: "“The Godfather” (movie, 18+) is hidden: no age is set for this profile, so results stop at 13+. Set an age in Refine to see it." The UI shows it under the results heading. When no item with that title is visible at all (for example "The Godfather" with the Movies filter and no age), the page answers with the visible titles most like it ("more like this" on the hidden film, through the same eligibility gate), and the notice adds "Showing similar titles you can see." Before this, a bare title the viewer could not see filled the page with popular PG-13 films the profile liked, which looked like a broken search.

Each result carries:
- **scores:** `score`, plus the per-signal `match_score`, `semantic_score`, `popularity_score`, `preference_bonus`, `graph_score`, `ema_score`, `kg_score` and `profile_score`, and `signal_weights` when a weight differs from the default;
- `kg_explanation` and `kg_path`: the strongest knowledge-graph path, in words and as steps;
- `similar_to` when the query was "movies like X";
- `match_kind` (creator / title / keyword) for lexical hits;
- the audience fields.

---

## Frontend

- **Stack:** React 19, Vite, Tailwind CSS, framer-motion and axios.
- **Size:** 21 components and 3 pages — the login page (email, Google, or a demo persona), the "pick a few things you like" page for a new account, and the main search page.
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
| `KG_WEIGHT` / `KG_SIMILAR_WEIGHT` | 0.2 / 0.8 |
| `POPULARITY_WEIGHT` | 0.3 |
| `CANDIDATE_POOL` | 300 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` (changing it needs the rebuild steps under Embeddings) |
| `EMBEDDING_QUERY_INSTRUCTION` | unset: BGE's retrieval instruction for `bge-*-en` models, none otherwise |
| `PLOT_EMBEDDINGS_PATH` / `PLOT_EMBEDDING_INDEX_PATH` | `embeddings/plot_embeddings.npy` / `embeddings/plot_embedding_index.csv` (optional) |
| `TMDB_MOVIES_PATH` / `TMDB_SHOWS_PATH` | `datasets/tmdb/tmdb_{movies,shows}.csv` — vote counts for the popularity prior; without them the catalogue's popularity column is used |
| `EMA_ALPHA` | 0.25 |
| `GOOGLE_CLIENT_ID` | — (Google button hidden). A "Web application" client from Google Cloud Console; add `http://localhost` and `http://localhost:5173` as authorized JavaScript origins |
| `SESSION_DAYS` | 30 |
| `COOKIE_SECURE` | 0 — set to 1 once the site is served over HTTPS |

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
python -m embeddings.build_plot_embeddings      # synopsis vectors for plot queries, about a minute
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

Or start the demo deployment in one command: it checks MySQL, builds the frontend for production, starts the API, waits until it is healthy, and serves the build on port 4173 with `/api` proxied to the API.

```powershell
.\scripts\demo_up.ps1              # http://localhost:4173
.\scripts\demo_up.ps1 -Public      # also a temporary https://*.trycloudflare.com link (email login works; Google sign-in does not)
.\scripts\demo_up.ps1 -Stop        # stop everything it started
```

---

## Repository layout

```
backend/            FastAPI app, Qdrant recommender, audience policy, domain registry,
                    MySQL store, LightGCN/EMA rerankers, typed knowledge graph
preprocessing/      Normalizers, audience tagging, catalog build and validation
embeddings/         Item and synopsis embedding builds, Qdrant collection build (FAISS builder kept for reference)
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
.\.venv\Scripts\python.exe -m pytest tests/ -q     # 459 passed
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
- accounts: sign-up, login, lockout (including after a restart, and with the lockout table down), Google linking, the per-account access guard and account deletion
- the MySQL connection pool: shared across store instances, falls back to a direct connection when exhausted or unbuildable
- the score breakdown: `semantic_score` excludes the personalisation bonus, which the API returns as `preference_bonus`
- the knowledge graph: typed edges by content type and source, longest-match entity linking, 2-hop path scores and their explanations, export completeness
- starter picks: one per genre, eligible for an unknown-age viewer, with cover art and a decent rating
- typo correction (title phrases beat commoner words, titles are never "corrected") and the nonsense gate
- NeuMF: the all-items scorer equals pairwise scoring, sampled negatives are never observed items, training learns a planted structure
- interaction toggles: an un-like is not counted as a second like, and the interaction-history profile query runs
- ranking quality: query intent ("books about…", "movies like X"), the popularity scale per source, an established relevant title beating an obscure namesake, only exact titles being navigational, brand and common-word creator guards, the profile-bonus cap, and "more like this" choosing a same-director film over closer text
- knowledge-graph candidates: entity candidates, graph neighbours, everyday genre words ("sci-fi", "rom com"), franchise names without "Collection"
- query linking: plural words reaching singular keywords ("balloons" → "balloon"), common words and one- or two-item entities not linking inside a sentence, a decade mask on entity candidates
- years, slang and duplicates: decade and year phrases (and years that belong to a title), browsing a decade by prominence, a decade filtering a topical search, slang spelled out for the encoder, "like" an unknown title searching the description, one-word creator queries ordered by prominence, copies of one work dropped (covers and remakes kept)
- synopsis vectors: exact search with a row mask, and a film whose synopsis matches outranking a title echo
- the encoder: BGE's query instruction on queries only (never items, never for MiniLM or bge-m3), and startup refusing item vectors built by a different model
- years as a setting: a decade inside a plot description ("kids in the 80s…") is when the story happens, not a release filter
- titles a viewer may not see: an 18+ title searched without an age is answered with the visible titles most like it, a visible namesake (Puzo's novel) keeps the search, and the notice names the title, its rating and why it is hidden

No server is required. `.github/workflows/ci.yml` runs the suite on Windows (the requirements pin `pywin32`) and lints and builds the frontend.

---

## Known limitations and roadmap

- **LightGCN artifact in the app:** the artifact shipped for the running app is trained on the 13 seeded demo personas, because the platform has no real user base. The Amazon benchmark is the evidence for the model.
- **Knowledge graph:** the graph lives in memory and its relation weights are hand-set. The Neo4j export is generated but not served from a graph database.
- **Ranking weights and yardstick:** the popularity, knowledge-graph and pool settings were chosen on a hand-made yardstick of 18 queries and 7 seed titles, which favours famous titles by construction. Moods the catalogue cannot see ("relaxing music for studying", "sad love story that makes you cry") are still weak: there are no audio features, and synopses rarely say how a story feels.
- **Popularity bias:** the prior helps the well-known titles people usually mean, at some cost to discovery of obscure ones; `POPULARITY_WEIGHT=0` turns it off.
- **What the encoder cannot know:** BGE-small only sees our text. Plot descriptions whose film is described in other words still miss: "a hacker discovers reality is a simulation" (The Matrix), "a farm boy joins rebels to fight an evil galactic empire" (Star Wars), "a man trapped in a reality tv show without knowing" (The Truman Show). Very established titles sometimes fill a top-3 slot they do not deserve (*The Notebook* for "old man flies his house with balloons"). Negation ("movies without romance") and non-English queries are not understood. A cross-encoder reranker (e.g. bge-reranker) over the top 50, or an LLM rewriting the query, are the next steps.
- **Embedded Qdrant latency:** embedded Qdrant checks the eligibility filter in Python for every point, so a filtered search takes about 1.6 s and a request about 1.8 s (`reports/latency_benchmark_embedded_sample.md`). The 363 ms p50 / 455 ms p95 figures were measured with Qdrant running as a server, where the search takes 85 ms.
- **Fusion and evaluation:**
  - Fusion weights are hand-set, and the full hybrid ranker has not been evaluated end to end; learning-to-rank needs real interaction logs.
  - The benchmark uses leave-one-out with a single seed. A global time split and confidence intervals would be stricter.
- **Hypothesis to test (Health result):** LightGCN is trained with a dot product but served (and evaluated) with cosine. Normalizing removes the popularity information carried in an item vector's length. That may explain the Health result.
- **Book and music maturity:** these labels are keyword heuristics; only movies carry board certifications.
- **EMA update:**
  - It is a read-modify-write with no locking.
  - It has no time decay or session boundary.
- **Accounts:** no email verification or password reset (both need an email sender).
- **Deployment:** the system runs on one machine. `scripts/demo_up.ps1 -Public` gives a temporary public link through a Cloudflare quick tunnel; there is no permanent host and no container image yet (Docker is not set up on the development machine).
- **LightGCN scoring:** the backend and the paper rank LightGCN by cosine. Ranked by the raw dot product BPR was trained with, Recall@20 rises from 0.0288 to 0.0454 on Movies & TV and from 0.0326 to 0.0589 on Industrial & Scientific (`reports/neumf_comparison.md`). Serving it needs a rescaling step first, because the other fused signals are on a 0-1 cosine scale.
- **Search mode:** embedded Qdrant is used for development. A production deployment would run Qdrant as a server with HNSW and payload indexes on the filter fields.
- **Language:** BGE-small (`-en`) is English-only: queries in other languages fall back to word overlap, and negation is ignored ("not sad" reads as "sad"). A multilingual model (e.g. bge-m3) would mean re-embedding the catalogue at about 17× the parameters.
