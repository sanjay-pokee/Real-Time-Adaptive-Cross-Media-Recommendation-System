# Interview Prep: Real-Time Adaptive Cross-Media Recommendation Engine

Checked against the code on 2026-10-04. Where an older doc (README, runbook, panel brief, paper) disagrees with the code, the code wins. The differences are listed in section 9.

How to use this: for every layer, know it at three depths — one sentence, one paragraph, and which file you would open. Section 2 is laid out that way.

---

## 0. Owning the whole project

- You built the system end to end, and the repo shows it: **74 of the 75 commits are yours**.
- The paper lists four authors and the course registers it as a four-person Phase-2 project. Say it in a way that matches that record: *"It was our four-person final-year project, and I built the system: the data pipeline, backend, models, frontend and evaluation."* If the interviewer has read the paper, your story and its author list then agree.
- The four-way split in `panel_prep_brief.html` assigned speaking time at the review. It says nothing about who wrote what. Ignore it for interviews.

---

## 1. The pitch

**30 seconds**
> I built a recommendation engine that works across domains: movies, TV, books and music, plus health, industrial and finance products — 120,747 items in one catalog. You type what you want. I retrieve by meaning and by keyword, filter by age and risk *before* anything is scored, then rerank with collaborative filtering, a live taste vector and profile signals. Like something, and your next search changes without retraining anything.

**2 minutes** (the section 2 walkthrough, compressed)
1. **Data.** I pull from live APIs (TMDb, Open Library, Last.fm) and public datasets (Spotify, Amazon Reviews 2023), and normalize everything into one 20-column catalog with a stable ID.
2. **Retrieval.** Sentence-BERT embeddings in Qdrant find things by meaning. A tiered keyword matcher finds names and titles, because an embedding can't find "F1".
3. **Eligibility.** Age, safe mode, domain and risk tier are compiled into the vector query itself, so a blocked item is never retrieved and can't be reranked back in.
4. **Personalization.** LightGCN for long-term collaborative taste, an EMA taste vector for in-session drift, and profile-based reranking.
5. **Product.** FastAPI and MySQL behind a React UI that shows why each item ranked where it did, including what would happen without each signal.
6. **Evidence.** LightGCN beats a popularity baseline 1.59x on Movies & TV and 1.24x on Industrial, and loses on Health (0.55x). That loss is the argument for a hybrid. 328 automated tests.

---

## 2. The system, layer by layer

### 2.1 Data acquisition — `scripts/fetch_*.py`, `config/datasets.yaml`

| Content type | Source | Rows | What's worth saying |
| --- | --- | --- | --- |
| movie | TMDb `/discover/movie`, live | 23,492 | Sliced by release year because `/discover` caps at 500 pages; ≥100 votes, from 1960. Credits and US certifications are fetched separately. |
| show | TMDb `/discover/tv` | 10,617 | ≥25 votes, because TV gets far fewer votes. |
| book | Open Library search API | 12,402 | 60 subjects, `sort=rating`, description and cover required. Asks for `description` as a search field, so 100 books cost one request. |
| music | Spotify songs CSV + Last.fm API | 31,133 | Spotify stops at 2020, so Last.fm adds current music. An adult-listing filter runs on ingest. |
| health, industrial | Amazon Reviews 2023 metadata | 20,000 each | One shared normalizer; only `content_type` differs. |
| finance | Amazon Software › Accounting & Finance, plus Open Library finance books | 3,103 | Finance isn't an Amazon top-level category. Book subjects are filtered with require/exclude patterns, so novels don't leak in. |

Decisions to be able to defend:
- **Replaced the 2017 Kaggle TMDB-5000 dump (4,803 rows) with the live API.** The ceiling was the file, not the source.
- **Replaced a Google Books scrape.** Only 55% of its rows had descriptions, and the top-ranked rows were SEO spam. A row with no description embeds to noise; a keyword-stuffed row matches every query weakly.
- **`sort=rating` has a recency blind spot** (ratings take years to accumulate). So 2024–2026 is topped up with a year window and `--merge`, which keeps existing IDs stable and so preserves logged interactions.
- **Fetchers are resilient.** Retrying HTTP sessions with backoff on 429/5xx, rate limiting, and incremental caches so an interrupted run resumes. Certifications use 8 worker threads, because serial throughput was ~100 requests/min.

### 2.2 One catalog — `preprocessing/build_content_catalog.py`

- Each dataset is dispatched to a normalizer named in config. Health, industrial and finance share `amazon_meta`, so a new Amazon vertical is a YAML entry, not code.
- **20 columns** = 15 content columns + 4 audience columns (`domain`, `maturity`, `audience_min_age`, `risk_tier`) + `backdrop_url`.
- **`global_id = content_type:source:source_id`** is the primary key everywhere: Qdrant, MySQL, LightGCN, the UI.
- **Two text fields:**
  - `metadata_text` holds everything, for storage and inspection.
  - `embedding_text` is curated per type. Movie = title + categories + description + director + top-5 cast. Book = title + authors + categories + description. Music = title + artist + genre + subgenre. Amazon = title + brand + categories + description.
- `text_hash` is the SHA-256 of `embedding_text` and drives the embedding cache.
- TMDb franchise names and keywords are appended to `categories`, so a keyword search for "avengers" reaches Iron Man and Black Panther.
- Cover art comes from the source where it exists, otherwise from a backfill cache, otherwise a generated gradient in the UI.

### 2.3 Audience labels — `preprocessing/audience_tagging.py`

- **Movies use the real board rating:** TMDb US certification, mapped G → all_ages, PG → child, PG-13 → teen, R → adult, NC-17 → restricted (TV ratings likewise). `NR`/Unrated means "no board rated it", not "harmless", so those fall through to the heuristic.
- **Everything else uses keyword rules** over categories + title:
  - Ordered most-restrictive first.
  - Matched at a left word boundary, so "teen" doesn't fire on "canteen".
  - A short list of whole-word rules. Every candidate keyword was measured against all titles first: "ass" would hit *Kick-Ass*, and "xxx" hits "XXX-Large" clothing sizes.
- **Restrict-only.** For board-rated types and regulated domains, a keyword can raise maturity but never lower it below the default. The motivating bug: "Animation" matched the all-ages rule, so *Akira* and *Grave of the Fireflies* were served to an 8-year-old.
- **Fails closed.** Unknown content type → adult. Unknown risk → tier 1. Unrecognised label → the most restrictive level.

### 2.4 Embeddings — `embeddings/build_embeddings.py`

- `all-MiniLM-L6-v2`: 384-d, L2-normalized (so dot product = cosine), batch 256, GPU if available.
- The whole matrix is 120,747 × 384 × 4 bytes ≈ **185 MB**, which fits in RAM on a CPU-only server.
- **Incremental rebuild.** The index is keyed on (`global_id`, model, version, `text_hash`), so only changed rows are re-encoded.
- **Bug I fixed:** the cache missed 100% of the time. The version `"1"` came back from CSV as the integer `1`, so no key ever matched. Both sides of the key are now coerced to strings.

### 2.5 Vector store — `embeddings/build_qdrant_collection.py`

- Qdrant with cosine distance. Point ID = `uuid5(global_id)`, so IDs are deterministic.
- The payload carries the display fields plus the audience fields as **real integers**. A Range filter skips points where the field is missing or a string, which would silently hide items.
- **Why Qdrant over FAISS** (the first version used FAISS): Qdrant filters *inside* the search. FAISS searches first and filters afterwards, which means blocked items get retrieved and pages come back short.
- **Rebuild bug:** in embedded mode, `delete_collection` leaves the on-disk sqlite file, and recreating the collection re-attached it. After movies were re-sourced, 4,803 stale points reappeared and films showed up twice. Fix: delete, close the client, purge the directory, recreate.
- Embedded Qdrant holds a single-process file lock. That is why the API opens the client once in its lifespan hook, and why you stop the server before rebuilding.

### 2.6 MySQL — `backend/mysql_store.py`

- **Tables:**
  - `content_entities`: primary key `global_id` with `utf8mb4_bin` collation, because IDs are case-sensitive and two different books collided under the default collation.
  - `user_profiles`: `preferences` JSON and `ema_vector` JSON.
  - `user_interactions`: index on `(user_id, timestamp)` for the hot "recent history" query; foreign key to the catalog with `ON DELETE CASCADE`.
- **Catalog loader:**
  - Chunked upserts of 2,000 rows, because one statement blew `max_allowed_packet` past ~100k rows.
  - Then it prunes rows no longer in the catalog with a temp table and a `LEFT JOIN` delete. The temp table uses the same collation, so the join uses the index.
- **Idempotent migrations** run on every init: `MEDIUMTEXT` for long Amazon descriptions, and collation changes made around the foreign key.
- **Interactions are an event log.** Like, bookmark etc. are toggles, and "currently active" means an odd count of events.

### 2.7 One request, end to end — `backend/app.py`, `backend/qdrant_recommender.py`

```
POST /recommend {query, user_id, age, safe_mode, domain, content_type, top_k}
 1  Pydantic validation (top_k 1-50, age 0-120)
 2  AudienceContext — ALWAYS built. No age → teen ceiling (13). safe_mode → child ceiling (7).
 3  Creator match   catalog scan, whole-name/prefix/word tiers, ≤5 hits, eligibility re-checked in Python
 4  Lexical match   title exact/prefix/contains, all tokens in title+categories, phrase in first
                    1,000 chars of description; ≤ half the page; tiers 2-4 ordered by
                    within-type popularity percentile; eligibility re-checked in Python
 5  Vector search   encode query → Qdrant search with the eligibility filter;
                    5×top_k candidates when personalised, so the rerankers have room
 6  Merge           creator + lexical hits on top (fixed scores 1.2-1.3, above any cosine), then vector hits, de-duplicated
 7  Rerank chain    profile bonus → LightGCN → EMA → KG overlap + profile match
 8  Return top_k (and the domain advisory for health/finance)
```

The creator and lexical paths read the catalog directly rather than going through Qdrant, so they apply the same policy with `is_eligible()`. Every path that can return an item is gated.

### 2.8 Ranking signals, with exact formulas

| Signal | How it's computed | Weight |
| --- | --- | --- |
| Retrieval | Cosine for vector hits. Lexical: `1.30 − 0.06 × tier`. Creator: `1.2 − 0.1 × tier` + a small popularity term. | 1.0 |
| Profile bonus (MySQL history, last 500 events) | `Σ min(category weight, 3) × 0.035 + min(type weight, 3) × 0.02`; already liked −0.15 (favors new items), skipped −0.35 | added directly |
| LightGCN | cosine(user embedding, item embedding) from the trained artifact | 0.20 |
| EMA | cosine(user taste vector, item embedding) | 0.15 |
| KG overlap | `|query tokens ∩ item tokens| / max(3, |query tokens|)` over title, description, creators, categories | 0.08 |
| Profile match | Terms from the user's `preferences` JSON (interests, skills, profession and age-group term lists): +0.16 per hit, −0.25 per "avoid" hit, +0.15 for a preferred type, clamped to ±1 | 0.10 |

That's **six** additive signals. The panel deck said five; if asked to list them, say six.

### 2.9 LightGCN and the benchmark pipeline — `models/graph/`, `notebooks/lightgcn_cloud.ipynb`

**The model**
- Users and items are nodes; positive interactions are edges. Propagation uses the symmetric normalized adjacency D^-½AD^-½.
- The only parameters are the embedding tables (d=64). There are no feature transforms and no nonlinearities.
- 3 propagation layers via sparse matmul; the final embedding is the mean of layers 0–3.
- **Loss: BPR**, `-log σ(s_pos − s_neg)`, plus L2 on the layer-0 embeddings. Adam, lr 1e-3, weight decay 1e-4, batch 4096, 60 epochs, seed 42.
- **Negative sampling** is uniform. Collisions are checked vectorized: each (user, item) pair is encoded as one int64 key in a sorted array and looked up with `searchsorted`, with up to 32 redraw rounds.
- **Interaction weights:** view 0.4, click 0.6, like/bookmark 1.0, complete 1.2, rating = (r − 3)/2, skip −1. Only positive weights become edges.

**The benchmark pipeline**
- Amazon Reviews 2023 ratings → ratings ≥ 4 become positives → de-duplicate.
- Iterative 10-core filter, implemented with `bincount` over factorized codes. Industrial uses 5-core, because it is 10× smaller and doesn't survive 10-core.
- Sample up to 50k users, drop items the sample left too sparse (under 5 interactions on the 10-core runs), and cache the result as parquet.
- Then three runs from that cache: the popularity baseline, the LightGCN holdout evaluation, and the final artifact.
- Runs on a Kaggle GPU. The notebook pulls `main`, so it has to be launched as a committed run.

**Evaluation protocol**
- Leave-last-one-out per user by timestamp.
- Every item is ranked, with each user's training items masked out.
- Scoring runs batched on GPU, with a fallback to CPU on out-of-memory.
- The popularity baseline ranks by *training-set* counts and also skips seen items, so the comparison is fair.

**Performance work.** The first benchmark runs looked hung. I replaced per-user Python loops with vectorized splits, `factorize` and `bincount`, and the runs became practical.

**In the app.** The artifact's embeddings are L2-normalized, and `graph_score` is a cosine. Users or items missing from the artifact get no graph score.

### 2.10 EMA taste vector — `backend/ema_recommender.py`

- `step = α · min(|w|, 1)`; `new = normalize((1 − step) · old + step · (±item_vector))`, with α = 0.25.
- A like moves the vector 25% of the way; a view moves it 10%. A skip blends toward the *negated* item vector, which pushes the profile away.
- The first event initialises the vector to the item itself.
- It lives in the same 384-d space as queries and items, so no conversion is needed anywhere.
- Each update is one read and one write to MySQL. Nothing is retrained and no item vector moves, so other users are unaffected.

### 2.11 API — `backend/app.py`

- **8 endpoints:**
  - `GET /`
  - `GET /health`
  - `GET /domains`
  - `GET /search/suggest`
  - `POST /recommend`
  - `POST /recommend/item` ("more like this", through the same age gate)
  - `POST /interactions`
  - `GET /interactions/{user}/{entity}`
- Heavy objects (model, catalog, Qdrant client) are created once in the `lifespan` hook. Endpoints get them via `Depends`, which is also what lets tests swap in fakes.
- **Errors:**
  - 400 for bad input (unknown domain or content type, empty query).
  - 503 when Qdrant, the model or the artifacts are unavailable.
  - 500 for an internal bug. It must never become a silent fallback; that is the fix made today (section 4).
- **`/interactions`** checks the item exists first, so a bad ID is a clean 400 rather than a foreign-key 500. It then logs the event and updates the EMA vector.
- **`/domains`** is built from `config/domains.yaml`, so the UI discovers verticals without a code change on either side.

### 2.12 Frontend — `frontend/src/`

- **Stack:** React 19, Vite, Tailwind, framer-motion, axios. 19 components, 2 pages: a demo login with personas, and the main page.
- **Calls** go through Vite's `/api` proxy, so there is no CORS in development. The axios timeout is 15 s, which is why the creator search had to get fast.
- **Search controls:** autocomplete (`/search/suggest`), query chips, a domain selector fed by `/domains`, audience controls (age slider, safe mode), a content filter, and an advisory banner for regulated domains.
- **Out-of-order responses.** Every search takes a sequence number, and a response that isn't the latest is dropped. Without this, a slow earlier request could overwrite the right results.
- **Result cards:**
  - a maturity badge and a "matched by" badge (creator / title / keyword)
  - a score breakdown bar
  - "Why this ranked here" (`ExplainPanel`)
  - interaction buttons with optimistic toggles that roll back on failure
  - a similar-items drawer and a detail modal
- **Counterfactuals.** Because the fusion is a plain weighted sum, removing a signal means subtracting its points from every item and re-sorting. It's exact and needs no model call. It shows when the *largest* contributor isn't the *decisive* one.
- **Signal and backend panels.** `AISignalsPanel` shows which signals are live for the top result. `BackendStatus` gives a clean offline state instead of a crash.

### 2.13 Testing and evaluation

- **328 test cases in 11 files.** The largest areas are audience tagging, preprocessing, the Open Library fetcher and the domain registry. Also covered: embeddings, eligibility filtering, LightGCN math, report consistency, the Last.fm adult filter and the API.
- **Know what each evaluation artifact proves, and what it doesn't:**
  1. **LightGCN benchmark** (`reports/domain_benchmark_table.md`). Real data, real protocol. It tests the collaborative-filtering component only, on Amazon data rather than the served catalog.
  2. **Constraint report** (`scripts/evaluate_constraints.py`).
     - It proves the cost: what each viewer can see (a child sees 4.1% of the catalog), and what an unfiltered ranking would leak (100% violations for a child).
     - Its "filter ON = 0.000" column is zero *by construction*: it filters with `is_eligible()` and then re-checks with the same function. Don't present that zero as evidence.
  3. **Scenario table** (`reports/recommendation_evaluation_table.md`). Eight hand-picked queries. These are sanity checks, not statistics.
- **What actually backs the age guarantee:**
  - The filter is part of the Qdrant query.
  - `tests/test_audience_filtering.py:153` checks that the compiled Range filter and `is_eligible()` agree across a grid of viewer ages × item minimum ages.
  - The live age-8 vs age-25 demo.
- **Missing:** a test that runs a real in-memory Qdrant search with the filter.

---

## 3. Numbers cheat sheet

| Fact | Value |
| --- | --- |
| Catalog | **120,747 × 20** — music 31,133 · movie 23,492 · health 20,000 · industrial 20,000 · book 12,402 · show 10,617 · finance 3,103 |
| Domains | entertainment (movie, show, book, music), health, industry, finance. Health and finance are risk tier 1 and carry an advisory. |
| Maturity ladder | all_ages 0 · child 7 · teen 13 · young_adult 16 · adult 18 · restricted 21 |
| Embeddings | MiniLM-L6-v2, 384-d, ≈185 MB for the catalog |
| Weights | retrieval 1.0 · LightGCN 0.20 · EMA 0.15 · KG 0.08 · profile 0.10 · plus the profile bonus |
| EMA | α = 0.25 |
| LightGCN | d = 64, L = 3, BPR, lr 1e-3, wd 1e-4, batch 4096, 60 epochs |
| API / UI | 8 endpoints · 19 components · 2 pages |
| Tests | **328 passing** |
| Coverage by viewer | child (8) 4.1% · teen / no age 54.5% · adult 100% |

**Benchmark** — Amazon Reviews 2023, ratings ≥ 4, leave-last-one-out, full ranking, K = 20:

| Domain | Users / Items / Interactions | Pop. R@20 | LightGCN R@20 | Lift | NDCG@20 |
| --- | --- | --- | --- | --- | --- |
| Movies & TV | 50,000 / 52,263 / 914,346 | 0.0182 | 0.0288 | **1.59x** | 0.0125 |
| Industrial & Sci. (5-core) | 33,959 / 19,024 / 238,752 | 0.0264 | 0.0326 | **1.24x** | 0.0138 |
| Health & Household | 50,000 / 42,640 / 809,212 | 0.0221 | 0.0121 | **0.55x** | 0.0046 |

- Say 1.59x, never "2x".
- `hit_rate == recall` because exactly one item is held out per user. Precision is recall/20 (0.02884 / 20 = 0.001442), which checks out.
- The numbers look small because the interaction matrices are about 0.04% dense. The claim is "beats popularity", not "the number is large".

---

## 4. Bugs I found and fixed

Pick three to tell well: one search/ML, one data, one systems. Each is symptom → cause → fix.

1. **"F1" returned a Punjabi song and a 3D-printer pad** while the film sat in the catalog. A two-character query carries no intent for an embedding. → Tiered lexical retrieval placed above the semantic band.
2. **A scar cream ("Skinception") ranked as a creator hit for *Inception*.** Substring matching. → Space-padded, whole-word haystacks.
3. **"racing" ranked a species-extinction documentary above *F1*, *Rush* and *Ford v Ferrari*.** Strict tier order, and descriptions weren't indexed. → Tiers 2–4 share one band ordered by popularity percentile, and the first 1,000 characters of each description are indexed.
4. **Creator search took 15–49 s per request** at 106k rows, past the UI's 15 s timeout. Row-by-row Python. → Vectorized pandas string operations on haystacks built once at startup.
5. ***Akira* and *Grave of the Fireflies* were served to an 8-year-old.** "Animation" matched the all-ages rule. → Real certifications plus the restrict-only rule.
6. **Unrated rows were treated as safe for newborns.** A CSV `NaN` became the string `"nan"`, which resolved to age 0. → `fillna` before stringifying, and unknown labels resolve to the most restrictive level.
7. **Viewers who declared no age were served 18+ items.** The API skipped the audience context when no age was sent. → It is always built, with the teen ceiling.
8. **The API could fail open** (fixed today, `backend/app.py`). On a `TypeError` it retried the search *without* the age gate and returned the unfiltered page with a 200. → Removed the fallback. The regression test `test_internal_error_is_not_retried_without_the_eligibility_gate` fails against the old code: it served "Adults Only" to an 8-year-old.
9. **Films appeared twice in results.** Embedded Qdrant kept the old sqlite file after `delete_collection`. → Purge the directory on rebuild; deterministic `uuid5` IDs.
10. **Every embedding rebuild re-encoded the whole catalog.** The cache key changed type on its CSV round-trip. → Coerce the key to strings.
11. **Two different books overwrote each other in MySQL.** Case-insensitive collation on case-sensitive IDs. → `utf8mb4_bin`.
12. **Every finance query returned a 500.** Pandas' chunked type inference turned "2010" into 2010.0, which failed response validation. → `low_memory=False`.
13. **The catalog load dropped its connection past ~100k rows.** One statement exceeded `max_allowed_packet`. → 2,000-row chunks.
14. **Stale search results overwrote fresh ones.** Responses arrived out of order. → A sequence-number guard.
15. **`graph_score` was null on every result.** The artifact referenced item IDs that no longer existed after re-sourcing. → Reseeded and retrained.

---

## 5. Limitations — say them before they're found

**Design limits**
1. **The LightGCN artifact in the app is trained on the 13 seeded demo personas** (347 items, 792 interactions).
   > The implementation is complete and tested. With no real user base, the artifact in the app is a smoke test, which is exactly why I benchmark the model on Amazon Reviews 2023.
2. **There is no knowledge graph.** `backend/knowledge_graph.py` scores token overlap. The paper calls it "in-memory knowledge-graph scoring", so be precise if they've read it.
   > Today it's entity-overlap scoring at weight 0.08. A typed graph with path-based scoring is the next step.
3. **LightGCN loses on Health (0.55x).**
   > Collaborative filtering needs taste variance, and household goods are commodity purchases where everyone buys the same best-sellers.
   - **The deeper hypothesis:** I train with a dot product but serve cosine. An item vector's length encodes its popularity, so normalizing throws that away, and Health is the domain where popularity matters most. It's untested; the experiment is dot-product scoring on Health.
   - It's also a trade-off: normalizing is what puts the signals on one scale so the weights mean anything.
4. **The weights are hand-set, and the full hybrid isn't evaluated end to end.** There are no real interaction logs to learn from. The next step is learning-to-rank once logs exist.
5. **The evaluation protocol is leave-one-out, not a global time split.** Training can contain other users' later interactions. It's the standard protocol in the LightGCN papers, but a stricter split would cut on a global date. Single seed, no confidence intervals.
6. **Book and music maturity labels are heuristics.** The policy guarantees unknowns fail closed, not that every label is certified.
7. **Login is a demo:** a hardcoded password and personas chosen on the client.

**Known bugs** — fix before the interview, or own them
8. **Un-liking is recorded as another like.** Toggling off posts a second `like` event. The state endpoint reads it correctly by parity, but the EMA update, the profile bonus and LightGCN training all count it as positive, so un-liking pulls the taste vector *toward* the item.
9. **The score bar's "Semantic" segment isn't purely semantic** once a persona is selected. `GraphEmbeddingStore.rerank` copies the running score into `semantic_score`. So that segment includes the profile bonus, and for title hits it shows the lexical constant (~1.3) labeled as Sentence-BERT similarity. The "Profile" segment is really the 0.10 × profile-match term.
10. **The EMA update is read-modify-write**, so two events landing at once can lose one. There's also no time decay or session boundary, so one accidental click drifts the vector permanently.
11. **No connection pooling.** A personalized search opens three MySQL connections.
12. **Missing scores count as neutral, but negative cosines are penalized.** So an item unknown to the graph can outrank one the graph dislikes. Defensible, but know it's there.

---

## 6. Questions to expect

**ML / recommender**
- *Why a hybrid?*
  - Content signals handle cold start and new items. Collaborative signals capture taste that descriptions can't.
  - EMA covers the hours LightGCN needs to retrain.
  - The Health result shows no single signal wins everywhere.
- *Why LightGCN rather than an MLP?* There are no input features to transform. The LightGCN paper showed that removing feature transforms and nonlinearities improves results. Be honest that the only baseline I ran is popularity, not matrix factorization.
- *Why BPR?* The data is implicit: we only see positives, never true negatives. So the model learns pairwise orderings instead of rating values.
- *Why 3 layers?* Each layer is one more hop through the graph. Much deeper, and embeddings over-smooth until users become indistinguishable.
- *Cold start?*
  - New item: semantic and lexical retrieval need no behavior.
  - New user: no graph or EMA score yet, so ranking falls back to semantic + profile. The first interaction creates their EMA vector.
- *Why Recall/NDCG at 20, not RMSE?* It's a ranking problem on implicit feedback; there are no ratings to regress on.
- *Why a popularity baseline?* It's the strongest naive baseline on implicit data. If a graph model can't beat it, it isn't earning its cost.
- *How do you combine scores on different scales?*
  - Everything cosine-based is normalized, so those share a scale.
  - Lexical hits get fixed scores above the cosine band, so an exact title wins.
  - The rest is a hand-weighted sum. It's the weakest part of the design, and it's what makes exact counterfactual explanations possible.

**Search / retrieval**
- *Why not just embeddings?* They can't find titles or names ("F1", "Christopher Nolan"). Keyword search can't find intent ("space adventure with aliens"). Each fails where the other works.
- *Why not BM25?* A tiered pandas matcher was enough at 120k items and gives exact titles priority. BM25 or sparse vectors is the scale-up path.
- *Why MiniLM?* 384-d, fast, CPU-friendly, ~185 MB for the whole catalog. A bigger model buys accuracy I haven't shown I need. A cross-encoder reranker is the natural next step.

**Data**
- *Biggest data-quality problem?* Descriptions. A row without one embeds to noise. That drove replacing both the Google Books scrape and the 2017 movie dump.
- *How did you assign age ratings?* Board certifications for movies, restrict-only keyword rules elsewhere, fail-closed defaults. Then tell the *Akira* story.

**System design / scaling**
- *Two-stage, filter in the query — why not filter the final list?*
  - A post-filter shrinks the page unpredictably.
  - A reranker can only promote what it was handed.
  - Putting the filter in the query makes "never shown" a property of retrieval, not of every downstream step.
- *How would you scale it?*
  - Qdrant in server mode with HNSW and payload indexes on the filter fields.
  - A MySQL connection pool, with the three per-request reads collapsed into one.
  - A real search engine for the lexical side, and precomputed autocomplete.
  - Scheduled offline LightGCN retraining with versioned artifacts.
  - EMA vectors in Redis with atomic updates.
- *Is Qdrant doing ANN here?* The app runs Qdrant embedded. As far as I know, embedded mode scans every vector exactly, which is fine at this size. HNSW applies in server mode. Verify before saying "ANN" about the running system.

**Frontend**
- *How does the explanation work?* Additive fusion lets me split each score into per-signal points, and re-rank the whole page with one signal removed. If asked about the breakdown's accuracy, see limitation 9.
- *Race conditions?* Each search carries a sequence number, and stale responses are dropped.

**Behavioral**
- *What was hardest?* The age layer. It isn't one feature; it's a property every retrieval path has to keep, and most bugs in section 4 were places where it leaked.
- *What did the project teach you?* The result where my model lost to popularity was the most useful one, because it forced a real explanation and a design argument.
- *What would you do differently?* Build the evaluation harness and the constraint layer first, and treat labels as data with their own tests from day one.

---

## 7. Demo script (about 90 s)

1. Start the backend (`uvicorn backend.app:app`, ~40 s to load) and the frontend (`cd frontend; npm run dev`).
2. Search "dark violent thriller" as age 8, then as age 25. Same query, same engine, different viewers.
3. Pick Health with no age declared. It returns nothing and says why (the teen ceiling), instead of an unexplained empty page.
4. As Sci-Fi Explorer, search "space adventure", like three sci-fi items, and search again. Point at the EMA segment, then say: "nothing was retrained between those two searches."
5. Open "Why this ranked here". Show that the largest contributor isn't always the decisive one.

If the backend dies, `.\.venv\Scripts\python.exe -m pytest tests/ -q` runs in seconds with no server.

---

## 8. Before the interview

- [x] **Fail-open fallback removed** (`backend/app.py`), with a regression test. The test fails against the old code and passes now; the suite is at 328.
- [ ] Commit or stash your work in progress (slides, `ref.bib`, `review_phase2/main.tex`, two untracked images) so the repo you share is clean.
- [ ] Decide on limitations 8 and 9 (the un-like bug and the breakdown labels). Each is a small fix; owning them is also fine.
- [ ] Optional: add an in-memory Qdrant test that runs a real filtered search. It closes the evidence gap in 2.13.

---

## 9. Where older docs are wrong

| Doc | Says | Actually |
| --- | --- | --- |
| `README.md` | 106,332 items; entertainment 65,607; no `show` type | 120,747; entertainment 77,644; 7 content types |
| `review_phase2/demo_runbook.md` | "195 passed", "106,332 × 20" | 328 passed, 120,747 × 20 |
| `review_phase2/demo_runbook.md` | `evaluate_constraints` "exits non-zero if any ineligible item reaches a viewer" | True, but its filter-ON check can't fail (see 2.13) |
| `scripts/evaluate_constraints.py` docstring | "Swap in real query rankings with --qdrant" | No `--qdrant` flag exists |
| Panel deck / brief | "five scores" | Six additive signals (2.8) |
| Paper | "Qdrant approximate nearest neighbors", "in-memory knowledge graph" | Embedded mode is likely an exact scan; the KG is token overlap |
| `build_content_catalog.py:420` comment | "Google Books ships a real cover URL" | Books now come from Open Library |
