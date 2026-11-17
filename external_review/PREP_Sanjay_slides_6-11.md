# Sanjay — Slides 6–11 + live API demo

**Order:** Aditya 1–5 → **you 6–11** → Himmu 12–17 → Eswar 18–25.
**Your time:** ~3.5 min of slides + ~60–75 s Swagger demo (on slide 9) = **~5 min**. Eswar does the UI demo on slide 19, so yours is the *backend* proof.

---

## 0. Before you walk in (do this at 9:25)

1. **Plug in the charger.** On battery the CPU throttles and a search takes 3–9 s instead of ~1.8 s. Set *Settings → System → Power → Power mode → Best performance*.
2. Start the stack: `.\scripts\demo_up.ps1` → wait for "App: http://localhost:4173".
3. Open two browser tabs: **http://127.0.0.1:8000/docs** (yours) and **http://localhost:4173** (Eswar's).
4. **Warm it up:** run `python -m scripts.api_demo` once. The first search after start-up is the slowest.
5. In Swagger, expand **POST /recommend** and click **Try it out** in advance, so you only paste and press Execute live.
6. Backup if Swagger misbehaves: `python -m scripts.api_demo --pause` (press Enter per step).

---

## 1. What to say, slide by slide

### Slide 6 — Software System Architecture (~40 s)
**Say:**
"The system has four tiers. The **React client** only talks JSON to a **FastAPI service** with 17 endpoints. The API calls the **engine**, which does eligibility first and relevance second. Underneath is **storage**: Qdrant holds 120,747 vectors of 384 dimensions, MySQL holds the catalogue mirror, interactions, profiles and accounts, plus the LightGCN embeddings and the knowledge graph in memory. At the bottom is the **offline pipeline** — fetch from TMDb, Open Library, Last.fm and Amazon, build the catalogue, encode with SBERT, load Qdrant and MySQL, train the models. Only the trained artefact crosses from offline into serving — the dashed arrow."
**Point at:** the dashed arrow, the "Stage 1: eligibility" box.

### Slide 7 — Request Lifecycle (~45 s)
**Say:**
"This is one search. The query is typo-corrected — 'hary poter' becomes 'harry potter' — and if no word is in our catalogue, we skip the vector search instead of returning random neighbours. Then the **eligibility filter**: age, safe mode, domain and risk tier are compiled **into** the Qdrant query, so an ineligible item is never retrieved. Retrieval combines two paths — exact creator and title matching, and SBERT semantic search — merged with the name/title hits capped at half the page. Then the reranking signals are **added**: profile bonus, LightGCN, EMA, knowledge graph and profile match. The bottom loop is the real-time part: a like goes to the event log and updates the user's EMA vector in 0.05 ms, so the next search already reflects it, with no retraining."
**Key sentence to land:** *"Eligibility before relevance — no reranker can bring an ineligible item back."*

### Slide 8 — Database Schema (~35 s)
**Say:**
"MySQL 8 with six tables. `content_entities` mirrors the catalogue, keyed by a case-sensitive global id like `movie:tmdb_api:27205`. `user_interactions` is an append-only event log — 988 events today — with a foreign key to the catalogue that **cascades on delete** and an index on (user, timestamp). `user_profiles` stores preferences and the 384-dimensional EMA vector. Accounts are `users` and `user_sessions` — we store only the SHA-256 of the session token. `login_failures` makes the login lockout survive a restart. Deleting an account removes everything in **one transaction**. Audience fields live in Qdrant's payload, where the filter runs."

### Slide 9 — API Contracts (~30 s + demo)
**Say:**
"17 FastAPI endpoints, every request validated by Pydantic. The main one is `POST /recommend`: query, user, age, safe mode, domain, top-k in; results out with the final score plus **six per-signal scores** and the knowledge-graph explanation. Errors are explicit — 400 for a bad domain, 422 for an invalid body, 401/403 for sessions, 429 for lockout, 503 if a dependency is down — and we **never fall back to an unfiltered search**. Let me show it live."
→ **Swagger demo (section 2).**

### Slide 10 — Implementation Overview (~35 s)
**Say:**
"Every layer is implemented and integrated — frontend, data pipeline, engine, backend, database, accounts. The one row at **90%** is models and evaluation: both models are built and benchmarked, but our own results show LightGCN ranks 1.6–1.8× better with dot-product scoring, and the app still uses cosine; and we want multi-seed runs. Deployment is one command, `demo_up.ps1`, which builds, starts and health-checks everything, and GitHub Actions runs all 390 tests on every push."
**Be ready:** "why 90%?" → the two open items above.

### Slide 11 — Data Pipeline & Catalogue (~40 s)
**Say:**
"Nine sources go into one 20-column schema — 120,747 items: 23,492 movies and 10,617 shows from the live TMDb API, 12,402 books from Open Library, 31,133 songs from Spotify plus Last.fm, 40,725 health, industrial and finance products from Amazon Reviews 2023, and 2,378 finance books from Open Library. Each source has a normaliser; a new Amazon category is just a config entry. **Age labels** for films come from real TMDb US certificates — G, PG, PG-13, R, NC-17 — and everything else from audited keyword rules; anything unknown falls to the **most** restrictive level. Embeddings are re-encoded only when an item's text changes, using a SHA-256 hash."
**Hand-off:** "Himmu will now take you through retrieval, ranking and the knowledge graph."

---

## 2. Live API demo in Swagger (~60–75 s, on slide 9)

Open **http://127.0.0.1:8000/docs**. Tell them: *"This is FastAPI's built-in interactive documentation — real HTTP calls to the running backend."*

**Call 1 — POST /recommend** → Try it out → paste → Execute:
```json
{"query": "christopher nolan", "user_id": "user_scifi", "top_k": 3, "age": 30}
```
Point at in the response: `"title": "The Dark Knight"`, `"score"`, the per-signal fields (`semantic_score`, `preference_bonus`, `ema_score`, `kg_score`, `profile_score`), `"maturity": "teen"`, and
`"kg_explanation": "Directed by Christopher Nolan (matches your search)"`. Also the **response code 200**.

**Call 2 — same endpoint, add safe mode** (change one field, Execute):
```json
{"query": "christopher nolan", "user_id": "user_scifi", "top_k": 3, "age": 30, "safe_mode": true}
```
Say: *"Same query, safe mode on — the ceiling drops to 7+. Every Nolan film is 13+ or 18+, so none can appear; it returns only all-ages and child titles like Lightyear."*

**Call 3 (if time) — bad input:**
```json
{"query": "x", "domain": "space", "top_k": 3}
```
→ **400** `Unknown domain 'space'. Use one of: entertainment, finance, health, industry` — *"it refuses rather than guessing."*

**Optional one-liners:** `GET /kg/stats` (260,109 nodes, 1,015,044 edges), `GET /health` (`{"status":"ok"}`).

**If a call takes a few seconds:** *"This laptop runs Qdrant in embedded mode, which evaluates the filter in Python; with Qdrant as a server it's 85 ms per search and 455 ms p95 end to end."*

**Why Swagger and not Postman:** it is generated from the same Pydantic models the code validates with, so the docs cannot drift from the API, and it needs no extra tool. (Postman would work too, but it is a separate tool and a hand-maintained collection.)

---

## 3. Deep knowledge for your slides

### Architecture
- **FastAPI**: async-capable, Pydantic v2 validation, auto Swagger/OpenAPI. Heavy objects (SBERT model, catalogue tables, Qdrant client, knowledge graph) are created **once** in the `lifespan` hook and injected with `Depends` — no reload per request; tests swap them for fakes.
- **Qdrant**: vector DB with **payload filtering inside the search**, which is why eligibility can't leak. Embedded mode (one process, files in `qdrant_storage/`, 605 MB) for the laptop; server mode for production.
- **MySQL**: relational data with integrity — foreign keys, transactions, unique emails.
- **Offline vs online**: fetching, encoding and training are offline; the request path never calls TMDb or trains anything.

### Request lifecycle — exact rules
- **Effective age**: no age → **13 (teen ceiling)**; safe mode → **7 (child)**; maturity ladder: all_ages 0, child 7, teen 13, young_adult 16, adult 18, restricted 21. Risk tier 2 is withheld by default.
- **Creator match**: whole-word, 3 tiers (full name, prefix, word). **Title match**: 5 tiers (exact, prefix, contains, all tokens, phrase in description). Exact hits get fixed scores 1.2–1.3, above the cosine range (≤ 1.0), so they rank first.
- **Fusion** = **weighted sum**, not a weighted average:
  `score = retrieval + profile_bonus + 0.20·LightGCN + 0.15·EMA + 0.08·KG + 0.10·profile` → can exceed 1 (e.g. 2.126). Only the order matters; dividing by the weight total (1.53) would not change the ranking.
- **EMA**: `v ← normalise((1 − αw)·v + αw·(±item))`, α = 0.25; a like moves 25% toward the item, a skip moves away.

### Database
| Table | Rows now | Key / index | Notes |
|---|---|---|---|
| content_entities | 120,747 | PK `global_id` (utf8mb4_bin), idx content_type, idx source | MEDIUMTEXT descriptions; loaded in 2,000-row chunks, stale rows pruned |
| user_interactions | 988 | PK id, idx (user_id, timestamp), idx entity_id, **FK → content_entities ON DELETE CASCADE** | like 352, rating 180, view 173, skip 143, bookmark 137 |
| user_profiles | 16 | PK user_id | preferences JSON + EMA vector JSON |
| users | 2 | PK user_id, **unique email, unique google_sub** | Argon2id hash |
| user_sessions | 1 | PK token_hash (SHA-256), idx user_id, FK → users cascade | 30-day sessions |
| login_failures | 0 | idx (email, failed_at) | lockout survives restart |

- Toggles (like/save/skip…) are read by **parity**: odd count = on. Un-liking no longer counts as a second like.
- **Connection pool** of 10 shared connections: 3.9 ms vs 18 ms per read; a burst beyond 10 falls back to a direct connection.

### API — all 17 endpoints
`GET /`, `/health`, `/domains`, `/kg/stats`, `/search/suggest` · `POST /recommend`, `/recommend/item` · `POST /interactions`, `GET /interactions/{user}/{item}` · `GET /auth/config`, `POST /auth/signup`, `/auth/login`, `/auth/google`, `/auth/logout`, `GET /auth/me`, `DELETE /auth/me` · `GET /onboarding/items`.
- **400 vs 422**: 422 = the body doesn't match the schema (Pydantic, e.g. empty query); 400 = valid shape but a business rule fails (unknown domain).
- Session cookie `nexus_session`: HttpOnly, SameSite=Lax; account ids need their own session (401 none / 403 someone else's); demo personas need no login.

### Data pipeline
| Content | Source | Items |
|---|---|---|
| Movies | TMDb API (+ credits, certificates, backdrops, franchises) | 23,492 |
| TV shows | TMDb API | 10,617 |
| Books | Open Library API | 12,402 |
| Music | Spotify CSV 28,352 + Last.fm API 2,781 | 31,133 |
| Health | Amazon Reviews 2023 | 20,000 |
| Industrial | Amazon Reviews 2023 | 20,000 |
| Finance | Amazon Software (725) + Open Library finance (2,378) | 3,103 |
- Filters at fetch: movies ≥ 100 votes, shows ≥ 25, books need a description and cover, adult music listings removed.
- **Embedding text** per type (movie = title + genres + overview + director + top-5 cast). SBERT `all-MiniLM-L6-v2`, 384-d, normalised, batch 256 on GPU; 177 MB matrix; cache key (global_id, model, version, text hash).
- Qdrant point id = `uuid5(global_id)` (stable across rebuilds).
- **Snapshot**: fetched 14 Sep 2026; newest film 3 Sep 2026. New releases need a fetch re-run.

---

## 4. Likely questions on your slides — short answers

1. **Why Qdrant, not FAISS?** FAISS has no payload filtering; we'd have to filter after ranking, which can leak. Qdrant filters inside the search. (We started with FAISS — the builder is still in the repo.)
2. **Why both MySQL and Qdrant?** Different jobs: Qdrant = similarity search with filters; MySQL = transactions, foreign keys, accounts, the event log.
3. **Why is a search 1.8 s here?** Embedded Qdrant checks the filter in Python per point; as a server it's 85 ms. Measured both.
4. **How do you stop a child seeing adult content?** The age ceiling is compiled into the vector query; no age = teen; unknown labels = most restrictive; 0 violations in the constraint evaluation.
5. **What if someone guesses passwords?** 5 wrong → 429 for 15 min, stored in MySQL, survives restart; Argon2id hashes.
6. **Is it REST?** Yes, JSON over HTTP, resource-style paths, standard status codes, OpenAPI spec auto-generated.
7. **Why does the score go above 1?** Weighted sum, not average; exact matches score 1.2–1.3 on purpose; only order matters.
8. **How are the 120,747 items kept consistent?** One 20-column schema, validation script, deterministic ids, chunked idempotent loads that prune stale rows.
9. **What happens when you add a new domain?** A YAML entry in `config/domains.yaml` (+ a normaliser config if new data); no code change to retrieval or filtering.
10. **How fresh is the data?** Snapshot from 14 Sep 2026, frozen for reproducible evaluation; fetchers are incremental and resumable.
11. **How do you deploy?** One script builds the client, starts the API, health-checks and serves; CI tests every push. Not containerised yet — Docker is future work.
12. **Why 90% on models?** Dot-product LightGCN (better offline) not yet served; single-seed benchmarks.
13. **Concurrency?** Pooled DB connections; known gap: the EMA update is an unlocked read-modify-write.
14. **What's in a recommend response?** Title, type, maturity, minimum age, final score, six signal scores, match kind, knowledge-graph path and sentence, advisory for health/finance.

---

## 5. Numbers to know cold
120,747 items · 7 types · 4 domains · 9 sources · 384-d SBERT · 17 endpoints · 6 tables · 988 events · 390 tests · CI on every push · 455 ms p95 (server) / ~1.8 s (embedded) · 85 ms Qdrant search (server) · 0 eligibility violations · 260,109 KG nodes · 1,015,044 edges · EMA α = 0.25 · weights 0.20 / 0.15 / 0.08 / 0.10 · lockout 5 tries / 15 min · pool of 10.
