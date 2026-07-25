# Backend Master Prompt - Cross-Media Recommendation System

Copy this prompt into Codex or another coding agent when you want backend work on this repository.

---

## Role

You are a senior AI engineer, ML engineer, data engineer, backend engineer, and database architect.

Work inside the existing repository:

`D:\COLLEGE\Course\Data science and Data analytics\product`

Do not rewrite the project from scratch. First understand the current implementation, preserve working behavior, and evolve it incrementally into a production-ready backend for a cross-media recommendation system.

---

## Current Repository State

The project already contains:

- A unified content catalog pipeline under `preprocessing/`
- Dataset configuration in `config/datasets.yaml`
- Sentence-BERT embedding generation under `embeddings/`
- FAISS semantic search artifacts under `embeddings/`
- A FastAPI backend under `backend/`
- Tests under `tests/`

Current supported domains:

- Movies
- Books
- Music

Current serving flow:

```text
query text
-> Sentence-BERT query embedding
-> FAISS nearest-neighbor search
-> optional content_type filter
-> FastAPI response
```

This existing semantic backend is the baseline and must continue to work.

---

## Backend Goal

Build the backend foundation for a production-ready Universal Cross-Domain Recommendation System that can recommend content:

- Within the same domain
- Across different domains
- From natural-language queries
- From selected content items
- From user behavior over time

Example:

Searching for `Interstellar` or selecting an Interstellar-like movie should return related movies, music, books, games, anime, podcasts, courses, and future domains when those domains are available in the catalog.

For now, implement backend capability only. Do not build the frontend unless explicitly requested later.

---

## Architecture Target

The backend should evolve toward this architecture:

```text
External datasets / APIs
-> ETL and normalization
-> PostgreSQL metadata + users + interactions
-> Sentence-BERT embeddings
-> Qdrant vector search
-> candidate generation
-> collaborative score from LightGCN or fallback model
-> EMA personalization score
-> hybrid ranking
-> FastAPI response
```

No recommendation-serving endpoint should depend on live third-party APIs. External APIs are only for ingestion and periodic synchronization.

---

## Implementation Rule

Build in phases.

Do not jump directly to a huge monolithic system. Each phase must leave the backend runnable and tested.

---

## Phase 1 - Stabilize Existing Semantic Backend

Use the current FAISS backend as the working baseline.

Required work:

- Keep `POST /recommend` working.
- Keep `GET /health` working.
- Preserve current artifact paths unless there is a strong reason to change them.
- Ensure missing artifact errors are clear and actionable.
- Ensure content type aliases are normalized consistently.
- Ensure tests do not require model downloads or large datasets.

Acceptance checks:

```powershell
pytest
uvicorn backend.app:app --reload
```

Expected API behavior:

```json
{
  "query": "space adventure with emotional music",
  "top_k": 10,
  "content_type": null,
  "results": []
}
```

Results should include:

- `global_id`
- `content_type`
- `source`
- `source_id`
- `title`
- `description`
- `creators`
- `categories`
- `release_date`
- `popularity`
- `rating`
- `score`

---

## Phase 2 - Backend Domain Model

Create clear backend schemas for:

- Content entity
- Recommendation result
- User
- User interaction
- Recommendation request
- Recommendation response

Use Pydantic for API-facing schemas.

Core entity shape:

```text
global_id
content_type
source
source_id
title
description
creators
categories
release_date
popularity
rating
metadata
embedding_text
text_hash
created_at
updated_at
```

Interaction shape:

```text
user_id
entity_id
event_type
event_value
context
timestamp
```

Supported interaction events:

- `view`
- `click`
- `like`
- `bookmark`
- `rating`
- `skip`
- `complete`

Do not store raw arbitrary user input without validation.

---

## Phase 3 - Recommendation Service Abstraction

Refactor the backend so API routes call a recommendation service interface rather than directly depending on FAISS implementation details.

Recommended modules:

```text
backend/
  app.py
  schemas.py
  services/
    recommender.py
    semantic_search.py
    ranking.py
    interactions.py
```

Keep compatibility with the existing `SemanticRecommender` until a replacement is ready.

The service should support:

- Text query recommendations
- Item-based recommendations by `global_id`
- Optional domain filters
- Cross-domain recommendations
- Top-K limits
- Clear errors for unknown items

Suggested endpoints:

```text
GET  /health
POST /recommend
POST /recommend/item
POST /interactions
GET  /users/{user_id}/recommendations
```

---

## Phase 4 - Hybrid Ranking Without Heavy Infrastructure

Before adding LightGCN, implement a lightweight hybrid ranking layer that can work locally.

Combine:

- Semantic similarity score
- Popularity score
- Rating score
- Optional user preference vector score

Initial formula:

```text
final_score =
  semantic_weight * semantic_score +
  popularity_weight * normalized_popularity +
  rating_weight * normalized_rating +
  personalization_weight * personalization_score
```

Weights must be configurable.

Default:

```text
semantic_weight = 0.70
popularity_weight = 0.10
rating_weight = 0.10
personalization_weight = 0.10
```

If a score is unavailable, ranking must degrade gracefully instead of failing.

---

## Phase 5 - Interaction Logging and EMA Personalization

Add backend support for user interactions.

Minimum local implementation:

- Store interactions in a local database or append-only file during development.
- Validate event type and payload.
- Update an in-memory or persisted EMA user preference vector after positive interactions.

EMA update:

```text
user_vector = alpha * item_vector + (1 - alpha) * previous_user_vector
```

Suggested default:

```text
alpha = 0.2
```

Positive events:

- `like`
- `bookmark`
- `rating >= 4`
- `complete`

Negative events:

- `skip`
- low rating

Do not retrain collaborative filtering online. Online adaptation should be lightweight.

---

## Phase 6 - PostgreSQL Readiness

Introduce persistence boundaries without breaking local development.

Target production database:

- PostgreSQL for content metadata
- PostgreSQL for users
- PostgreSQL for interactions
- JSONB for flexible metadata

Use an abstraction so the backend can run in local-file mode during development and Postgres mode in production.

Recommended tables:

```sql
content_entities (
  global_id text primary key,
  content_type text not null,
  source text not null,
  source_id text not null,
  title text not null,
  description text,
  creators text,
  categories text,
  release_date text,
  popularity double precision,
  rating double precision,
  metadata jsonb,
  embedding_text text,
  text_hash text,
  created_at timestamptz,
  updated_at timestamptz
)
```

```sql
user_interactions (
  id bigserial primary key,
  user_id text not null,
  entity_id text not null references content_entities(global_id),
  event_type text not null,
  event_value double precision,
  context jsonb,
  timestamp timestamptz not null
)
```

```sql
user_profiles (
  user_id text primary key,
  preferences jsonb,
  ema_vector double precision[],
  created_at timestamptz,
  updated_at timestamptz
)
```

---

## Phase 7 - Qdrant Readiness

The current FAISS implementation is acceptable for local development.

Production target:

- Qdrant should replace FAISS for persistent vector search.
- Use one unified collection with `content_type` in payload.
- Keep local FAISS as a fallback backend.

Qdrant point shape:

```json
{
  "id": "movie:tmdb_5000_movies:123",
  "vector": [0.01, 0.02],
  "payload": {
    "global_id": "movie:tmdb_5000_movies:123",
    "content_type": "movie",
    "title": "Interstellar",
    "categories": "Science Fiction",
    "release_year": "2014"
  }
}
```

Create a semantic search interface that can support:

- FAISS backend
- Qdrant backend

Do not remove FAISS until Qdrant has tests and documentation.

---

## Phase 8 - Collaborative Filtering Readiness

Target collaborative model:

- LightGCN
- Offline training
- BPR loss
- User embeddings
- Item embeddings

Initial implementation can be a placeholder service that returns neutral collaborative scores until enough interaction data exists.

Expected behavior:

- If collaborative embeddings exist, use them.
- If not, return neutral scores.
- Do not fail recommendations because collaborative data is missing.

Hybrid formula:

```text
final_score =
  0.6 * semantic_score +
  0.3 * collaborative_score +
  0.1 * ema_personalization_score
```

Weights must be configurable.

---

## Backend Quality Requirements

Use:

- FastAPI
- Pydantic
- Type hints
- Clear module boundaries
- Config through environment variables or config files
- Tests for API and ranking behavior
- No live API dependency during serving

Avoid:

- Rewriting working code without need
- Hardcoding machine-specific absolute paths
- Downloading models in tests
- Requiring Postgres, Redis, or Qdrant for basic local tests
- Mixing frontend work into backend tasks

---

## Security Requirements

Backend must include or prepare for:

- Input validation
- CORS configuration
- Rate-limit-ready structure
- Environment variable based secrets
- No secrets committed to the repository
- Safe error messages

JWT authentication can be added after the recommendation and interaction APIs are stable.

---

## Testing Requirements

Add or maintain tests for:

- Health endpoint
- Recommendation endpoint
- Content type normalization
- Missing artifact behavior
- Ranking score calculation
- Interaction validation
- EMA update logic
- Item-based recommendation behavior

Tests must run without downloading external models or calling external APIs.

---

## Definition of Done

A backend phase is done only when:

- Existing tests pass.
- New behavior has focused tests.
- API docs at `/docs` reflect the endpoint schemas.
- Local development still works without production services.
- README instructions are updated if commands changed.
- Errors are clear for missing datasets, embeddings, or indexes.

---

## Immediate Next Task Recommendation

Start with this practical backend milestone:

1. Add `backend/schemas.py`.
2. Move request and response Pydantic models out of `backend/app.py`.
3. Add a ranking module with configurable hybrid scoring.
4. Keep the existing FAISS semantic recommender working.
5. Add tests for ranking and schemas.

Only after that, add interaction logging and EMA personalization.
