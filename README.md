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

External API ingestion helpers are available for TMDb movies, Open Library books, and MusicBrainz recordings. Set `TMDB_API_KEY` before using TMDb. Open Library and MusicBrainz do not need API keys, but MusicBrainz requires a descriptive `MUSICBRAINZ_USER_AGENT` for production use.

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


