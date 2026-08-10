"""
Fast targeted re-embed script.
- Only re-encodes movies whose embedding_text changed (by text_hash comparison).
- Patches those rows directly into the existing content_embeddings.npy.
- Leaves book/music embeddings untouched.
- Then rebuilds the embedding index and Qdrant collection.

Run:
    python patch_movie_embeddings.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_PATH  = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
NPY_PATH      = PROJECT_ROOT / "embeddings" / "content_embeddings.npy"
INDEX_PATH    = PROJECT_ROOT / "embeddings" / "content_embedding_index.csv"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_VERSION = "1"
BATCH_SIZE = 128


# ── Load artifacts ──────────────────────────────────────────────────────────
print("Loading catalog …")
catalog = pd.read_csv(CATALOG_PATH)

print("Loading existing embedding index …")
old_index = pd.read_csv(INDEX_PATH)

print(f"Loading existing embedding matrix  shape={np.load(NPY_PATH).shape} …")
matrix = np.load(NPY_PATH)          # shape (N_old, 384)

# Build fast lookup: global_id → (old_row, old_hash)
old_lookup = {
    row["global_id"]: (int(row["embedding_row"]), row["text_hash"])
    for _, row in old_index.iterrows()
}

# ── Identify stale / new rows ────────────────────────────────────────────────
needs_encode_idx   = []   # positions in catalog that need new vectors
reuse_old_row      = {}   # catalog_pos → old_matrix_row

for pos, (_, row) in enumerate(catalog.iterrows()):
    gid  = row["global_id"]
    hash_ = row["text_hash"]
    if gid in old_lookup:
        old_row, old_hash = old_lookup[gid]
        if old_hash == hash_:
            reuse_old_row[pos] = old_row   # reuse
            continue
    needs_encode_idx.append(pos)

print(f"\nRows to re-encode : {len(needs_encode_idx)}")
print(f"Rows to reuse     : {len(catalog) - len(needs_encode_idx)}")

# ── Encode only the stale texts ──────────────────────────────────────────────
if needs_encode_idx:
    texts = [str(catalog.iloc[pos]["embedding_text"]) for pos in needs_encode_idx]
    print(f"\nLoading model: {EMBEDDING_MODEL}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not installed.")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Encoding {len(texts)} texts (batch_size={BATCH_SIZE}) …")
    new_vecs = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True, device=device,
    ).astype(np.float32)
else:
    new_vecs = None
    print("Nothing to encode — all embeddings are fresh.")

# ── Assemble full matrix (same row order as catalog) ─────────────────────────
dim = matrix.shape[1]
full = np.zeros((len(catalog), dim), dtype=np.float32)

new_cursor = 0
for pos in range(len(catalog)):
    if pos in reuse_old_row:
        full[pos] = matrix[reuse_old_row[pos]]
    else:
        full[pos] = new_vecs[new_cursor]
        new_cursor += 1

# ── Build updated index ───────────────────────────────────────────────────────
from datetime import datetime, timezone
now = datetime.now(timezone.utc).isoformat()
records = []
for pos, (_, row) in enumerate(catalog.iterrows()):
    records.append({
        "global_id":        row["global_id"],
        "content_type":     row["content_type"],
        "source":           row["source"],
        "source_id":        row["source_id"],
        "embedding_model":  EMBEDDING_MODEL,
        "embedding_version": EMBEDDING_VERSION,
        "text_hash":        row["text_hash"],
        "embedding_row":    pos,
        "created_at":       now,
    })
new_index = pd.DataFrame(records)

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving patched matrix and index …")
np.save(NPY_PATH, full)
new_index.to_csv(INDEX_PATH, index=False)
print(f"  Matrix : {NPY_PATH}  shape={full.shape}")
print(f"  Index  : {INDEX_PATH}  rows={len(new_index)}")

# ── Rebuild Qdrant collection ─────────────────────────────────────────────────
print("\nRebuilding Qdrant collection …")
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:
    sys.exit("qdrant-client not installed.")

from backend.settings import get_settings
settings = get_settings()

if settings.qdrant_path:
    client = QdrantClient(path=settings.qdrant_path)
else:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

client.recreate_collection(
    collection_name=settings.qdrant_collection,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
)

points = []
QDRANT_BATCH = 256

def clean(v):
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except Exception: pass
    if isinstance(v, np.generic): return v.item()
    return v

for pos, (_, row) in enumerate(catalog.iterrows()):
    payload = {k: clean(row.get(k, "")) for k in
               ["global_id","content_type","source","source_id",
                "title","description","creators","categories",
                "release_date","popularity","rating"]}
    points.append(PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(row["global_id"]))),
        vector=full[pos].tolist(),
        payload=payload,
    ))
    if len(points) >= QDRANT_BATCH:
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        points = []

if points:
    client.upsert(collection_name=settings.qdrant_collection, points=points)

client.close()
print(f"\n✅ Done!")
print(f"  Qdrant collection '{settings.qdrant_collection}' rebuilt with {len(catalog)} points.")
print(f"  Re-encoded   : {len(needs_encode_idx)} vectors (movies with cast+director)")
print(f"  Reused       : {len(catalog) - len(needs_encode_idx)} vectors (unchanged)")
print("\nRestart uvicorn to pick up the new embeddings.")
