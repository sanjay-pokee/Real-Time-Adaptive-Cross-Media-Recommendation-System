"""
Rebuild Qdrant collection only (embeddings .npy already patched).
Run this with uvicorn STOPPED.
"""
from __future__ import annotations
import sys, uuid
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
NPY_PATH     = PROJECT_ROOT / "embeddings" / "content_embeddings.npy"
INDEX_PATH   = PROJECT_ROOT / "embeddings" / "content_embedding_index.csv"

print("Loading catalog …")
catalog = pd.read_csv(CATALOG_PATH)
print(f"  {len(catalog)} rows")

print("Loading embedding matrix …")
matrix = np.load(NPY_PATH).astype(np.float32)
print(f"  shape={matrix.shape}")

assert matrix.shape[0] == len(catalog), "Matrix/catalog row count mismatch!"

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:
    sys.exit("pip install qdrant-client")

from backend.settings import get_settings
settings = get_settings()

print(f"\nConnecting to Qdrant at: {settings.qdrant_path or settings.qdrant_url}")
if settings.qdrant_path:
    client = QdrantClient(path=settings.qdrant_path)
else:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

dim = matrix.shape[1]
col = settings.qdrant_collection

print(f"Recreating collection '{col}' (dim={dim}) …")
# Use new API to avoid deprecation warning
if client.collection_exists(col):
    client.delete_collection(col)
client.create_collection(col, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

def clean(v):
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except Exception: pass
    if isinstance(v, np.generic): return v.item()
    return v

BATCH = 512   # larger batch = faster upsert
points = []
total = len(catalog)
uploaded = 0

print(f"Uploading {total} points in batches of {BATCH} …")
for pos, (_, row) in enumerate(catalog.iterrows()):
    payload = {k: clean(row.get(k, "")) for k in
               ["global_id","content_type","source","source_id",
                "title","description","creators","categories",
                "release_date","popularity","rating"]}
    points.append(PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(row["global_id"]))),
        vector=matrix[pos].tolist(),
        payload=payload,
    ))
    if len(points) >= BATCH:
        client.upsert(collection_name=col, points=points)
        uploaded += len(points)
        points = []
        pct = uploaded / total * 100
        print(f"  {uploaded}/{total}  ({pct:.0f}%)", flush=True)

if points:
    client.upsert(collection_name=col, points=points)
    uploaded += len(points)

client.close()
print(f"\n✅ Qdrant collection '{col}' rebuilt with {uploaded} points.")
print("Now restart uvicorn.")
