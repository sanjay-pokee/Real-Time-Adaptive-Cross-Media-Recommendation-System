"""Diagnose why Iron Man / Avengers aren't showing for 'robert downey jr' query."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
NPY_PATH     = PROJECT_ROOT / "embeddings" / "content_embeddings.npy"

catalog = pd.read_csv(CATALOG_PATH)
matrix  = np.load(NPY_PATH)

# 1. Check if Iron Man / key RDJ movies are in the dataset
rdj_keywords = ["iron man", "avengers", "tropic thunder", "sherlock holmes",
                "the judge", "chaplin", "zodiac", "kiss kiss"]
print("=== RDJ movies in catalog ===")
movies = catalog[catalog["content_type"] == "movie"]
for kw in rdj_keywords:
    hits = movies[movies["title"].str.lower().str.contains(kw)]
    for _, r in hits.iterrows():
        print(f"  FOUND: '{r['title']}' | creators: {str(r['creators'])[:80]}")

print()

# 2. Encode the query and get top-30 semantic scores for movies only
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q = model.encode(["robert downey jr"], convert_to_numpy=True,
                 normalize_embeddings=True).astype("float32")[0]

movie_mask = (catalog["content_type"] == "movie").values
movie_rows = np.where(movie_mask)[0]
movie_vecs = matrix[movie_rows]

scores = movie_vecs @ q
top30_idx = np.argsort(scores)[::-1][:30]

print("=== Top-30 semantic matches (movies only) for 'robert downey jr' ===")
for rank, idx in enumerate(top30_idx, 1):
    cat_row = catalog.iloc[movie_rows[idx]]
    rdj = "** RDJ **" if "downey" in str(cat_row["creators"]).lower() else ""
    print(f"  #{rank:2d}  score={scores[idx]:.4f}  {rdj}  '{cat_row['title']}'")
    print(f"         creators: {str(cat_row['creators'])[:70]}")
