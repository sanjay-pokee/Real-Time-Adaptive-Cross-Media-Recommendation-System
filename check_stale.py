import pandas as pd

cat = pd.read_csv("data/processed/content_catalog.csv")
idx = pd.read_csv("embeddings/content_embedding_index.csv")

movies_cat = cat[cat["content_type"] == "movie"][["global_id", "text_hash"]].rename(columns={"text_hash": "cat_hash"})
movies_idx = idx[idx["content_type"] == "movie"][["global_id", "text_hash"]].rename(columns={"text_hash": "idx_hash"})
merged = movies_cat.merge(movies_idx, on="global_id", how="left")
stale = (merged["cat_hash"] != merged["idx_hash"]).sum()

print(f"Total movies in catalog  : {len(movies_cat)}")
print(f"Movies with STALE embeds : {stale}")
print(f"Movies already fresh     : {len(movies_cat) - stale}")

# Sample embedding_text with credits
print("\n--- Sample embedding_text (with credits) ---")
for _, r in cat[cat["content_type"] == "movie"].head(3).iterrows():
    print(f"\n[{r['title']}]\n{str(r['embedding_text'])[:200]}")
