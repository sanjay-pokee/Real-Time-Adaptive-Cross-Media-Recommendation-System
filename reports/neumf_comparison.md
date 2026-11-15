# NeuMF vs LightGCN vs popularity (Amazon Reviews 2023)

Answers the Phase-2 panel's recommendation to try Neural Collaborative Filtering.
NeuMF (He et al., 2017: GMF + MLP towers, pointwise log loss, 4 sampled negatives
per positive) runs through the *same* preparation, split and metrics as the
LightGCN benchmark: ratings >= 4, k-core filter, leave-last-one-out by timestamp,
full-catalogue ranking with training items masked, K = 20, seed 42. Every row of
a domain ran on the same graph (checked when this table is built).

| Domain | Model | Recall@20 | NDCG@20 | MRR@20 | vs popularity | vs LightGCN (paper) |
| --- | --- | --- | --- | --- | --- | --- |
| Movies & TV | Popularity | 0.0182 | 0.0080 | 0.0052 | 1.00x | 0.63x |
| Movies & TV | LightGCN, cosine (Kaggle run, as in the paper) | 0.0288 | 0.0125 | 0.0080 | 1.59x | 1.00x |
| Movies & TV | NeuMF | 0.0344 | 0.0138 | 0.0081 | 1.90x | 1.19x |
| | *graph: 50,000 users, 52,263 items, 914,346 train interactions* | | | | | |
| Industrial & Scientific | Popularity | 0.0264 | 0.0111 | 0.0070 | 1.00x | 0.81x |
| Industrial & Scientific | LightGCN, cosine (Kaggle run, as in the paper) | 0.0326 | 0.0138 | 0.0087 | 1.24x | 1.00x |
| Industrial & Scientific | LightGCN, cosine (rerun on this laptop) | 0.0326 | 0.0138 | 0.0087 | 1.24x | 1.00x |
| Industrial & Scientific | LightGCN, dot product (rerun on this laptop) | 0.0589 | 0.0254 | 0.0162 | 2.23x | 1.81x |
| Industrial & Scientific | NeuMF | 0.0416 | 0.0180 | 0.0115 | 1.58x | 1.27x |
| | *graph: 33,959 users, 19,024 items, 238,752 train interactions* | | | | | |

- **NeuMF:** GMF dim 32, MLP 2x64 -> 64 -> 32 -> 16, Adam lr 1e-3, 20 epochs, batch 4096 positives.
- **LightGCN:** d = 64, 3 layers, BPR, 60 epochs, batch 4096. The backend and the paper rank by
  *cosine* (normalised embeddings); the *dot product* rows rank by the raw vectors BPR was trained
  with, whose length carries item popularity. Local reruns used an RTX 3050 laptop GPU.

Caveats: one seed per model, no hyperparameter tuning for either model, and Health & Household
was not rerun (its rating file is not on this machine).

Rebuild: `python -m scripts.build_neumf_table`. The per-run commands are in the README.
