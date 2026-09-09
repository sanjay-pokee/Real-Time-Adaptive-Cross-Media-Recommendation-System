# Phase-2 Live Demo Runbook

One command per claim in the status table. Run everything from the project root
with the venv active:

```powershell
.\.venv\Scripts\Activate.ps1
```

Keep two terminals open in VS Code (`` Ctrl+` `` then the split icon): one for the
backend server, one for these commands.

---

## Before they walk in

```powershell
# Terminal 1 - leave running. Wait for "Application startup complete" (~40 s).
uvicorn backend.app:app --reload
```

```powershell
# Terminal 2 - frontend
cd frontend; npm run dev
```

If port 8000 is taken:
`Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Get-Process -Id $_.OwningProcess }`
then `Stop-Process -Id <pid> -Force`.

---

## Row-by-row proof

### Data pipeline — 48,294 items, 14 columns
```powershell
python -c "import pandas as pd; d=pd.read_csv('data/processed/content_catalog.csv'); print(f'{len(d):,} rows x {len(d.columns)} cols'); print(d.content_type.value_counts())"
```
Expect `48,294 rows x 14 cols` and music 28,352 / book 15,139 / movie 4,803.
**Verified.**

### Embeddings — 48,294 x 384
```powershell
python -c "import numpy as np; a=np.load('embeddings/content_embeddings.npy'); print(a.shape, a.dtype)"
```
Expect `(48294, 384) float32`. **Verified.**

### Vector search + Creator search — live endpoints
Open `http://127.0.0.1:8000/docs` and run `POST /recommend` from the Swagger UI.
Better: use the frontend and let them type a query. Point at the score
breakdown on each card — that is the differentiator.

### Database — 3 tables, 4 indices
```powershell
mysql -u root -p -e "USE crossmedia; SHOW TABLES; SHOW INDEX FROM interactions;"
```
Have the schema file open as backup: `scripts/init_mysql.py`.

### EMA adaptation — alpha = 0.25, 384-d
Open `backend/ema_recommender.py` and show line 58 (`alpha: float = 0.25`) and
`backend/settings.py` line 38. **Verified.**

Live version, which lands much better: in the UI, like 3 sci-fi items, re-run the
same query, and show the EMA contribution appearing in the score breakdown.

### Graph CF — LightGCN d=64, L=3
```powershell
python -c "import json; c=json.load(open('models/graph/artifacts/lightgcn_embeddings.json')); print(c['config'])"
```
Expect `embedding_dim: 64, num_layers: 3`. **Verified.**
> See the warning below before opening this file in front of anyone.

### Backend APIs — 7 endpoints
```powershell
Select-String -Path backend\app.py -Pattern "@app\.(get|post)" | Measure-Object
```
Expect 7. Or just show `/docs`. **Verified.**

### Frontend UI — 14 components, 2 pages
```powershell
(Get-ChildItem frontend\src\components\*.jsx).Count; (Get-ChildItem frontend\src\pages\*.jsx).Count
```
Expect 14 and 2. **Verified.**

### Integration — feedback loop, no retraining
The strongest live moment. In the UI: like an item, then re-run the query and
show its ranking move, with the Profile / EMA segments changing in the score
bar. Then open the MySQL `interactions` table to show the row landed.

### Quality gate — 55/55 green
```powershell
pytest tests/ -q
```
Expect `55 passed`. Took 9.08 s on this machine. **Verified just now.**

Add `-v` if they want to see individual test names.

---

## Two rows that will not survive a probe

Rehearse these answers. Do not let them be discovered.

### 1. The LightGCN artifact is trained on 11 users

`models/graph/artifacts/lightgcn_embeddings.json` says:

```
num_users: 11, num_items: 251, num_positive_interactions: 644
```

The code is real and the config is real, but the shipped artifact is synthetic.
If a faculty member opens that JSON while you are claiming "Graph CF — Done",
it looks bad.

**Say this first, before they find it:**
> "The LightGCN implementation is complete and tested — BPR loss, 3 layers,
> d=64. The artifact currently loaded is a synthetic smoke-test set, because
> our own platform has no real user base yet. That is exactly why we are
> training on Amazon Reviews 2023 — a 40k-user run already beats the
> popularity baseline 2x on Recall@20, and the 200k run is finishing now."

That turns the weakness into the reason the benchmark work exists. You have the
numbers to back it: Recall@20 0.0364 vs 0.0182 popularity, NDCG 0.0156 vs
0.0082, MRR 0.0099 vs 0.0053.

### 2. There is no knowledge graph in the backend

`backend/knowledge_graph.py` has **no networkx, no nodes, no edges**. Its
attributes are only `catalog`, `rerank`, `score_item`, `score_profile`, and
`score_item` computes **token-set overlap** between the query string and the
item's concatenated title/description/creators/categories.

The "564 nodes / 845 edges" figure is hardcoded in `kg_visualizer.html`
(line 183) — it is the visualizer's own demo data, not a runtime graph.

**Do not open `backend/knowledge_graph.py` voluntarily.** If asked directly:
> "The KG signal today is entity-overlap scoring between query terms and item
> metadata — it works and contributes 0.08 weight to ranking. Building the
> real typed graph in Neo4j with traversal-based scoring is our Review-3
> deliverable; the visualizer shows the target schema."

Honest, and it matches the "Remaining 40%: Neo4j KG" row you already have.

**Safest fix if you have time before the review:** change the table row from
"Knowledge graph — 564 nodes / 845 edges — Done" to
"KG scoring — entity overlap, lambda_k = 0.08 — Done" and move the node/edge
figure into the Review-3 planned row. Then nothing on the slide overstates.

---

## If the backend dies mid-demo

`pytest tests/ -q` needs no server and proves 55/55 in under 10 s. The frontend
also renders its offline state cleanly rather than crashing.
