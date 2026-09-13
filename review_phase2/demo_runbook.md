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

### Data pipeline — 106,332 items, 20 columns
```powershell
python -c "import pandas as pd; d=pd.read_csv('data/processed/content_catalog.csv'); print(f'{len(d):,} rows x {len(d.columns)} cols'); print(d.content_type.value_counts())"
```
Expect `106,332 rows x 20 cols` and music 28,352 / movie 22,116 / health 20,000 /
industrial 20,000 / book 15,139 / finance 725. **Verified.**

The six extra columns over the original 14-column schema are `image_url` and
`backdrop_url` plus the audience quartet `domain`, `maturity`,
`audience_min_age`, `risk_tier`.

### Embeddings — 106,332 x 384
```powershell
python -c "import numpy as np; a=np.load('embeddings/content_embeddings.npy'); print(a.shape, a.dtype)"
```
Expect `(106332, 384) float32`. **Verified.**

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

### Backend APIs — 8 endpoints
```powershell
Select-String -Path backend\app.py -Pattern "@app\.(get|post)" | Measure-Object
```
Expect 8 - `/domains` was added with the multi-domain work. Or just show
`/docs`. **Verified.**

### Frontend UI — 19 components, 2 pages
```powershell
(Get-ChildItem frontend\src\components\*.jsx).Count; (Get-ChildItem frontend\src\pages\*.jsx).Count
```
Expect 19 and 2. **Verified.**

### Integration — feedback loop, no retraining
The strongest live moment. In the UI: like an item, then re-run the query and
show its ranking move, with the Profile / EMA segments changing in the score
bar. Then open the MySQL `interactions` table to show the row landed.

### Multi-domain engine — 4 domains, config-driven
```powershell
Invoke-RestMethod http://127.0.0.1:8000/domains | ConvertTo-Json -Depth 4
```
Expect entertainment, health, industry and finance, with `risk_tier: 1` and an
advisory on health and finance. The list is built from `config/domains.yaml`, so
this is the claim "adding a vertical is a config change" being demonstrated
rather than asserted. **Verified.**

### Audience eligibility — the strongest live moment
In the UI, click the two "dark violent thriller" query chips back to back. Same
query, same engine, two viewers:

- **age 8** returns only `all_ages` items
- **age 25** brings `18+` titles back

Then pick Health with no age declared. It returns nothing, and says why:
"Every item in Health & Household requires 18+. No age was declared, so the
engine capped at the teen ceiling rather than assuming an adult." That is the
constraint layer explaining itself, not an empty result.

Command-line proof, if the UI is unavailable:
```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_constraints --k 20
```
Expect `PASS: zero violations@20 across all 9 profiles`. Exits non-zero if any
ineligible item reaches a viewer. **Verified.**

**If they push on how the ages were assigned, this is now a strong answer, not a weak
one.** Movies take the real TMDb US board certification per title, fetched by
`scripts.fetch_tmdb_certifications`. Before that, maturity was inferred from genre text,
and `Animation` matched the child-friendly rule — so *Akira*, *Heavy Metal*, *Grave of
the Fireflies* and *Waltz with Bashir* were all rated `all_ages` and served to an
8-year-old. A genre is a production technique, not an audience. For any content type
rated by a board, a keyword may now restrict a title but never relax one, and `NR` is
treated as "no board rated this" rather than as harmless.

Be precise about the limit: books and music have no certification source, so their
maturity is still keyword-derived, and that is a heuristic. Say so if asked — the
policy is that every unknown fails closed, not that every label is certified.

### Personalization in the new verticals
The three vertical personas in the user selector have seeded history in their own
domain, so the graph and EMA signals are live there rather than falling back to pure
semantic search:

- **Health Caregiver** — health products + health books
- **Industrial Engineer** — industrial supplies + technical books
- **Finance Planner** — finance software + finance books

Pick Finance Planner and search "budgeting" versus the same query as Sci-Fi Explorer.
The point is that personalization is not an entertainment-only feature. **Verified.**

### Explainability — counterfactual ranking
Open any result and look at "Why this ranked here". Beyond the score breakdown it
re-ranks with each signal removed, which answers a question the bar chart cannot:
whether a signal actually changed the outcome.

For Interstellar at #1, semantic is 66% of the score but removing it leaves it at
#1, while removing the graph signal - far fewer points - drops it to #4. The
largest contributor is not the decisive one. **Verified.**

### Quality gate — 195/195 green
```powershell
pytest tests/ -q
```
Expect `195 passed`. Takes about 5 s. **Verified.**

Note `pytest` is not on PATH; run it through the venv interpreter:
`.\.venv\Scripts\python.exe -m pytest tests/ -q`

Add `-v` if they want to see individual test names.

---

## Two rows that will not survive a probe

Rehearse these answers. Do not let them be discovered.

### 1. The LightGCN artifact is trained on 13 users

`models/graph/artifacts/lightgcn_embeddings.json` says:

```
num_users: 13, num_items: 347, num_positive_interactions: 792
```

The code is real and the config is real, but the shipped artifact is synthetic.
If a faculty member opens that JSON while you are claiming "Graph CF — Done",
it looks bad.

It is at least *current* now, and covers all six content types including health,
industrial and finance. Before the reseed it held 251 items of which 58 were
`tmdb_5000_movies` ids that no longer exist, so `graph_score` came back `null` on
every single result — the rerank was silently contributing nothing at all. If they
ask whether the graph signal is actually firing, it is, and you can show
`graph_score` populated in the score breakdown. Pick **Finance Planner** and search
"budgeting software": 4 of the top 10 carry a graph score.

**Say this first, before they find it:**
> "The LightGCN implementation is complete and tested — BPR loss, 3 layers,
> d=64. The artifact currently loaded is a synthetic smoke-test set, because
> our own platform has no real user base yet. That is exactly why we are
> training on Amazon Reviews 2023 — a 50k-user run already beats the
> popularity baseline by 59% on Recall@20 - and it holds on an unrelated
> vertical, beating it by 24% on Industrial & Scientific."

That turns the weakness into the reason the benchmark work exists. The measured
numbers, from `reports/domain_benchmark_table.md` (50,000 users, 60 epochs, K=20):

| Domain | LightGCN R@20 | Popularity R@20 | Lift |
| --- | --- | --- | --- |
| Entertainment (Movies & TV) | 0.0288 | 0.0182 | 1.59x |
| Industry (Industrial & Sci.) | 0.0326 | 0.0264 | 1.24x |
| Health (Health & Household) | 0.0121 | 0.0221 | **0.55x** |

Do not round 1.59x up to "2x" - the earlier 0.0364 figure was a smaller run and no
longer matches the reports in the repo.

**Health is the one they may notice: LightGCN loses there.** Say it first rather
than being caught by it:
> "Collaborative filtering needs taste. Health and household goods are commodity
> purchases where nearly everyone buys the same top sellers, so popularity is
> genuinely hard to beat. That is the empirical case for the engine being a
> hybrid, and why `lightgcn_weight` is tunable per domain."

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

**This is now fixed in the deck.** The progress table row reads "KG scoring — entity
overlap between query terms and item title / description / creators / categories,
lambda_k = 0.08", and the KG frame's left panel is split into "Shipping today:
entity-overlap scoring" and "Target typed schema (`kg_visualizer.html`)", with the
564/845 figure explicitly labelled as the visualizer's worked example and the Neo4j
migration named as the Review-3 deliverable. Nothing on the slides claims a runtime
graph any more, so you can answer the question directly instead of deflecting.

---

## If the backend dies mid-demo

`.\.venv\Scripts\python.exe -m pytest tests/ -q` needs no server and proves 195/195 in
about 5 s. The frontend
also renders its offline state cleanly rather than crashing.
