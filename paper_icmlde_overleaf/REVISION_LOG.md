# Paper 1343 — Revision Log (ICMLDE 5.0)

Running record of every change made in response to the reviewers. This becomes the
point-by-point response appended to `1343_revised_version.pdf`.

- **Internal deadline:** Wed 14 Oct 2026 (faculty) · **Conference deadline:** 7 days from notice
- **Status key:** ✅ done · 🔄 in progress · ⏳ pending
- **Location:** final page numbers are filled in after the last compile; until then the
  section and `main.tex` line are recorded.

---

## Reviewer #1

| # | Comment | Status | What we changed | Where (paper) | Where (code / files) |
|---|---|---|---|---|---|
| 1.1 | Figure 1 caption is written twice | ⏳ | | | |
| 1.2 | Different text fonts on pages 4 and 5 | ⏳ | | | |
| 1.3 | Check the caption of Figure 2 | ⏳ | | | |
| 1.4 | Correct the captions of all tables | ⏳ | | | |
| 1.5 | Cite papers from 2025 onwards | ⏳ | | | |
| 1.6 | Spelling mistakes (e.g., in conclusion) | ⏳ | | | |
| 1.7 | Ensure the paper is in the template | ⏳ | | | |

## Reviewer #2

| # | Comment | Status | What we changed | Where (paper) | Where (code / files) |
|---|---|---|---|---|---|
| 2.1 | Strengthen novelty with a detailed comparison to recent cross-domain and hybrid approaches | ⏳ | | | |
| 2.2 | Larger-scale evaluation with Recall@K, NDCG@K, MAP, MRR | 🔄 | Experiment code written: chronological split on 3 Amazon Reviews 2023 domains (up to 50k users / 52k items), Recall/NDCG/MAP/MRR @10 and @20. Awaiting Kaggle run. | — | `models/graph/evaluate_hybrid_ablation.py`, `notebooks/hybrid_ablation_kaggle.ipynb` |
| 2.3 | Ablation study of semantic, EMA, LightGCN, KG/profile; fixed weights | 🔄 | Each signal alone + tuned hybrid with each signal removed. Weights now selected by grid search on a validation split (sem ∈ {0, 0.5, 1}; graph/EMA/KG ∈ {0, 0.05, …, 1.6}; selected by validation NDCG@20) and compared against the original fixed weights. Wilcoxon signed-rank significance test per variant. | — | `models/graph/evaluate_hybrid_ablation.py` (`grid_search`, `evaluate_configs`, `paired_tests`) |
| 2.4 | Implement semantic-only / graph-only / KG-only baselines quantitatively (Table 3) | 🔄 | Popularity, Semantic-only, EMA-only, LightGCN-only, KG-only implemented under identical split, candidates and metric code. Will replace the qualitative rows of Table 3. | — | `models/graph/evaluate_hybrid_ablation.py` |
| 2.5 | Reproducibility + real-time claims: data split, LightGCN params, EMA settings, weight selection, latency/throughput | 🔄 | Split, LightGCN config, EMA α and grid are recorded in each JSON report. Latency/throughput benchmark ⏳. | — | report JSON `protocol`, `lightgcn`, `timings` fields |

## Reviewer #3

| # | Comment | Status | What we changed | Where (paper) | Where (code / files) |
|---|---|---|---|---|---|
| 3.1 | LightGCN training settings: layers, embedding size, negative sampling, stopping criteria | ⏳ | | | Source of truth: `models/graph/lightgcn.py` (`LightGCNConfig`), notebook config (64-d, 3 layers, Adam lr 1e-3, L2 1e-4, batch 4096, 60 epochs, 1 uniform negative per positive with collision resampling, fixed epoch budget) |
| 3.2 | How positive interactions are selected and event types weighted | ⏳ | | | `models/graph/lightgcn.py` `POSITIVE_EVENT_WEIGHTS` (view 0.4, click 0.6, like/bookmark/rating 1.0, complete 1.2), `NEGATIVE_EVENT_WEIGHTS` (skip −1.0); Amazon: rating ≥ 4 → positive |
| 3.3 | EMA initialisation and how skips/negative events update it | ⏳ | | | `backend/ema_recommender.py` `update_profile_vector` (first event initialises; step = α·min(\|w\|,1), α = 0.25; negative events move toward −e_i; L2-normalised after each update) |
| 3.4 | Precise procedure for constructing KG entities, relations, user–item paths | ⏳ | | | `backend/knowledge_graph.py` |
| 3.5 | Reconcile cross-domain output rate (60%) with scenario results; define it | ⏳ | | | |
| 3.6 | Ablation of semantic, EMA, graph, KG/profile signals | 🔄 | Same experiment as 2.3. | — | `models/graph/evaluate_hybrid_ablation.py` |

## Editorial (committee instructions)

| Item | Status | Notes |
|---|---|---|
| English proofreading | ⏳ | |
| Cite figures taken from published work | ⏳ | |
| Strict Procedia Computer Science template (header, footer, margins, captions, equation numbers) | ⏳ | |
| Similarity ≤ 10% + upload similarity report | ⏳ | |
| Single file `1343_revised_version.pdf` with this response appended | ⏳ | |
| Acknowledgement email to info@icmlde.org | ⏳ | |

---

## Change history (chronological)

| Date | Change | Files | Reviewer point |
|---|---|---|---|
| 2026-10-10 | Added hybrid baseline + ablation evaluation (5 baselines, 2 hybrids, 4 leave-one-signal-out variants, Recall/NDCG/MAP/MRR @10/@20, validation grid search over weights, Wilcoxon tests) | `models/graph/evaluate_hybrid_ablation.py` (new) | 2.2, 2.3, 2.4, 2.5, 3.6 |
| 2026-10-10 | Metadata fetcher can fetch text for exactly the benchmark graph's items (`--asins-from`) so semantic/EMA/KG signals exist offline | `scripts/fetch_amazon_metadata.py` | 2.2, 2.4 |
| 2026-10-10 | Kaggle notebook running the ablation on Movies & TV, Health & Household, Industrial & Scientific | `notebooks/hybrid_ablation_kaggle.ipynb` (new) | 2.2, 2.3, 2.4 |
