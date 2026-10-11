# Paper 1343 — Revision Log (ICMLDE 5.0)

Running record of every change made in response to the reviewers. This becomes the
point-by-point response appended to `1343_revised_version.pdf`.

- **Internal deadline:** Wed 14 Oct 2026 (faculty) · **Conference deadline:** 7 days from notice
- **Status key:** ✅ done · 🔄 in progress · ⏳ pending
- **Locations** are section / table / figure / equation numbers and **page numbers of the revised
  10-page PDF** (local build of 2026-10-11 with pdfLaTeX's exact txfonts metrics, matching Overleaf's page count; re-check after the final Overleaf
  compile, since a page boundary can shift by a few lines).
- **Revised layout:** §1 Introduction p.1–2 · §2 Related Work p.2–3 (Table 1 p.3) · §3 Implementation
  p.3–6 (Fig. 1 p.3, §3.3 p.4, §3.4 p.4, §3.5 p.4–5, Fig. 2 p.5, §3.6 p.5–6, §3.7 p.6) · §4 Experimental
  Setup p.6–7 (Tables 2–3 p.7) · §5 Results p.7–9 (Tables 4–5 p.8, Table 6 p.9) · §6 Conclusion p.9–10 ·
  References p.10. Equations: (1) catalog, (2) EMA update, (3) LightGCN propagation, (4) BPR loss,
  (5) KG paths, (6) KG score, (7) profile score, (8) hybrid score, (9) preference bonus; the semantic,
  EMA and graph cosine scores are inline.

---

## Reviewer #1

| # | Comment | Status | What we changed | Where (paper) | Where (files) |
|---|---|---|---|---|---|
| 1.1 | Figure 1 caption is written twice | ✅ | Root cause confirmed by compiling the submitted version: the architecture image itself contained the text "Fig. 1. Proposed architecture of …", and LaTeX added a second, long caption below it (submitted p.4). Figure 1 is now redrawn in TikZ (no embedded text, document font) with a single one-line caption, and also updated to the current data sources (7 content types). The paragraph that repeated the caption's layer list was rewritten. | §3.1, Fig. 1, p.3 | `main.tex` |
| 1.2 | Different text fonts on pages 4 and 5 | ✅ | Root cause confirmed in the submitted PDF: typewriter font (`\texttt`) for field names, queries, event types and relation names (submitted p.4–5, and 8 of 10 pages overall), plus two raster figures in two different fonts (Computer Modern in Fig. 1, bold Times in Fig. 2). All `\texttt` removed, URLs set in the body font, both figures redrawn in TikZ in the document font, `titlesec` removed. Font check of the revised PDF: Times on every page, no typewriter font. | Whole paper; Figs. 1–2, p.3 and p.5 | `main.tex` |
| 1.3 | Check the caption of Figure 2 | ✅ | The submitted Fig. 2 had a 4-line caption, and its edge labels contradicted the text (e.g., "User → Genre: belongs", "User → Book: directed/acted bookmarked"). Fig. 2 is redrawn in TikZ to match the KG procedure exactly (has, liked/bookmarked/completed, directed_by/written_by/performed_by, belongs_to_genre, related_to_profession, suitable_for), with a one-line caption; `figure*` → `figure`. | §3.6, Fig. 2, p.5 | `main.tex` |
| 1.4 | Correct the captions of all tables | ✅ | All six table captions are one line, above the table, in template style; explanations moved to the text. Table rules use the template's `\toprule` / `\colrule` / `\botrule` instead of `booktabs`. | Tables 1–6, p.3, 7, 8, 9 | `main.tex` |
| 1.5 | Cite papers from 2025 onwards | ✅ | Added 6 verified 2025–26 papers (LGKAT, Sci. Rep. 2025; LightKG, KDD 2025; LLMCDSR, ACM TOIS 2025; LLM4CDSR, SIGIR 2025; LeCDSR, Inf. Fusion 2026; KLLMRec, Inf. Fusion 2026) and the Amazon Reviews 2023 dataset paper; details checked on Crossref / publisher pages. 27 references, 9 of them from 2025–26. | §2, p.2–3; Table 1, p.3; References, p.10 | `main.tex` |
| 1.6 | Spelling mistakes (e.g., in conclusion) | ✅ | Fixed "papaer", "differnent", "retrival", "choosen", "go show" and others; the broken citation `kangg2018sasrec` (printed as **[?]**) fixed; duplicated catalog paragraph removed; whole paper rewritten for grammar and concision, American spelling. Final proofread due Day 5. | Whole paper | `main.tex` |
| 1.7 | Ensure the paper is in the template | ✅ | All layout overrides removed (`titlesec`, float/caption/display spacing, `\arraystretch`, `enumitem`, `\linespread{1.4}\scriptsize` in references, `\vspace` hacks, duplicate `hyperref`); running head "Author name" → "Sanjay A. K. et al."; undefined affiliation label fixed; corresponding authors via template `\corref`/`\cortext`/`\email`; references in Procedia style, in citation order. Length kept at the 10-page limit of earlier ICMLDE editions (submitted: 10 pages). | Front matter p.1; References p.10 | `main.tex` |

## Reviewer #2

| # | Comment | Status | What we changed | Where (paper) | Where (files) |
|---|---|---|---|---|---|
| 2.1 | Strengthen novelty with a detailed comparison to recent cross-domain and hybrid approaches | ✅ | New paragraph on 2025–26 knowledge-aware graph and LLM-based cross-domain recommenders; Table 1 compares LightGCN, KGAT, LGKAT, LightKG, LLMCDSR, LLM4CDSR and this work on domains, text semantics, collaborative graph, KG, short-term adaptation, deployment; the novelty paragraph states the trade-offs (entity-overlap KG, compact encoder) and why these models are not reimplemented. | §2, p.2–3; Table 1, p.3 | `main.tex` |
| 2.2 | Larger-scale evaluation with Recall@K, NDCG@K, MAP, MRR | ✅ | Benchmark on 3 Amazon Reviews 2023 categories (Movies & TV 50,000 users / 52,213 items; Health & Household 50,000 / 42,621; Industrial & Scientific 33,959 / 18,804), full-catalog ranking, Recall/NDCG/MAP/MRR @10 and @20. Tuned hybrid best on Movies & TV and Industrial & Scientific on every measure (NDCG@20 +21.6% / +26.9% over the best single signal, 2.3× / 1.9× over popularity). | §4, p.6–7; Table 2, p.7; §5.1, p.7–8; Table 4, p.8 | `models/graph/evaluate_hybrid_ablation.py`, `notebooks/hybrid_ablation_kaggle.ipynb` |
| 2.3 | Ablation study; fixed weights | ✅ | Removing LightGCN costs 14.6–21.8% NDCG@20, semantic 19.2–31.6%, EMA 0.7–2.7%, KG 1.5–3.0%; all significant (Wilcoxon p < 0.01). Validation-selected weights beat the hand-set ones by 5.4% / 3.6% / 6.5% (p < 1e-4); selected weights are stable across domains (λg ≈ 0.8, λe 0.2–0.4, λk 0.05–0.1 relative to semantic), i.e. the hand-set graph weight was 4× too low. | §4 (Baselines and weight selection), p.7; §5.1, p.7–8; Table 4, p.8 | `evaluate_hybrid_ablation.py` |
| 2.4 | Implement baselines quantitatively (Table 3) | ✅ | Most-popular, semantic-only, EMA-only, LightGCN-only and KG-only baselines run under the identical split, candidates and metric code (Table 4), replacing the qualitative rows of the old Table 3. Exception reported: on Health & Household KG-only is best (+14.4% over the hybrid), discussed as a limitation. | §5.1, p.7–8; Table 4, p.8; §6, p.10 | `evaluate_hybrid_ablation.py` |
| 2.5 | Reproducibility + real-time claims | ✅ | §4 gives datasets, filtering, chronological split, candidates, offline signal definitions, metrics, weight grid and the settings table (Table 3), plus LightGCN training time (17.1 / 13.2 / 1.2 min on a T4). The full request pipeline is described, including the creator/title matching stage and the preference bonus (Eq. 9) that were in the code but not in the paper. §5.2 + Table 5: 700 requests over the 120,747-item catalog on a laptop (Ryzen 5 5600H, RTX 3050, MySQL 8.0, Qdrant 1.12.6 HNSW server): p50 363 ms, p95 455 ms, p99 507 ms; per-stage breakdown; EMA update 0.05 ms; 2.7–3.3 req/s per process; bottleneck and embedded-mode result stated. | §3.3, p.4; §3.7 Eqs. 8–9, p.6; §4, p.6–7; Table 3, p.7; §5.2, p.8–9; Table 5, p.8 | `scripts/benchmark_latency.py`, `reports/latency_benchmark.{json,md}` |

## Reviewer #3

| # | Comment | Status | What we changed | Where (paper) | Where (files) |
|---|---|---|---|---|---|
| 3.1 | LightGCN training settings: layers, embedding size, negative sampling, stopping | ✅ | Table 3: 64-d, 3 layers averaged (Eq. 3), N(0, 0.1²) init, Adam lr 1e-3, L2 1e-4, batch 4096, fixed 60 epochs without early stopping (validation/test never influence training), 1 uniform negative per positive resampled if observed, seed 42. §3.5 states the sampling rule; the graph score is the cosine of normalized embeddings (corrected from a raw dot product to match the code). | §3.5, p.4–5; §4, p.7; Table 3, p.7 | `models/graph/lightgcn.py`, `backend/graph_recommender.py` |
| 3.2 | How positive interactions are selected and event types weighted | ✅ | Event weights: view 0.4, click 0.6, like/bookmark 1.0, complete 1.2, skip −1.0, rating r → (r−3)/2. Positive = weight > 0; duplicates merged into one unweighted edge; skips/low ratings excluded from the graph; Amazon: rating ≥ 4. | §3.4, p.4; §3.5, p.4–5; Table 3, p.7 | `models/graph/lightgcn.py` |
| 3.3 | EMA initialization and how skips update it | ✅ | Eq. 2 matches the code: step η = α·min(\|w\|,1), α = 0.25, move toward sign(w)·e_i, renormalize. Profile empty until the first non-neutral event (u₁ = sign(w)·e_i); users without events get no EMA score; skips push the profile away. | §3.4, Eq. 2, p.4 | `backend/ema_recommender.py` |
| 3.4 | Procedure for KG entities, relations, user–item paths | ✅ | §3.6 as a 3-step procedure: (i) entities, (ii) relations with examples, (iii) length-two paths through shared entities (Eq. 5), counted by set intersection; floor of 3 in Eq. 6 explained; every symbol of Eq. 7 defined; Fig. 2 redrawn to match. | §3.6, p.5–6; Eqs. 5–7; Fig. 2, p.5 | `backend/knowledge_graph.py` |
| 3.5 | Reconcile cross-domain output rate with scenario results; define it | ✅ | The old scenario numbers came from a script that never called the deployed system. Eight scenarios are now run through the deployed API as demo users with declared age (and domain for health/industry/finance). Cross-domain rate defined and computed from Table 6: 4/8 overall, 4/5 without a domain filter (3/8 and 3/5 if movies and TV shows count as one medium). | §5.3, p.9; Table 6, p.9 | `scripts/evaluate_scenarios_live.py`, `reports/scenario_evaluation_live.{json,md}` |
| 3.6 | Ablation of semantic, EMA, graph, KG/profile signals | ✅ | Same experiment as 2.3; §4 explains that the declared-profile score and the preference bonus have no offline counterpart and the profile signal is covered by the scenarios. | §4, p.7; §5.1, p.7–8; Table 4, p.8 | `evaluate_hybrid_ablation.py` |

## Other corrections found during revision (not raised by reviewers)

| Change | Where |
|---|---|
| LLMRec reference had the wrong authors ("L. Wu et al."); corrected to Wei W, Ren X, et al. (WSDM 2024) | References |
| Graph score used a raw dot product in the paper; the system uses the cosine of normalized embeddings | §3.5, p.5 |
| Contribution 3 claimed improvements over single-signal recommenders without evidence; now describes the actual evaluation | §1, p.2 |
| Old Fig. 3 (bar chart restating scenario averages) and old Table 2 (module summary) removed to stay within 10 pages | — |
| Data sources were out of date (books: Open Library API; music: Spotify export + Last.fm API; TV shows, health, industrial, finance not mentioned); rewritten with per-type counts | §3.2, p.3–4 |
| Creator/title matching (5 tiers, fixed score 1.30 − 0.06·tier) and the MySQL preference bonus were part of the ranking but not described | §3.3, p.4; §3.7, Eq. 9, p.6 |
| The audience/domain eligibility filter (undeclared viewer capped at teen, adult default for health/industry/finance) described | §3.1, p.3 |
| The deployed LightGCN model covers only the demo log (13 users, 347 items); its score was available for 1 of 80 scenario results — stated, with its effect measured on the benchmark | §5.3, p.9; §6, p.10 |

## Editorial (committee instructions)

| Item | Status | Notes |
|---|---|---|
| English proofreading | 🔄 | Grammar pass done on the whole paper; final proofread Day 5 |
| Cite figures taken from published work | ✅ | Both figures are original, drawn by the authors in TikZ |
| Strict Procedia template | ✅ | See 1.7; template spacing kept as supplied — `elsarticle.cls` sets 22 pt above and below every display equation and `ecrc.sty` forces `\flushbottom`, so the white space around equations is the template's own (the submission had overridden it to 5 pt); template header/footer untouched (note: the conference's `ecrc.sty` prints "(2025)" in the running head and copyright line — left as supplied) |
| Page limit | ✅ | 10 pages (ICMLDE 2022–23 limit: 10, extra pages charged) |
| Similarity ≤ 10% + upload similarity report | ⏳ | Day 5 |
| Single file `1343_revised_version.pdf` with this response appended | 🔄 | Draft built: `reports/1343_revised_version_DRAFT.pdf` (4-page response + 10-page paper); final version after the Overleaf compile |
| Acknowledgement email to info@icmlde.org | ⏳ | |

---

## Change history (chronological)

| Date | Change | Files | Reviewer point |
|---|---|---|---|
| 2026-10-10 | Added hybrid baseline + ablation evaluation (5 baselines, 2 hybrids, 4 leave-one-signal-out variants, Recall/NDCG/MAP/MRR @10/@20, validation grid search over weights, Wilcoxon tests) | `models/graph/evaluate_hybrid_ablation.py` (new) | 2.2, 2.3, 2.4, 2.5, 3.6 |
| 2026-10-10 | Metadata fetcher can fetch text for exactly the benchmark graph's items (`--asins-from`) | `scripts/fetch_amazon_metadata.py` | 2.2, 2.4 |
| 2026-10-10 | Kaggle notebook running the ablation on Movies & TV, Health & Household, Industrial & Scientific; run started | `notebooks/hybrid_ablation_kaggle.ipynb` (new) | 2.2, 2.3, 2.4 |
| 2026-10-10 | Template compliance: preamble overrides removed, template table rules, running head, author/affiliation labels, corresponding-author footnote | `main.tex` preamble, front matter | 1.7, 1.4 |
| 2026-10-10 | Captions: Fig. 1, Fig. 2 and all table captions shortened; Fig. 1 duplicate text removed; `figure*` → `figure` | `main.tex` | 1.1, 1.3, 1.4 |
| 2026-10-10 | All `\texttt` removed from body and captions | `main.tex` | 1.2 |
| 2026-10-10 | Language pass: abstract, §1, §2, §3, conclusion; spelling and broken citation fixed | `main.tex` | 1.6 |
| 2026-10-10 | EMA (event weights, init, negative updates), LightGCN (positives, negative sampling, cosine score) and KG construction procedure written precisely from the code | `main.tex` §3.5–§3.7 | 3.1–3.4 |
| 2026-10-10 | New §4 Experimental Setup with dataset and settings tables; Results restructured into baselines/ablation, latency, prototype scenarios; scenario metrics and cross-domain rate defined | `main.tex` §4–§5 | 2.2–2.5, 3.1, 3.5 |
| 2026-10-10 | 2025–26 related work, novelty comparison table, bibliography in Procedia style and citation order, LLMRec author fix | `main.tex` §2, References | 1.5, 2.1 |
| 2026-10-10 | Fixed notebook cell broken by shell escaping; domain loop made failure-tolerant; Kaggle run restarted | `notebooks/hybrid_ablation_kaggle.ipynb` | 2.2 |
| 2026-10-10 | Lexical stage and preference bonus added to the method description; hybrid equation completed (Eqs. 12–13); Table 2 rows updated; λ_s clarified in §4 | `main.tex` §3.4, §3.8, §4 | 2.5, 2.3 |
| 2026-10-10 | Latency/throughput benchmark (TestClient through the real POST /recommend; refuses to run if MySQL is down, since the recommender silently skips personalization then) | `scripts/benchmark_latency.py` (new) | 2.5 |
| 2026-10-10 | Generator that writes the ablation, K=10, dataset and weight tables from the Kaggle JSON reports | `scripts/build_ablation_table.py` (new) | 2.2–2.4, 3.6 |
| 2026-10-11 | Kaggle ablation run completed for all three domains (Movies & TV 29.7 min, Health & Household 23.4 min, Industrial & Scientific 5.4 min on a T4) | Kaggle output `*_ablation.json` | 2.2–2.4, 3.6 |
| 2026-10-11 | Results written into the paper: Table 3 (datasets, coverage, test users), new Table 5 (12 measures × 11 methods), §5.1 discussion including the Health & Household exception, weight-selection findings, LightGCN training time, abstract and conclusion sentences, new limitation | `main.tex` §4, §5.1, Abstract, §6 | 2.2–2.4, 3.6 |
| 2026-10-11 | Table generator: shorter labels and 3pt column spacing so 12 numeric columns fit the single-column width; marks non-significant cells instead of significant ones; test-user column | `scripts/build_ablation_table.py` | 2.2 |
| 2026-10-11 | Latency benchmark, embedded-Qdrant sample (28 requests): ~1.8 s per request, 85% in Qdrant local mode (exact search in Python, no HNSW). Kept as `reports/latency_benchmark_embedded_sample.*`; server-mode measurement pending | `reports/` | 2.5 |
| 2026-10-10 | Qdrant 1.12.6 server (official Windows release, data in `C:\Users\sanjay\qdrant-bench`) loaded with the 120,747 catalog vectors via `embeddings/build_qdrant_collection.py` (`QDRANT_PATH=""`); project `qdrant_storage/` and `.env` unchanged | — | 2.5 |
| 2026-10-10 | Full latency benchmark against the server (700 requests + 3×300 throughput); §5.2 and Table 6 written, abstract/conclusion latency sentences, throughput limitation added | `reports/latency_benchmark.*`, `main.tex` §5.2, Abstract, §6 | 2.5 |
| 2026-10-11 | Scenario evaluation through the deployed API (new script); first run without declared age/domain showed the teen-ceiling gate hiding health/industry/finance items, rerun with the frontend's age/domain declarations | `scripts/evaluate_scenarios_live.py`, `reports/scenario_evaluation_live.*` | 3.5 |
| 2026-10-11 | §5.3 and Table 7 rewritten from the live report; data sources, catalog scope, LightGCN-coverage limitation, Last.fm reference | `main.tex` §3.2, §5.3, Abstract, §1, §6, References | 3.5, 2.5 |
| 2026-10-11 | Submitted Overleaf source compiled locally (Tectonic) to reproduce what reviewers saw: confirmed the caption baked into the Fig. 1 image (submitted p.4), typewriter font on p.4–5, and the [?] citation | — | 1.1, 1.2, 1.6 |
| 2026-10-11 | Revised paper compiled and checked page by page: 15 pages → trimmed to 10 (prose condensed, module table and old Fig. 3 dropped, minor equations inline, conference names abbreviated in references); Figures 1–2 redrawn in TikZ; float placement fixed; Times on every page | `main.tex` | 1.1–1.4, 1.7 |
| 2026-10-11 | Overleaf compile gave 11 pages (one reference over) with stretched gaps around equations. Reproduced locally with pdfLaTeX's exact txfonts metrics; cause: Times metrics differ from the earlier build, and `\flushbottom` (forced by `ecrc.sty`) inflates the template's 22 pt display spacing on short pages. Fix: semantic cosine score made inline (Eq. 2 removed, later equations renumbered); result 10 pages with ~15 lines spare and no inflated gaps | `main.tex` | 1.7 |
| 2026-10-11 | Overlap check: 2.3% of the revised text matches the submitted version in 8-word runs (references excluded) | — | Editorial (similarity) |
| 2026-10-11 | Kaggle report JSONs saved to the repo; Tables 2 and 4 regenerated from them and checked row by row against main.tex (14/14 rows identical); every percentage and training time in the text recomputed and confirmed | `reports/benchmarks/*_ablation.json`, `reports/benchmarks/ablation_*.tex` | 2.2-2.4, 3.6 |
| 2026-10-11 | Point-by-point response letter written (`response.tex`): every comment quoted verbatim, response, and location; section/table/figure/equation/page numbers read from the paper build via xr-hyper so they cannot drift; labels added to subsections 3.5, 3.6, 5.1-5.3 (no layout change). Draft single PDF assembled | `paper_icmlde_overleaf/response.tex`, `main.tex`, `reports/1343_revised_version_DRAFT.pdf` | all |
