# Paper 1343 — Revision Log (ICMLDE 5.0)

Running record of every change made in response to the reviewers. This becomes the
point-by-point response appended to `1343_revised_version.pdf`.

- **Internal deadline:** Wed 14 Oct 2026 (faculty) · **Conference deadline:** 7 days from notice
- **Status key:** ✅ done · 🔄 in progress · ⏳ pending · 👀 done in source, confirm in compiled PDF
- **Location:** final page numbers are filled in after the last Overleaf compile; until then
  the section and `main.tex` line (as of 2026-10-10) are recorded.
- `[TBD]` markers in `main.tex` (16) are placeholders for the benchmark/latency numbers.

---

## Reviewer #1

| # | Comment | Status | What we changed | Where (paper) | Where (files) |
|---|---|---|---|---|---|
| 1.1 | Figure 1 caption is written twice | 👀 | The long Fig. 1 caption repeated the paragraph below it (both listed the retrieval / storage / personalization / presentation layers). Caption shortened to one line; the layer description now lives only in the text, as one sentence instead of the broken "Figure 1retrieval, …". **Still to confirm:** whether `chad.png` itself contains caption text. | §3.1, Fig. 1 (`main.tex` l. 236, 240) | `main.tex` |
| 1.2 | Different text fonts on pages 4 and 5 | 👀 | Removed every `\texttt{}` (typewriter font) from body text and captions — field names, event types, relation names, endpoints, model name, queries (11 uses in §3.2–§3.9, Fig. 2 caption, results). Removed `titlesec`, which overrides the template's heading fonts. One body font now throughout. | §3.2–§3.9, §5 | `main.tex` |
| 1.3 | Check the caption of Figure 2 | 👀 | Fig. 2 caption reduced from a 4-line paragraph (with typewriter-font relation names) to one line; the description moved into the new KG construction text. `figure*` → `figure` (single-column template) so it cannot float out of numbering order. | §3.7, Fig. 2 (l. 353) | `main.tex` |
| 1.4 | Correct the captions of all tables | ✅ | All table captions are now one-line, placed above the table, in template caption style; explanatory sentences moved to the text. Table rules switched from `booktabs` (`\midrule`, `\bottomrule`) to the template's own `\toprule` / `\colrule` / `\botrule`. Old Table 3 (qualitative baseline rows) replaced; see 2.4. | Tables 1–6 | `main.tex` |
| 1.5 | Cite papers from 2025 onwards | ✅ | Added 6 verified 2025–2026 papers (LGKAT, Sci. Rep. 2025; LightKG, KDD 2025; LLMCDSR, ACM TOIS 2025; LLM4CDSR, SIGIR 2025; LeCDSR, Inf. Fusion 2026; KLLMRec, Inf. Fusion 2026), plus the Amazon Reviews 2023 dataset paper (Hou et al.). Bibliographic details checked against Crossref / publisher pages. Now 27 references, 8 from 2025–26. | §2 new paragraph (l. 197), Table 1, References | `main.tex` |
| 1.6 | Spelling mistakes (e.g., in conclusion) | ✅ | Fixed "papaer", "differnent", "retrival", "choosen", "go show", "Relative systems", stray spaces before punctuation, run-on sentences; broken citation `\cite{kangg2018sasrec}` (rendered as **[?]**) → `kang2018sasrec`; duplicated catalog paragraph in §3.2 removed. Abstract, introduction, §2, §3 and conclusion rewritten for grammar; American spelling throughout. Full proofread still due on Day 5. | Abstract, §1–§3, §6 | `main.tex` |
| 1.7 | Ensure the paper is in the template | 👀 | Removed all layout overrides (`titlesec`, float/caption/display spacing, `\arraystretch`, `enumitem` nosep, `\linespread{1.4}\scriptsize` in references, `\vspace` hacks, duplicate `hyperref`); `booktabs` → template rules; `\runauth{Author name}` → `Sanjay A. K. et al.`; undefined affiliation label `[ca]` → `[a]`; ad-hoc e-mail block after the front matter → template `\corref` / `\cortext` / `\email` footnote (corresponding authors only, per template); references converted to Procedia style (Surname Initials. Title. Journal Year;Vol(Issue):pages) and ordered by first citation. | Front matter, preamble, References | `main.tex` |

## Reviewer #2

| # | Comment | Status | What we changed | Where (paper) | Where (files) |
|---|---|---|---|---|---|
| 2.1 | Strengthen novelty with a detailed comparison to recent cross-domain and hybrid approaches | ✅ | New paragraph on 2025–26 knowledge-aware graph and LLM-based cross-domain recommenders; new **Table 1** compares LightGCN, KGAT, LGKAT, LightKG, LLMCDSR, LLM4CDSR and this work on domains, text semantics, collaborative graph, KG, short-term adaptation, deployed system; paragraph states the trade-offs honestly (lightweight entity-overlap KG, compact encoder) and why these models are not reimplemented. Replaces the old metric-mapping table. | §2 (l. 197–224), Table 1 | `main.tex` |
| 2.2 | Larger-scale evaluation with Recall@K, NDCG@K, MAP, MRR | ✅ | Benchmark on 3 Amazon Reviews 2023 categories (Movies & TV 50,000 users / 52,213 items; Health & Household 50,000 / 42,621; Industrial & Scientific 33,959 / 18,804), full-catalog ranking, Recall/NDCG/MAP/MRR @10 and @20. New Table 5; dataset Table 3 filled from the run reports. Tuned hybrid best on Movies & TV and Industrial & Scientific on every measure (NDCG@20 +21.6% / +26.9% over the best single signal, 2.3× / 1.9× over popularity). | §4, Table 3, §5.1, Table 5, Abstract, §6 | `models/graph/evaluate_hybrid_ablation.py`, `notebooks/hybrid_ablation_kaggle.ipynb` |
| 2.3 | Ablation study; fixed weights | ✅ | Ablation: removing LightGCN costs 14.6–21.8% NDCG@20, semantic 19.2–31.6%, EMA 0.7–2.7%, KG 1.5–3.0%; all significant (Wilcoxon p < 0.01). Validation-selected weights beat the hand-set ones by 5.4% / 3.6% / 6.5% (p < 1e-4); selected weights are stable across domains (λg ≈ 0.8, λe 0.2–0.4, λk 0.05–0.1 relative to semantic), showing the hand-set graph weight was 4× too low. | §4 (Weight selection), §5.1, Table 5 | `evaluate_hybrid_ablation.py` |
| 2.4 | Implement baselines quantitatively (Table 3) | ✅ | Most-popular, semantic-only, EMA-only, LightGCN-only and KG-only baselines run under the identical split, candidate set and metric code; reported quantitatively in Table 5, replacing the qualitative rows of old Table 3. Honest exception reported: on Health & Household KG-only is best (+14.4% over the hybrid) and is discussed as a limitation. | §5.1, Table 5, §6 | `evaluate_hybrid_ablation.py` |
| 2.5 | Reproducibility + real-time claims | ✅ | §4 gives datasets, filtering, chronological split, candidate set, offline signal definitions, metrics, weight grid and a LightGCN/EMA settings table (Table 4), plus LightGCN training time (17.1 / 13.2 / 1.2 min on a T4). The full request pipeline is described completely: the lexical title/keyword stage (§3.4) and the MySQL preference bonus (Eq. 13) were in the code but missing from the paper. New §5.2 + Table 6: 700 POST /recommend requests over the 120,747-item catalog on a laptop (Ryzen 5 5600H, RTX 3050, MySQL 8.0, Qdrant 1.12.6 server with HNSW m=16, ef_construct=100): p50 363 ms, p95 455 ms, p99 507 ms; per-stage breakdown; EMA update 0.05 ms per event; throughput 2.7–3.3 req/s per process. Bottleneck (pandas lexical scan, 191 ms) and the embedded-Qdrant result (1.5 s per query, exact search) stated honestly. Abstract, conclusion and limitations updated. | §3.4, §3.8 (Eqs. 12–13), Table 2, §4, Tables 3–4, §5.2, Table 6, Abstract, §6 | `scripts/benchmark_latency.py`, `reports/latency_benchmark.{json,md}`, `reports/latency_benchmark_embedded_sample.{json,md}` |

## Reviewer #3

| # | Comment | Status | What we changed | Where (paper) | Where (files) |
|---|---|---|---|---|---|
| 3.1 | LightGCN training settings: layers, embedding size, negative sampling, stopping | ✅ | Table 4: 64-d, 3 layers (mean of layer outputs), N(0, 0.1²) init, Adam lr 1e-3, L2 1e-4, batch 4096, fixed 60-epoch budget with no early stopping (so validation/test never influence training), 1 uniform negative per positive resampled if observed, seed 42, T4 GPU. §3.6 states the negative-sampling rule and that graph scores use L2-normalized embeddings (Eq. 8 corrected from raw dot product to cosine, matching the code). | §3.6 (l. 327), §4, Table 4 | `models/graph/lightgcn.py`, `backend/graph_recommender.py` |
| 3.2 | How positive interactions are selected and event types weighted | ✅ | Event weights stated: view 0.4, click 0.6, like 1.0, bookmark 1.0, complete 1.2, skip −1.0, rating r → (r−3)/2. Positive = weight > 0; duplicates merged into one unweighted edge; skips/low ratings excluded from the graph (not used as explicit negatives); Amazon: rating ≥ 4. | §3.5 (l. 282), §3.6 (l. 300), Table 4 | `models/graph/lightgcn.py` |
| 3.3 | EMA initialization and how skips update it | ✅ | Eq. 3 rewritten to match the code: step η = α·min(\|w\|,1), α = 0.25, move toward sign(w)·e_i, then L2-renormalize. Profile empty until first non-neutral event, which sets u₁ = sign(w)·e_i; users without events get no EMA score; skips push the profile away; renormalization prevents collapse. | §3.5 (l. 282–296), Eq. 3 | `backend/ema_recommender.py` |
| 3.4 | Procedure for KG entities, relations, user–item paths | ✅ | §3.7 rewritten as a 3-step procedure: (i) entities (item, creator, genre, term tokens ≥ 3 chars after Unicode normalization; user, profession, age group, interest/skill), (ii) relations (directed_by / written_by / performed_by, belongs_to_genre, liked / bookmarked / completed, related_to_profession, suitable_for, unsuitable terms) with examples, (iii) length-two paths through shared entities (Eq. 9), counted by set intersection. Explains the floor of 3 in Eq. 10 and defines every symbol of Eq. 11. | §3.7 (l. 335–373) | `backend/knowledge_graph.py` |
| 3.5 | Reconcile cross-domain output rate with scenario results; define it | ✅ | The old scenario numbers came from a script that never called the deployed system (own substring candidate pool selected with each scenario's expected terms, KG rerank only). Scenarios are now run through the deployed POST /recommend (Qdrant server, MySQL profiles, EMA, LightGCN, KG) as the demo users, with the profile's declared age and, for health/industry/finance, the declared domain, as the frontend does. 8 scenarios (5 original + 3 domain-scoped). Cross-domain rate defined and computed from the table: 4/8 = 0.50 overall, 4/5 without a domain filter (3/8 and 3/5 if movies and TV shows count as one medium); domain-scoped scenarios are single-domain by design. PersonHit@5 redefined as top-5 results crediting the person (4/5 and 5/5). Abstract and conclusion use the same numbers. | §5.3, Table 7, Abstract, §6 | `scripts/evaluate_scenarios_live.py` (new), `reports/scenario_evaluation_live.{json,md}` |
| 3.6 | Ablation of semantic, EMA, graph, KG/profile signals | ✅ | Same experiment as 2.3: semantic, EMA, LightGCN and KG each removed from the tuned hybrid and each run alone. §4 explains that the declared-profile score and the preference bonus have no offline counterpart; the profile signal is covered by the prototype scenarios (§5.3). | §4, §5.1, Table 5 | `evaluate_hybrid_ablation.py` |

## Other corrections found during revision (not raised by reviewers)

| Change | Where |
|---|---|
| LLMRec reference had the wrong authors ("L. Wu et al."); corrected to Wei W, Ren X, et al. (WSDM 2024), verified on Crossref | References |
| Graph score equation used a raw dot product; the system uses cosine of normalized embeddings — equation corrected | §3.6, Eq. 8 |
| Contribution 3 claimed "quantitative and qualitative improvements over single-signal recommenders" without evidence; now describes the actual evaluation | §1 |
| Paragraph after Table 1 said full benchmarking was future work; now points to §4 | §2 |
| Old Fig. 3 (bar chart restating the scenario averages) removed; a new results figure may replace it | §5 |
| Data sources were out of date: books come from the Open Library API (not a Google Books-style dataset); music from a Spotify songs export plus the Last.fm API; TV shows, health, industrial and finance items were not mentioned. Paragraph rewritten with per-type counts (120,747 items, 7 content types); Last.fm reference added | §3.2, References |
| Catalog scope stated consistently (movies, TV shows, books, music, extended to health, industrial and finance) | Abstract, §1, contribution 1, §6 |
| The deployed LightGCN model covers only the demo log (13 users, 347 items), so its score was available for 1 of 80 scenario results; stated in §5.3 and as a limitation, with its effect measured on the benchmark instead | §5.3, §6 |
| Lexical title/keyword matching (5 tiers, fixed score 1.30 − 0.06·tier, at most max(3, ⌊K/2⌋) slots) was part of the system but not described | §3.4 |
| MySQL preference bonus B_pref (category / content-type weights from the last 500 events, −0.15 / −0.35 for already-seen items) and 5K candidate pool for known users were part of the ranking but missing from Eq. 12 | §3.8, Eqs. 12–13, Table 2 |

## Editorial (committee instructions)

| Item | Status | Notes |
|---|---|---|
| English proofreading | 🔄 | Grammar pass done on all rewritten sections; full proofread Day 5 |
| Cite figures taken from published work | ✅ | Fig. 1 and Fig. 2 are original (drawn by the authors); no external figures used |
| Strict Procedia template | 👀 | See 1.7; confirm header/footer, margins, page count in the Overleaf PDF |
| Similarity ≤ 10% + upload similarity report | ⏳ | Day 5 |
| Single file `1343_revised_version.pdf` with this response appended | ⏳ | Day 5 |
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
