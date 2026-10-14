# Paper 1343 — Revision Log (ICMLDE 5.0)

Running record of every change made in response to the reviewers. This becomes the
point-by-point response appended to `1343_revised_version.pdf`.

- **Internal deadline:** Wed 14 Oct 2026 (faculty) · **Conference deadline:** 7 days from notice
- **Status key:** ✅ done · 🔄 in progress · ⏳ pending
- **Locations** are section / table / figure / equation numbers and **page numbers of the revised
  10-page PDF** (verified against the final Overleaf PDF of 2026-10-11: all 10 page boundaries and every section/table/figure page identical; re-check after the final Overleaf
  compile, since a page boundary can shift by a few lines).
- **Revised layout:** §1 Introduction p.1–2 · §2 Related Work p.2–3 (Table 1 p.3) · §3 Implementation
  p.3–6 (Fig. 1 p.3, §3.3 p.4, §3.4 p.4, §3.5 p.4–5, Fig. 2 p.5, §3.6 p.5–6, §3.7 p.6) · §4 Experimental
  Setup p.6–7 (Tables 2–3 p.7) · §5 Results p.7–9 (Tables 4–5 p.8, Table 6 p.9) · §6 Conclusion p.9–10 ·
  References p.10. Equations: (1) catalog, (2) EMA update, (3) LightGCN propagation, (4) BPR loss,
  (5) KG paths, (6) KG score, (7) profile score, (8) hybrid score, (9) preference bonus; the semantic,
  EMA and graph cosine scores are inline.

---

## Round 2 (2026-10-13): faculty's 16-combination ablation + team review of the response sheet

**Status (2026-10-14):** `response.tex` rewritten ✅ · 17 scenarios + rank metrics ✅ · 16-combination Kaggle run
✅ and written into the paper (§4, §5.1, Table 4, abstract, contributions, conclusion) and the letter (Table B) ✅ ·
**10 pages** in the faithful build ✅ · ⏳ compile both files on Overleaf and check the page numbers (see "Remaining").

| Request (faculty / team, 2026-10-13) | Outcome |
|---|---|
| Faculty: put the 16 combinations in the paper | `models/graph/evaluate_hybrid_ablation.py` now evaluates all 2⁴ on/off combinations of {semantic S, EMA E, LightGCN G, KG K}, **each with its own validation-tuned weights** (first signal = 1, others from {0.05…3.2}, positive only); empty = popularity, S+E+G+K = the tuned hybrid (same run). `scripts/build_ablation_table.py` writes Table 4 (16 rows + hand-set row, R/N/MAP/MRR@20 × 3 datasets), `ablation_response_table.tex` (Table B of the letter) and prints the numbers for the text. Old keys of the JSON are unchanged, a `factorial` block is added. |
| Mention old vs new numbering for the reviewers | `response.tex`: numbering map (sections, tables, figures, equations; submitted page → revised page) and a **Submitted version / Revised version** line under every comment. Revised numbers/pages still come from `main.aux` (xr-hyper); submitted numbers/pages typed from the submitted PDF. |
| "Table 1 is updated, no new table added" | R2.1 now says Table 1 was *updated*; the map says submitted Table 1 → revised Table 1 (updated), submitted Table 2 (modules) removed, revised Tables 2, 3, 5 are new, submitted Table 3 replaced by revised Tables 4 and 6. |
| R1.2 "just write we changed…", rephrase 2.1 / 1.2 | Both rewritten to what/where, no explanation of causes. |
| R1.6 "tell all 5 mistakes", write the words | The five: "papaer" (Conclusion, p.10), "differnent" and "choosen" (§2, p.3), "retrival" (§2, p.2), "go show" (§1, p.2); the "[?]" citation (§2, p.3) listed separately. Verified by diffing the submitted text against the revised one. |
| "Whole" → "entire" | `response.tex` ("Entire manuscript"). |
| Additional corrections: say "updated", not "reviewers missed it" | Section renamed **Additional updates**: "We also updated the following points…". |
| R2.2 "not only 2–3 but at least 5" + "increase scenarios", "show results with the required metrics", "increase metrics" | Read as: scenarios (we had added only 3 to the original 5). Now **17 scenarios** (+12): 5 person queries (James Cameron, Taylor Swift, Christopher Nolan, Stephen King, Coldplay), 6 cross-media intents, 6 vertical queries (2 each health / industry / finance). New metrics per scenario: P/R/NDCG/MAP/MRR/HR@10 against rule-based relevance (person in creators, or ≥2 intent terms in title/categories/creators/description, inside the declared vertical; metadata only, never a ranker score). Paper: §5.3 + Table 6 grouped by kind of scenario (4 rows); response: Table A, one row per scenario. Caveat written in §5.3: 11 of the 12 topical queries score 1.00 (550–6,558 relevant items each), so only the person queries discriminate; recall is bounded by 10/\|relevant\|. Offline: HitRate/Precision@K were already in the JSON; shown in the text at integration. |
| R2.3 "add table" | Table B in the letter = NDCG@20 and Recall@20 of the 16 combinations (filled from the Kaggle JSONs by `scripts.inline_response_tables`). |
| R2.4 "qualitative and quantitative, don't overlap" | R2.4 now has a Quantitative part (first rows of Table 4) and a Qualitative part (what each baseline captures, §4; per-dataset behaviour, §5.1) and covers only the baselines; R2.3 covers only the 16 combinations and weights. |
| "Table should not be in the top of the page" | All paper tables are now `[!ht]` (were `[!t]`): Tables 2, 3, 5 and 6 sit under their text. Tables 1 and 4 still move to the next page top when they do not fit under their paragraph (normal float behaviour; `[H]` would leave blank half-pages with `\flushbottom`). In the letter, tables are inline with `\Needspace` so a caption never separates from its table. |

**Scenario results (live API, 17 scenarios, `reports/scenario_evaluation_live.json`, `scenario_ablation_live.json`):**
the original 8 reproduce the earlier numbers exactly. Cross-domain 8/17 overall, 8/11 without a domain filter
(6/17 and 6/11 with movies + TV shows as one medium); PersonHit@5 0.88 (4/5, 5/5, 5/5, 4/5, 4/5; the misses are
items with the name in the title but not in `creators`); ProfileAlign@5 1.00; mean KG@5 0.697, profile@5 0.858;
mean P/NDCG/MAP/MRR@10 0.947/0.943/0.913/0.941 (person queries only 0.84/0.82/0.73/0.80); EMA score on 170/170
returned items, LightGCN on 7. Ablation: w/o profile ProfileAlign@5 1.00→0.96, profile@5 0.858→0.795, NDCG@10
0.943→0.916, 0.71 of the top 5 change (Taylor Swift 5/5→2/5); w/o KG KG@5 0.697→0.664, NDCG@10 0.943→0.931, 0.47
change; person matching unchanged, cross-domain 8/17→9/17 in both (one scenario gains a TV show), 0 unsuitable items.
Regenerate with `python -m scripts.build_scenario_table`.

### Numbering of the submitted vs the revised manuscript

| Item | Submitted (10 p.) | Revised |
|---|---|---|
| Sections | 1 Intro p.2 · 2 Related work p.2–3 · 3 Implementation p.4–8 (3.1 p.4, 3.2 p.4, 3.3 p.5, 3.4 p.5, 3.5 EMA p.6, 3.6 LightGCN p.6, 3.7 KG p.7, 3.8 pipeline p.8, 3.9 API/frontend p.8) · 4 Results p.9 · 5 Conclusion p.10 | 1 · 2 · 3 (3.1 overview, 3.2 catalog, 3.3 = old 3.3+3.4, 3.4 EMA, 3.5 LightGCN, 3.6 KG, 3.7 hybrid; old 3.9 merged into 3.1) · **4 Experimental setup (new)** · 5 Results (5.1 baselines/ablation, 5.2 latency, 5.3 scenarios) · 6 Conclusion |
| Tables | 1 prior families/measures p.3 · 2 modules p.8 · 3 prototype results + qualitative baselines p.9 | 1 comparison with recent work (**updated** old 1) · 2 benchmark graphs (new) · 3 LightGCN/EMA settings (new) · 4 baselines + 16 combinations (replaces the qualitative rows of old 3) · 5 latency (new) · 6 scenarios (old 3, per-scenario block). Old 2 removed |
| Figures | 1 architecture p.4 · 2 KG p.7 · 3 result bar chart p.9 | 1 (redrawn) · 2 (redrawn) · old 3 removed |
| Equations | (1) catalog (2) semantic (3) EMA update (4) EMA score (5)(6) LightGCN (7) BPR (8) graph score (9) paths (10) KG (11) profile (12) hybrid | (1) catalog (2) EMA update (3) LightGCN (4) BPR (5) paths (6) KG (7) profile (8) hybrid (9) preference bonus; semantic/EMA/graph scores inline |

### 16-combination results (Kaggle run of 2026-10-14, `reports/benchmarks/*_ablation.json`; round-1 files in `round1/`)

Same graphs, seed and selected weights as round 1 (ratios S:G:E:K = 1:0.8:0.2:0.05 Movies & TV, 1:0.8:0.4:0.1 the other two); the
full hybrid's NDCG@20 is unchanged (0.0227 / 0.0126 / 0.0283); LightGCN training 17.0 / 13.2 / 1.2 min on a T4.
- **Movies & TV, Industrial & Scientific:** S+E+G+K is the best of the 16 on R/N/MAP/MRR at K=10 and 20 (HitRate@10 on I&S tied with
  S+E+G). NDCG@20 +21.6% / +26.9% over the best single signal (semantic), ×2.3 / ×1.9 over popularity; HitRate@20 0.051→0.063,
  0.068→0.089. Best pair everywhere: S+G.
- **Removing a signal (re-tuned triples), NDCG@20:** semantic −22.0 / −8.6 / −20.6%; LightGCN −13.4 / −21.9 / −19.6%; KG −1.5 / −3.0 /
  −2.1% (p<1e-6); EMA −0.7 / +0.5 (n.s., p=0.47) / −2.7%. Mean gain from adding a signal to the 7 other non-empty combinations:
  S +86/+41/+45%, G +50/+30/+26%, KG +4/+22/+14%, EMA +39/+6/+7%.
- **Health & Household:** K alone best on every measure (+14.4% NDCG@20 over the hybrid); hybrid ties S+G+K (p=0.47, the only
  non-significant difference of all 45) and beats the other 13.
- **Tuned vs hand-set:** +5.4 / +3.6 / +6.5% (p<1e-4); tie on H&H at K=10.
- Wilcoxon: every combination vs S+E+G+K p<0.01 except S+G+K on H&H.

### Page budget (decision 2026-10-14: 10 pages, as in round 1)

Cuts, all wording or layout, no result or claim dropped: §5.3 compressed (the PA sentence and the rank-metric averages are in
Table 6); §5.2 embedded-Qdrant aside and p99 parenthesis; §3.2 SHA-256 sentence; abstract opening sentence and stack
clause; one clause each in §3.1, §3.4, §4 opening, §4 baselines paragraph, §5.1; the textbook BPR justification in §3.5;
Table 5 rows "MySQL preference bonus" and "LightGCN rerank" moved into the §5.2 sentence (6.4 ms, 0.07 ms); "Proc." for
all nine conference proceedings in the references; Fig. 1 vertical spacing 1.1 → 1.0 cm, Fig. 2 genre node 0.26 cm higher.
Faithful build: **10 pages**, p.10 nearly full (≈2–3 lines spare). Table 1 stays at the top of p.3 and Table 4 at the top of
p.8: both are too large to fit under their paragraph.

### Remaining

1. Upload `main.tex` and `response.tex` to Overleaf; compile `main.tex`, then `response.tex`.
2. Check that the Overleaf paper is 10 pages (p.10 has only ≈2–3 lines spare; if it spills, the next cut is the §2 LLM
   sentence) and that the letter's page numbers match it.
3. Build `1343_revised_version.pdf` (letter first, then the paper) from the Overleaf PDFs.
4. Similarity report (≤10%) and acknowledgement email to info@icmlde.org (see Editorial).

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
| 2.3 | Ablation study; fixed weights | ✅ round 2: all 16 combinations, each re-tuned (Table 4, letter Table B; see "16-combination results" above). The rest of this row is the round-1 fixed-weight leave-one-out, superseded | Removing LightGCN costs 14.6–21.8% NDCG@20, semantic 19.2–31.6%, EMA 0.7–2.7%, KG 1.5–3.0%; all significant (Wilcoxon p < 0.01). Validation-selected weights beat the hand-set ones by 5.4% / 3.6% / 6.5% (p < 1e-4); selected weights are stable across domains (λg ≈ 0.8, λe 0.2–0.4, λk 0.05–0.1 relative to semantic), i.e. the hand-set graph weight was 4× too low. | §4 (Baselines and weight selection), p.7; §5.1, p.7–8; Table 4, p.8 | `evaluate_hybrid_ablation.py` |
| 2.4 | Implement baselines quantitatively (Table 3) | ✅ round 2: qualitative description of each baseline added to §4; quantitative rows are the single-signal rows of Table 4 | Most-popular, semantic-only, EMA-only, LightGCN-only and KG-only baselines run under the identical split, candidates and metric code (Table 4), replacing the qualitative rows of the old Table 3. Exception reported: on Health & Household KG-only is best (+14.4% over the hybrid), discussed as a limitation. | §5.1, p.7–8; Table 4, p.8; §6, p.10 | `evaluate_hybrid_ablation.py` |
| 2.5 | Reproducibility + real-time claims | ✅ | §4 gives datasets, filtering, chronological split, candidates, offline signal definitions, metrics, weight grid and the settings table (Table 3), plus LightGCN training time (17.1 / 13.2 / 1.2 min on a T4). The full request pipeline is described, including the creator/title matching stage and the preference bonus (Eq. 9) that were in the code but not in the paper. §5.2 + Table 5: 700 requests over the 120,747-item catalog on a laptop (Ryzen 5 5600H, RTX 3050, MySQL 8.0, Qdrant 1.12.6 HNSW server): p50 363 ms, p95 455 ms, p99 507 ms; per-stage breakdown; EMA update 0.05 ms; 2.7–3.3 req/s per process; bottleneck and embedded-mode result stated. | §3.3, p.4; §3.7 Eqs. 8–9, p.6; §4, p.6–7; Table 3, p.7; §5.2, p.8–9; Table 5, p.8 | `scripts/benchmark_latency.py`, `reports/latency_benchmark.{json,md}` |

## Reviewer #3

| # | Comment | Status | What we changed | Where (paper) | Where (files) |
|---|---|---|---|---|---|
| 3.1 | LightGCN training settings: layers, embedding size, negative sampling, stopping | ✅ | Table 3: 64-d, 3 layers averaged (Eq. 3), N(0, 0.1²) init, Adam lr 1e-3, L2 1e-4, batch 4096, fixed 60 epochs without early stopping (validation/test never influence training), 1 uniform negative per positive resampled if observed, seed 42. §3.5 states the sampling rule; the graph score is the cosine of normalized embeddings (corrected from a raw dot product to match the code). | §3.5, p.4–5; §4, p.7; Table 3, p.7 | `models/graph/lightgcn.py`, `backend/graph_recommender.py` |
| 3.2 | How positive interactions are selected and event types weighted | ✅ | Event weights: view 0.4, click 0.6, like/bookmark 1.0, complete 1.2, skip −1.0, rating r → (r−3)/2. Positive = weight > 0; duplicates merged into one unweighted edge; skips/low ratings excluded from the graph; Amazon: rating ≥ 4. | §3.4, p.4; §3.5, p.4–5; Table 3, p.7 | `models/graph/lightgcn.py` |
| 3.3 | EMA initialization and how skips update it | ✅ | Eq. 2 matches the code: step η = α·min(\|w\|,1), α = 0.25, move toward sign(w)·e_i, renormalize. Profile empty until the first non-neutral event (u₁ = sign(w)·e_i); users without events get no EMA score; skips push the profile away. | §3.4, Eq. 2, p.4 | `backend/ema_recommender.py` |
| 3.4 | Procedure for KG entities, relations, user–item paths | ✅ | §3.6 as a 3-step procedure: (i) entities, (ii) relations with examples, (iii) length-two paths through shared entities (Eq. 5), counted by set intersection; floor of 3 in Eq. 6 explained; every symbol of Eq. 7 defined; Fig. 2 redrawn to match. | §3.6, p.5–6; Eqs. 5–7; Fig. 2, p.5 | `backend/knowledge_graph.py` |
| 3.5 | Reconcile cross-domain output rate with scenario results; define it | ✅ | The old scenario numbers came from a script that never called the deployed system. Eight scenarios are now run through the deployed API as demo users with declared age (and domain for health/industry/finance). Cross-domain rate defined and computed from Table 6: 4/8 overall, 4/5 without a domain filter (3/8 and 3/5 if movies and TV shows count as one medium). | §5.3, p.9; Table 6, p.9 | `scripts/evaluate_scenarios_live.py`, `reports/scenario_evaluation_live.{json,md}` |
| 3.6 | Ablation of semantic, EMA, graph, KG/profile signals | ✅ | Offline: same experiment as 2.3. Profile and KG additionally ablated in the 8 scenarios through the deployed API (weights set to 0): without profile, ProfileAlign@5 1.00 → 0.93, mean profile@5 0.892 → 0.791, ~1 of the top-5 items changes (pop user 5/5 → 2/5); without KG, mean KG@5 0.702 → 0.679, 0.38 items change; no unsuitable items in any variant because the eligibility filter removes them first. §4 explains that the declared-profile score and the preference bonus have no offline counterpart and the profile signal is covered by the scenarios. | §4, p.7; §5.1, p.7–8; Table 4, p.8 | `evaluate_hybrid_ablation.py` |

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
| Page limit | ✅ | 10 pages in round 1 and round 2 (ICMLDE 2022–23 limit: 10, extra pages charged); round 2 needed the cuts listed under "Page budget"; re-check on Overleaf |
| Similarity ≤ 10% + upload similarity report | ⏳ | Day 5 |
| Single file `1343_revised_version.pdf` with this response appended | ✅ | `reports/submission/1343_revised_version.pdf`: 5-page response + the 10-page Overleaf PDF; standalone `1343_response_to_reviewers.pdf` alongside |
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
| 2026-10-11 | Profile/KG ablation in the deployed scenarios (4 variants via the real API, Qdrant server + MySQL); paragraph added to §5.3; responses R2.3 and R3.6 updated; still 10 pages | `scripts/evaluate_scenarios_live.py --ablation`, `reports/scenario_ablation_live.{json,md}`, `main.tex` §5.3, `response.tex` | 2.3, 3.6 |
| 2026-10-11 | Final Overleaf PDF checked against the local build: same 10 pages, all 22 sections/tables/figures and all 10 page endings identical, so the response letter's page numbers hold. Final submission files built (response letter first, then the Overleaf paper) | `reports/submission/1343_revised_version.pdf`, `reports/submission/1343_response_to_reviewers.pdf` | all |
| 2026-10-13 | Full-factorial ablation: all 16 on/off combinations of the four signals, each tuned on validation (one pass over the validation users; replaces the single 1,029-point grid, so the tuned hybrid is the S+E+G+K row); `factorial` block + markdown table in the JSON; fixed-weight leave-one-out kept. Smoke-tested on Industrial & Scientific (3 epochs, partial metadata: numbers meaningless, pipeline verified: singles = baselines, tuned = full) | `models/graph/evaluate_hybrid_ablation.py` | 2.3, 3.6 |
| 2026-10-13 | Table generator writes the 16-combination Table 4 (16 rows + hand-set), a compact NDCG/Recall table for the letter, a HitRate/Precision table and a printed summary (best single, best pair, drop when a signal is removed, mean gain from adding each signal) | `scripts/build_ablation_table.py` | 2.3, 3.6 |
| 2026-10-13 | Kaggle notebook text updated for the 16-combination run (same cells; pulls `origin/main`) | `notebooks/hybrid_ablation_kaggle.ipynb` | 2.3 |
| 2026-10-13 | Scenarios 8 → 17 (3 person queries, 3 cross-media intents, 3 vertical queries added); rule-based relevance judgments and P/R/NDCG/MAP/MRR/HR@10; ablation re-run on 17 scenarios; scenario table builders (paper summary by kind, per-scenario for the letter) | `scripts/evaluate_recommendation_table.py`, `scripts/evaluate_scenarios_live.py`, `scripts/build_scenario_table.py`, `reports/scenario_*_live.*`, `reports/scenario_*table.tex` | 2.2, 3.5, 3.6 |
| 2026-10-13 | §5.3 rewritten for 17 scenarios and the rank metrics; Table 6 now grouped by kind of scenario (per-scenario table moved to the letter, Table A); all tables `[!ht]`; labels `sec:overview`, `sec:catalog`, `sec:hybrid` added (no layout effect). Faithful build: 11 pages (3 lines on p.11) | `paper_icmlde_overleaf/main.tex` | 2.2, 3.5, 3.6, "table at top" |
| 2026-10-13 | Response letter rewritten: numbering map, Submitted/Revised location under every comment, shortened responses (what + where), five spelling mistakes listed, "Entire manuscript", Table 1 "updated", R2.3/R2.4 separated, "Additional updates"; Table A (17 scenarios) inline, Table B (16 combinations) pending Kaggle; `scripts/inline_response_tables.py` refreshes both from the reports | `paper_icmlde_overleaf/response.tex`, `scripts/inline_response_tables.py` | all |
| 2026-10-14 | Page-budget trims (wording only, no result removed): §5.3 compressed, §5.2 embedded-Qdrant aside and p99 parenthesis, §3.2 SHA-256 sentence, abstract stack clause, §3.1 clause, Fig. 1 spacing; R1.2 in the letter cut to one sentence. Paper still 10 pages with the §5.1 skeleton; estimated 11 pages (≈8 lines) with the finished §5.1 | `main.tex`, `response.tex` | R1.2, 2.2 |
| 2026-10-14 | Kaggle 16-combination run (needed Internet switched on in the notebook; 25.3 / 19.7 / 3.9 min). Round-1 JSONs and tables moved to `reports/benchmarks/round1/`; new JSONs in place (Kaggle download came as `.txt`, copied to `.json`) | `reports/benchmarks/*_ablation.json`, `round1/` | 2.3, 3.6 |
| 2026-10-14 | Builder: "mean gain from adding a signal" now averages over the 7 non-empty combinations without it (going from the empty combination replaces popularity, it does not add to it); hand-set row label shortened | `scripts/build_ablation_table.py` | 2.3 |
| 2026-10-14 | Results written: Table 4 (16 combinations + hand-set, inlined), §5.1 rewritten (3 paragraphs), significance/K=10 sentence, training time 17.1 → 17.0 min, abstract, contribution 3 and conclusion now state the 16-combination result; letter Table B inlined and one result sentence added to R2.3 | `main.tex`, `response.tex` | 2.3, 2.4, 3.6 |
| 2026-10-14 | Page-budget cuts (see "Page budget"): paper back to 10 pages in the faithful build; previews `reports/paper_round2_preview.pdf`, `reports/response_round2_preview.pdf` | `main.tex`, `reports/` | editorial |
