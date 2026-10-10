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
| 2.2 | Larger-scale evaluation with Recall@K, NDCG@K, MAP, MRR | 🔄 | Experiment on 3 Amazon Reviews 2023 categories (up to 50k users / 52k items), Recall/NDCG/MAP/MRR @10 and @20. New §4 Experimental Setup written; §5.1 numbers pending the Kaggle run. | §4, §5.1 | `models/graph/evaluate_hybrid_ablation.py`, `notebooks/hybrid_ablation_kaggle.ipynb` |
| 2.3 | Ablation study; fixed weights | 🔄 | Each signal alone + tuned hybrid with each signal removed; weights selected by validation grid search and compared with the original hand-set weights; Wilcoxon tests. Procedure written in §4 ("Weight selection"); results pending. | §4, §5.1, §3.8 (l. 375+) | `evaluate_hybrid_ablation.py` |
| 2.4 | Implement baselines quantitatively (Table 3) | 🔄 | Qualitative baseline rows removed from the scenario table; Popularity, Semantic-only, EMA-only, LightGCN-only, KG-only to be reported in the new ablation table under the same split, candidates and metric code. | §5.1, Table 5 (new) | `evaluate_hybrid_ablation.py` |
| 2.5 | Reproducibility + real-time claims | 🔄 | §4 now gives datasets, filtering, chronological split, candidate set, offline signal definitions, metrics, weight grid, and a LightGCN/EMA settings table (Table 3). Latency/throughput measurement ⏳. | §4 (l. 422–473), Tables 3–4, §5.2 | report JSON `protocol` / `lightgcn` / `timings` |

## Reviewer #3

| # | Comment | Status | What we changed | Where (paper) | Where (files) |
|---|---|---|---|---|---|
| 3.1 | LightGCN training settings: layers, embedding size, negative sampling, stopping | ✅ | Table 4: 64-d, 3 layers (mean of layer outputs), N(0, 0.1²) init, Adam lr 1e-3, L2 1e-4, batch 4096, fixed 60-epoch budget with no early stopping (so validation/test never influence training), 1 uniform negative per positive resampled if observed, seed 42, T4 GPU. §3.6 states the negative-sampling rule and that graph scores use L2-normalized embeddings (Eq. 8 corrected from raw dot product to cosine, matching the code). | §3.6 (l. 327), §4, Table 4 | `models/graph/lightgcn.py`, `backend/graph_recommender.py` |
| 3.2 | How positive interactions are selected and event types weighted | ✅ | Event weights stated: view 0.4, click 0.6, like 1.0, bookmark 1.0, complete 1.2, skip −1.0, rating r → (r−3)/2. Positive = weight > 0; duplicates merged into one unweighted edge; skips/low ratings excluded from the graph (not used as explicit negatives); Amazon: rating ≥ 4. | §3.5 (l. 282), §3.6 (l. 300), Table 4 | `models/graph/lightgcn.py` |
| 3.3 | EMA initialization and how skips update it | ✅ | Eq. 3 rewritten to match the code: step η = α·min(\|w\|,1), α = 0.25, move toward sign(w)·e_i, then L2-renormalize. Profile empty until first non-neutral event, which sets u₁ = sign(w)·e_i; users without events get no EMA score; skips push the profile away; renormalization prevents collapse. | §3.5 (l. 282–296), Eq. 3 | `backend/ema_recommender.py` |
| 3.4 | Procedure for KG entities, relations, user–item paths | ✅ | §3.7 rewritten as a 3-step procedure: (i) entities (item, creator, genre, term tokens ≥ 3 chars after Unicode normalization; user, profession, age group, interest/skill), (ii) relations (directed_by / written_by / performed_by, belongs_to_genre, liked / bookmarked / completed, related_to_profession, suitable_for, unsuitable terms) with examples, (iii) length-two paths through shared entities (Eq. 9), counted by set intersection. Explains the floor of 3 in Eq. 10 and defines every symbol of Eq. 11. | §3.7 (l. 335–373) | `backend/knowledge_graph.py` |
| 3.5 | Reconcile cross-domain output rate with scenario results; define it | 🔄 | Defined: cross-domain rate = fraction of scenarios whose top-10 contains ≥ 2 content types = 3/5 = 0.60 (Taylor Swift, business analytics, space adventure), matching the Domains@10 column; all scenario metrics defined precisely (PersonHit@5 = creator-match items, averages over top-5 as in the evaluation script); abstract wording aligned. **Open issue:** the scenario numbers come from an older catalog; rerunning `scripts/evaluate_recommendation_table.py` on the current catalog gives different values — decision pending. | §5.3 (l. 483–506), Table 6, Abstract | `scripts/evaluate_recommendation_table.py`, `reports/recommendation_evaluation_table.csv` |
| 3.6 | Ablation of semantic, EMA, graph, KG/profile signals | 🔄 | Same experiment as 2.3; §4 explains that the profile signal has no offline counterpart and is covered by the scenarios. | §4, §5.1 | `evaluate_hybrid_ablation.py` |

## Other corrections found during revision (not raised by reviewers)

| Change | Where |
|---|---|
| LLMRec reference had the wrong authors ("L. Wu et al."); corrected to Wei W, Ren X, et al. (WSDM 2024), verified on Crossref | References |
| Graph score equation used a raw dot product; the system uses cosine of normalized embeddings — equation corrected | §3.6, Eq. 8 |
| Contribution 3 claimed "quantitative and qualitative improvements over single-signal recommenders" without evidence; now describes the actual evaluation | §1 |
| Paragraph after Table 1 said full benchmarking was future work; now points to §4 | §2 |
| Old Fig. 3 (bar chart restating the scenario averages) removed; a new results figure may replace it | §5 |

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
