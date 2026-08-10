ICMLDE / Elsevier Procedia Overleaf Instructions

Files in this folder
- main.tex: final paper LaTeX source.
- elsarticle.cls, ecrc.sty, framed.sty: required Elsevier/Procedia template files.
- Procs.pdf, elsevier-logo-3p.pdf, SDlogo-3p.pdf: template logo/assets.
- elsarticle-harv.bst: included from template, although references are currently written directly in main.tex.

How to use in Overleaf
1. Open https://www.overleaf.com/.
2. Create a new blank project.
3. Upload every file from this folder into the project root.
4. Make sure main.tex is selected as the main file.
5. In Menu -> Compiler, select pdfLaTeX.
6. Click Recompile.

What was changed from the older paper
- Removed Team 22 wording.
- Removed the old publication-trend Fig. 3.
- Removed separate evaluation-metrics, prototype-status, unit-testing, and repeated status sections.
- Moved research gap and open challenges below the introduction.
- Kept the survey compact, around the first two pages.
- Made the rest implementation-focused: data pipeline, schema, Sentence-BERT, Qdrant, person search, MySQL interactions, EMA adaptation, LightGCN, API, frontend, KG/profile future extension.
- Merged limitations, future work, and conclusion into the final discussion/conclusion flow.

How to check page count
- After Overleaf compiles, check the PDF page count shown in the preview/downloaded PDF.
- Target is 10 to 12 pages.
- If it is below 10 pages, add one implementation figure/table or expand Section 7/8 slightly.
- If it is above 12 pages, first shorten the bibliography, then shorten Section 2 literature review or Section 11 future work.

Important editing notes
- Replace author names/affiliation if your faculty wants exact official formatting.
- Add email/corresponding author only if ICMLDE asks for it.
- Keep compiler as pdfLaTeX because the ICMLDE template is based on Elsevier Procedia.
- Do not upload the old PDF as the source. Use main.tex as the source.
