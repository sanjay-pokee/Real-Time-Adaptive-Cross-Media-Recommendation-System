"""Fill the ICMLDE 2026 template with paper 1343 (camera-ready content).

Edits the template in place: every run keeps the template's own formatting
(Times New Roman, theme colours); only text, sizes where needed, two pictures
and three native charts change. Conference branding is untouched apart from
the paper ID in the footer.

Usage (project venv; the template is the organisers' ICMLDE_2026_Paper_Presentation_Template.pptx):
    python reports/submission/presentation/build_slides.py <template.pptx> <out.pptx> \
        external_review/figures/search_results.jpg

Numbers come from the camera-ready paper (Tables 4, 5 and 6) and, for the unrounded
signal-removal percentages, reports/benchmarks/*_ablation.json ("factorial").
"""
import sys
from copy import deepcopy

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION,
                             XL_TICK_LABEL_POSITION)
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

TEMPLATE, OUT, SCREENSHOT = sys.argv[1], sys.argv[2], sys.argv[3]
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

PURPLE, DEEP, MAGENTA, CYAN, LAVENDER, INK, GRID = (
    "5B21B6", "4C1D95", "C026D3", "0891B2", "C7CBEA", "3F4466", "E2E5F3")
FONT = "Times New Roman"

prs = Presentation(TEMPLATE)
S = list(prs.slides)


def shape(slide, sid):
    for s in slide.shapes:
        if s.shape_id == sid:
            return s
    raise KeyError(sid)


def remove(slide, *sids):
    for sid in sids:
        el = shape(slide, sid)._element
        el.getparent().remove(el)


def _set_color(rPr, hexcolor):
    for f in rPr.findall(A + "solidFill"):
        rPr.remove(f)
    fill = etree.SubElement(rPr, A + "solidFill")
    etree.SubElement(fill, A + "srgbClr").set("val", hexcolor)
    # solidFill must precede latin/ea/cs/sym
    rPr.remove(fill)
    rPr.insert(0, fill)


def fill(shp, paras, size=None, tmpl_index=0, spc_after=None):
    """Replace the text of a shape, cloning the template paragraph/run formatting.

    paras: list of items; an item is a str, a list of (text, opts) runs, or a
    dict {"runs": [...], "size": n, "bullet": False, "spc_after": pts, "italic": b}.
    run opts: bold, italic, sub, sup, color, size.
    """
    body = shp.text_frame._txBody
    ps = body.findall(A + "p")
    tmpl = deepcopy(ps[tmpl_index])
    for p in ps:
        body.remove(p)
    for item in paras:
        if isinstance(item, str):
            item = {"runs": [(item, {})]}
        elif isinstance(item, list):
            item = {"runs": item}
        p = deepcopy(tmpl)
        runs = p.findall(A + "r")
        r0 = deepcopy(runs[0]) if runs else None
        for el in list(p):
            if el.tag in (A + "r", A + "br", A + "fld"):
                p.remove(el)
        pPr = p.find(A + "pPr")
        if item.get("bullet") is False and pPr is not None:
            for tag in ("buChar", "buAutoNum", "buNone"):
                for b in pPr.findall(A + tag):
                    pPr.remove(b)
            etree.SubElement(pPr, A + "buNone")
            pPr.set("marL", "0")
            pPr.set("indent", "0")
        sa = item.get("spc_after", spc_after)
        if sa is not None and pPr is not None:
            spc = pPr.find(A + "spcAft")
            if spc is not None:
                spc.find(A + "spcPts").set("val", str(int(sa * 100)))
        end = p.find(A + "endParaRPr")
        sz = item.get("size", size)
        for k, (text, o) in enumerate(item["runs"]):
            r = deepcopy(r0)
            rPr = r.find(A + "rPr")
            run_sz = o.get("size", sz)
            if run_sz:
                rPr.set("sz", str(int(run_sz * 100)))
            if "bold" in o:
                rPr.set("b", "1" if o["bold"] else "0")
            it = o.get("italic", item.get("italic"))
            if it is not None:
                rPr.set("i", "1" if it else "0")
            if o.get("sub"):
                rPr.set("baseline", "-25000")
            if o.get("sup"):
                rPr.set("baseline", "30000")
            if o.get("color"):
                _set_color(rPr, o["color"])
            r.find(A + "t").text = text
            if end is not None:
                end.addprevious(r)
            else:
                p.append(r)
        if end is not None and sz:
            end.set("sz", str(int(sz * 100)))
        body.append(p)
    return shp


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def style_chart(chart, title=None, legend=True):
    chart.font.name = FONT
    chart.font.size = Pt(12)
    chart.font.color.rgb = RGBColor.from_string(INK)
    if title:
        chart.has_title = True
        tf = chart.chart_title.text_frame
        tf.text = title
        run = tf.paragraphs[0].runs[0]
        run.font.size, run.font.bold, run.font.name = Pt(13), True, FONT
        run.font.color.rgb = RGBColor.from_string(DEEP)
    else:
        chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(11)
        chart.legend.font.name = FONT
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor.from_string(GRID)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(11)
    va.tick_labels.font.name = FONT
    ca = chart.category_axis
    ca.format.line.color.rgb = RGBColor.from_string(LAVENDER)
    ca.tick_labels.font.size = Pt(12)
    ca.tick_labels.font.name = FONT
    ca.has_major_gridlines = False


def color_series(chart, colors):
    for ser, c in zip(chart.plots[0].series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = RGBColor.from_string(c)
        ser.invert_if_negative = False


def labels(plot, fmt, size=10, pos=XL_LABEL_POSITION.OUTSIDE_END):
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = pos
    dl.font.size = Pt(size)
    dl.font.name = FONT
    dl.font.color.rgb = RGBColor.from_string(INK)


def rounded_box(slide, x, y, w, h, fill_hex, name):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.adjustments[0] = 0.12
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor.from_string(fill_hex)
    box.line.fill.background()
    box.shadow.inherit = False
    box.name = name
    return box


# ---------------------------------------------------------------- footer ID
for layout in prs.slide_layouts:
    for shp in layout.shapes:
        if shp.has_text_frame:
            for para in shp.text_frame.paragraphs:
                for run in para.runs:
                    if "ICMLDE-2026-XXX" in run.text:
                        run.text = run.text.replace("ICMLDE-2026-XXX", "ICMLDE-2026-1343")

# ---------------------------------------------------------------- 1 Title
s = S[0]
fill(shape(s, 33), ["Real-Time Adaptive Cross-Media Recommendation Using Semantic "
                    "Embeddings, Knowledge Graph Scoring, and Graph Collaborative Learning"], size=24)
authors = shape(s, 34)
fill(authors, [
    {"runs": [(", ".join(n.replace(" ", " ") for n in (
        "Sanjay A. K.", "Eswara Raj M.", "Aditya Vijay", "Himaneesh Reddy Jangalapalli",
        "Bagavathi Sivakumar P.", "Krishna Priya G.")), {})], "size": 15},
], tmpl_index=0)
body = authors.text_frame._txBody
# affiliation and paper-ID lines reuse their own template paragraphs
src = Presentation(TEMPLATE).slides[0]
src_ps = shape(src, 34).text_frame._txBody.findall(A + "p")
aff, pid = deepcopy(src_ps[1]), deepcopy(src_ps[2])
body.append(aff)
body.append(pid)
from pptx.text.text import _Paragraph  # noqa: E402
for p_el, text in ((aff, "Department of Computer Science and Engineering, Amrita School of Computing, "
                         "Coimbatore, Amrita Vishwa Vidyapeetham, India"),
                   (pid, "Paper ID: ICMLDE-2026-1343")):
    runs = p_el.findall(A + "r")
    for r in runs[1:]:
        p_el.remove(r)
    runs[0].find(A + "t").text = text
notes(s, "(~20 s) Good morning. I am Sanjay A. K. from Amrita Vishwa Vidyapeetham, Coimbatore. "
         "Our paper, ID 1343, presents a real-time recommender that ranks movies, TV shows, books, "
         "music and health, industrial and finance products in one catalog, combining semantic "
         "embeddings, a knowledge graph and LightGCN collaborative filtering.")

# ---------------------------------------------------------------- 2 Introduction
s = S[1]
fill(shape(s, 45), [
    "Recommenders usually stay in one medium: films, music or books",
    "Interests cross media: Interstellar, a space novel, a similar soundtrack",
    "We built one real-time recommender for 7 content types (120,747 items)",
    "It joins semantic search, a knowledge graph, LightGCN and live feedback",
], size=17, spc_after=10)
shape(s, 45).height = Inches(3.6)
frame = shape(s, 46)
frame.top, frame.height = Inches(1.5), Inches(3.15)
remove(s, 47)
pic = s.shapes.add_picture(SCREENSHOT, Inches(5.9), Inches(1.6), width=Inches(3.5))
pic.name = "Prototype screenshot"
pic._element.nvPicPr.cNvPr.set("descr", "Prototype search for christopher nolan with per-signal score bar")
cap = shape(s, 48)
cap.left, cap.top, cap.width, cap.height = Inches(5.95), Inches(3.9), Inches(3.4), Inches(0.65)
fill(cap, ["Deployed prototype: “christopher nolan” returns his films, each with a "
           "score bar split by signal"], size=12)
banner = rounded_box(s, 0.5, 5.05, 9.0, 1.05, PURPLE, "Topic banner")
tf = banner.text_frame
tf.word_wrap = True
tf.margin_left = tf.margin_right = Inches(0.3)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.LEFT
r1 = p.add_run()
r1.text = "This paper: "
r2 = p.add_run()
r2.text = ("one hybrid ranker over one catalog that adapts to every like or skip, "
           "across movies, TV, books, music and products")
for r, b in ((r1, True), (r2, False)):
    r.font.name, r.font.size, r.font.bold = FONT, Pt(17), b
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
notes(s, "(~50 s) Recommender systems are how people find content, but each platform recommends "
         "one medium. Human interest does not follow these boundaries: someone who liked "
         "Interstellar may enjoy a novel about space exploration and a soundtrack of similar tone. "
         "We built one working pipeline that puts seven content types, 120,747 items, into one "
         "catalog, and ranks them with semantic retrieval, a knowledge graph, LightGCN and a profile "
         "that updates on every like or skip. The screenshot is the deployed prototype: every result "
         "shows how much each signal contributed. I will cover the problem, the method, the setup, "
         "the results and the conclusions.")

# ---------------------------------------------------------------- 3 Problem
s = S[2]
fill(shape(s, 59), ["Recommenders are built per medium. None of the compared methods ranks films, books, "
                    "music and products in one running system using meaning, collaborative signals, "
                    "entity links and instant feedback."], size=17)
fill(shape(s, 64), ["Users whose tastes span films, books, music and products"])
fill(shape(s, 69), ["120,747 items in 7 content types; benchmark graphs ~0.04% dense"])
fill(shape(s, 74), ["Models cover one or a fixed set of domains and are trained offline"])
notes(s, "(~50 s) The problem: most deployed recommenders serve a single medium, and research "
         "models rarely combine meaning-based retrieval, collaborative learning, knowledge-graph "
         "evidence and per-event adaptation in one running system. This affects users whose tastes "
         "span media and platforms with many verticals. The scale is real: our catalog has 120,747 "
         "items in seven content types, and the interaction graphs we evaluate on are only about "
         "0.04 percent dense. Existing graph models are trained on one domain at a time, and "
         "cross-domain models are trained offline on a fixed set of domains.")

# ---------------------------------------------------------------- 4 Background
s = S[3]
tbl = shape(s, 82).table
rows = [
    ("He et al. (2020), LightGCN", "Graph convolution on the user–item graph", "One domain; no text semantics"),
    ("Hassanzadeh et al. (2025), LGKAT", "LightGCN + knowledge-aware attention", "Trained and tested per domain"),
    ("Liu et al. (2025), LLM4CDSR", "LLM item representations across domains", "Offline, fixed domains; not deployed"),
]
for i, row in enumerate(rows, start=1):
    tbl.rows[i].height = Inches(0.36)
    for j, text in enumerate(row):
        para = tbl.cell(i, j).text_frame.paragraphs[0]
        para.runs[0].text = text
        para.runs[0].font.size = Pt(13)
        for r in para.runs[1:]:
            r.text = ""
# the table is now shorter: lift the three cards and give them more height
for sid in range(83, 98):
    shape(s, sid).top -= Inches(0.3)
for sid in (83, 87, 91):
    shape(s, sid).height += Inches(0.3)
for sid in (95, 96, 97):
    shape(s, sid).height = Inches(1.9)
fill(shape(s, 95), ["Sentence-BERT cosine, LightGCN with BPR loss, entity-overlap KG scores and an EMA "
                    "user profile"], size=15)
fill(shape(s, 96), ["None of these joins text retrieval, a collaborative graph, KG/profile scores and "
                    "per-event adaptation in one running system"], size=15)
fill(shape(s, 97), ["Unified 7-type cross-media platform", "Explainable 4-signal hybrid ranker",
                    "16-combination ablation and live tests"], size=15)
notes(s, "(~60 s) LightGCN simplifies graph convolution to neighbourhood propagation, but on one "
         "domain and without text. LGKAT adds knowledge-aware attention to LightGCN, still trained "
         "per domain. LLM4CDSR uses large language models across domains, trained offline on a fixed "
         "set of domains. We build on Sentence-BERT similarity, LightGCN trained with the BPR loss, "
         "entity-overlap knowledge-graph scores and an exponential moving average of item embeddings. "
         "The gap: none of these joins all four in one running system over a multi-domain catalog. "
         "Our contributions "
         "are the unified platform, the explainable hybrid ranker, and an evaluation with all 16 "
         "signal combinations, latency measurements and live scenario tests. Our contribution is "
         "this combination, not a new neural layer.")

# ---------------------------------------------------------------- 5 Methodology
s = S[4]
for sid, text in ((108, "Unified Catalog"), (113, "Candidate Retrieval"), (118, "Hybrid Ranking"),
                  (123, "Live Feedback")):
    fill(shape(s, sid), [text])
B = {"bold": True}
sub = {"sub": True}
fill(shape(s, 124), [
    [("Catalog: ", B), ("TMDb, Open Library, Spotify + Last.fm, Amazon → one schema, 120,747 items", {})],
    [("Retrieval: ", B), ("Sentence-BERT (all-MiniLM-L6-v2, 384-d) + Qdrant HNSW; creator and title matching", {})],
    [("Ranking: ", B), ("S = S", {}), ("sem", sub), (" + B", {}), ("pref", sub), (" + λ", {}), ("g", sub),
     ("S", {}), ("graph", sub), (" + λ", {}), ("e", sub), ("S", {}), ("ema", sub), (" + λ", {}),
     ("k", sub), ("S", {}), ("kg", sub), (" + λ", {}), ("p", sub), ("S", {}), ("profile", sub)],
    [("Feedback: ", B), ("u ← (1−η)u + η·sign(w)·e", {}), ("i", sub),
     (" on every like or skip (0.05 ms); LightGCN retrained periodically", {})],
], size=16, spc_after=8)
notes(s, "(~90 s) Every source is mapped into one schema, so adding a domain only needs a new "
         "mapper. Each item's title, categories, description and creators are encoded by "
         "Sentence-BERT into 384-dimensional vectors, and Qdrant searches them with an HNSW index; "
         "creator and title matching run first, so a query like Christopher Nolan returns his "
         "films. Candidates are then reranked by the hybrid score: semantic similarity, a short-term "
         "preference bonus, the LightGCN graph score, the EMA profile score, the knowledge-graph "
         "score and the declared-profile score. Feedback is logged in MySQL; each event moves the "
         "user's EMA profile toward or away from the item in 0.05 milliseconds, so the next request "
         "already reflects it, while LightGCN is retrained periodically. An eligibility filter on "
         "declared age and domain removes unsuitable items before ranking.")

# ---------------------------------------------------------------- 6 Setup
s = S[5]
for sid, text in ((133, "3"), (134, "Amazon datasets"), (136, "134K"), (137, "users"),
                  (139, "16"), (140, "signal combinations"), (142, "6"), (143, "ranking metrics")):
    fill(shape(s, sid), [text])
fill(shape(s, 147), ["Amazon Reviews 2023: Movies & TV, Health, Industrial; ratings ≥ 4; "
                     "last 2 items per user tested"])
fill(shape(s, 151), ["PyTorch, FastAPI, Qdrant, MySQL; training on a T4 GPU; latency on a Ryzen 5 laptop"])
fill(shape(s, 155), ["Popularity, the 4 single signals and all 16 combinations, each tuned on validation"])
fill(shape(s, 159), ["Recall, NDCG, MAP, MRR, HitRate, Precision @10/@20; Wilcoxon; 17 live scenarios"])
notes(s, "(~50 s) The offline benchmark uses three Amazon Reviews 2023 categories: Movies and TV, "
         "Health and Household, and Industrial and Scientific, with up to 50,000 users each, about "
         "134,000 users in total. Ratings of four or five are positives; for each user the last two "
         "items are the test set and the one before is validation. Every method ranks the full "
         "catalog. We compare popularity, the four single signals and all 16 combinations of "
         "semantic, EMA, LightGCN and KG, each with weights tuned on 10,000 validation users; the "
         "test set is used once. LightGCN is trained for 60 epochs on a T4 GPU. Differences are tested "
         "with the Wilcoxon signed-rank test. We also measured latency and ran 17 live scenarios "
         "through the deployed API.")

# ---------------------------------------------------------------- 7 Results
s = S[6]
old = shape(s, 167)
x, y, w, h = old.left, old.top, old.width, old.height
rid = old._element.xpath(".//c:chart/@r:id")[0]
remove(s, 167)
s.part.drop_rel(rid)
cd = CategoryChartData()
cd.categories = ["Movies & TV", "Health & Household", "Industrial & Scientific"]
cd.add_series("Popularity", (0.0097, 0.0115, 0.0147))
cd.add_series("Best single signal", (0.0187, 0.0144, 0.0223))
cd.add_series("Semantic + LightGCN", (0.0222, 0.0120, 0.0267))
cd.add_series("Hybrid, all four (tuned)", (0.0227, 0.0126, 0.0283))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, x, y, w, h, cd)
gf.name = "NDCG@20 chart"
ch = gf.chart
style_chart(ch, "NDCG@20 on the test set (higher is better)")
ch.category_axis.reverse_order = True
ch.value_axis.maximum_scale = 0.034
ch.value_axis.minimum_scale = 0.0
ch.value_axis.major_unit = 0.01
ch.value_axis.tick_labels.number_format = "0.00"
ch.value_axis.tick_labels.number_format_is_linked = False
ch.plots[0].gap_width = 45
ch.plots[0].overlap = 0
color_series(ch, [LAVENDER, CYAN, PURPLE, MAGENTA])
labels(ch.plots[0], "0.0000", size=10)
fill(shape(s, 169), ["+21.6%"])
fill(shape(s, 170), ["NDCG@20 over the best single signal on Movies & TV (+26.9% on Industrial)"],
     size=14)
shape(s, 170).height = Inches(1.0)
b = shape(s, 171)
b.top, b.height = Inches(4.0), Inches(2.4)
fill(b, ["Tuned hybrid is best of all 16 combinations on 2 of 3 datasets",
         "Semantic + LightGCN is the strongest pair on all 3",
         "Health & Household: KG overlap alone is best"], size=14, spc_after=6)
notes(s, "(~90 s) This chart shows NDCG at 20 for popularity, the best single signal, the best pair "
         "and the tuned four-signal hybrid. On Movies and TV and on Industrial and Scientific the "
         "hybrid is the best of all 16 combinations on recall, NDCG, MAP and MRR at both cut-offs, "
         "improving NDCG@20 over the strongest single signal, semantic similarity, by 21.6 and 26.9 "
         "percent, and reaching 2.3 and 1.9 times popularity. Semantic plus LightGCN is the strongest "
         "pair on all three datasets, even though LightGCN alone is weaker than semantic retrieval. "
         "Health and Household is the exception: knowledge-graph overlap alone, matching category "
         "paths and brands, is best, 14.4 percent above the hybrid; we attribute this to repeat "
         "purchases within the same brand or category. "
         "Validation-tuned weights beat our hand-set weights by 3.6 to 6.5 percent. Absolute values "
         "are low, as usual for full-catalog ranking on sparse purchase data.")

# ---------------------------------------------------------------- 8 Comparative
s = S[7]
remove(s, 181, 182, 184, 185)
f1, f2 = shape(s, 179), shape(s, 180)
cd = CategoryChartData()
cd.categories = ["Semantic", "LightGCN", "KG", "EMA"]
cd.add_series("Movies & TV", (22.0, 13.4, 1.5, 0.7))
cd.add_series("Health & Household", (8.6, 21.9, 3.0, -0.5))
cd.add_series("Industrial & Scientific", (20.6, 19.6, 2.1, 2.7))
pad = Inches(0.12)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, f1.left + pad, f1.top + pad,
                        f1.width - 2 * pad, f1.height - 2 * pad, cd)
gf.name = "Signal removal chart"
ch = gf.chart
style_chart(ch, "NDCG@20 lost without the signal (%)")
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
ch.value_axis.minimum_scale = -5
ch.value_axis.maximum_scale = 25
ch.value_axis.major_unit = 5
ch.value_axis.tick_labels.number_format = "0"
ch.value_axis.tick_labels.number_format_is_linked = False
ch.plots[0].gap_width = 60
ch.plots[0].overlap = 0
color_series(ch, [PURPLE, CYAN, MAGENTA])
labels(ch.plots[0], "0.0", size=9)
ch.legend.font.size = Pt(10)

cd = CategoryChartData()
stages = [("Title/keyword scan", 190.8), ("Qdrant HNSW search", 82.6), ("Creator matching", 44.4),
          ("KG/profile rerank", 13.1), ("Query encoding", 12.0), ("EMA rerank", 7.3)]
cd.categories = [n for n, _ in stages]
cd.add_series("Median (ms)", [v for _, v in stages])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, f2.left + pad, f2.top + pad,
                        f2.width - 2 * pad, f2.height - 2 * pad, cd)
gf.name = "Latency chart"
ch = gf.chart
style_chart(ch, "Median time per stage (ms)", legend=False)
ch.category_axis.reverse_order = True
ch.category_axis.tick_labels.font.size = Pt(11)
ch.value_axis.maximum_scale = 230
ch.value_axis.minimum_scale = 0
ch.value_axis.major_unit = 50
ch.plots[0].gap_width = 50
color_series(ch, [PURPLE])
pt = ch.plots[0].series[0].points[0]
pt.format.fill.solid()
pt.format.fill.fore_color.rgb = RGBColor.from_string(MAGENTA)
labels(ch.plots[0], "0", size=10)
fill(shape(s, 183), ["Fig. 1: Semantic and LightGCN carry most of the gain"])
fill(shape(s, 186), ["Fig. 2: Whole request 363 ms median (p95 455 ms); EMA update 0.05 ms"])
notes(s, "(~70 s) Figure 1 removes one signal at a time from the tuned hybrid. Dropping semantic "
         "similarity costs 8.6 to 22 percent of NDCG@20 and dropping LightGCN 13.4 to 21.9 percent, "
         "so these two carry most of the gain. Removing the KG score costs 1.5 to 3 percent on every "
         "dataset, significant at p below 10 to the minus 6, and removing EMA 0.7 and 2.7 percent on "
         "two datasets, with no significant effect on Health and Household. Figure 2 is the latency "
         "of a recommendation over the full catalog on a laptop with a Qdrant server: 363 "
         "milliseconds at the median and 455 at p95. Most of the time is the title scan, a pass over "
         "the catalog, and the Qdrant query. An EMA update takes 0.05 ms, so feedback needs no "
         "retraining. One process serves about three requests per second; an inverted title index "
         "is the main fix.")

# ---------------------------------------------------------------- 9 Discussion
s = S[8]
fill(shape(s, 198), ["Each signal sees different evidence (meaning, co-interaction, shared entities, "
                     "recent taste), so a tuned mix should beat any one alone"], size=15)
fill(shape(s, 203), ["On two datasets: best of 16, and removing any signal costs NDCG@20 (p < 0.01). "
                     "Health & Household favours KG overlap, likely from repeat brand purchases"], size=15)
fill(shape(s, 208), ["Runs live: 363 ms over 120,747 items, feedback applied in 0.05 ms; no unsuitable "
                     "item in 17 live scenarios"], size=15)
for sid in (198, 203, 208):
    shape(s, sid).height = Inches(0.6)
notes(s, "(~50 s) We expected the signals to be complementary because each sees different evidence. "
         "The evidence supports this on two of three datasets: the tuned hybrid is best and every "
         "removal costs NDCG. Health and Household shows the limit: simple entity overlap wins there, "
         "which we attribute to repeat purchases inside one brand or category, so weights should be "
         "tuned per domain. "
         "In practice the system runs in real time. In 17 live scenarios through the API, person "
         "queries reach a mean precision at 10 of 0.84, eight of the eleven unfiltered scenarios "
         "return more than one content type, and the eligibility filter keeps unsuitable items out. "
         "The deployed LightGCN covers only the demo users, so its effect is measured offline.")

# ---------------------------------------------------------------- 10 Conclusion
s = S[9]
fill(shape(s, 220), ["One real-time ranker for 7 content types and 120,747 items",
                     "+21.6% / +26.9% NDCG@20 over the best single signal on 2 datasets",
                     "363 ms median per request; feedback applied at once"], size=16, spc_after=8)
fill(shape(s, 225), [
    "Tune the weights per domain",
    "Inverted title index; more worker processes",
    "More domains and user studies",
    {"runs": [("Since submission: ", {"bold": True}),
              ("BGE-small encoder, typed knowledge graph with explanations, NeuMF baseline", {})],
     "bullet": False, "size": 12, "italic": True},
], size=16, spc_after=8)
notes(s, "(~40 s) To conclude: one real-time recommender ranks seven content types from one catalog; "
         "the tuned hybrid beat every single signal by 21.6 and 26.9 percent on two datasets; and a "
         "request takes 363 ms, with feedback applied immediately. Next we will tune weights per "
         "domain, add an inverted title index and more worker processes, and extend to more domains "
         "and user studies. Since submitting, we have moved the encoder to BGE-small, typed the "
         "knowledge graph so each result is explained, and added a NeuMF baseline; these are not "
         "part of the paper's reported results.")

# ---------------------------------------------------------------- 11 References
s = S[10]
fill(shape(s, 233), [
    "[1] X. He et al., “LightGCN: simplifying and powering graph convolution network for "
    "recommendation,” in Proc. ACM SIGIR, 2020, pp. 639–648.",
    "[2] N. Reimers and I. Gurevych, “Sentence-BERT: sentence embeddings using Siamese "
    "BERT-networks,” in Proc. EMNLP-IJCNLP, 2019, pp. 3982–3992.",
    "[3] Y. Hou et al., “Bridging language and items for retrieval and recommendation,” "
    "arXiv:2403.03952, 2024.",
    "[4] X. Wang et al., “KGAT: knowledge graph attention network for recommendation,” in "
    "Proc. ACM SIGKDD, 2019, pp. 950–958.",
    "[5] R. Hassanzadeh et al., “A novel recommender system using light graph convolutional network "
    "and personalized knowledge-aware attention sub-network,” Sci. Rep., vol. 15, 15693, 2025.",
    "[6] Q. Liu et al., “Bridge the domains: large language models enhanced cross-domain sequential "
    "recommendation,” in Proc. ACM SIGIR, 2025, pp. 1582–1592.",
], size=15, spc_after=8)
notes(s, "(~10 s) These are the main references: LightGCN, Sentence-BERT, the Amazon Reviews 2023 "
         "corpus, KGAT, LGKAT and LLM4CDSR. The full list is in the paper.")

# ---------------------------------------------------------------- 12 Thank you
s = S[11]
t = shape(s, 241)
ps = t.text_frame.paragraphs
ps[1].runs[0].text = "Sanjay A. K.  ·  Amrita Vishwa Vidyapeetham, Coimbatore"
for r in ps[1].runs[1:]:
    r.text = ""
notes(s, "(~10 s) Thank you. We are happy to take questions.")

prs.save(OUT)
print("saved", OUT)
