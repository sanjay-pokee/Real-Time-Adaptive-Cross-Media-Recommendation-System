"""Turn the *_ablation.json reports into the paper's LaTeX tables.

Reads every ``<domain>_ablation.json`` written by
``models.graph.evaluate_hybrid_ablation`` and writes, next to them:

  ablation_table.tex       Recall/NDCG/MAP/MRR @20, all methods x all domains
  ablation_table_k10.tex   the same at K = 10
  ablation_datasets.tex    rows for the benchmark-graph table
  ablation_weights.tex     validation-selected weights per domain

and prints a short summary (best single signal, tuned vs paper weights,
leave-one-out drops) for writing the results paragraph. Nothing is retyped
by hand, so the paper and the JSON cannot drift apart.

Usage:
    python -m scripts.build_ablation_table
    python -m scripts.build_ablation_table --reports reports/benchmarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "reports" / "benchmarks"

DOMAIN_LABELS = {
    "movies_and_tv": "Movies \\& TV",
    "health_and_household": "Health \\& Household",
    "industrial_and_scientific": "Industrial \\& Scientific",
}
METHOD_ORDER = [
    "Popularity",
    "Semantic only",
    "EMA only",
    "LightGCN only",
    "KG only",
    "Hybrid (paper weights)",
    "Hybrid (tuned)",
    "Hybrid (tuned) w/o sem",
    "Hybrid (tuned) w/o ema",
    "Hybrid (tuned) w/o graph",
    "Hybrid (tuned) w/o kg",
]
METHOD_LABELS = {
    "Popularity": "Most popular",
    "Semantic only": "Semantic only",
    "EMA only": "EMA only",
    "LightGCN only": "LightGCN only",
    "KG only": "KG only",
    "Hybrid (paper weights)": "Hybrid, hand-set",
    "Hybrid (tuned)": "Hybrid, tuned",
    "Hybrid (tuned) w/o sem": "\\quad $-$ semantic",
    "Hybrid (tuned) w/o ema": "\\quad $-$ EMA",
    "Hybrid (tuned) w/o graph": "\\quad $-$ LightGCN",
    "Hybrid (tuned) w/o kg": "\\quad $-$ KG",
}
METRICS = ("recall", "ndcg", "map", "mrr")
METRIC_LABELS = {"recall": "R", "ndcg": "N", "map": "MAP", "mrr": "MRR"}
SINGLE_SIGNALS = ("Semantic only", "EMA only", "LightGCN only", "KG only")
ALPHA = 0.05


def load_reports(directory: Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(directory.glob("*_ablation.json")):
        slug = path.name[: -len("_ablation.json")]
        reports[slug] = json.loads(path.read_text(encoding="utf-8"))
    if not reports:
        raise SystemExit(f"No *_ablation.json files in {directory}")
    ordered = [s for s in DOMAIN_LABELS if s in reports] + [s for s in reports if s not in DOMAIN_LABELS]
    return {slug: reports[slug] for slug in ordered}


def results_table(reports: dict[str, dict], k: int) -> str:
    domains = list(reports)
    columns = [(d, m) for d in domains for m in METRICS]
    best = {}
    for d, m in columns:
        values = [reports[d]["results"][name][f"{m}@{k}"] for name in METHOD_ORDER if name in reports[d]["results"]]
        best[(d, m)] = max(values)

    head_domains = " & ".join(
        f"\\multicolumn{{4}}{{c}}{{{DOMAIN_LABELS.get(d, d)}}}" for d in domains
    )
    head_metrics = " & ".join(f"{METRIC_LABELS[m]}@{k}" for _ in domains for m in METRICS)
    lines = [
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular*}}{{\\hsize}}{{@{{\\extracolsep{{\\fill}}}}l{'c' * len(columns)}@{{}}}}",
        "\\toprule",
        f"Method & {head_domains} \\\\",
        f" & {head_metrics} \\\\",
        "\\colrule",
    ]
    for name in METHOD_ORDER:
        if not all(name in reports[d]["results"] for d in domains):
            continue
        cells = []
        for d, m in columns:
            value = reports[d]["results"][name][f"{m}@{k}"]
            text = f"{value:.4f}"
            if abs(value - best[(d, m)]) < 5e-5:
                text = f"\\textbf{{{text}}}"
            p = reports[d]["wilcoxon_vs_tuned_ndcg@20"].get(name)
            # Usually every row differs significantly from the tuned hybrid, so
            # the exceptions are what is worth marking.
            if m == "ndcg" and k == 20 and p is not None and p >= ALPHA:
                text += "$^{\\circ}$"
            cells.append(text)
        lines.append(f"{METHOD_LABELS[name]} & " + " & ".join(cells) + " \\\\")
        if name in ("KG only", "Hybrid (tuned)"):
            lines.append("\\colrule")
    lines += ["\\botrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def datasets_rows(reports: dict[str, dict]) -> str:
    rows = []
    for slug, report in reports.items():
        g = report["graph"]
        rows.append(
            f"{DOMAIN_LABELS.get(slug, slug)} & {g['users']:,} & {g['items']:,} & {g['train']:,} & "
            f"{report['protocol']['test_users']:,} & "
            f"{g['items_with_text']:,} ({100 * g['items_with_text'] / g['items']:.1f}\\%) \\\\"
        )
    return "\n".join(rows) + "\n"


def weights_rows(reports: dict[str, dict]) -> str:
    rows = []
    for slug, report in reports.items():
        w = report["tuned_weights"]
        rows.append(
            f"{DOMAIN_LABELS.get(slug, slug)} & {w['sem']:g} & {w['graph']:g} & {w['ema']:g} & {w['kg']:g} \\\\"
        )
    return "\n".join(rows) + "\n"


def summary(reports: dict[str, dict]) -> str:
    out = []
    for slug, report in reports.items():
        r = report["results"]
        tuned = r["Hybrid (tuned)"]["ndcg@20"]
        best_single = max(SINGLE_SIGNALS, key=lambda n: r[n]["ndcg@20"])
        single = r[best_single]["ndcg@20"]
        paper = r["Hybrid (paper weights)"]["ndcg@20"]
        pop = r["Popularity"]["ndcg@20"]
        p = report["wilcoxon_vs_tuned_ndcg@20"]
        out.append(f"== {slug}  (tuned weights {report['tuned_weights']})")
        out.append(f"  NDCG@20 tuned hybrid {tuned:.4f} | best single = {best_single} {single:.4f} "
                   f"({100 * (tuned / single - 1):+.1f}%, p={p.get(best_single, float('nan')):.1e})")
        out.append(f"  paper weights {paper:.4f} ({100 * (tuned / paper - 1):+.1f}% for tuning, p={p['Hybrid (paper weights)']:.1e}) "
                   f"| popularity {pop:.4f}")
        for signal in ("sem", "ema", "graph", "kg"):
            name = f"Hybrid (tuned) w/o {signal}"
            drop = r[name]["ndcg@20"]
            out.append(f"  w/o {signal:5s}: {drop:.4f} ({100 * (drop / tuned - 1):+.1f}%, p={p[name]:.1e})")
        t = report.get("timings", {})
        out.append(f"  timings: train {t.get('lightgcn_train_s')}, grid {t.get('grid_search_s', 0):.0f}s, "
                   f"rerank {t.get('rerank_ms_per_user', 0):.3f} ms/user/config")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reports", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    reports = load_reports(args.reports)
    outputs = {
        "ablation_table.tex": results_table(reports, 20),
        "ablation_table_k10.tex": results_table(reports, 10),
        "ablation_datasets.tex": datasets_rows(reports),
        "ablation_weights.tex": weights_rows(reports),
    }
    for name, text in outputs.items():
        (args.reports / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.reports / name}")
    print()
    print(summary(reports))


if __name__ == "__main__":
    main()
