"""Turn the *_ablation.json reports into the paper's LaTeX tables.

Reads every ``<domain>_ablation.json`` written by
``models.graph.evaluate_hybrid_ablation`` and writes, next to them:

  ablation_table.tex       Recall/NDCG/MAP/MRR @20, all methods x all domains
  ablation_table_k10.tex   the same at K = 10
  ablation_datasets.tex    rows for the benchmark-graph table
  ablation_weights.tex     validation-selected weights per domain

When the reports carry the full-factorial block (all 16 on/off combinations of
the four signals, each tuned on validation), the first two tables list those 16
combinations plus the hand-set hybrid instead of the single signals and the
fixed-weight leave-one-out rows, and two more files are written:

  ablation_response_table.tex   NDCG@20 and Recall@20 of the 16 combinations
                                (compact, for the response letter)
  ablation_extra_metrics.tex    HitRate/Precision @20 of the main methods

It prints a short summary (best single signal, tuned vs paper weights,
drops when a signal is removed, average gain from adding each signal) for
writing the results paragraph. Nothing is retyped by hand, so the paper and
the JSON cannot drift apart.

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


def has_factorial(reports: dict[str, dict]) -> bool:
    return all("factorial" in report for report in reports.values())


def _factorial_rows(reports: dict[str, dict]) -> list[str]:
    return list(next(iter(reports.values()))["factorial"]["order"])


def _row_label(label: str) -> str:
    return "None (popularity)" if label == "none" else label


def _combination_result(report: dict, label: str) -> dict:
    return report["factorial"]["results"][label]


def factorial_table(reports: dict[str, dict], k: int, metrics: tuple[str, ...] = METRICS) -> str:
    """16 signal combinations (and the hand-set hybrid) x domains x metrics."""
    domains = list(reports)
    columns = [(d, m) for d in domains for m in metrics]
    labels = _factorial_rows(reports)
    best = {}
    for d, m in columns:
        values = [_combination_result(reports[d], label)[f"{m}@{k}"] for label in labels]
        values.append(reports[d]["results"]["Hybrid (paper weights)"][f"{m}@{k}"])
        best[(d, m)] = max(values)

    span = len(metrics)
    head_domains = " & ".join(f"\\multicolumn{{{span}}}{{c}}{{{DOMAIN_LABELS.get(d, d)}}}" for d in domains)
    head_metrics = " & ".join(f"{METRIC_LABELS[m]}@{k}" for _ in domains for m in metrics)
    lines = [
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular*}}{{\\hsize}}{{@{{\\extracolsep{{\\fill}}}}l{'c' * len(columns)}@{{}}}}",
        "\\toprule",
        f"Signals & {head_domains} \\\\",
        f" & {head_metrics} \\\\",
        "\\colrule",
    ]

    def render(source_for) -> str:
        out = []
        for d in domains:
            result, p = source_for(reports[d])
            for m in metrics:
                value = result[f"{m}@{k}"]
                text = f"{value:.4f}"
                if abs(value - best[(d, m)]) < 5e-5:
                    text = f"\\textbf{{{text}}}"
                if m == "ndcg" and k == 20 and p is not None and p >= ALPHA:
                    text += "$^{\\circ}$"
                out.append(text)
        return " & ".join(out)

    previous_size = None
    for label in labels:
        size = 0 if label == "none" else label.count("+") + 1
        if previous_size is not None and size != previous_size:
            lines.append("\\colrule")  # rule between none / singles / pairs / triples / all four
        previous_size = size
        row = render(lambda r, lab=label: (
            _combination_result(r, lab),
            r["factorial"]["wilcoxon_vs_full_ndcg@20"].get(lab),
        ))
        lines.append(f"{_row_label(label)} & {row} \\\\")
    lines.append("\\colrule")
    row = render(lambda r: (
        r["results"]["Hybrid (paper weights)"],
        r["wilcoxon_vs_tuned_ndcg@20"].get("Hybrid (paper weights)"),
    ))
    lines.append(f"Hand-set weights & {row} \\\\")
    lines += ["\\botrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def response_table(reports: dict[str, dict]) -> str:
    """Compact version for the response letter: Recall@20 and NDCG@20 of the 16 combinations."""
    return factorial_table(reports, 20, metrics=("recall", "ndcg"))


def extra_metrics_table(reports: dict[str, dict]) -> str:
    """HitRate@20 and Precision@20 of the main methods."""
    domains = list(reports)
    rows = [("none", "Most popular"), ("S", "Semantic only"), ("E", "EMA only"), ("G", "LightGCN only"),
            ("K", "KG only"), ("S+E+G+K", "Hybrid, tuned")]
    lines = [
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular*}}{{\\hsize}}{{@{{\\extracolsep{{\\fill}}}}l{'cc' * len(domains)}@{{}}}}",
        "\\toprule",
        "Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DOMAIN_LABELS.get(d, d)}}}" for d in domains) + " \\\\",
        " & " + " & ".join("HR@20 & P@20" for _ in domains) + " \\\\",
        "\\colrule",
    ]
    for label, name in rows:
        cells = []
        for d in domains:
            result = _combination_result(reports[d], label)
            cells += [f"{result['hit_rate@20']:.4f}", f"{result['precision@20']:.4f}"]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\botrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def factorial_summary(reports: dict[str, dict]) -> str:
    out = []
    signals = {"S": "semantic", "E": "EMA", "G": "LightGCN", "K": "KG"}
    for slug, report in reports.items():
        f = report["factorial"]
        n20 = {label: f["results"][label]["ndcg@20"] for label in f["order"]}
        p = f["wilcoxon_vs_full_ndcg@20"]
        full = n20[f["full"]]
        out.append(f"== {slug}: 16 combinations, NDCG@20 (weights = ratios to the first signal)")
        for label in f["order"]:
            w = ", ".join(f"{k}={v:g}" for k, v in f["weights"][label].items()) or "popularity"
            pv = "--" if label not in p else f"{p[label]:.1e}"
            out.append(f"  {label:8s} {n20[label]:.4f}  ({100 * (n20[label] / full - 1):+6.1f}% vs full, p={pv})  {w}")
        beaten = [lab for lab in f["order"] if lab != f["full"] and n20[lab] >= full]
        out.append(f"  combinations at or above the full hybrid: {beaten or 'none'}")
        single_labels = [lab for lab in f["order"] if lab in signals]
        best_single = max(single_labels, key=lambda lab: n20[lab])
        out.append(f"  best single signal: {best_single} ({signals[best_single]}) {n20[best_single]:.4f}; "
                   f"full is {100 * (full / n20[best_single] - 1):+.1f}% (p={p[best_single]:.1e}); "
                   f"popularity {n20['none']:.4f} (x{full / n20['none']:.2f})")
        best_pair = max((lab for lab in f["order"] if lab.count("+") == 1), key=lambda lab: n20[lab])
        out.append(f"  best pair: {best_pair} {n20[best_pair]:.4f}; full is {100 * (full / n20[best_pair] - 1):+.1f}% "
                   f"(p={p[best_pair]:.1e})")
        for letter, name in signals.items():
            rest = [lab for lab in f["order"] if lab.count("+") == 2 and letter not in lab.split("+")]
            drop_label = rest[0] if rest else None
            # Over the 7 non-empty combinations without the signal: going from the empty
            # combination to a single signal replaces popularity rather than adding to it.
            gains = []
            for lab in f["order"]:
                if lab == "none":
                    continue
                parts = lab.split("+")
                if letter in parts:
                    continue
                with_label = "+".join(sorted(parts + [letter], key=lambda s: "SEGK".index(s)))
                base = n20[lab]
                gains.append(n20[with_label] / base - 1 if base else float("nan"))
            out.append(f"  w/o {name:8s}: {n20[drop_label]:.4f} ({100 * (n20[drop_label] / full - 1):+.1f}% vs full, "
                       f"p={p[drop_label]:.1e}) | mean gain from adding it to the 7 other combinations "
                       f"{100 * sum(gains) / len(gains):+.1f}% (min {100 * min(gains):+.1f}%, max {100 * max(gains):+.1f}%)")
        hand = report["results"]["Hybrid (paper weights)"]["ndcg@20"]
        out.append(f"  hand-set weights {hand:.4f}: tuned full is {100 * (full / hand - 1):+.1f}% "
                   f"(p={report['wilcoxon_vs_tuned_ndcg@20']['Hybrid (paper weights)']:.1e})")
        for metric in ("hit_rate@20", "precision@20"):
            vals = {lab: f["results"][lab][metric] for lab in ("none", "S", "E", "G", "K", "S+E+G+K")}
            out.append(f"  {metric}: " + ", ".join(f"{lab} {v:.4f}" for lab, v in vals.items()))
        t = report.get("timings", {})
        out.append(f"  timings: LightGCN {t.get('lightgcn_train_s')}, weight tuning {t.get('weight_tuning_s', 0):.0f}s")
    return "\n".join(out)


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
    factorial = has_factorial(reports)
    build = factorial_table if factorial else results_table
    outputs = {
        "ablation_table.tex": build(reports, 20),
        "ablation_table_k10.tex": build(reports, 10),
        "ablation_datasets.tex": datasets_rows(reports),
        "ablation_weights.tex": weights_rows(reports),
    }
    if factorial:
        outputs["ablation_response_table.tex"] = response_table(reports)
        outputs["ablation_extra_metrics.tex"] = extra_metrics_table(reports)
    for name, text in outputs.items():
        (args.reports / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.reports / name}")
    print()
    print(factorial_summary(reports) if factorial else summary(reports))


if __name__ == "__main__":
    main()
