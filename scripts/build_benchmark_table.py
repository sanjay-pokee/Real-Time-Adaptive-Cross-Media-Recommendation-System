"""Collect per-domain benchmark JSONs into one results table.

Each ``models.graph.evaluate_lightgcn --report <file>`` run drops a JSON. This
walks a directory of them and emits the cross-domain comparison in Markdown (for
the repo) and LaTeX (for the beamer deck), so the numbers in the paper are the
numbers the runs produced rather than something retyped.

Usage:
    python -m scripts.build_benchmark_table
    python -m scripts.build_benchmark_table --reports reports/benchmarks --k 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "benchmarks"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports"

METRIC_COLUMNS = [
    ("recall", "Recall@{k}"),
    ("ndcg", "NDCG@{k}"),
    ("mrr", "MRR@{k}"),
    ("hit_rate", "HitRate@{k}"),
]

# Maps an Amazon category file back to the vertical it stands in for, so the
# table reads as a domain comparison rather than a list of shopping categories.
DOMAIN_OF_DATASET = {
    "movies_and_tv": ("Entertainment", "Movies & TV"),
    "books": ("Entertainment", "Books"),
    "cds_and_vinyl": ("Entertainment", "Music"),
    "industrial_and_scientific": ("Industry", "Industrial & Scientific"),
    "health_and_household": ("Health", "Health & Household"),
}


def load_reports(reports_dir: Path) -> list[dict[str, Any]]:
    if not reports_dir.exists():
        raise FileNotFoundError(
            f"No reports directory at {reports_dir}.\n"
            "Run the Kaggle notebook (it passes --report), then copy the JSONs here."
        )
    reports = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc
    if not reports:
        raise FileNotFoundError(f"{reports_dir} contains no .json reports.")
    return reports


def _dataset_key(report: dict[str, Any]) -> str:
    stem = Path(str(report.get("dataset", ""))).stem.lower()
    for prefix in ("prepared_", "fake_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    # prepared_<category>_<max_users> -> <category>
    for known in DOMAIN_OF_DATASET:
        if stem.startswith(known):
            return known
    return stem


def group_reports(reports: list[dict[str, Any]]) -> dict[str, dict[str, dict]]:
    """Index as ``{dataset_key: {model: report}}``, keeping the newest per pair."""
    grouped: dict[str, dict[str, dict]] = {}
    for report in reports:
        key = _dataset_key(report)
        grouped.setdefault(key, {})[str(report.get("model", "unknown"))] = report
    return grouped


def build_rows(grouped: dict[str, dict[str, dict]], k: int) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(grouped, key=lambda name: list(DOMAIN_OF_DATASET).index(name)
                      if name in DOMAIN_OF_DATASET else 99):
        models = grouped[key]
        domain, label = DOMAIN_OF_DATASET.get(key, ("Other", key.replace("_", " ").title()))
        lightgcn = models.get("lightgcn")
        popularity = models.get("popularity")
        if lightgcn is None:
            continue

        graph = lightgcn.get("graph", {})
        row = {
            "domain": domain,
            "dataset": label,
            "users": graph.get("users", 0),
            "items": graph.get("items", 0),
            "interactions": graph.get("interactions", 0),
        }
        for metric, _ in METRIC_COLUMNS:
            row[f"lightgcn_{metric}"] = lightgcn["metrics"].get(metric)
            row[f"popularity_{metric}"] = (
                popularity["metrics"].get(metric) if popularity else None
            )
        # The headline claim: how much the graph model beats most-popular by.
        base = row.get("popularity_recall")
        lift = row.get("lightgcn_recall")
        row["recall_lift"] = (lift / base) if base else None
        rows.append(row)
    return rows


def to_markdown(rows: list[dict[str, Any]], k: int, note: str = "") -> str:
    header = (
        "| Domain | Dataset | Users | Items | Interactions | "
        f"Recall@{k} (Pop.) | Recall@{k} (LightGCN) | NDCG@{k} (LightGCN) | "
        f"MRR@{k} (LightGCN) | Lift |"
    )
    divider = "| " + " | ".join(["---"] * 10) + " |"
    lines = [header, divider]
    for row in rows:
        lift = row["recall_lift"]
        lines.append(
            f"| {row['domain']} | {row['dataset']} | {row['users']:,} | {row['items']:,} | "
            f"{row['interactions']:,} | {_fmt(row['popularity_recall'])} | "
            f"{_fmt(row['lightgcn_recall'])} | {_fmt(row['lightgcn_ndcg'])} | "
            f"{_fmt(row['lightgcn_mrr'])} | {f'{lift:.2f}x' if lift else '--'} |"
        )
    if note:
        lines.append("")
        lines.append("_" + note + "_")
    return "\n".join(lines)


def to_latex(rows: list[dict[str, Any]], k: int, note: str = "") -> str:
    """A booktabs table sized for the beamer deck."""
    lines = [
        "% Generated by scripts.build_benchmark_table -- do not edit by hand.",
        r"\begin{table}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Dataset} & \textbf{Users} & "
        rf"\textbf{{Recall@{k}}} & \textbf{{NDCG@{k}}} & \textbf{{Lift}} \\",
        r"\midrule",
    ]
    for row in rows:
        lift = row["recall_lift"]
        lines.append(
            f"{_tex(row['domain'])} & {_tex(row['dataset'])} & {row['users']:,} & "
            f"{_fmt(row['lightgcn_recall'])} & {_fmt(row['lightgcn_ndcg'])} & "
            + (f"{lift:.2f}$\\times$" if lift else "--")
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{LightGCN vs.\ most-popular across domains (K={k}). "
        r"Lift is Recall@K relative to the popularity baseline."
        + (" " + _tex(note) if note else "")
        + r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


# Dataset labels contain "&" ("Movies & TV", "Health & Household"), which LaTeX
# reads as a column separator; unescaped it silently mangles the table.
_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex(text: str) -> str:
    return "".join(_TEX_ESCAPES.get(character, character) for character in str(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cross-domain results table.")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--note",
        default="",
        help="Caveat appended to the caption, e.g. a differing k-core filter.",
    )
    args = parser.parse_args()

    reports = load_reports(args.reports)
    rows = build_rows(group_reports(reports), args.k)
    if not rows:
        raise SystemExit(
            f"Found {len(reports)} report(s) but no LightGCN result among them. "
            "Each domain needs a run without --baseline."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(rows, args.k, args.note)
    (args.out_dir / "domain_benchmark_table.md").write_text(markdown + "\n", encoding="utf-8")
    (args.out_dir / "domain_benchmark_table.tex").write_text(
        to_latex(rows, args.k, args.note) + "\n", encoding="utf-8"
    )

    print(markdown)
    print()
    print(f"wrote {args.out_dir / 'domain_benchmark_table.md'}")
    print(f"wrote {args.out_dir / 'domain_benchmark_table.tex'}")
    missing = [row["dataset"] for row in rows if row["popularity_recall"] is None]
    if missing:
        print(f"\nno popularity baseline yet for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
