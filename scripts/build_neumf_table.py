"""Build reports/neumf_comparison.md from the benchmark JSON files.

Rows come straight from reports/benchmarks/<domain>_<model>.json, so no number
is retyped by hand. Every model in a domain must have run on the same graph;
the script refuses to build the table otherwise.

    python -m scripts.build_neumf_table
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = PROJECT_ROOT / "reports" / "benchmarks"
OUTPUT = PROJECT_ROOT / "reports" / "neumf_comparison.md"

DOMAINS = [("movies_and_tv", "Movies & TV"), ("industrial_and_scientific", "Industrial & Scientific")]
# (file suffix, label). Missing files are skipped, so the local reruns are optional.
MODELS = [
    ("popularity", "Popularity"),
    ("lightgcn", "LightGCN, cosine (Kaggle run, as in the paper)"),
    ("lightgcn_cosine_local", "LightGCN, cosine (rerun on this laptop)"),
    ("lightgcn_dot_local", "LightGCN, dot product (rerun on this laptop)"),
    ("neumf", "NeuMF"),
]


def _load(slug: str, suffix: str) -> dict | None:
    path = BENCHMARKS / f"{slug}_{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build() -> str:
    lines = [
        "| Domain | Model | Recall@20 | NDCG@20 | MRR@20 | vs popularity | vs LightGCN (paper) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for slug, name in DOMAINS:
        runs = {suffix: run for suffix, _ in MODELS if (run := _load(slug, suffix))}
        graph = runs["popularity"]["graph"]
        for suffix, run in runs.items():
            if (run["graph"]["users"], run["graph"]["items"]) != (graph["users"], graph["items"]):
                raise SystemExit(f"{slug}_{suffix} ran on a different graph: {run['graph']} vs {graph}")
        pop = runs["popularity"]["metrics"]["recall"]
        paper = runs["lightgcn"]["metrics"]["recall"]
        for suffix, label in MODELS:
            if suffix not in runs:
                continue
            m = runs[suffix]["metrics"]
            lines.append(
                f"| {name} | {label} | {m['recall']:.4f} | {m['ndcg']:.4f} | {m['mrr']:.4f} "
                f"| {m['recall'] / pop:.2f}x | {m['recall'] / paper:.2f}x |"
            )
        lines.append(
            f"| | *graph: {graph['users']:,} users, {graph['items']:,} items, "
            f"{graph['interactions']:,} train interactions* | | | | | |"
        )

    return f"""# NeuMF vs LightGCN vs popularity (Amazon Reviews 2023)

Answers the Phase-2 panel's recommendation to try Neural Collaborative Filtering.
NeuMF (He et al., 2017: GMF + MLP towers, pointwise log loss, 4 sampled negatives
per positive) runs through the *same* preparation, split and metrics as the
LightGCN benchmark: ratings >= 4, k-core filter, leave-last-one-out by timestamp,
full-catalogue ranking with training items masked, K = 20, seed 42. Every row of
a domain ran on the same graph (checked when this table is built).

{chr(10).join(lines)}

- **NeuMF:** GMF dim 32, MLP 2x64 -> 64 -> 32 -> 16, Adam lr 1e-3, 20 epochs, batch 4096 positives.
- **LightGCN:** d = 64, 3 layers, BPR, 60 epochs, batch 4096. The backend and the paper rank by
  *cosine* (normalised embeddings); the *dot product* rows rank by the raw vectors BPR was trained
  with, whose length carries item popularity. Local reruns used an RTX 3050 laptop GPU.

Caveats: one seed per model, no hyperparameter tuning for either model, and Health & Household
was not rerun (its rating file is not on this machine).

Rebuild: `python -m scripts.build_neumf_table`. The per-run commands are in the README.
"""


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(OUTPUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
