"""Measure what the audience constraint layer actually enforces, and what it costs.

Two numbers matter to a panel, and they pull against each other:

*Violation rate@K* -- how often an ineligible item reaches a viewer. With the
pre-filter on this should be exactly zero, because ineligible points are never
retrieved. Reporting it next to the filter-off column is the point: it shows the
zero is enforcement, not an accident of the data.

*Utility cost* -- what the constraint takes away. Measured as catalog coverage
(how much of the catalog a viewer may see) and as overlap@K against the same
ranking without the filter. A safety layer that costs nothing is usually a
safety layer that is not doing anything.

The ranking itself is deliberately simple (popularity over the catalog), because
this measures the *constraint*, not the recommender. Swap in real query rankings
with --qdrant once a collection is built from a tagged catalog.

Usage:
    python -m scripts.evaluate_constraints
    python -m scripts.evaluate_constraints --k 20 --catalog data/processed/content_catalog.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from backend.audience import (
    AudienceContext,
    explain_ineligibility,
    is_eligible,
)
from backend.domains import get_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports"

# The viewer profiles the report is built around. Named so the slide can quote
# them directly.
PROFILES: list[tuple[str, dict[str, Any]]] = [
    ("Child (8)", {"age": 8}),
    ("Teen (15)", {"age": 15}),
    ("Adult (25)", {"age": 25}),
    ("Adult, safe mode", {"age": 35, "safe_mode": True}),
    ("Age not supplied", {"age": None}),
    ("Adult, health domain", {"age": 30, "domain": "health"}),
    ("Adult, entertainment", {"age": 30, "domain": "entertainment"}),
]


def load_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {path}\n"
            "Run: python -m preprocessing.build_content_catalog"
        )
    catalog = pd.read_csv(path)
    missing = [
        column
        for column in ("content_type", "domain", "maturity", "audience_min_age", "risk_tier")
        if column not in catalog.columns
    ]
    if missing:
        raise ValueError(
            f"Catalog is missing audience columns ({', '.join(missing)}).\n"
            "Rebuild it: python -m preprocessing.build_content_catalog"
        )
    return catalog


def rank_unconstrained(catalog: pd.DataFrame, k: int) -> list[dict[str, Any]]:
    """The baseline ranking a viewer would get with no constraint layer at all."""
    ordered = catalog.sort_values("popularity", ascending=False, na_position="last")
    return ordered.head(k).to_dict("records")


def rank_constrained(catalog: pd.DataFrame, context: AudienceContext, k: int) -> list[dict[str, Any]]:
    """Pre-filter, then rank -- the order the engine itself uses."""
    rows = catalog.to_dict("records")
    eligible = [row for row in rows if is_eligible(row, context)]
    eligible.sort(key=lambda row: _popularity(row), reverse=True)
    return eligible[:k]


def _popularity(row: dict[str, Any]) -> float:
    try:
        value = float(row.get("popularity") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(value) else value


def evaluate_profile(
    catalog: pd.DataFrame,
    name: str,
    kwargs: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    context = AudienceContext(**kwargs)
    rows = catalog.to_dict("records")

    eligible_rows = [row for row in rows if is_eligible(row, context)]
    constrained = rank_constrained(catalog, context, k)
    unconstrained = rank_unconstrained(catalog, k)

    # Filter ON: every returned item re-checked by the Python predicate.
    violations_on = sum(1 for row in constrained if not is_eligible(row, context))
    # Filter OFF: what the same viewer would have been shown without the layer.
    violations_off = sum(1 for row in unconstrained if not is_eligible(row, context))

    constrained_ids = {row.get("global_id") for row in constrained}
    unconstrained_ids = {row.get("global_id") for row in unconstrained}
    overlap = len(constrained_ids & unconstrained_ids) / max(len(unconstrained_ids), 1)

    return {
        "profile": name,
        "effective_age": context.effective_age,
        "domain": context.domain or "all",
        "catalog_items": len(rows),
        "eligible_items": len(eligible_rows),
        "coverage": len(eligible_rows) / max(len(rows), 1),
        "violations_filter_on": violations_on,
        "violation_rate_on": violations_on / max(len(constrained), 1),
        "violations_filter_off": violations_off,
        "violation_rate_off": violations_off / max(len(unconstrained), 1),
        "overlap_with_unconstrained": overlap,
        "advisory": context.advisory() or "",
    }


def sample_withheld(
    catalog: pd.DataFrame,
    kwargs: dict[str, Any],
    limit: int = 3,
) -> list[str]:
    """A few concrete 'why was this hidden' lines, for the appendix slide."""
    context = AudienceContext(**kwargs)
    reasons = []
    for row in catalog.to_dict("records"):
        why = explain_ineligibility(row, context)
        if why:
            reasons.append(f"{row.get('title', '?')} ({row.get('content_type', '?')}): {why}")
        if len(reasons) >= limit:
            break
    return reasons


def to_markdown(results: list[dict[str, Any]], k: int) -> str:
    lines = [
        f"| Viewer profile | Effective age | Domain | Catalog coverage | "
        f"Violation rate@{k} (filter ON) | Violation rate@{k} (filter OFF) | Overlap@{k} |",
        "| " + " | ".join(["---"] * 7) + " |",
    ]
    for row in results:
        lines.append(
            f"| {row['profile']} | {row['effective_age']} | {row['domain']} | "
            f"{row['coverage']:.1%} ({row['eligible_items']:,}/{row['catalog_items']:,}) | "
            f"{row['violation_rate_on']:.3f} | {row['violation_rate_off']:.3f} | "
            f"{row['overlap_with_unconstrained']:.1%} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the audience constraint layer: enforcement and its cost."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    registry = get_registry()

    print(f"catalog: {args.catalog.name}  ({len(catalog):,} items)")
    print(f"domains: {', '.join(registry.domain_names())}")
    print()

    results = [
        evaluate_profile(catalog, name, kwargs, args.k) for name, kwargs in PROFILES
    ]
    markdown = to_markdown(results, args.k)
    print(markdown)

    failures = [row for row in results if row["violations_filter_on"] > 0]
    print()
    if failures:
        print("FAIL: the pre-filter let ineligible items through for:")
        for row in failures:
            print(f"  - {row['profile']}: {row['violations_filter_on']} violation(s)")
    else:
        print(f"PASS: zero violations@{args.k} across all {len(results)} profiles.")

    print()
    print("Examples of withheld items (child profile):")
    for reason in sample_withheld(catalog, {"age": 8}):
        print(f"  - {reason}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.out_dir / "constraint_evaluation.csv", index=False)
    (args.out_dir / "constraint_evaluation.md").write_text(markdown + "\n", encoding="utf-8")
    print()
    print(f"wrote {args.out_dir / 'constraint_evaluation.csv'}")
    print(f"wrote {args.out_dir / 'constraint_evaluation.md'}")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
