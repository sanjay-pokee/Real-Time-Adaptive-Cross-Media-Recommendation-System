"""Derive audience metadata for every catalogue row.

The constraint layer in ``backend.audience`` needs four columns on each item:
``domain``, ``maturity``, ``audience_min_age`` and ``risk_tier``. None of the
source datasets ship them, so they are derived here, in one place, after the
per-dataset normalizers have run. That keeps the normalizers domain-shaped and
means a new vertical inherits audience tagging for free.

**These are heuristics, not certified ratings.** The signal is a keyword match
over the catalogue's own category text plus the content type's declared default
from ``config/domains.yaml``. That is honest enough for ranking and for the
constraint evaluation, and it fails *closed*: a row whose categories say nothing
falls back to its content type's default rather than to "unrestricted". Anywhere
a real certification is available (a BBFC/CBFC rating, an explicit-lyrics flag,
an Rx-only marker) it should replace this, and ``item_min_age`` already prefers
an explicit ``audience_min_age`` when one is present.
"""

from __future__ import annotations

import re

import pandas as pd

from backend.audience import AUDIENCE_COLUMNS
from backend.domains import DomainRegistry, get_registry

# Ordered most-restrictive first: the first rule that matches a row's category
# text wins, so "young adult horror" is tagged adult rather than teen.
MATURITY_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "restricted",
        ("adults only", "adult only", "pornograph", "x-rated"),
    ),
    (
        "adult",
        (
            "erotic",
            "explicit",
            "horror",
            "mature content",
            "true crime",
            "gore",
        ),
    ),
    (
        "teen",
        ("young adult", "teen", "juvenile nonfiction"),
    ),
    (
        "all_ages",
        (
            "animation",
            "children",
            "juvenile fiction",
            "kids",
            "family",
            "picture book",
            "nursery",
            "fairy tale",
            "early reader",
        ),
    ),
]


def _compile(rules: list[tuple[str, tuple[str, ...]]]) -> list[tuple[str, re.Pattern[str]]]:
    return [
        (maturity, re.compile("|".join(re.escape(keyword) for keyword in keywords)))
        for maturity, keywords in rules
    ]


_COMPILED_RULES = _compile(MATURITY_CATEGORY_RULES)


def maturity_for_row(
    content_type: str,
    category_text: str,
    registry: DomainRegistry | None = None,
) -> str:
    """Best-effort maturity label for one row.

    Falls back to the content type's declared default when the category text
    carries no signal, which is the fail-closed behaviour the constraint layer
    relies on.
    """
    registry = registry or get_registry()
    try:
        spec = registry.content_type(content_type)
    except (ValueError, KeyError):
        return "adult"  # An unrecognised type is gated, not waved through.

    if spec.maturity_source != "fixed":
        haystack = str(category_text or "").lower()
        if haystack:
            for maturity, pattern in _COMPILED_RULES:
                if pattern.search(haystack):
                    return maturity

    return spec.default_maturity


def annotate_audience(
    catalog: pd.DataFrame,
    registry: DomainRegistry | None = None,
) -> pd.DataFrame:
    """Add the audience columns to a catalogue frame.

    Existing non-empty values are preserved, so a dataset that *does* carry a
    real rating keeps it and only the gaps are filled.
    """
    registry = registry or get_registry()
    annotated = catalog.copy()

    content_types = annotated["content_type"].astype(str).str.strip().str.lower()

    # The category text a maturity rule matches against: categories plus title,
    # since some datasets put "(Children's Edition)" only in the title.
    category_text = (
        annotated.get("categories", pd.Series("", index=annotated.index)).astype(str)
        + " "
        + annotated.get("title", pd.Series("", index=annotated.index)).astype(str)
    )

    # One rule evaluation per distinct (content_type, category_text) pair rather
    # than per row: catalogues repeat category strings heavily.
    pairs = pd.DataFrame({"content_type": content_types, "category_text": category_text})
    unique_pairs = pairs.drop_duplicates()
    resolved = {
        (row.content_type, row.category_text): maturity_for_row(
            row.content_type, row.category_text, registry
        )
        for row in unique_pairs.itertuples(index=False)
    }
    derived_maturity = pd.Series(
        [resolved[(ct, text)] for ct, text in zip(pairs["content_type"], pairs["category_text"])],
        index=annotated.index,
    )

    if "maturity" in annotated.columns:
        existing = annotated["maturity"].astype(str).str.strip()
        annotated["maturity"] = existing.where(existing != "", derived_maturity)
    else:
        annotated["maturity"] = derived_maturity

    annotated["domain"] = [
        _domain_for(content_type, registry) for content_type in content_types
    ]
    annotated["audience_min_age"] = [
        registry.minimum_age(maturity) for maturity in annotated["maturity"]
    ]
    annotated["risk_tier"] = [
        _risk_tier_for(content_type, registry) for content_type in content_types
    ]
    return annotated


def _domain_for(content_type: str, registry: DomainRegistry) -> str:
    try:
        return registry.domain_of(content_type).name
    except (ValueError, KeyError):
        return ""


def _risk_tier_for(content_type: str, registry: DomainRegistry) -> int:
    try:
        return registry.domain_of(content_type).risk_tier
    except (ValueError, KeyError):
        # Unknown provenance is treated as consequential, not as harmless.
        return 1


def audience_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    """Row counts per domain and maturity, for the build log and the report."""
    missing = [column for column in AUDIENCE_COLUMNS if column not in catalog.columns]
    if missing:
        raise ValueError(f"Catalog is missing audience columns: {', '.join(missing)}")
    return (
        catalog.groupby(["domain", "content_type", "maturity"], dropna=False)
        .size()
        .reset_index(name="items")
        .sort_values(["domain", "content_type", "maturity"])
        .reset_index(drop=True)
    )
