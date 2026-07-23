"""Validate content_catalog.csv against the required schema.

Raises ``SystemExit`` (exit code 1) on any validation failure so that the
script can be used in CI pipelines.

Usage:
    python -m preprocessing.validate_content_catalog
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"

REQUIRED_COLUMNS = [
    "global_id",
    "content_type",
    "source",
    "source_id",
    "title",
    "description",
    "creators",
    "categories",
    "release_date",
    "popularity",
    "rating",
    "metadata_text",
    "embedding_text",
    "text_hash",
]


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_catalog(df: pd.DataFrame) -> None:
    """
    Validate a content catalog DataFrame.

    Raises ``ValueError`` for any schema or data quality violation.
    Intended to be called programmatically or via ``main()``.
    """
    errors: list[str] = []

    # 1. Required columns present
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        # Cannot continue further checks without the required columns.
        _fail(errors)

    # 2. global_id uniqueness
    dup_count = df["global_id"].duplicated().sum()
    if dup_count > 0:
        errors.append(
            f"global_id is not unique: {dup_count} duplicate(s) found. "
            f"Examples: {df[df['global_id'].duplicated(keep=False)]['global_id'].head(3).tolist()}"
        )

    # 3. global_id format: {content_type}:{source}:{source_id}
    bad_format = df[~df["global_id"].str.match(r"^[^:]+:[^:]+:[^:]+$")]
    if not bad_format.empty:
        errors.append(
            f"global_id format invalid for {len(bad_format)} row(s). "
            f"Expected '{{type}}:{{source}}:{{source_id}}'. "
            f"Examples: {bad_format['global_id'].head(3).tolist()}"
        )

    # 4. No empty titles
    empty_titles = (df["title"].fillna("").str.strip() == "").sum()
    if empty_titles > 0:
        errors.append(f"Empty 'title' in {empty_titles} row(s).")

    # 5. No empty embedding_text
    empty_embed = (df["embedding_text"].fillna("").str.strip() == "").sum()
    if empty_embed > 0:
        errors.append(f"Empty 'embedding_text' in {empty_embed} row(s).")

    # 6. No empty text_hash
    empty_hash = (df["text_hash"].fillna("").str.strip() == "").sum()
    if empty_hash > 0:
        errors.append(f"Empty 'text_hash' in {empty_hash} row(s).")

    # 7. text_hash looks like a SHA-256 hex string (64 chars)
    bad_hash = df[~df["text_hash"].fillna("").str.match(r"^[0-9a-f]{64}$")]
    if not bad_hash.empty:
        errors.append(
            f"Invalid text_hash (expected 64-char hex) in {len(bad_hash)} row(s)."
        )

    # 8. No empty metadata_text
    empty_meta = (df["metadata_text"].fillna("").str.strip() == "").sum()
    if empty_meta > 0:
        errors.append(f"Empty 'metadata_text' in {empty_meta} row(s).")

    if errors:
        _fail(errors)

    print(f"[OK] Catalog validated successfully. {len(df)} rows, {len(df.columns)} columns.")
    print(f"     Content types: {df['content_type'].value_counts().to_dict()}")


def _fail(errors: list[str]) -> None:
    message = "\n".join(f"  - {e}" for e in errors)
    raise ValueError(f"Content catalog validation failed:\n{message}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Content catalog not found at: {CATALOG_PATH}\n"
            "Run: python -m preprocessing.build_content_catalog"
        )

    df = pd.read_csv(CATALOG_PATH)
    print(f"Loaded catalog: {df.shape[0]} rows × {df.shape[1]} columns")

    try:
        validate_catalog(df)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()