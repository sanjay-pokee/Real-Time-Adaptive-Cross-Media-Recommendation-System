from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"

REQUIRED_COLUMNS = [
    "content_id",
    "content_type",
    "title",
    "description",
    "categories",
    "creators",
    "release_date",
    "popularity",
    "rating",
    "metadata_text",
    "source",
]


def validate_catalog() -> None:
    df = pd.read_csv(CATALOG_PATH)

    print("\nCatalog shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    print("\nMissing required columns:")
    print(missing_columns)

    print("\nContent types:")
    print(df["content_type"].value_counts())

    print("\nMissing titles:")
    print(df["title"].isna().sum())

    print("\nEmpty metadata_text:")
    print((df["metadata_text"].fillna("").str.strip() == "").sum())

    print("\nDuplicate content_id:")
    print(df["content_id"].duplicated().sum())

    print("\nSample rows:")
    print(df[["content_id", "content_type", "title", "metadata_text"]].head())


if __name__ == "__main__":
    validate_catalog()