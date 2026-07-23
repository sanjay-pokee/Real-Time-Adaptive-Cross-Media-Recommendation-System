"""Build the unified content_catalog.csv from all registered datasets.

Usage:
    python -m preprocessing.build_content_catalog
"""

from pathlib import Path

import pandas as pd

try:
    from .cleaners import clean_text, join_non_empty, make_text_hash, parse_name_list
    from .content_schema import CONTENT_COLUMNS, make_global_id
    from .loaders import DEFAULT_CONFIG_PATH, load_dataset_config, resolve_project_path
except ImportError:
    from cleaners import clean_text, join_non_empty, make_text_hash, parse_name_list
    from content_schema import CONTENT_COLUMNS, make_global_id
    from loaders import DEFAULT_CONFIG_PATH, load_dataset_config, resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_content_catalog(config_path: Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    """Load, normalize, and concatenate all registered datasets into one catalog."""
    dataset_config = load_dataset_config(config_path)

    if not dataset_config:
        raise ValueError("No datasets are registered in the dataset config.")

    catalog_parts = [
        normalize_dataset(dataset_name, config)
        for dataset_name, config in dataset_config.items()
    ]

    catalog = pd.concat(catalog_parts, ignore_index=True)

    # Final dedup and quality filters on the combined catalog.
    catalog = catalog.drop_duplicates(subset=["global_id"])
    catalog = catalog[catalog["title"].str.strip() != ""]
    catalog = catalog[catalog["embedding_text"].str.strip() != ""]

    return catalog[CONTENT_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def normalize_dataset(dataset_name: str, dataset_config: dict) -> pd.DataFrame:
    dataset_path = resolve_project_path(dataset_config["path"])
    raw_df = pd.read_csv(dataset_path)

    if dataset_name == "movies":
        return normalize_movies(raw_df, dataset_config)

    if dataset_name == "books":
        return normalize_books(raw_df, dataset_config)

    if dataset_name == "music":
        return normalize_music(raw_df, dataset_config)

    raise ValueError(f"No normalizer found for dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Per-media normalizers
# ---------------------------------------------------------------------------

def normalize_movies(raw_df: pd.DataFrame, dataset_config: dict) -> pd.DataFrame:
    df = pd.DataFrame()

    source = dataset_config["source"]
    content_type = dataset_config["content_type"]

    source_id = raw_df[dataset_config["id_column"]].fillna("").astype(str)
    df["source_id"] = source_id
    df["source"] = source
    df["content_type"] = content_type
    df["global_id"] = source_id.apply(
        lambda sid: make_global_id(content_type, source, sid)
    )

    df["title"] = raw_df[dataset_config["title_column"]].apply(clean_text)
    df["description"] = raw_df[dataset_config["description_column"]].apply(clean_text)

    genres = raw_df["genres"].apply(lambda v: ", ".join(parse_name_list(v)))
    keywords = raw_df["keywords"].apply(lambda v: ", ".join(parse_name_list(v)))

    df["categories"] = genres
    df["creators"] = ""
    df["release_date"] = raw_df[dataset_config["release_date_column"]]
    df["popularity"] = raw_df[dataset_config["popularity_column"]]
    df["rating"] = raw_df[dataset_config["rating_column"]]

    # metadata_text — richest text (includes keywords for internal search)
    df["metadata_text"] = [
        join_non_empty(vals)
        for vals in zip(df["title"], genres, keywords, df["description"])
    ]

    # embedding_text — clean semantic text for the model (no raw keyword dumps)
    df["embedding_text"] = [
        join_non_empty(vals)
        for vals in zip(df["title"], genres, df["description"])
    ]

    df["text_hash"] = df["embedding_text"].apply(make_text_hash)

    df = _filter_rows(df, content_type, source)
    return df[CONTENT_COLUMNS]


def normalize_books(raw_df: pd.DataFrame, dataset_config: dict) -> pd.DataFrame:
    df = pd.DataFrame()

    source = dataset_config["source"]
    content_type = dataset_config["content_type"]

    source_id = raw_df[dataset_config["id_column"]].fillna("").astype(str)
    df["source_id"] = source_id
    df["source"] = source
    df["content_type"] = content_type
    df["global_id"] = source_id.apply(
        lambda sid: make_global_id(content_type, source, sid)
    )

    df["title"] = raw_df[dataset_config["title_column"]].apply(clean_text)
    df["description"] = raw_df[dataset_config["description_column"]].apply(clean_text)

    categories = raw_df["categories"].apply(clean_text)
    search_category = raw_df["search_category"].apply(clean_text)
    subtitle = raw_df["subtitle"].apply(clean_text)

    df["categories"] = [
        join_non_empty(vals, separator=", ")
        for vals in zip(categories, search_category)
    ]
    df["creators"] = raw_df["authors"].apply(clean_text)
    df["release_date"] = raw_df[dataset_config["release_date_column"]]
    df["popularity"] = raw_df[dataset_config["popularity_column"]]
    df["rating"] = raw_df[dataset_config["rating_column"]]

    # metadata_text — subtitle included for internal richness
    df["metadata_text"] = [
        join_non_empty(vals)
        for vals in zip(
            df["title"], subtitle, df["creators"], df["categories"], df["description"]
        )
    ]

    # embedding_text — title + authors + categories + description
    df["embedding_text"] = [
        join_non_empty(vals)
        for vals in zip(
            df["title"], df["creators"], df["categories"], df["description"]
        )
    ]

    df["text_hash"] = df["embedding_text"].apply(make_text_hash)

    df = _filter_rows(df, content_type, source)
    return df[CONTENT_COLUMNS]


def normalize_music(raw_df: pd.DataFrame, dataset_config: dict) -> pd.DataFrame:
    df = pd.DataFrame()

    source = dataset_config["source"]
    content_type = dataset_config["content_type"]

    source_id = raw_df[dataset_config["id_column"]].fillna("").astype(str)
    df["source_id"] = source_id
    df["source"] = source
    df["content_type"] = content_type
    df["global_id"] = source_id.apply(
        lambda sid: make_global_id(content_type, source, sid)
    )

    df["title"] = raw_df[dataset_config["title_column"]].apply(clean_text)
    df["description"] = ""

    playlist_genre = raw_df["playlist_genre"].apply(clean_text)
    playlist_subgenre = raw_df["playlist_subgenre"].apply(clean_text)
    album_name = raw_df["track_album_name"].apply(clean_text)

    df["categories"] = [
        join_non_empty(vals, separator=", ")
        for vals in zip(playlist_genre, playlist_subgenre)
    ]
    df["creators"] = raw_df["track_artist"].apply(clean_text)
    df["release_date"] = raw_df[dataset_config["release_date_column"]]
    df["popularity"] = raw_df[dataset_config["popularity_column"]]
    df["rating"] = ""

    # metadata_text — album name included for internal richness
    df["metadata_text"] = [
        join_non_empty(vals)
        for vals in zip(
            df["title"], df["creators"], album_name, playlist_genre, playlist_subgenre
        )
    ]

    # embedding_text — title + artist + genre + subgenre
    df["embedding_text"] = [
        join_non_empty(vals)
        for vals in zip(df["title"], df["creators"], playlist_genre, playlist_subgenre)
    ]

    df["text_hash"] = df["embedding_text"].apply(make_text_hash)

    df = _filter_rows(df, content_type, source)
    return df[CONTENT_COLUMNS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_rows(df: pd.DataFrame, content_type: str, source: str) -> pd.DataFrame:
    """Drop rows with empty or malformed global_id, title, or embedding_text."""
    empty_global_id = f"{content_type}:{source}:"
    df = df[df["global_id"] != empty_global_id]
    df = df[df["title"].str.strip() != ""]
    df = df[df["embedding_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["global_id"])
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    catalog = build_content_catalog()
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(DEFAULT_OUTPUT_PATH, index=False)

    print("Content catalog built successfully.")
    print(f"  Rows    : {len(catalog)}")
    print(f"  Columns : {catalog.columns.tolist()}")
    print(f"  Output  : {DEFAULT_OUTPUT_PATH}")
    print("\nContent types:")
    print(catalog["content_type"].value_counts().to_string())
    print("\nSample global_ids:")
    for ctype in catalog["content_type"].unique():
        sample = catalog[catalog["content_type"] == ctype]["global_id"].iloc[0]
        print(f"  {sample}")


if __name__ == "__main__":
    main()
