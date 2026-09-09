"""Build the unified content_catalog.csv from all registered datasets.

Usage:
    python -m preprocessing.build_content_catalog
"""

from pathlib import Path

import pandas as pd

try:
    from .audience_tagging import annotate_audience
    from .cleaners import clean_text, join_non_empty, make_text_hash, parse_name_list
    from .content_schema import CATALOG_COLUMNS, CONTENT_COLUMNS, make_global_id
    from .loaders import DEFAULT_CONFIG_PATH, load_dataset_config, resolve_project_path
except ImportError:
    from audience_tagging import annotate_audience
    from cleaners import clean_text, join_non_empty, make_text_hash, parse_name_list
    from content_schema import CATALOG_COLUMNS, CONTENT_COLUMNS, make_global_id
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

    # Derive the audience metadata the constraint layer filters on. Runs once
    # on the combined catalog so every dataset is tagged by the same rules.
    catalog = annotate_audience(catalog[CONTENT_COLUMNS])
    return catalog[CATALOG_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def normalize_dataset(dataset_name: str, dataset_config: dict) -> pd.DataFrame:
    dataset_path = resolve_project_path(dataset_config["path"])
    raw_df = pd.read_csv(dataset_path)

    # A dataset may name its normalizer explicitly, which lets several verticals
    # share one. Every Amazon category ships the same metadata schema, so health
    # and industrial both use "amazon_meta" and adding a further vertical stays a
    # config change. Falls back to the dataset name for the original three.
    normalizer = dataset_config.get("normalizer", dataset_name)

    if normalizer == "movies":
        return normalize_movies(raw_df, dataset_config)

    if normalizer == "books":
        return normalize_books(raw_df, dataset_config)

    if normalizer == "music":
        return normalize_music(raw_df, dataset_config)

    if normalizer == "amazon_meta":
        return normalize_amazon_meta(raw_df, dataset_config)

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
    df["release_date"] = raw_df[dataset_config["release_date_column"]]
    df["popularity"] = raw_df[dataset_config["popularity_column"]]
    df["rating"] = raw_df[dataset_config["rating_column"]]

    # --- Load and merge credits (cast + director) ---
    top_n = int(dataset_config.get("credits_top_cast", 5))
    cast_series, director_series = _load_credits(
        dataset_config, raw_df[dataset_config["id_column"]], top_n
    )
    df["creators"] = [
        join_non_empty(vals, separator=", ")
        for vals in zip(director_series, cast_series)
    ]

    # metadata_text — richest text (includes keywords, cast, director)
    df["metadata_text"] = [
        join_non_empty(vals)
        for vals in zip(df["title"], genres, keywords, df["description"], df["creators"])
    ]

    # embedding_text — semantic text for the model (title + genres + description + director + top cast)
    df["embedding_text"] = [
        join_non_empty(vals)
        for vals in zip(df["title"], genres, df["description"], df["creators"])
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

def normalize_amazon_meta(raw_df: pd.DataFrame, dataset_config: dict) -> pd.DataFrame:
    """Normalize an Amazon Reviews 2023 metadata export.

    Shared by every Amazon vertical, because the metadata schema is identical
    across categories: which vertical a file becomes is decided purely by
    ``content_type`` in config/datasets.yaml.
    """
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
    df["categories"] = raw_df["categories"].apply(clean_text)
    # "store" is the brand or manufacturer, the closest analogue a physical
    # product has to an author or an artist.
    df["creators"] = raw_df["store"].apply(clean_text)
    # Amazon item metadata carries no release date.
    df["release_date"] = None
    df["popularity"] = raw_df[dataset_config["popularity_column"]]
    df["rating"] = raw_df[dataset_config["rating_column"]]

    main_category = raw_df["main_category"].apply(clean_text)

    df["metadata_text"] = [
        join_non_empty(vals)
        for vals in zip(
            df["title"], df["creators"], main_category, df["categories"], df["description"]
        )
    ]
    df["embedding_text"] = [
        join_non_empty(vals)
        for vals in zip(
            df["title"], df["creators"], df["categories"], df["description"]
        )
    ]

    df["text_hash"] = df["embedding_text"].apply(make_text_hash)

    df = _filter_rows(df, content_type, source)
    return df[CONTENT_COLUMNS]


def _load_credits(
    dataset_config: dict,
    id_series: pd.Series,
    top_n: int,
) -> tuple[pd.Series, pd.Series]:
    """Load the credits CSV and return (cast_series, director_series) aligned to id_series.

    Each element is a comma-joined string of names (empty string when unavailable).
    """
    credits_path_str = dataset_config.get("credits_path", "")
    if not credits_path_str:
        empty = pd.Series([""] * len(id_series), dtype=str)
        return empty.reset_index(drop=True), empty.reset_index(drop=True)

    credits_path = resolve_project_path(credits_path_str)
    if not credits_path.exists():
        print(f"  [WARNING] Credits file not found: {credits_path}. Skipping.")
        empty = pd.Series([""] * len(id_series), dtype=str)
        return empty.reset_index(drop=True), empty.reset_index(drop=True)

    print(f"  Loading credits: {credits_path}")
    credits_df = pd.read_csv(credits_path)
    # Build lookup: movie_id → (cast_str, director_str)
    cast_map: dict[str, str] = {}
    director_map: dict[str, str] = {}
    for _, row in credits_df.iterrows():
        mid = str(row["movie_id"])
        cast_map[mid] = _parse_credits_cast(row.get("cast", ""), top_n)
        director_map[mid] = _parse_credits_director(row.get("crew", ""))

    id_strs = id_series.fillna("").astype(str)
    cast_series = id_strs.map(lambda mid: cast_map.get(mid, ""))
    director_series = id_strs.map(lambda mid: director_map.get(mid, ""))
    return cast_series.reset_index(drop=True), director_series.reset_index(drop=True)


def _parse_credits_cast(value: object, top_n: int) -> str:
    """Return a comma-joined string of the top-N cast names from the JSON blob."""
    names = parse_name_list(value)  # reuse existing AST parser
    return ", ".join(names[:top_n])


def _parse_credits_director(value: object) -> str:
    """Extract the first Director name from the crew JSON blob."""
    text = clean_text(value)
    if not text:
        return ""
    try:
        import ast as _ast
        crew = _ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ""
    if not isinstance(crew, list):
        return ""
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            name = clean_text(member.get("name", ""))
            if name:
                return name
    return ""


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
