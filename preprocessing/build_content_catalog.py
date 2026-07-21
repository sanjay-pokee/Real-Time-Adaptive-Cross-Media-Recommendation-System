from pathlib import Path

import pandas as pd

try:
    from .cleaners import clean_text, join_non_empty, parse_name_list
    from .content_schema import CONTENT_COLUMNS
    from .loaders import DEFAULT_CONFIG_PATH, load_dataset_config, resolve_project_path
except ImportError:
    from cleaners import clean_text, join_non_empty, parse_name_list
    from content_schema import CONTENT_COLUMNS
    from loaders import DEFAULT_CONFIG_PATH, load_dataset_config, resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"


def build_content_catalog(config_path: Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    dataset_config = load_dataset_config(config_path)

    if not dataset_config:
        raise ValueError("No datasets are registered in the dataset config.")

    catalog_parts = [
        normalize_dataset(dataset_name, config)
        for dataset_name, config in dataset_config.items()
    ]

    catalog = pd.concat(catalog_parts, ignore_index=True)
    catalog = catalog.drop_duplicates(subset=["content_id"])
    catalog = catalog[catalog["title"].str.strip() != ""]
    catalog = catalog[catalog["metadata_text"].str.strip() != ""]

    return catalog[CONTENT_COLUMNS].reset_index(drop=True)


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


def normalize_movies(raw_df: pd.DataFrame, dataset_config: dict) -> pd.DataFrame:
    df = pd.DataFrame()

    df["content_id"] = (
        dataset_config["content_type"]
        + ":"
        + raw_df[dataset_config["id_column"]].fillna("").astype(str)
    )
    df["content_type"] = dataset_config["content_type"]
    df["title"] = raw_df[dataset_config["title_column"]].apply(clean_text)
    df["description"] = raw_df[dataset_config["description_column"]].apply(clean_text)
    df["categories"] = raw_df["genres"].apply(lambda value: ", ".join(parse_name_list(value)))
    df["creators"] = ""
    df["release_date"] = raw_df[dataset_config["release_date_column"]]
    df["popularity"] = raw_df[dataset_config["popularity_column"]]
    df["rating"] = raw_df[dataset_config["rating_column"]]

    keywords = raw_df["keywords"].apply(lambda value: ", ".join(parse_name_list(value)))

    df["metadata_text"] = [
        join_non_empty(values)
        for values in zip(df["title"], df["categories"], keywords, df["description"])
    ]
    df["source"] = dataset_config["source"]

    df = df[df["content_id"].str.strip() != "movie:"]
    df = df[df["title"].str.strip() != ""]
    df = df[df["metadata_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["content_id"])

    return df[CONTENT_COLUMNS]


def normalize_books(raw_df: pd.DataFrame, dataset_config: dict) -> pd.DataFrame:
    df = pd.DataFrame()

    df["content_id"] = (
        dataset_config["content_type"]
        + ":"
        + raw_df[dataset_config["id_column"]].fillna("").astype(str)
    )
    df["content_type"] = dataset_config["content_type"]
    df["title"] = raw_df[dataset_config["title_column"]].apply(clean_text)
    df["description"] = raw_df[dataset_config["description_column"]].apply(clean_text)

    categories = raw_df["categories"].apply(clean_text)
    search_category = raw_df["search_category"].apply(clean_text)
    df["categories"] = [
        join_non_empty(values, separator=", ")
        for values in zip(categories, search_category)
    ]

    df["creators"] = raw_df["authors"].apply(clean_text)
    df["release_date"] = raw_df[dataset_config["release_date_column"]]
    df["popularity"] = raw_df[dataset_config["popularity_column"]]
    df["rating"] = raw_df[dataset_config["rating_column"]]

    subtitle = raw_df["subtitle"].apply(clean_text)

    df["metadata_text"] = [
        join_non_empty(values)
        for values in zip(
            df["title"],
            subtitle,
            df["creators"],
            df["categories"],
            df["description"],
        )
    ]
    df["source"] = dataset_config["source"]

    df = df[df["content_id"].str.strip() != "book:"]
    df = df[df["title"].str.strip() != ""]
    df = df[df["metadata_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["content_id"])

    return df[CONTENT_COLUMNS]


def normalize_music(raw_df: pd.DataFrame, dataset_config: dict) -> pd.DataFrame:
    df = pd.DataFrame()

    df["content_id"] = (
        dataset_config["content_type"]
        + ":"
        + raw_df[dataset_config["id_column"]].fillna("").astype(str)
    )
    df["content_type"] = dataset_config["content_type"]
    df["title"] = raw_df[dataset_config["title_column"]].apply(clean_text)
    df["description"] = ""

    playlist_genre = raw_df["playlist_genre"].apply(clean_text)
    playlist_subgenre = raw_df["playlist_subgenre"].apply(clean_text)
    df["categories"] = [
        join_non_empty(values, separator=", ")
        for values in zip(playlist_genre, playlist_subgenre)
    ]

    df["creators"] = raw_df["track_artist"].apply(clean_text)
    df["release_date"] = raw_df[dataset_config["release_date_column"]]
    df["popularity"] = raw_df[dataset_config["popularity_column"]]
    df["rating"] = ""

    album_name = raw_df["track_album_name"].apply(clean_text)

    df["metadata_text"] = [
        join_non_empty(values)
        for values in zip(
            df["title"],
            df["creators"],
            album_name,
            playlist_genre,
            playlist_subgenre,
        )
    ]
    df["source"] = dataset_config["source"]

    df = df[df["content_id"].str.strip() != "music:"]
    df = df[df["title"].str.strip() != ""]
    df = df[df["metadata_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["content_id"])

    return df[CONTENT_COLUMNS]


def main() -> None:
    catalog = build_content_catalog()
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(DEFAULT_OUTPUT_PATH, index=False)

    print("Content catalog built successfully.")
    print(f"Rows: {len(catalog)}")
    print("Content types:")
    print(catalog["content_type"].value_counts())
    print(f"Columns: {catalog.columns.tolist()}")
    print(f"Output path: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()


