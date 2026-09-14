import pandas as pd

try:
    from .loaders import load_dataset_config, resolve_project_path
except ImportError:
    from loaders import load_dataset_config, resolve_project_path

# Read the registry rather than a hardcoded list. The paths here were written
# against the original Kaggle exports and went stale silently as each source
# moved to a live API - the movies path still pointed at tmdb_5000_movies.csv
# months after that file was replaced.
datasets = {
    name.title(): resolve_project_path(config["path"])
    for name, config in load_dataset_config().items()
}

for name, path in datasets.items():
    print("\n" + "="*60)
    print(f"{name} Dataset")
    print("="*60)

    df = pd.read_csv(path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)