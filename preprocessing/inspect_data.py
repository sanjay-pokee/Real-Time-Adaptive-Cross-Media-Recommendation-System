import pandas as pd

datasets = {
    "Movies": "datasets/movies/movies/tmdb_5000_movies.csv",
    "Books": "datasets/books/archive/google_books_dataset.csv",
    "Music": "datasets/music/songs/spotify_songs.csv"
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