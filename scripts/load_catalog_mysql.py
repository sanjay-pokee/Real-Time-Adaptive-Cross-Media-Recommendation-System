"""Load data/processed/content_catalog.csv into MySQL."""

from pathlib import Path

from backend.mysql_store import MySQLStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"


def main() -> None:
    store = MySQLStore()
    store.init_schema(create_database=True)
    count = store.upsert_content_catalog(CATALOG_PATH)
    print(f"Loaded {count} content rows into MySQL.")


if __name__ == "__main__":
    main()
