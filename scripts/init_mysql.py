"""Create the MySQL database and backend tables from environment settings."""

from backend.mysql_store import MySQLStore


def main() -> None:
    MySQLStore().init_schema(create_database=True)
    print("MySQL schema is ready.")


if __name__ == "__main__":
    main()
