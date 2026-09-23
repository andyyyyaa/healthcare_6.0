"""Create missing application tables without deleting existing records."""

from pathlib import Path
import sqlite3


def init_database(path=None):
    database = Path(path) if path is not None else Path(__file__).resolve().parent / "user_db.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS user (name, password)")
        connection.execute("CREATE TABLE IF NOT EXISTS fracture (name, age, sex, fracture)")
        connection.execute("CREATE TABLE IF NOT EXISTS heart (name, age, sex, target)")
    return database


if __name__ == "__main__":
    print(f"Database ready: {init_database()}")
