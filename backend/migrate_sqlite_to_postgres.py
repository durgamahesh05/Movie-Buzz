from __future__ import annotations

import argparse
import sqlite3
from itertools import islice
from pathlib import Path
from typing import Iterable

from db import DB_BACKEND, get_db
from recommender import init_db as init_movie_tables
from trailer_router import _ensure_trailer_table
from user_model import init_users_table

DEFAULT_SOURCE_DB = Path(__file__).resolve().parent / "moviebuzz.db"
SERVING_TABLES = [
    "movies",
    "tags",
    "genome_scores",
    "omdb_cache",
    "user_feedback",
    "rating_timestamps",
    "model_metrics",
    "users",
    "wishlist",
    "user_interactions",
    "user_profiles",
    "recommendation_impressions",
    "trailer_cache",
]


def _batched(rows: Iterable[tuple], batch_size: int) -> Iterable[list[tuple]]:
    iterator = iter(rows)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def _sqlite_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row[1]) for row in rows]


def _truncate_destination_tables() -> None:
    with get_db() as conn:
        for table_name in reversed(SERVING_TABLES):
            conn.execute(f"DELETE FROM {table_name}")


def _copy_table(source_conn: sqlite3.Connection, table_name: str, batch_size: int) -> None:
    columns = _sqlite_columns(source_conn, table_name)
    if not columns:
        return

    column_sql = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    query = f"SELECT {column_sql} FROM {table_name}"
    rows = source_conn.execute(query)

    copied = 0
    with get_db() as dest_conn:
        for batch in _batched(rows, batch_size):
            dest_conn.executemany(
                f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})",
                batch,
            )
            copied += len(batch)
            print(f"{table_name}: copied {copied} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the serving tables from SQLite into the configured Postgres database.",
    )
    parser.add_argument(
        "--source-db",
        default=str(DEFAULT_SOURCE_DB),
        help="Path to the source SQLite moviebuzz.db file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Rows to insert per batch.",
    )
    parser.add_argument(
        "--skip-truncate",
        action="store_true",
        help="Append into the destination tables instead of clearing them first.",
    )
    args = parser.parse_args()

    if DB_BACKEND != "postgres":
        raise SystemExit(
            "DATABASE_URL must point to Supabase/Postgres before running this migration script."
        )

    source_db = Path(args.source_db).expanduser().resolve()
    if not source_db.exists():
        raise SystemExit(f"Source SQLite database not found: {source_db}")

    init_movie_tables()
    init_users_table()
    _ensure_trailer_table()

    if not args.skip_truncate:
        _truncate_destination_tables()

    with sqlite3.connect(str(source_db)) as source_conn:
        for table_name in SERVING_TABLES:
            _copy_table(source_conn, table_name, max(1, int(args.batch_size)))

    print("SQLite to Postgres migration complete.")


if __name__ == "__main__":
    main()
