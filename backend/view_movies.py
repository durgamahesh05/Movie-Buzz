from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from config import env_path


DEFAULT_COLUMNS = [
    "movieId",
    "title",
    "genres",
    "avg_rating",
    "num_ratings",
    "source",
]
DB_PATH = env_path(
    "DATABASE_URL",
    "DB_PATH",
    "MOVIEBUZZ_DB_PATH",
    default=Path(__file__).resolve().parent / "moviebuzz.db",
)


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if value is None:
        return ""
    return str(value)


def format_table(columns: list[str], rows: list[tuple[object, ...]]) -> str:
    widths = []
    for index, column in enumerate(columns):
        max_cell_width = max((len(format_cell(row[index])) for row in rows), default=0)
        widths.append(max(len(column), max_cell_width))

    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    divider = "-+-".join("-" * widths[index] for index in range(len(columns)))
    body = [
        " | ".join(format_cell(row[index]).ljust(widths[index]) for index in range(len(columns)))
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def fetch_rows(
    limit: int | None,
    title_query: str | None,
    genre_query: str | None,
    min_ratings: int | None,
) -> list[tuple[object, ...]]:
    query = f"SELECT {', '.join(DEFAULT_COLUMNS)} FROM movies"
    filters: list[str] = []
    params: list[object] = []

    if title_query:
        filters.append("title LIKE ?")
        params.append(f"%{title_query}%")
    if genre_query:
        filters.append("genres LIKE ?")
        params.append(f"%{genre_query}%")
    if min_ratings is not None:
        filters.append("num_ratings >= ?")
        params.append(min_ratings)

    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += " ORDER BY num_ratings DESC, movieId ASC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with sqlite3.connect(DB_PATH) as connection:
        return connection.execute(query, tuple(params)).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the movies table from backend/moviebuzz.db")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Show only the first N rows after filtering. Default: 20.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Filter movies by title text.",
    )
    parser.add_argument(
        "--genre",
        default=None,
        help="Filter movies by genre text.",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=None,
        help="Show only movies with at least N ratings.",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    rows = fetch_rows(args.limit, args.title, args.genre, args.min_ratings)
    if not rows:
        print("No rows found in movies with the current filters.")
        return

    print(format_table(DEFAULT_COLUMNS, rows))


if __name__ == "__main__":
    main()
