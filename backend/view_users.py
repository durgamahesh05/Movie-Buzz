from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from config import env_path


DEFAULT_COLUMNS = ["id", "name", "email", "role", "verified"]
DB_PATH = env_path(
    "DATABASE_URL",
    "DB_PATH",
    "MOVIEBUZZ_DB_PATH",
    default=Path(__file__).resolve().parent / "moviebuzz.db",
)


def format_table(columns: list[str], rows: list[tuple[object, ...]]) -> str:
    widths = []
    for index, column in enumerate(columns):
        max_cell_width = max((len(str(row[index])) for row in rows), default=0)
        widths.append(max(len(column), max_cell_width))

    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    divider = "-+-".join("-" * widths[index] for index in range(len(columns)))
    body = [
        " | ".join(str(row[index]).ljust(widths[index]) for index in range(len(columns)))
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def fetch_rows(limit: int | None, role: str | None) -> list[tuple[object, ...]]:
    query = f"SELECT {', '.join(DEFAULT_COLUMNS)} FROM users ORDER BY id"
    params: list[object] = []
    if role:
        query = f"SELECT {', '.join(DEFAULT_COLUMNS)} FROM users WHERE role = ? ORDER BY id"
        params.append(role)
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with sqlite3.connect(DB_PATH) as connection:
        return connection.execute(query, tuple(params)).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the users table from moviebuzz.db")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show only the first N rows.",
    )
    parser.add_argument(
        "--role",
        choices=["admin", "user"],
        default=None,
        help="Filter rows by role.",
    )
    parser.add_argument(
        "--admins-only",
        action="store_true",
        help="Shortcut for --role admin.",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    selected_role = "admin" if args.admins_only else args.role
    rows = fetch_rows(args.limit, selected_role)
    if not rows:
        if selected_role:
            print(f"No rows found in users for role={selected_role!r}.")
        else:
            print("No rows found in users.")
        return

    print(format_table(DEFAULT_COLUMNS, rows))


if __name__ == "__main__":
    main()
