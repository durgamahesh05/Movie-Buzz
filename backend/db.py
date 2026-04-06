from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import pandas as pd

from config import env

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent / "moviebuzz.db"


def _supabase_database_url_from_env() -> str:
    project_ref = env("SUPABASE_PROJECT_REF", default="").strip()
    host = env(
        "SUPABASE_DB_HOST",
        default=(f"db.{project_ref}.supabase.co" if project_ref else ""),
    ).strip()
    password = env("SUPABASE_DB_PASSWORD", default="").strip()
    if not host or not password:
        return ""

    user = env("SUPABASE_DB_USER", default="postgres").strip() or "postgres"
    db_name = env("SUPABASE_DB_NAME", default="postgres").strip() or "postgres"
    port = env("SUPABASE_DB_PORT", default="5432").strip() or "5432"
    sslmode = env("SUPABASE_DB_SSLMODE", default="require").strip() or "require"
    return (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(db_name, safe='')}?sslmode={quote(sslmode, safe='')}"
    )


def _normalize_database_target(raw_value: str) -> tuple[str, str | Path]:
    value = (raw_value or "").strip()
    if not value:
        supabase_url = _supabase_database_url_from_env()
        if supabase_url:
            return "postgres", supabase_url
        return "sqlite", DEFAULT_SQLITE_PATH

    if value.startswith("postgres://"):
        return "postgres", "postgresql://" + value[len("postgres://") :]
    if value.startswith("postgresql://"):
        return "postgres", value
    if value.startswith("sqlite:///"):
        return "sqlite", Path(value[len("sqlite:///") :]).expanduser()

    return "sqlite", Path(value).expanduser()


_backend, _target = _normalize_database_target(
    env(
        "DATABASE_URL",
        "DB_PATH",
        "MOVIEBUZZ_DB_PATH",
        default="",
    )
)

DB_BACKEND = _backend
DB_TARGET = _target


def is_postgres() -> bool:
    return DB_BACKEND == "postgres"


def is_sqlite() -> bool:
    return DB_BACKEND == "sqlite"


def format_db_target() -> str:
    if is_sqlite():
        return str(DB_TARGET)

    parsed = urlsplit(str(DB_TARGET))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    path = parsed.path.lstrip("/") or "postgres"
    return f"postgresql://{host}:{port}/{path}"


class CompatRow(Mapping[str, Any]):
    def __init__(self, columns: Iterable[str], values: Iterable[Any]):
        self._columns = tuple(columns)
        self._values = tuple(values)
        self._mapping = dict(zip(self._columns, self._values))

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping.get(key, default)

    def keys(self) -> tuple[str, ...]:
        return self._columns

    def as_dict(self) -> dict[str, Any]:
        return dict(self._mapping)


class EmptyCursor:
    columns: tuple[str, ...] = ()

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []

    def __iter__(self) -> Iterator[Any]:
        return iter(())


class CompatCursor:
    def __init__(self, cursor: Any):
        self._cursor = cursor
        description = getattr(cursor, "description", None) or ()
        self.columns = tuple(col[0] for col in description)

    def _wrap(self, row: Any) -> CompatRow | None:
        if row is None:
            return None
        if isinstance(row, CompatRow):
            return row
        if isinstance(row, sqlite3.Row):
            return CompatRow(row.keys(), tuple(row))
        if isinstance(row, Mapping):
            return CompatRow(row.keys(), [row[key] for key in row.keys()])
        return CompatRow(self.columns, tuple(row))

    def fetchone(self) -> CompatRow | None:
        return self._wrap(self._cursor.fetchone())

    def fetchall(self) -> list[CompatRow]:
        return [row for row in (self._wrap(item) for item in self._cursor.fetchall()) if row is not None]

    def __iter__(self) -> Iterator[CompatRow]:
        for item in self._cursor:
            wrapped = self._wrap(item)
            if wrapped is not None:
                yield wrapped


_NAMED_PARAM_RE = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")


def _translate_query(query: str, params: Any) -> str:
    if is_sqlite():
        return query
    if isinstance(params, Mapping):
        return _NAMED_PARAM_RE.sub(r"%(\1)s", query)
    return query.replace("?", "%s")


class DatabaseConnection:
    def __init__(self) -> None:
        if is_sqlite():
            db_path = Path(str(DB_TARGET))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        else:
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL support requires psycopg. Install backend requirements first."
                ) from exc
            self._conn = psycopg.connect(str(DB_TARGET))

    def execute(self, query: str, params: Any = None) -> CompatCursor | EmptyCursor:
        sql = _translate_query(query, params)
        if is_sqlite():
            if params is None:
                cursor = self._conn.execute(query)
            else:
                cursor = self._conn.execute(query, params)
            return CompatCursor(cursor)

        cursor = self._conn.cursor()
        if params is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql, params)
        return CompatCursor(cursor)

    def executemany(self, query: str, rows: Iterable[Any]) -> None:
        rows = list(rows)
        if not rows:
            return

        if is_sqlite():
            self._conn.executemany(query, rows)
            return

        sql = _translate_query(query, rows[0])
        cursor = self._conn.cursor()
        cursor.executemany(sql, rows)

    def executescript(self, script: str) -> None:
        if is_sqlite():
            self._conn.executescript(script)
            return

        statements = [stmt.strip() for stmt in script.split(";") if stmt.strip()]
        for statement in statements:
            self.execute(statement)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DatabaseConnection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()


def get_db() -> DatabaseConnection:
    return DatabaseConnection()


def get_table_columns(conn: DatabaseConnection, table_name: str) -> set[str]:
    if is_sqlite():
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table_name,),
    ).fetchall()
    return {str(row["column_name"]) for row in rows}


def read_sql_df(conn: DatabaseConnection, query: str, params: Any = None) -> pd.DataFrame:
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame(columns=list(cursor.columns))
    return pd.DataFrame([row.as_dict() for row in rows], columns=list(cursor.columns))
