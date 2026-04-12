from __future__ import annotations

import logging
from time import monotonic
from typing import Any, Iterable
from urllib.parse import urlsplit

import pandas as pd
from pymongo import ASCENDING, DESCENDING, TEXT, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure
from pymongo.server_api import ServerApi

from config import env, env_int  # type: ignore

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency
    certifi = None

log = logging.getLogger(__name__)

DB_BACKEND = "mongodb"
DB_NAME = (
    env(
        "DATABASE_NAME",
        "MONGODB_DATABASE",
        "MONGO_DB_NAME",
        default="moviebuzz",
    ).strip()
    or "moviebuzz"
)
DB_URI = (
    env(
        "MONGODB_URI",
        "MONGO_URI",
        default="mongodb://localhost:27017",
    ).strip()
    or "mongodb://localhost:27017"
)
SERVER_SELECTION_TIMEOUT_MS = env_int(
    "MONGODB_SERVER_SELECTION_TIMEOUT_MS",
    "MONGO_SERVER_SELECTION_TIMEOUT_MS",
    "MOVIEBUZZ_MONGO_SERVER_SELECTION_TIMEOUT_MS",
    default=3_000,
)
CONNECT_TIMEOUT_MS = env_int(
    "MONGODB_CONNECT_TIMEOUT_MS",
    "MONGO_CONNECT_TIMEOUT_MS",
    "MOVIEBUZZ_MONGO_CONNECT_TIMEOUT_MS",
    default=3_000,
)
SOCKET_TIMEOUT_MS = env_int(
    "MONGODB_SOCKET_TIMEOUT_MS",
    "MONGO_SOCKET_TIMEOUT_MS",
    "MOVIEBUZZ_MONGO_SOCKET_TIMEOUT_MS",
    default=5_000,
)
FAILURE_COOLDOWN_SECONDS = env_int(
    "MONGODB_FAILURE_COOLDOWN_SECONDS",
    "MOVIEBUZZ_MONGO_FAILURE_COOLDOWN_SECONDS",
    default=5,
)
def is_mongodb() -> bool:
    return True


def format_db_target() -> str:
    try:
        parsed = urlsplit(DB_URI)
    except Exception:
        return f"mongodb://***@unknown/{DB_NAME}"

    scheme = parsed.scheme or "mongodb"
    host = parsed.hostname or "localhost"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{scheme}://{host}{port}/{DB_NAME}"


MongoDocument = dict[str, Any]


class MongoDBService:
    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name
        self._client: MongoClient[MongoDocument] | None = None
        self._db: Database[MongoDocument] | None = None
        self._indexes_ready = False
        self._unavailable_until = 0.0
        self._last_error = ""

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "serverSelectionTimeoutMS": SERVER_SELECTION_TIMEOUT_MS,
            "connectTimeoutMS": CONNECT_TIMEOUT_MS,
            "socketTimeoutMS": SOCKET_TIMEOUT_MS,
            "server_api": ServerApi("1"),
        }
        if self.uri.startswith("mongodb+srv://"):
            kwargs["tls"] = True
            kwargs["tlsAllowInvalidCertificates"] = True
            if certifi is not None:
                kwargs["tlsCAFile"] = certifi.where()
        return kwargs

    def connect(self) -> Database[MongoDocument]:
        if self._unavailable_until > monotonic():
            raise RuntimeError(self._last_error or "MongoDB is temporarily unavailable")
        if self._db is None:
            self._client = MongoClient(self.uri, **self._client_kwargs())
            self._db = self._client[self.db_name]
            try:
                self._client.admin.command("ping")
                self._last_error = ""
                self._unavailable_until = 0.0
            except Exception as exc:
                self._last_error = str(exc)
                self._unavailable_until = monotonic() + max(1, FAILURE_COOLDOWN_SECONDS)
                self.close()
                raise
        return self._db

    @property
    def db(self) -> Database[MongoDocument]:
        return self.connect()

    def collection(self, name: str) -> Collection[MongoDocument]:
        return self.db[name]

    def _safe_create_index(
        self,
        collection_name: str,
        keys: list[tuple[str, Any]],
        **kwargs: Any,
    ) -> None:
        try:
            self.collection(collection_name).create_index(keys, **kwargs)
        except OperationFailure as exc:
            log.warning(
                "MongoDB index creation skipped for %s %s: %s",
                collection_name,
                keys,
                exc,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._db = None
        self._indexes_ready = False

    def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return

        self._safe_create_index("movies", [("movieId", ASCENDING)], unique=True)
        self._safe_create_index("movies", [("source", ASCENDING), ("movieId", DESCENDING)])
        self._safe_create_index("movies", [("num_ratings", DESCENDING), ("avg_rating", DESCENDING)])
        self._safe_create_index("movies", [("title", TEXT), ("genres", TEXT)])

        self._safe_create_index("ratings", [("movieId", ASCENDING), ("userId", ASCENDING)])
        self._safe_create_index("ratings", [("userId", ASCENDING), ("movieId", ASCENDING)])
        self._safe_create_index("tags", [("movieId", ASCENDING)])
        self._safe_create_index("genome_scores", [("movieId", ASCENDING), ("tagId", ASCENDING)])
        self._safe_create_index("genome_scores", [("movieId", ASCENDING)])
        self._safe_create_index("rating_timestamps", [("movieId", ASCENDING)], unique=True)

        self._safe_create_index("omdb_cache", [("title", ASCENDING)], unique=True)
        self._safe_create_index("model_metrics", [("run_id", ASCENDING)], unique=True)
        self._safe_create_index("model_metrics", [("ts", DESCENDING)])
        self._safe_create_index("trailer_cache", [("movie_id", ASCENDING)], unique=True)

        self._safe_create_index("users", [("email", ASCENDING)], unique=True)
        self._safe_create_index("users", [("role", ASCENDING)])
        self._safe_create_index("users", [("created_at", DESCENDING)])

        self._safe_create_index(
            "wishlists",
            [("user_email", ASCENDING), ("movie_key", ASCENDING)],
            unique=True,
        )
        self._safe_create_index("wishlists", [("created_at", DESCENDING)])

        self._safe_create_index("user_interactions", [("user_id", ASCENDING), ("ts", DESCENDING)])
        self._safe_create_index("user_interactions", [("movieId", ASCENDING), ("ts", DESCENDING)])
        self._safe_create_index("user_profiles", [("user_id", ASCENDING)], unique=True)
        self._safe_create_index("recommendation_impressions", [("request_id", ASCENDING)])
        self._safe_create_index("recommendation_impressions", [("user_id", ASCENDING), ("ts", DESCENDING)])
        self._safe_create_index(
            "user_feedback",
            [("user_id", ASCENDING), ("movieId", ASCENDING)],
            unique=True,
        )

        self._indexes_ready = True


_mongo_service: MongoDBService | None = None


def get_db() -> MongoDBService:
    global _mongo_service
    if _mongo_service is None:
        _mongo_service = MongoDBService(DB_URI, DB_NAME)
    return _mongo_service


def get_collection(name: str) -> Collection[MongoDocument]:
    return get_db().collection(name)


def read_collection_df(
    name: str,
    query: MongoDocument | None = None,
    projection: MongoDocument | None = None,
    sort: list[tuple[str, int]] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    cursor = get_collection(name).find(query or {}, projection or None)
    if sort:
        cursor = cursor.sort(sort)
    if limit is not None:
        cursor = cursor.limit(int(limit))
    rows = list(cursor)
    if not rows:
        requested_columns = [
            key for key, value in (projection or {}).items() if key != "_id" and value
        ]
        return pd.DataFrame(columns=requested_columns)
    return _documents_to_df(rows)


def _documents_to_df(rows: Iterable[MongoDocument]) -> pd.DataFrame:
    payload: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        normalized.pop("_id", None)
        payload.append(normalized)
    return pd.DataFrame(payload)


def get_table_columns(*_: Any, **__: Any) -> set[str]:
    return set()


def read_sql_df(*_: Any, **__: Any) -> pd.DataFrame:
    raise RuntimeError("SQL access has been removed. Use MongoDB collection helpers instead.")
