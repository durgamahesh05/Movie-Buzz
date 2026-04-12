"""
auth/user_model.py  -  MongoDB-backed user store for MovieBuzz
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from pymongo import DESCENDING, UpdateOne

from db import format_db_target, get_collection, get_db

log = logging.getLogger(__name__)

INTERACTION_WEIGHT_MAP = {
    "impression": 0.05,
    "view": 0.20,
    "click": 0.35,
    "search": 0.15,
    "search_match": 0.25,
    "wishlist_add": 1.00,
    "wishlist_remove": -0.50,
    "trailer_open": 0.45,
    "watch_time": 0.25,
    "rating_like": 1.20,
    "rating_dislike": -0.80,
    "rating_neutral": 0.05,
}
DB_PATH = format_db_target()


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _normalize_login_identifier(value: str) -> str:
    return str(value or "").strip().lower()


def _compact_login_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_login_identifier(value))


def _slug_login_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize_login_identifier(value)).strip("_")


def _login_aliases_for_user(doc: dict[str, Any] | None) -> set[str]:
    payload = _normalize_user_doc(doc)
    if not payload:
        return set()

    email = str(payload.get("email") or "").strip().lower()
    local_part = email.split("@", 1)[0] if "@" in email else email
    name = str(payload.get("name") or "").strip()
    aliases: set[str] = set()

    for candidate in (email, local_part, name):
        normalized = _normalize_login_identifier(candidate)
        compact = _compact_login_identifier(candidate)
        slug = _slug_login_identifier(candidate)
        if normalized:
            aliases.add(normalized)
        if compact:
            aliases.add(compact)
        if slug:
            aliases.add(slug)

    return aliases


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except Exception:
            return []
        raw_items = parsed if isinstance(parsed, list) else []
    else:
        return []

    items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = str(item or "").strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        items.append(cleaned)
    return items


def _wishlist_projection() -> dict[str, int]:
    return {
        "_id": 0,
        "user_email": 1,
        "movie_key": 1,
        "movie_id": 1,
        "title": 1,
        "clean_title": 1,
        "year": 1,
        "genres": 1,
        "poster": 1,
        "plot": 1,
        "cast": 1,
        "director": 1,
        "imdb_rating": 1,
        "runtime": 1,
        "rating": 1,
        "youtube_link": 1,
        "created_at": 1,
    }


def _normalize_user_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None

    payload = dict(doc)
    payload.pop("_id", None)
    payload["email"] = _normalize_email(str(payload.get("email") or ""))
    payload["verified"] = bool(payload.get("verified"))
    payload["otp"] = payload.get("otp")
    payload["otp_expiry"] = str(payload.get("otp_expiry") or "").strip() or None
    payload["otp_purpose"] = str(payload.get("otp_purpose") or "").strip() or None
    payload["role"] = str(payload.get("role") or "user").strip() or "user"
    payload["preferred_genres"] = _json_list(payload.get("preferred_genres"))
    payload["preferred_moods"] = _json_list(payload.get("preferred_moods"))

    raw_age = payload.get("age")
    try:
        payload["age"] = int(raw_age) if raw_age not in (None, "") else None
    except Exception:
        payload["age"] = None

    payload["created_at"] = str(payload.get("created_at") or "").strip()
    payload["updated_at"] = str(payload.get("updated_at") or "").strip()
    return payload


def _normalize_wishlist_doc(doc: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(doc or {})
    payload.pop("_id", None)
    payload["user_email"] = _normalize_email(str(payload.get("user_email") or ""))
    payload["movie_key"] = str(payload.get("movie_key") or "").strip()
    payload["movie_id"] = payload.get("movie_id")
    payload["title"] = str(payload.get("title") or "").strip()
    payload["clean_title"] = str(payload.get("clean_title") or "").strip()
    payload["year"] = str(payload.get("year") or "").strip()
    payload["genres"] = str(payload.get("genres") or "").strip()
    payload["poster"] = str(payload.get("poster") or "").strip()
    payload["plot"] = str(payload.get("plot") or "").strip()
    payload["cast"] = str(payload.get("cast") or "").strip()
    payload["director"] = str(payload.get("director") or "").strip()
    payload["imdb_rating"] = str(payload.get("imdb_rating") or "").strip()
    payload["runtime"] = str(payload.get("runtime") or "").strip()
    payload["rating"] = str(payload.get("rating") or "").strip()
    payload["youtube_link"] = str(payload.get("youtube_link") or "").strip()
    payload["created_at"] = str(payload.get("created_at") or "").strip()
    return payload


def _preferences_from_user(user: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _normalize_user_doc(user)
    if not normalized:
        return {
            "age": None,
            "preferred_genres": [],
            "preferred_moods": [],
        }
    return {
        "age": normalized.get("age"),
        "preferred_genres": list(normalized.get("preferred_genres") or []),
        "preferred_moods": list(normalized.get("preferred_moods") or []),
    }


def init_users_table():
    """Ensure MongoDB collections and indexes exist."""
    try:
        get_db().ensure_indexes()
    except Exception as exc:  # pragma: no cover - depends on live MongoDB
        log.warning("MongoDB user-store index check failed: %s", exc)


def find_one(email: str) -> dict | None:
    try:
        doc = get_collection("users").find_one({"email": _normalize_email(email)})
    except Exception as exc:
        log.warning("MongoDB find_one failed for %s: %s", _normalize_email(email), exc)
        return None
    return _normalize_user_doc(doc)


def find_one_by_login_identifier(identifier: str) -> dict | None:
    normalized_identifier = _normalize_login_identifier(identifier)
    if not normalized_identifier:
        return None

    direct_match = find_one(normalized_identifier)
    if direct_match:
        return direct_match

    if "@" in normalized_identifier:
        return None

    try:
        rows = list(
            get_collection("users").find(
                {"role": {"$in": ["admin", "mod"]}},
                {
                    "_id": 0,
                    "name": 1,
                    "email": 1,
                    "password": 1,
                    "verified": 1,
                    "role": 1,
                    "age": 1,
                    "preferred_genres": 1,
                    "preferred_moods": 1,
                    "otp": 1,
                    "otp_expiry": 1,
                    "otp_purpose": 1,
                    "created_at": 1,
                    "updated_at": 1,
                },
            )
        )
    except Exception as exc:
        log.warning(
            "MongoDB admin login lookup failed for %s: %s",
            normalized_identifier,
            exc,
        )
        return None

    for row in rows:
        if normalized_identifier in _login_aliases_for_user(row):
            return _normalize_user_doc(row)

    return None


def insert_one(data: dict):
    email = _normalize_email(str(data.get("email") or ""))
    payload = {
        "name": str(data.get("name") or "").strip(),
        "email": email,
        "password": str(data.get("password") or ""),
        "verified": bool(data.get("verified")),
        "otp": data.get("otp"),
        "otp_expiry": str(data.get("otp_expiry") or "").strip() or None,
        "otp_purpose": str(data.get("otp_purpose") or "").strip() or None,
        "role": str(data.get("role") or "user").strip() or "user",
        "age": data.get("age"),
        "preferred_genres": _json_list(data.get("preferred_genres")),
        "preferred_moods": _json_list(data.get("preferred_moods")),
        "created_at": str(data.get("created_at") or _utc_now_iso()).strip(),
        "updated_at": _utc_now_iso(),
    }
    get_collection("users").insert_one(payload)


def get_preferences(email: str) -> dict[str, Any]:
    return _preferences_from_user(find_one(email))


def update_preferences(
    email: str,
    age: int | None = None,
    preferred_genres: list[str] | None = None,
    preferred_moods: list[str] | None = None,
):
    get_collection("users").update_one(
        {"email": _normalize_email(email)},
        {
            "$set": {
                "age": age,
                "preferred_genres": _json_list(preferred_genres or []),
                "preferred_moods": _json_list(preferred_moods or []),
                "updated_at": _utc_now_iso(),
            }
        },
    )


def set_verified(email: str):
    get_collection("users").update_one(
        {"email": _normalize_email(email)},
        {
            "$set": {
                "verified": True,
                "updated_at": _utc_now_iso(),
            },
            "$unset": {
                "otp": "",
                "otp_expiry": "",
                "otp_purpose": "",
            },
        },
    )


def set_otp(email: str, otp: str, otp_expiry: str, purpose: str = "verify"):
    get_collection("users").update_one(
        {"email": _normalize_email(email)},
        {
            "$set": {
                "otp": str(otp or "").strip(),
                "otp_expiry": str(otp_expiry or "").strip(),
                "otp_purpose": str(purpose or "verify").strip() or "verify",
                "updated_at": _utc_now_iso(),
            }
        },
    )


def clear_otp(email: str):
    get_collection("users").update_one(
        {"email": _normalize_email(email)},
        {
            "$unset": {
                "otp": "",
                "otp_expiry": "",
                "otp_purpose": "",
            },
            "$set": {"updated_at": _utc_now_iso()},
        },
    )


def update_password(email: str, password: str):
    get_collection("users").update_one(
        {"email": _normalize_email(email)},
        {"$set": {"password": password, "updated_at": _utc_now_iso()}},
    )


def update_name(email: str, name: str):
    get_collection("users").update_one(
        {"email": _normalize_email(email)},
        {"$set": {"name": str(name or "").strip(), "updated_at": _utc_now_iso()}},
    )


def get_all_users() -> list:
    try:
        rows = list(
            get_collection("users").find(
                {},
                {
                    "_id": 0,
                    "name": 1,
                    "email": 1,
                    "verified": 1,
                    "role": 1,
                    "created_at": 1,
                },
            ).sort([("created_at", DESCENDING), ("email", 1)])
        )
    except Exception as exc:
        log.warning("MongoDB get_all_users failed: %s", exc)
        return []
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        items.append(
            {
                "id": int(row.get("id") or index),
                "name": str(row.get("name") or "").strip(),
                "email": _normalize_email(str(row.get("email") or "")),
                "verified": bool(row.get("verified")),
                "role": str(row.get("role") or "user").strip() or "user",
                "created_at": str(row.get("created_at") or "").strip(),
            }
        )
    return items


def delete_user(email: str):
    normalized_email = _normalize_email(email)
    get_collection("wishlists").delete_many({"user_email": normalized_email})
    get_collection("user_interactions").delete_many({"user_id": normalized_email})
    get_collection("user_profiles").delete_many({"user_id": normalized_email})
    get_collection("recommendation_impressions").delete_many({"user_id": normalized_email})
    get_collection("user_feedback").delete_many({"user_id": normalized_email})
    get_collection("users").delete_one({"email": normalized_email})


def update_role(email: str, role: str):
    get_collection("users").update_one(
        {"email": _normalize_email(email)},
        {"$set": {"role": str(role or "user").strip() or "user", "updated_at": _utc_now_iso()}},
    )


def get_wishlist(email: str) -> list[dict[str, Any]]:
    rows = list(
        get_collection("wishlists").find(
            {"user_email": _normalize_email(email)},
            _wishlist_projection(),
        ).sort([("created_at", DESCENDING)])
    )
    return [_normalize_wishlist_doc(row) for row in rows]


def get_all_wishlist_items() -> list[dict[str, Any]]:
    rows = list(
        get_collection("wishlists").find({}, _wishlist_projection()).sort(
            [("created_at", DESCENDING)]
        )
    )
    return [_normalize_wishlist_doc(row) for row in rows]


def upsert_wishlist_item(email: str, movie: dict[str, Any]):
    normalized_email = _normalize_email(email)
    movie_key = str(movie.get("movie_key") or "").strip()
    payload = {
        "user_email": normalized_email,
        "movie_key": movie_key,
        "movie_id": movie.get("movie_id"),
        "title": str(movie.get("title") or "").strip(),
        "clean_title": str(movie.get("clean_title") or "").strip(),
        "year": str(movie.get("year") or "").strip(),
        "genres": str(movie.get("genres") or "").strip(),
        "poster": str(movie.get("poster") or "").strip(),
        "plot": str(movie.get("plot") or movie.get("description") or "").strip(),
        "cast": str(movie.get("cast") or "").strip(),
        "director": str(movie.get("director") or "").strip(),
        "imdb_rating": str(movie.get("imdb_rating") or "").strip(),
        "runtime": str(movie.get("runtime") or "").strip(),
        "rating": str(movie.get("rating") or "").strip(),
        "youtube_link": str(movie.get("youtube_link") or "").strip(),
        "created_at": _utc_now_iso(),
    }
    get_collection("wishlists").update_one(
        {"user_email": normalized_email, "movie_key": movie_key},
        {"$set": payload},
        upsert=True,
    )


def remove_wishlist_item(email: str, movie_key: str):
    get_collection("wishlists").delete_one(
        {
            "user_email": _normalize_email(email),
            "movie_key": str(movie_key or "").strip(),
        }
    )


def _normalise_interaction_event_type(event_type: str) -> str:
    return (
        str(event_type or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _coerce_movie_id(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def _derive_interaction_weight(
    event_type: str,
    event_value: float,
    query_text: str = "",
) -> float:
    normalized_type = _normalise_interaction_event_type(event_type)
    base_weight = INTERACTION_WEIGHT_MAP.get(normalized_type)

    if normalized_type == "watch_time":
        seconds = max(0.0, float(event_value or 0.0))
        base_weight = 0.20 + min(seconds, 300.0) / 300.0 * 0.80
    elif normalized_type == "search":
        base_weight = 0.15 + min(len(str(query_text or "").split()), 5) * 0.02

    if base_weight is None:
        base_weight = float(event_value or 0.0)

    return round(max(-1.5, min(float(base_weight), 1.5)), 4)


def record_interaction(
    user_id: str,
    event_type: str,
    movie_id: Any = None,
    event_value: Any = 1.0,
    session_id: str = "",
    query_text: str = "",
    source_page: str = "",
    metadata: Any = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    normalized_user_id = str(user_id or "").strip().lower()
    if not normalized_user_id:
        raise ValueError("user_id is required")

    normalized_event_type = _normalise_interaction_event_type(event_type)
    if not normalized_event_type:
        raise ValueError("event_type is required")

    try:
        resolved_event_value = float(event_value or 0.0)
    except Exception:
        resolved_event_value = 0.0

    resolved_movie_id = _coerce_movie_id(movie_id)
    resolved_ts = str(timestamp or _utc_now_iso()).strip()
    if isinstance(metadata, str):
        metadata_json = metadata.strip()
    elif metadata not in (None, ""):
        try:
            metadata_json = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"))
        except Exception:
            metadata_json = json.dumps({"value": str(metadata)}, ensure_ascii=True)
    else:
        metadata_json = ""

    resolved_weight = _derive_interaction_weight(
        normalized_event_type,
        resolved_event_value,
        query_text=str(query_text or ""),
    )

    get_collection("user_interactions").insert_one(
        {
            "user_id": normalized_user_id,
            "movieId": resolved_movie_id,
            "event_type": normalized_event_type,
            "event_value": resolved_event_value,
            "weight": resolved_weight,
            "session_id": str(session_id or "").strip(),
            "query_text": str(query_text or "").strip(),
            "source_page": str(source_page or "").strip(),
            "metadata_json": metadata_json,
            "ts": resolved_ts,
        }
    )
    get_collection("user_profiles").update_one(
        {"user_id": normalized_user_id},
        {
            "$set": {
                "last_active_at": resolved_ts,
                "updated_at": resolved_ts,
            },
            "$inc": {"total_events": 1},
            "$setOnInsert": {
                "genre_profile_json": "",
                "actor_profile_json": "",
                "keyword_profile_json": "",
            },
        },
        upsert=True,
    )

    return {
        "user_id": normalized_user_id,
        "movie_id": resolved_movie_id,
        "event_type": normalized_event_type,
        "event_value": resolved_event_value,
        "weight": resolved_weight,
        "ts": resolved_ts,
    }


def record_recommendation_impressions(
    user_id: str,
    request_id: str,
    items: list[dict[str, Any]],
    timestamp: str | None = None,
) -> int:
    normalized_user_id = str(user_id or "").strip().lower()
    normalized_request_id = str(request_id or "").strip()
    if not normalized_user_id or not normalized_request_id or not items:
        return 0

    resolved_ts = str(timestamp or _utc_now_iso()).strip()
    operations: list[UpdateOne] = []

    for rank_position, item in enumerate(items, start=1):
        movie_id = _coerce_movie_id(item.get("movie_id") or item.get("movieId"))
        if movie_id is None:
            continue
        try:
            score = float(item.get("score") or 0.0)
        except Exception:
            score = 0.0
        operations.append(
            UpdateOne(
                {
                    "request_id": normalized_request_id,
                    "user_id": normalized_user_id,
                    "movieId": movie_id,
                    "rank_position": rank_position,
                },
                {
                    "$set": {
                        "generator": str(item.get("generator") or item.get("source") or "").strip(),
                        "score": score,
                        "ts": resolved_ts,
                    }
                },
                upsert=True,
            )
        )

    if not operations:
        return 0

    get_collection("recommendation_impressions").bulk_write(operations, ordered=False)
    return len(operations)


def get_user_store_overview() -> dict[str, int]:
    try:
        users = get_collection("users")
        wishlists = get_collection("wishlists")
        return {
            "total_users": int(users.count_documents({})),
            "verified_users": int(users.count_documents({"verified": True})),
            "wishlist_items": int(wishlists.count_documents({})),
        }
    except Exception as exc:
        log.warning("MongoDB user overview failed: %s", exc)
        return {
            "total_users": 0,
            "verified_users": 0,
            "wishlist_items": 0,
        }


try:
    init_users_table()
except Exception as exc:  # pragma: no cover - depends on live MongoDB
    log.warning("MongoDB user store bootstrap skipped: %s", exc)
