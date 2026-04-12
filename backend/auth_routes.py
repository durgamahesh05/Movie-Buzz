"""
auth/auth_routes.py  –  MovieBuzz Authentication (MongoDB)
"""

import json
import logging
import random
from datetime import datetime, timedelta

import bcrypt
from fastapi import APIRouter, Request

from config import env
from email_service import (
    has_email_configuration,
    send_account_created_email,
    send_account_deleted_email,
    send_account_deletion_otp_email,
    send_password_reset_otp_email,
    send_verification_otp_email,
)
from recommender import (
    MOOD_GENRE_MAP,
    _clean_title,
    _curated_seed_metadata,
    _fallback_movie_description,
    _generated_poster_url,
    _is_missing_poster,
)
from user_model import (
    clear_otp,
    delete_user,
    find_one,
    find_one_by_login_identifier,
    get_all_users,
    get_preferences,
    get_wishlist,
    insert_one,
    remove_wishlist_item,
    set_otp,
    set_verified,
    update_preferences,
    update_name,
    update_password,
    update_role,
    upsert_wishlist_item,
)

log = logging.getLogger(__name__)
auth_router = APIRouter()
VERIFY_OTP_MINUTES = 5
SENSITIVE_OTP_MINUTES = 10
LOCAL_REQUEST_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
PASSWORD_POLICY_MESSAGE = (
    "Password must be at least 6 characters and include 1 uppercase letter, "
    "1 number, and 1 special character"
)


def _load_system_admin_accounts() -> list[dict[str, str]]:
    raw_value = env("MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON", default="").strip()
    if not raw_value:
        return []

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        log.warning("Invalid MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON: %s", exc)
        return []

    if not isinstance(payload, list):
        log.warning("MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON must be a JSON array")
        return []

    accounts: list[dict[str, str]] = []
    for index, entry in enumerate(payload, start=1):
        if not isinstance(entry, dict):
            log.warning("Skipping non-object admin entry at index %d", index - 1)
            continue

        name = str(entry.get("name") or f"Admin{index}").strip()
        email = str(entry.get("email") or "").strip().lower()
        password = str(entry.get("password") or "")
        if not email or not password:
            log.warning("Skipping admin entry %d because email or password is missing", index)
            continue

        accounts.append({
            "name": name,
            "email": email,
            "password": password,
        })

    return accounts


SYSTEM_ADMIN_ACCOUNTS = _load_system_admin_accounts()
SURVEY_GENRE_OPTIONS = [
    "All",
    "Action",
    "Comedy",
    "Drama",
    "Sci-Fi",
    "Thriller",
    "Horror",
    "Romance",
    "Animation",
    "Fantasy",
    "Crime",
]
SURVEY_MOOD_OPTIONS = list(MOOD_GENRE_MAP.keys())


# ── helpers ───────────────────────────────────────────────────────────────────
def generate_otp() -> str:
    return str(random.randint(100000, 999999))


def _normalise_email(value: str) -> str:
    return value.strip().lower()


def _env_flag(name: str, default: str = "0") -> bool:
    return env(name, default=default).strip().lower() not in {"", "0", "false", "no", "off"}


def _is_local_request(request: Request) -> bool:
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip().lower()
    return host in LOCAL_REQUEST_HOSTS


def _allow_local_auth_fallback(request: Request) -> bool:
    return (
        _is_local_request(request)
        and not has_email_configuration()
        and _env_flag(
            "MOVIEBUZZ_ALLOW_LOCAL_AUTH_FALLBACK",
            default="1",
        )
    )


def _local_otp_delivery_response(otp: str, action_label: str) -> dict[str, object]:
    return {
        "success": True,
        "msg": (
            f"{action_label} Email delivery is unavailable on localhost. "
            f"Use OTP {otp} to continue."
        ),
        "delivery": "local-fallback",
        "dev_otp": otp,
    }


def _otp_expiry(minutes: int) -> str:
    return (datetime.utcnow() + timedelta(minutes=minutes)).isoformat()


def _password_policy_error(password: str) -> str:
    if len(password) < 6:
        return PASSWORD_POLICY_MESSAGE
    if not any(char.isupper() for char in password):
        return PASSWORD_POLICY_MESSAGE
    if not any(char.isdigit() for char in password):
        return PASSWORD_POLICY_MESSAGE
    if not any(not char.isalnum() and not char.isspace() for char in password):
        return PASSWORD_POLICY_MESSAGE
    return ""


def _normalise_multi_select(
    raw_values: object,
    allowed_values: list[str],
    *,
    allow_all_exclusive: bool = False,
) -> list[str]:
    if raw_values is None:
        values: list[object] = []
    elif isinstance(raw_values, list):
        values = raw_values
    else:
        raise ValueError("Selections must be provided as a list")

    allowed_lookup = {value.lower(): value for value in allowed_values}
    normalized: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = str(value or "").strip()
        lowered = cleaned.lower()
        if not cleaned:
            continue
        if lowered not in allowed_lookup:
            raise ValueError(f"Unsupported value: {cleaned}")
        canonical = allowed_lookup[lowered]
        if canonical.lower() in seen:
            continue
        seen.add(canonical.lower())
        normalized.append(canonical)

    if allow_all_exclusive and "all" in {item.lower() for item in normalized}:
        return ["All"]

    return normalized


def _normalise_age(raw_value: object) -> int | None:
    if raw_value in (None, ""):
        return None
    try:
        age = int(str(raw_value).strip())
    except Exception as exc:
        raise ValueError("Age must be a whole number") from exc
    if age < 1 or age > 120:
        raise ValueError("Age must be between 1 and 120")
    return age


def _parse_survey_preferences(data: dict) -> tuple[int | None, list[str], list[str]]:
    age = _normalise_age(data.get("age"))
    preferred_genres = _normalise_multi_select(
        data.get("preferred_genres"),
        SURVEY_GENRE_OPTIONS,
        allow_all_exclusive=True,
    )
    preferred_moods = _normalise_multi_select(
        data.get("preferred_moods"),
        SURVEY_MOOD_OPTIONS,
    )
    return age, preferred_genres, preferred_moods


def _merge_survey_preferences(
    data: dict,
    current_preferences: dict,
) -> tuple[int | None, list[str], list[str]]:
    age = current_preferences.get("age")
    preferred_genres = list(current_preferences.get("preferred_genres") or [])
    preferred_moods = list(current_preferences.get("preferred_moods") or [])

    if "age" in data:
        age = _normalise_age(data.get("age"))

    if "preferred_genres" in data:
        preferred_genres = _normalise_multi_select(
            data.get("preferred_genres"),
            SURVEY_GENRE_OPTIONS,
            allow_all_exclusive=True,
        )

    if "preferred_moods" in data:
        preferred_moods = _normalise_multi_select(
            data.get("preferred_moods"),
            SURVEY_MOOD_OPTIONS,
        )

    return age, preferred_genres, preferred_moods


def _otp_is_valid(user: dict | None, otp: str, purpose: str) -> bool:
    if not user or not otp.strip():
        return False
    if user.get("otp_purpose") != purpose:
        return False
    if not user.get("otp"):
        return False
    try:
        expiry = datetime.fromisoformat(user["otp_expiry"])
    except Exception:
        return False
    if datetime.utcnow() > expiry:
        return False
    return user["otp"] == otp.strip()


def ensure_system_admins() -> int:
    ready_accounts = 0

    for account in _load_system_admin_accounts() or SYSTEM_ADMIN_ACCOUNTS:
        email = _normalise_email(account["email"])
        password = account["password"]
        existing = find_one(email)

        if existing:
            stored_password = str(existing.get("password") or "")
            password_matches = False
            if stored_password:
                try:
                    password_matches = bcrypt.checkpw(
                        password.encode(),
                        stored_password.encode(),
                    )
                except ValueError:
                    password_matches = False

            if not password_matches:
                update_password(
                    email,
                    bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
                )
            if str(existing.get("name") or "").strip() != account["name"]:
                update_name(email, account["name"])
            if existing.get("role") != "admin":
                update_role(email, "admin")
            if not existing.get("verified"):
                set_verified(email)
        else:
            insert_one({
                "name": account["name"],
                "email": email,
                "password": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
                "verified": True,
                "otp": None,
                "otp_expiry": None,
                "otp_purpose": None,
                "role": "admin",
                "created_at": datetime.utcnow().isoformat(),
            })

        ready_accounts += 1

    return ready_accounts


# ═══════════════════════════════════════════════════════════════════════════════
#  USER ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@auth_router.post("/signup")
async def signup(request: Request):
    data     = await request.json()
    name     = data.get("name", "").strip()
    email    = _normalise_email(data.get("email", ""))
    password = data.get("password", "")

    try:
        age, preferred_genres, preferred_moods = _parse_survey_preferences(data)
    except ValueError as exc:
        return {"success": False, "msg": str(exc)}

    if not name or not email or not password:
        return {"success": False, "msg": "All fields are required"}
    password_error = _password_policy_error(password)
    if password_error:
        return {"success": False, "msg": password_error}

    existing_user = find_one(email)
    if existing_user and existing_user.get("verified"):
        return {"success": False, "msg": "Email already registered"}

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    otp    = generate_otp()
    expiry = _otp_expiry(VERIFY_OTP_MINUTES)

    try:
        if existing_user:
            update_name(email, name)
            update_password(email, hashed)
            update_preferences(
                email,
                age=age,
                preferred_genres=preferred_genres,
                preferred_moods=preferred_moods,
            )
            set_otp(email, otp, expiry, "verify")
        else:
            insert_one({
                "name":       name,
                "email":      email,
                "password":   hashed,
                "verified":   False,
                "otp":        otp,
                "otp_expiry": expiry,
                "otp_purpose": "verify",
                "role":       "user",
                "age":        age,
                "preferred_genres": preferred_genres,
                "preferred_moods": preferred_moods,
                "created_at": datetime.utcnow().isoformat(),
            })

        if send_verification_otp_email(email, otp, name):
            return {"success": True, "msg": "OTP sent to your email"}
        if _allow_local_auth_fallback(request):
            log.warning("Local signup OTP fallback used for %s", email)
            return _local_otp_delivery_response(otp, "Account created.")
        return {"success": False, "msg": "Unable to send verification email right now"}
    except Exception:
        return {"success": False, "msg": "Unable to create account right now"}


@auth_router.post("/verify-otp")
async def verify_otp(request: Request):
    data  = await request.json()
    email = _normalise_email(data.get("email", ""))
    otp   = data.get("otp", "").strip()

    user = find_one(email)
    if not user:
        return {"success": False, "msg": "User not found"}
    if user.get("verified"):
        return {"success": False, "msg": "Already verified. Please login."}
    if not user.get("otp") or user.get("otp_purpose") != "verify":
        return {"success": False, "msg": "OTP expired. Please register again."}
    if not _otp_is_valid(user, otp, "verify"):
        return {"success": False, "msg": "Invalid OTP"}

    set_verified(email)
    welcome_email_sent = send_account_created_email(email)
    return {
        "success": True,
        "msg": "Account verified!",
        "welcome_email_sent": welcome_email_sent,
        "next_target": "/preferences-setup",
    }


@auth_router.post("/resend-otp")
async def resend_otp(request: Request):
    data  = await request.json()
    email = _normalise_email(data.get("email", ""))

    user = find_one(email)
    if not user:
        return {"success": False, "msg": "User not found"}
    if user.get("verified"):
        return {"success": False, "msg": "Already verified"}

    otp    = generate_otp()
    expiry = _otp_expiry(VERIFY_OTP_MINUTES)
    set_otp(email, otp, expiry, "verify")
    if send_verification_otp_email(email, otp, user.get("name", "")):
        return {"success": True, "msg": "New OTP sent"}
    if _allow_local_auth_fallback(request):
        log.warning("Local resend OTP fallback used for %s", email)
        return _local_otp_delivery_response(otp, "New OTP generated.")
    return {"success": False, "msg": "Unable to send verification email right now"}


@auth_router.post("/login")
async def login(request: Request):
    data     = await request.json()
    login_identifier = _normalise_email(data.get("email", ""))
    password = data.get("password", "")

    user = find_one_by_login_identifier(login_identifier)
    if not user and _allow_local_auth_fallback(request):
        ensure_system_admins()
        user = find_one_by_login_identifier(login_identifier)
    if not user:
        return {"success": False, "msg": "User not found"}
    if not user.get("verified"):
        return {"success": False, "msg": "Please verify your OTP first"}
    if not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return {"success": False, "msg": "Incorrect password"}

    resolved_email = _normalise_email(user.get("email", ""))
    preferences = get_preferences(resolved_email)

    return {
        "success": True,
        "msg":     "Login successful",
        "name":    user["name"],
        "email":   resolved_email,
        "role":    user.get("role", "user"),
        "age":     preferences.get("age"),
        "preferred_genres": preferences.get("preferred_genres", []),
        "preferred_moods": preferences.get("preferred_moods", []),
    }


@auth_router.get("/preferences/{email}")
def read_preferences(email: str):
    normalized_email = _normalise_email(email)
    user = find_one(normalized_email)
    if not user:
        return {"success": False, "msg": "User not found"}
    return {
        "success": True,
        "name": str(user.get("name") or "").strip(),
        "email": str(user.get("email") or normalized_email).strip(),
        **get_preferences(normalized_email),
    }


@auth_router.post("/preferences")
async def save_preferences(request: Request):
    data = await request.json()
    email = _normalise_email(data.get("email", ""))
    user = find_one(email)
    if not user:
        return {"success": False, "msg": "User not found"}

    current_preferences = get_preferences(email)

    try:
        age, preferred_genres, preferred_moods = _merge_survey_preferences(
            data,
            current_preferences,
        )
    except ValueError as exc:
        return {"success": False, "msg": str(exc)}

    try:
        update_preferences(
            email,
            age=age,
            preferred_genres=preferred_genres,
            preferred_moods=preferred_moods,
        )
    except Exception:
        return {"success": False, "msg": "Unable to save preferences right now"}

    return {
        "success": True,
        "msg": "Preferences saved",
        "name": str(user.get("name") or "").strip(),
        "email": str(user.get("email") or email).strip(),
        "age": age,
        "preferred_genres": preferred_genres,
        "preferred_moods": preferred_moods,
    }


@auth_router.post("/forgot-password/request-otp")
async def forgot_password_request_otp(request: Request):
    data = await request.json()
    email = _normalise_email(data.get("email", ""))
    user = find_one(email)

    if not user:
        return {"success": False, "msg": "User not found"}
    if not user.get("verified"):
        return {"success": False, "msg": "Please verify your account first"}

    otp = generate_otp()
    expiry = _otp_expiry(SENSITIVE_OTP_MINUTES)
    set_otp(email, otp, expiry, "password_reset")
    if send_password_reset_otp_email(email, otp):
        return {"success": True, "msg": "Password reset OTP sent to your email"}
    if _allow_local_auth_fallback(request):
        log.warning("Local password reset OTP fallback used for %s", email)
        return _local_otp_delivery_response(otp, "Password reset OTP generated.")
    return {"success": False, "msg": "Unable to send password reset email right now"}


@auth_router.post("/forgot-password/verify-otp")
async def forgot_password_verify_otp(request: Request):
    data = await request.json()
    email = _normalise_email(data.get("email", ""))
    otp = data.get("otp", "").strip()
    user = find_one(email)

    if not user:
        return {"success": False, "msg": "User not found"}
    if not _otp_is_valid(user, otp, "password_reset"):
        return {"success": False, "msg": "Invalid or expired OTP"}

    return {"success": True, "msg": "OTP verified"}


@auth_router.post("/forgot-password/reset")
async def forgot_password_reset(request: Request):
    data = await request.json()
    email = _normalise_email(data.get("email", ""))
    otp = data.get("otp", "").strip()
    new_password = data.get("new_password", "")
    user = find_one(email)

    if not user:
        return {"success": False, "msg": "User not found"}
    if not new_password:
        return {"success": False, "msg": "New password is required"}
    password_error = _password_policy_error(new_password)
    if password_error:
        return {"success": False, "msg": password_error}
    if not _otp_is_valid(user, otp, "password_reset"):
        return {"success": False, "msg": "Invalid or expired OTP"}

    update_password(email, bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode())
    clear_otp(email)
    return {"success": True, "msg": "Password reset successful"}


@auth_router.post("/delete/request-otp")
async def request_delete_account_otp(request: Request):
    data = await request.json()
    email = _normalise_email(data.get("email", ""))
    user = find_one(email)

    if not user:
        return {"success": False, "msg": "User not found"}
    if not user.get("verified"):
        return {"success": False, "msg": "Please verify your account first"}

    otp = generate_otp()
    expiry = _otp_expiry(SENSITIVE_OTP_MINUTES)
    set_otp(email, otp, expiry, "delete_account")
    if send_account_deletion_otp_email(email, otp):
        return {"success": True, "msg": "Account deletion OTP sent to your email"}
    if _allow_local_auth_fallback(request):
        log.warning("Local account deletion OTP fallback used for %s", email)
        return _local_otp_delivery_response(otp, "Account deletion OTP generated.")
    return {"success": False, "msg": "Unable to send deletion OTP right now"}


@auth_router.post("/delete/confirm")
async def confirm_delete_account(request: Request):
    data = await request.json()
    email = _normalise_email(data.get("email", ""))
    otp = data.get("otp", "").strip()
    user = find_one(email)

    if not user:
        return {"success": False, "msg": "User not found"}
    if not _otp_is_valid(user, otp, "delete_account"):
        return {"success": False, "msg": "Invalid or expired OTP"}

    delete_user(email)
    send_account_deleted_email(email)
    return {"success": True, "msg": "Account deleted successfully"}


@auth_router.get("/wishlist/{email}")
def wishlist_items(email: str):
    normalized_email = _normalise_email(email)
    user = find_one(normalized_email)
    if not user:
        return {"success": False, "msg": "User not found"}

    items = []
    for movie in get_wishlist(normalized_email):
        item = dict(movie)
        clean_title = str(item.get("clean_title", "")).strip()
        year = str(item.get("year", "")).strip()
        title = str(item.get("title", "")).strip()

        if not clean_title or not year:
            inferred_clean_title, inferred_year = _clean_title(clean_title or title)
            clean_title = clean_title or inferred_clean_title
            year = year or inferred_year
            item["clean_title"] = clean_title
            item["year"] = year

        seed_metadata = _curated_seed_metadata(clean_title or title, year)
        genres = str(item.get("genres", "")).strip() or str(seed_metadata.get("genres", "")).strip()
        item["genres"] = genres
        poster = str(item.get("poster", "")).strip()
        if _is_missing_poster(poster):
            item["poster"] = _generated_poster_url(clean_title or title, year, genres)

        if not str(item.get("plot", "")).strip():
            description = _fallback_movie_description(clean_title or title, year, genres)
            item["plot"] = description
            item["description"] = description
        else:
            item["description"] = str(item.get("plot", "")).strip()

        if not str(item.get("rating", "")).strip() and seed_metadata.get("rating"):
            item["rating"] = str(seed_metadata["rating"])
        if not str(item.get("imdb_rating", "")).strip() and str(item.get("rating", "")).strip():
            item["imdb_rating"] = str(item["rating"]).strip()

        items.append(item)

    return {"success": True, "items": items}


@auth_router.post("/wishlist")
async def wishlist_add(request: Request):
    data = await request.json()
    email = _normalise_email(data.get("email", ""))
    movie = data.get("movie") or {}

    user = find_one(email)
    if not user:
        return {"success": False, "msg": "User not found"}
    if not movie or not str(movie.get("movie_key", "")).strip():
        return {"success": False, "msg": "Movie details are required"}

    upsert_wishlist_item(email, movie)
    return {"success": True, "msg": "Movie added to wishlist"}


@auth_router.post("/wishlist/remove")
async def wishlist_remove(request: Request):
    data = await request.json()
    email = _normalise_email(data.get("email", ""))
    movie_key = str(data.get("movie_key", "")).strip()

    user = find_one(email)
    if not user:
        return {"success": False, "msg": "User not found"}
    if not movie_key:
        return {"success": False, "msg": "Movie key is required"}

    remove_wishlist_item(email, movie_key)
    return {"success": True, "msg": "Movie removed from wishlist"}


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN ROUTES  (user management)
# ═══════════════════════════════════════════════════════════════════════════════

@auth_router.get("/admin/users")
def admin_list_users():
    """Admin dashboard: list all registered users."""
    return get_all_users()


@auth_router.delete("/admin/users/{email}")
def admin_delete_user(email: str):
    """Admin: delete a user by email."""
    normalized_email = _normalise_email(email)
    delete_user(normalized_email)
    return {"success": True, "msg": f"User {normalized_email} deleted"}


@auth_router.patch("/admin/users/{email}/role")
async def admin_update_role(email: str, request: Request):
    """Admin: change a user's role (user / mod / admin)."""
    data = await request.json()
    role = data.get("role", "user")
    if role not in ("user", "mod", "admin"):
        return {"success": False, "msg": "Invalid role"}
    update_role(_normalise_email(email), role)
    return {"success": True, "msg": f"Role updated to {role}"}


@auth_router.get("/test")
def test():
    return {"status": "auth working", "db": "MongoDB", "smtp": "configured"}
