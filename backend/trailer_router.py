"""
trailer_router.py
==================
FastAPI router for movie trailer functionality.

Flow:
  1. Client calls GET /api/trailer/{movie_id}
  2. Check SQLite cache (trailer_cache table) — return immediately if hit
  3. On cache miss → call OMDB API for that movie
  4. OMDB returns a "Website" or embedded trailer URL (YouTube link)
  5. Extract YouTube video ID from the URL
  6. Cache video_id in SQLite with movie_id key
  7. Return { video_id, embed_url, title } to frontend

Mount this router in your main app:
    from trailer_router import router as trailer_router
    app.include_router(trailer_router)
"""

import os
import re
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import env, env_path

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Config  (reads from env vars — set these in your .env / systemd unit)
# ─────────────────────────────────────────────────────────────────────────────
OMDB_API_KEY = env("OMDB_API_KEY", "MOVIEBUZZ_OMDB_API_KEY", default="")
OMDB_BASE    = "https://www.omdbapi.com/"
DB_PATH      = env_path(
    "DATABASE_URL",
    "DB_PATH",
    "MOVIEBUZZ_DB_PATH",
    default=Path(__file__).resolve().parent / "moviebuzz.db",
)

router = APIRouter(prefix="/api", tags=["trailer"])

# ─────────────────────────────────────────────────────────────────────────────
#  DB helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    import os

    db_dir = os.path.dirname(str(DB_PATH)) or "."
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_trailer_table():
    """Create trailer_cache table if it doesn't exist yet."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trailer_cache (
                movie_id   INTEGER PRIMARY KEY,
                imdb_id    TEXT,
                title      TEXT,
                year       TEXT,
                video_id   TEXT,          -- YouTube video ID (null if not found)
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


_ensure_trailer_table()


def _movie_table_columns() -> set[str]:
    with _get_conn() as conn:
        rows = conn.execute("PRAGMA table_info(movies)").fetchall()
    return {str(row["name"]) for row in rows}


def _get_movie_lookup(movie_id: int) -> Optional[dict[str, Any]]:
    columns = _movie_table_columns()
    select_columns = ["title"]
    if "imdb_id" in columns:
        select_columns.append("imdb_id")

    query = f"SELECT {', '.join(select_columns)} FROM movies WHERE movieId = ?"
    with _get_conn() as conn:
        row = conn.execute(query, (movie_id,)).fetchone()
    if not row:
        return None
    payload = {
        "title": row["title"],
        "imdb_id": row["imdb_id"] if "imdb_id" in row.keys() else None,
    }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
#  YouTube video ID extraction
# ─────────────────────────────────────────────────────────────────────────────
_YT_PATTERNS = [
    # Standard watch URL:  https://www.youtube.com/watch?v=VIDEO_ID
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_\-]{11})",
]

def _extract_video_id(url: Optional[str]) -> Optional[str]:
    """Extract 11-char YouTube video ID from any YouTube URL format."""
    if not url:
        return None
    for pattern in _YT_PATTERNS:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  OMDB fetch
# ─────────────────────────────────────────────────────────────────────────────
async def _fetch_omdb(title: str, year: Optional[str] = None) -> dict:
    """
    Fetch movie data from OMDB by title.
    Returns the raw OMDB response dict.
    """
    params = {
        "apikey": OMDB_API_KEY,
        "t":      title,
        "plot":   "short",
        "type":   "movie",
    }
    if year:
        params["y"] = year

    if not OMDB_API_KEY:
        log.info("OMDB API key is missing; using trailer fallback for '%s'", title)
        return {}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(OMDB_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("OMDB request failed for title='%s': %s", title, exc)
        return {}

    if data.get("Response") == "False":
        log.warning(f"OMDB not found for title='{title}': {data.get('Error')}")
        return {}

    return data


async def _fetch_omdb_by_imdb(imdb_id: str) -> dict:
    """Fetch OMDB data by IMDb ID (more reliable when you have it)."""
    params = {
        "apikey": OMDB_API_KEY,
        "i":      imdb_id,
        "plot":   "short",
    }
    if not OMDB_API_KEY:
        log.info("OMDB API key is missing; using trailer fallback for IMDb id '%s'", imdb_id)
        return {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(OMDB_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        log.warning("OMDB request failed for imdb_id='%s': %s", imdb_id, exc)
        return {}

    if data.get("Response") == "False":
        return {}
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Response model
# ─────────────────────────────────────────────────────────────────────────────
class TrailerResponse(BaseModel):
    movie_id:  int
    title:     str
    year:      Optional[str]
    video_id:  Optional[str]          # None if no trailer found
    embed_url: Optional[str]          # Full iframe src URL
    found:     bool


# ─────────────────────────────────────────────────────────────────────────────
#  Main endpoint
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/trailer/{movie_id}", response_model=TrailerResponse)
async def get_trailer(movie_id: int):
    """
    Returns YouTube trailer info for a given MovieLens movie_id.

    Frontend usage:
        const res = await fetch(`/api/trailer/${movieId}`)
        const { embed_url, title, found } = await res.json()
        // embed_url → use as iframe src
    """

    # ── 1. Cache check ───────────────────────────────────────────────────────
    with _get_conn() as conn:
        cached = conn.execute(
            "SELECT * FROM trailer_cache WHERE movie_id = ?", (movie_id,)
        ).fetchone()

    if cached:
        video_id = cached["video_id"]
        return TrailerResponse(
            movie_id  = movie_id,
            title     = cached["title"] or "",
            year      = cached["year"],
            video_id  = video_id,
            embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=1&rel=0&modestbranding=1"
                        if video_id else None,
            found     = bool(video_id),
        )

    # ── 2. Look up movie title + imdb_id from our movies table ───────────────
    movie_row = _get_movie_lookup(movie_id)

    if not movie_row:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found in DB")

    raw_title = str(movie_row["title"] or "")
    imdb_id = str(movie_row["imdb_id"]).strip() if movie_row.get("imdb_id") else None

    # Strip year from MovieLens title format: "Toy Story (1995)" → "Toy Story", "1995"
    year_match = re.search(r"\((\d{4})\)\s*$", raw_title)
    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw_title).strip()
    year        = year_match.group(1) if year_match else None

    # ── 3. Fetch from OMDB ───────────────────────────────────────────────────
    if imdb_id:
        omdb_data = await _fetch_omdb_by_imdb(imdb_id)
    else:
        omdb_data = await _fetch_omdb(clean_title, year)

    # OMDB returns a "Website" field that sometimes contains a YouTube URL
    # It can also return a "Trailer" field in some extended responses
    trailer_url = (
        omdb_data.get("Trailer")  or   # extended OMDB trailer field
        omdb_data.get("Website")  or   # sometimes a YouTube link
        None
    )

    video_id = _extract_video_id(trailer_url)

    # ── 4. Fallback: construct YouTube search embed URL ───────────────────────
    # If OMDB doesn't return a direct YouTube URL,
    # use YouTube's search embed — no API key required.
    # Format: https://www.youtube.com/results?search_query=...
    # For iframe we use the search trick via youtube-nocookie embed
    if not video_id:
        # We'll return a search-based embed so the player still shows something
        search_query = f"{clean_title} {year or ''} official trailer".strip()
        # NOTE: YouTube doesn't allow direct search iframes — so we return
        # video_id=None and embed_url as a YouTube search URL.
        # The frontend will handle this by opening YouTube search in a new tab
        # as a graceful fallback. See TrailerPlayer.tsx comments.
        embed_url = None
        fallback_search_url = (
            f"https://www.youtube.com/results?search_query="
            f"{search_query.replace(' ', '+')}"
        )
        log.info(f"No trailer URL from OMDB for '{clean_title}' — fallback search ready")
    else:
        embed_url = (
            f"https://www.youtube.com/embed/{video_id}"
            f"?autoplay=1&rel=0&modestbranding=1&color=white"
        )
        fallback_search_url = None
        log.info(f"Trailer found for '{clean_title}': {video_id}")

    # ── 5. Cache result ───────────────────────────────────────────────────────
    with _get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO trailer_cache
                (movie_id, imdb_id, title, year, video_id)
            VALUES (?, ?, ?, ?, ?)
        """, (movie_id, imdb_id, clean_title, year, video_id))
        conn.commit()

    return TrailerResponse(
        movie_id  = movie_id,
        title     = clean_title,
        year      = year,
        video_id  = video_id,
        embed_url = embed_url,
        found     = bool(video_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Bulk pre-warm endpoint (optional — call once to pre-cache popular movies)
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/trailer/prewarm")
async def prewarm_trailers(limit: int = 100):
    """
    Pre-fetch trailers for the top-N most popular movies.
    Call this once after deployment to warm the cache.
    Usage: POST /api/trailer/prewarm?limit=500
    """
    with _get_conn() as conn:
        # Get top movies by rating count (already computed during training)
        rows = conn.execute("""
            SELECT m.movieId, m.title
            FROM movies m
            LEFT JOIN trailer_cache tc ON m.movieId = tc.movie_id
            WHERE tc.movie_id IS NULL
            ORDER BY m.num_ratings DESC
            LIMIT ?
        """, (limit,)).fetchall()

    results = {"cached": 0, "failed": 0, "total": len(rows)}
    for row in rows:
        try:
            await get_trailer(row["movieId"])
            results["cached"] += 1
        except Exception as e:
            log.warning(f"Prewarm failed for movie {row['movieId']}: {e}")
            results["failed"] += 1

    return results
