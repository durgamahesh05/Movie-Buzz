"""
app.py  –  MovieBuzz FastAPI Backend v3
"""

import os
import io, csv
from typing import Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recommender import (
    search_movies,
    recommend_movies,
    browse_mood,
    record_feedback,
    save_movie_rating,
    add_movies_to_db,
    delete_movie_from_db,
    get_home_movies,
    get_movie_details,
    list_admin_movies,
    MOOD_GENRE_MAP,
    _clean_title,
    _curated_seed_metadata,
    _fallback_movie_description,
    _generated_poster_url,
    _is_missing_poster,
    _movie_key,
    get_model_metrics,
    render_model_metrics_plot,
    get_db as get_movie_db,
    sync_eval_report_to_db,
)
from auth_routes import auth_router, ensure_system_admins
from trailer_router import router as trailer_router
from user_model import get_all_wishlist_items, get_user_store_overview


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = str(os.getenv(name, "")).strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "on"}


def _auto_load_dataset_on_startup() -> bool:
    return _env_flag("MOVIEBUZZ_AUTO_LOAD_DATA", default=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("MOVIEBUZZ_SKIP_STARTUP") == "1":
        yield
        return
    import recommender
    startup_db_ready = False
    try:
        recommender.log.info("Initialising DB …")
        recommender.init_db()
        if _auto_load_dataset_on_startup():
            recommender.log.info("Auto-loading MovieLens dataset during startup …")
            recommender.load_ml25m_to_db()
        else:
            recommender.log.info(
                "Skipping automatic MovieLens import on startup. "
                "Run bootstrap_backend.py or run_training.py --skip-engine to load data manually."
            )
        metrics_synced = sync_eval_report_to_db(log_errors=True)
        if metrics_synced:
            recommender.log.info("Current evaluation report synced to MongoDB")
        seeded_admins = ensure_system_admins()
        recommender.log.info("System admin accounts ready: %d", seeded_admins)
        startup_db_ready = True
    except Exception as exc:
        recommender.log.warning(
            "DB startup bootstrap skipped; backend will continue with file-based fallbacks where possible: %s",
            exc,
        )

    if startup_db_ready and os.getenv("MOVIEBUZZ_WARM_ENGINE") == "1":
        recommender.log.info("Warming engine …")
        recommender.RecommenderEngine.get()
    elif startup_db_ready:
        recommender.log.info("Skipping engine warmup; lightweight search and recommendations stay available")
    else:
        recommender.log.info("Skipping engine warmup because MongoDB bootstrap did not complete")
    yield

app = FastAPI(title="MovieBuzz API", version="3.0", lifespan=lifespan)

# Configure CORS based on environment
CORS_ORIGINS = [
    "http://localhost:3000",      # Frontend local dev (Next.js default)
    "http://localhost:5173",      # Frontend local dev (Vite default)
    "http://localhost:8000",      # Backend local
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]

# Add production URLs
production_frontend = os.getenv("FRONTEND_URL")
if production_frontend:
    CORS_ORIGINS.append(production_frontend)
if os.getenv("RENDER") or os.getenv("FLY"):  # Render/Fly deployment detected
    CORS_ORIGINS.extend([
        "https://moviebuzz-frontend.onrender.com",
        "https://moviebuzz-frontend.fly.dev",
    ])

# Development: Allow all origins for easier testing
if os.getenv("MOVIEBUZZ_DEV"):
    CORS_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")
app.include_router(trailer_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _normalize_admin_filter_value(value: str | None) -> str:
    return " ".join(
        str(value or "")
        .lower()
        .replace("-", " ")
        .replace(":", " ")
        .replace(",", " ")
        .replace("|", " ")
        .split()
    )


def _movie_matches_admin_filters(
    movie: dict[str, Any],
    search: Optional[str] = None,
    genre: Optional[str] = None,
) -> bool:
    search_tokens = [
        token
        for token in _normalize_admin_filter_value(search).split()
        if token
    ]
    genre_tokens = [
        token
        for token in _normalize_admin_filter_value(genre).split()
        if token
    ]

    haystack = _normalize_admin_filter_value(
        " ".join(
            [
                str(movie.get("title") or ""),
                str(movie.get("clean_title") or ""),
                str(movie.get("genres") or ""),
                str(movie.get("year") or ""),
            ]
        )
    )
    genre_text = _normalize_admin_filter_value(str(movie.get("genres") or ""))

    if search_tokens and not all(token in haystack for token in search_tokens):
        return False
    if genre_tokens and not all(token in genre_text for token in genre_tokens):
        return False
    return True


def _wishlist_movie_items(
    search: Optional[str] = None,
    genre: Optional[str] = None,
) -> list[dict]:
    rows = get_all_wishlist_items()

    items: list[dict] = []
    seen: set[str] = set()

    for row in rows:
        movie_key = str(row["movie_key"] or "").strip()
        if movie_key and movie_key in seen:
            continue

        title = str(row["title"] or "").strip()
        clean_title = str(row["clean_title"] or "").strip()
        year = str(row["year"] or "").strip()
        if not clean_title or not year:
            inferred_clean_title, inferred_year = _clean_title(clean_title or title)
            clean_title = clean_title or inferred_clean_title
            year = year or inferred_year

        if not movie_key:
            movie_key = _movie_key(clean_title or title, year)
        seen.add(movie_key)

        seed_metadata = _curated_seed_metadata(clean_title or title, year)
        genres = str(row["genres"] or "").strip() or str(seed_metadata.get("genres", "")).strip()
        poster = str(row["poster"] or "").strip()
        if _is_missing_poster(poster):
            poster = _generated_poster_url(clean_title or title, year, genres)

        rating = str(row["rating"] or "").strip()
        if not rating and seed_metadata.get("rating"):
            rating = str(seed_metadata["rating"])
        imdb_rating = str(row["imdb_rating"] or "").strip() or rating
        description = str(row["plot"] or "").strip() or _fallback_movie_description(
            clean_title or title,
            year,
            genres,
        )

        movie_entry = {
            "movie_key": movie_key,
            "movie_id": None,
            "title": title or clean_title,
            "clean_title": clean_title or title,
            "year": year,
            "genres": genres,
            "poster": poster,
            "description": description,
            "plot": description,
            "cast": str(row["cast"] or "").strip(),
            "director": str(row["director"] or "").strip(),
            "runtime": str(row["runtime"] or "").strip(),
            "imdb_rating": imdb_rating,
            "rating": rating or imdb_rating or "N/A",
            "youtube_link": str(row["youtube_link"] or "").strip(),
            "source": "wishlist",
            "source_label": "Wishlist",
            "can_delete": False,
            "created_at": str(row["created_at"] or "").strip(),
        }

        if _movie_matches_admin_filters(movie_entry, search=search, genre=genre):
            items.append(movie_entry)

    return items


def _admin_movie_items(
    limit: Optional[int] = 1000,
    offset: int = 0,
    search: Optional[str] = None,
    genre: Optional[str] = None,
) -> dict[str, Any]:
    safe_limit = None if limit is None else max(1, int(limit))
    safe_offset = max(0, int(offset))
    catalog_payload = list_admin_movies(
        limit=safe_limit,
        offset=safe_offset,
        search=search,
        genre=genre,
    )
    catalog_items = list(catalog_payload.get("items") or [])
    total = int(catalog_payload.get("total") or 0)
    available_genres = sorted(
        {
            *catalog_payload.get("genres", []),
        }
    )

    return {
        "items": catalog_items,
        "total": total,
        "limit": safe_limit if safe_limit is not None else total,
        "offset": safe_offset,
        "has_more": (safe_offset + len(catalog_items)) < total,
        "genres": available_genres,
    }


# ── Search ────────────────────────────────────────────────────────────────────
@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=50),
):
    return search_movies(q, limit=limit)


@app.get("/movies/home")
def home_movies(
    limit: int = Query(default=50, ge=1, le=50),
    genre: Optional[str] = Query(default=None),
    user_email: Optional[str] = Query(default=None),
):
    return {"results": get_home_movies(limit, genre, user_email=user_email)}


@app.get("/movies/{movie_id}/details")
def movie_details(
    movie_id: int,
    user_email: Optional[str] = Query(default=None),
):
    movie = get_movie_details(movie_id, user_email=user_email)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


@app.get("/admin/overview")
def admin_overview():
    overview = get_user_store_overview()
    total_users = overview.get("total_users", 0)
    verified_users = overview.get("verified_users", 0)
    wishlist_items = overview.get("wishlist_items", 0)

    try:
        movie_store = get_movie_db()
        catalog_movies = int(movie_store.collection("movies").count_documents({}))
    except Exception:
        catalog_movies = 0

    return {
        "total_users": int(total_users or 0),
        "verified_users": int(verified_users or 0),
        "catalog_movies": int(catalog_movies or 0),
        "wishlist_items": int(wishlist_items or 0),
    }


@app.get("/admin/model-metrics")
def admin_model_metrics():
    return get_model_metrics()


@app.get("/admin/model-metrics/plot")
def admin_model_metrics_plot(
    kind: str = Query(..., pattern="^(comparison|loss|availability|engine)$"),
    theme: str = Query(default="light"),
):
    plot_bytes = render_model_metrics_plot(kind=kind, theme=theme)
    if not plot_bytes:
        raise HTTPException(
            status_code=404,
            detail="Model metric plot is unavailable",
        )
    return Response(content=plot_bytes, media_type="image/png")


@app.get("/admin/movies")
def admin_movies(
    limit: int = Query(default=1000, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = Query(default=None),
    genre: Optional[str] = Query(default=None),
):
    return _admin_movie_items(limit=limit, offset=offset, search=search, genre=genre)


@app.delete("/admin/movies/{movie_id}")
def admin_delete_movie(movie_id: int):
    if not delete_movie_from_db(movie_id):
        raise HTTPException(
            status_code=404,
            detail="Admin-added movie not found",
        )
    return {"success": True, "msg": "Movie deleted"}


# ── Recommend (hybrid ensemble) ───────────────────────────────────────────────
@app.get("/recommend")
def recommend(
    title:   str           = Query(..., min_length=1),
    user_id: int           = Query(default=1),
    mood:    Optional[str] = Query(default=None),
    user_email: Optional[str] = Query(default=None),
    limit:   int           = Query(default=50, ge=1, le=50),
):
    """
    Returns hybrid recommendations.
    mood: happy | sad | excited | scared | romantic | thoughtful | adventurous | relaxed
    Response includes resolved_title (NLP-corrected) and results[].
    """
    return recommend_movies(
        title,
        user_id,
        mood=mood,
        top_n=limit,
        user_email=user_email,
    )


# ── Browse by mood ────────────────────────────────────────────────────────────
@app.get("/mood/{mood}")
def mood_browse(mood: str):
    """Get trending movies for a mood. Valid moods: happy, sad, excited, etc."""
    if mood.lower() not in MOOD_GENRE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mood. Valid: {list(MOOD_GENRE_MAP.keys())}"
        )
    return browse_mood(mood)


@app.get("/moods")
def list_moods():
    return {"moods": list(MOOD_GENRE_MAP.keys())}


# ── User feedback ─────────────────────────────────────────────────────────────
class FeedbackBody(BaseModel):
    user_id:  str
    movie_id: int
    feedback: str   # "like" | "dislike" | "neutral"


class RatingBody(BaseModel):
    user_id: str
    movie_id: int
    rating: int

@app.post("/feedback")
def feedback(body: FeedbackBody):
    """
    Record user like / dislike for a movie.
    Affects future recommendations for that user.
    """
    ok = record_feedback(body.user_id, body.movie_id, body.feedback)
    if not ok:
        raise HTTPException(status_code=400,
                            detail="feedback must be 'like', 'dislike', or 'neutral'")
    return {"status": "recorded"}


@app.post("/feedback/rating")
def feedback_rating(body: RatingBody):
    ok = save_movie_rating(body.user_id, body.movie_id, body.rating)
    if not ok:
        raise HTTPException(status_code=400, detail="rating must be an integer between 1 and 5")
    return {"status": "recorded", "rating": int(body.rating)}


# ── Admin: manual add ─────────────────────────────────────────────────────────
class MovieEntry(BaseModel):
    title:   str
    genres:  Optional[str]  = ""
    rating:  Optional[float] = 0.0
    year:    Optional[str]  = ""
    poster:  Optional[str]  = ""

@app.post("/admin/movies/manual")
def admin_add_manual(movies: List[MovieEntry]):
    inserted = add_movies_to_db([m.model_dump() for m in movies])
    return {"inserted": inserted, "status": "ok"}


# ── Admin: CSV upload ─────────────────────────────────────────────────────────
@app.post("/admin/movies/csv")
async def admin_upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files accepted")

    text   = (await file.read()).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    movies_list = [
        {
            "title":  (row := {k.strip().lower(): v.strip() for k, v in r.items()}).get("title", ""),
            "genres": row.get("genres", ""),
            "rating": row.get("rating", 0),
            "year":   row.get("year", ""),
            "poster": row.get("poster", ""),
        }
        for r in reader
    ]
    inserted = add_movies_to_db(movies_list)
    return {"inserted": inserted, "status": "ok", "filename": file.filename}


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "MovieBuzz backend v3 running",
        "models":  ["TF-IDF", "SBERT", "SVD", "ALS", "NCF", "Genome", "Sentiment"],
        "signals": ["content", "collaborative", "trending", "mood", "feedback"],
    }
