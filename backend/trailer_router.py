"""
trailer_router.py
==================
FastAPI router for movie trailer functionality backed by MongoDB.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import DESCENDING

from config import env
from db import get_collection, get_db

log = logging.getLogger(__name__)

OMDB_API_KEY = env("OMDB_API_KEY", "MOVIEBUZZ_OMDB_API_KEY", default="")
OMDB_BASE = "https://www.omdbapi.com/"
TMDB_API_KEY = env("TMDB_API_KEY", "THEMOVIEDB_API_KEY", "MOVIEBUZZ_TMDB_API_KEY", default="")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_VIDEOS_URL_TEMPLATE = "https://api.themoviedb.org/3/movie/{movie_id}/videos"

router = APIRouter(prefix="/api", tags=["trailer"])


def _ensure_trailer_indexes() -> None:
    try:
        get_db().ensure_indexes()
    except Exception as exc:  # pragma: no cover - depends on live MongoDB
        log.warning("MongoDB trailer index check failed: %s", exc)


def _trailer_cache():
    return get_collection("trailer_cache")


def _movies():
    return get_collection("movies")


def _get_movie_lookup(movie_id: int) -> Optional[dict[str, Any]]:
    row = _movies().find_one(
        {"movieId": int(movie_id)},
        {"_id": 0, "title": 1, "imdb_id": 1},
    )
    if not row:
        return None
    return {
        "title": str(row.get("title") or "").strip(),
        "imdb_id": str(row.get("imdb_id") or "").strip() or None,
    }


_YT_PATTERNS = [
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_\-]{11})",
]


def _extract_video_id(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    for pattern in _YT_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _normalize_lookup_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _tmdb_match_score(item: dict[str, Any], clean_title: str, year: Optional[str]) -> float:
    candidate_title = str(item.get("title") or item.get("original_title") or "")
    normalized_title = _normalize_lookup_text(clean_title)
    normalized_candidate = _normalize_lookup_text(candidate_title)

    score = 0.0
    if normalized_title and normalized_title == normalized_candidate:
        score += 4.0
    elif normalized_title and normalized_title in normalized_candidate:
        score += 2.0
    elif normalized_title and normalized_candidate in normalized_title:
        score += 1.5

    release_year = str(item.get("release_date") or "")[:4]
    if year and release_year == year:
        score += 2.5
    elif not year and release_year:
        score += 0.3

    try:
        score += min(float(item.get("popularity") or 0) / 1000.0, 0.5)
    except Exception:
        pass

    return score


def _pick_tmdb_video_key(results: list[dict[str, Any]]) -> Optional[str]:
    youtube_results = [
        item
        for item in results
        if str(item.get("site") or "").strip().lower() == "youtube"
        and str(item.get("key") or "").strip()
    ]
    if not youtube_results:
        return None

    def score(item: dict[str, Any]) -> tuple[int, int, int, str]:
        type_value = str(item.get("type") or "").strip().lower()
        return (
            1 if bool(item.get("official")) else 0,
            2 if type_value == "trailer" else 1 if type_value == "teaser" else 0,
            int(item.get("size") or 0),
            str(item.get("published_at") or ""),
        )

    best = max(youtube_results, key=score)
    key = str(best.get("key") or "").strip()
    return key or None


async def _fetch_tmdb_video_id(clean_title: str, year: Optional[str] = None) -> Optional[str]:
    if not TMDB_API_KEY:
        return None

    search_params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
        "include_adult": "false",
        "language": "en-US",
    }
    if year:
        search_params["year"] = year

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            search_response = await client.get(TMDB_SEARCH_URL, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            results = list(search_data.get("results") or [])

            if not results and year:
                fallback_params = dict(search_params)
                fallback_params.pop("year", None)
                search_response = await client.get(TMDB_SEARCH_URL, params=fallback_params)
                search_response.raise_for_status()
                search_data = search_response.json()
                results = list(search_data.get("results") or [])

            if not results:
                return None

            best_match = max(
                results[:8],
                key=lambda item: _tmdb_match_score(item, clean_title, year),
            )
            tmdb_movie_id = int(best_match.get("id") or 0)
            if not tmdb_movie_id:
                return None

            videos_response = await client.get(
                TMDB_VIDEOS_URL_TEMPLATE.format(movie_id=tmdb_movie_id),
                params={"api_key": TMDB_API_KEY, "language": "en-US"},
            )
            videos_response.raise_for_status()
            videos_data = videos_response.json()
    except Exception as exc:
        log.warning("TMDB trailer lookup failed for title='%s': %s", clean_title, exc)
        return None

    return _pick_tmdb_video_key(list(videos_data.get("results") or []))


async def _fetch_omdb(title: str, year: Optional[str] = None) -> dict[str, Any]:
    params = {
        "apikey": OMDB_API_KEY,
        "t": title,
        "plot": "short",
        "type": "movie",
    }
    if year:
        params["y"] = year

    if not OMDB_API_KEY:
        log.info("OMDB API key is missing; using trailer fallback for '%s'", title)
        return {}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(OMDB_BASE, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        log.warning("OMDB request failed for title='%s': %s", title, exc)
        return {}

    if data.get("Response") == "False":
        return {}
    return data


async def _fetch_omdb_by_imdb(imdb_id: str) -> dict[str, Any]:
    params = {
        "apikey": OMDB_API_KEY,
        "i": imdb_id,
        "plot": "short",
    }
    if not OMDB_API_KEY:
        log.info("OMDB API key is missing; using trailer fallback for IMDb id '%s'", imdb_id)
        return {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(OMDB_BASE, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        log.warning("OMDB request failed for imdb_id='%s': %s", imdb_id, exc)
        return {}
    if data.get("Response") == "False":
        return {}
    return data


class TrailerResponse(BaseModel):
    movie_id: int
    title: str
    year: Optional[str]
    video_id: Optional[str]
    embed_url: Optional[str]
    found: bool


@router.get("/trailer/{movie_id}", response_model=TrailerResponse)
async def get_trailer(movie_id: int):
    _ensure_trailer_indexes()

    cached = _trailer_cache().find_one(
        {"movie_id": int(movie_id)},
        {"_id": 0, "title": 1, "year": 1, "video_id": 1},
    )
    cached_video_id = str(cached.get("video_id") or "").strip() or None if cached else None
    if cached and cached_video_id:
        return TrailerResponse(
            movie_id=movie_id,
            title=str(cached.get("title") or "").strip(),
            year=str(cached.get("year") or "").strip() or None,
            video_id=cached_video_id,
            embed_url=(
                f"https://www.youtube.com/embed/{cached_video_id}?autoplay=1&rel=0&modestbranding=1"
                if cached_video_id
                else None
            ),
            found=True,
        )

    movie_row = _get_movie_lookup(movie_id)
    if not movie_row:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found in DB")

    raw_title = str(movie_row.get("title") or "")
    imdb_id = str(movie_row.get("imdb_id") or "").strip() or None

    year_match = re.search(r"\((\d{4})\)\s*$", raw_title)
    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw_title).strip()
    year = year_match.group(1) if year_match else None

    video_id = await _fetch_tmdb_video_id(clean_title, year)
    if not video_id:
        omdb_data = await (
            _fetch_omdb_by_imdb(imdb_id) if imdb_id else _fetch_omdb(clean_title, year)
        )
        trailer_url = omdb_data.get("Trailer") or omdb_data.get("Website") or None
        video_id = _extract_video_id(trailer_url)

    if video_id:
        embed_url = (
            f"https://www.youtube.com/embed/{video_id}"
            f"?autoplay=1&rel=0&modestbranding=1&color=white"
        )
    else:
        embed_url = None
        log.info("No direct trailer URL from OMDB for '%s'", clean_title)

    _trailer_cache().update_one(
        {"movie_id": int(movie_id)},
        {
            "$set": {
                "movie_id": int(movie_id),
                "imdb_id": imdb_id,
                "title": clean_title,
                "year": year,
                "video_id": video_id,
                "provider": "tmdb" if video_id else "fallback",
                "fetched_at": datetime.utcnow().isoformat(),
            }
        },
        upsert=True,
    )

    return TrailerResponse(
        movie_id=movie_id,
        title=clean_title,
        year=year,
        video_id=video_id,
        embed_url=embed_url,
        found=bool(video_id),
    )


@router.post("/trailer/prewarm")
async def prewarm_trailers(limit: int = 100):
    _ensure_trailer_indexes()
    rows = list(
        _movies().find(
            {},
            {"_id": 0, "movieId": 1, "title": 1},
        ).sort(
            [("num_ratings", DESCENDING), ("avg_rating", DESCENDING), ("title", 1)]
        ).limit(max(1, int(limit)))
    )

    results = {"cached": 0, "failed": 0, "total": len(rows)}
    for row in rows:
        movie_id = int(row.get("movieId") or 0)
        if _trailer_cache().count_documents({"movie_id": movie_id}, limit=1):
            results["cached"] += 1
            continue
        try:
            await get_trailer(movie_id)
            results["cached"] += 1
        except Exception as exc:
            log.warning("Prewarm failed for movie %s: %s", movie_id, exc)
            results["failed"] += 1

    return results


_ensure_trailer_indexes()
