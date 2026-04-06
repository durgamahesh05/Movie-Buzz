"""
recommender.py  –  MovieBuzz Recommendation Engine v4
=======================================================
Deep Learning Stack:
  NCF   : GMF + MLP with ReLU / LeakyReLU / Sigmoid activations
           Loss: Binary Cross-Entropy  +  BPR (Bayesian Personalised Ranking)
  Boost : XGBoost ensemble re-ranker over all model scores
  SVD   : Regularised matrix factorisation (Surprise)
  ALS   : Implicit feedback ALS (implicit library)
  SBERT : Sentence-Transformer embeddings (MiniLM-L6-v2)
  TF-IDF: Genre + Tag + OMDB plot content similarity
  Genome: MovieLens tag-genome cosine similarity
Signals : Trending, Sentiment, Mood, User Feedback
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os, re, pickle, logging, requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher, get_close_matches
from html import escape
from pathlib import Path
from typing import Optional, List, Dict, Any, cast
from urllib.parse import quote, quote_plus

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from config import env
from db import DB_BACKEND, format_db_target, get_db as open_db, is_postgres, read_sql_df
from user_model import get_preferences as get_user_preferences
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("recommender")


def _load_optional_module(module_name: str) -> Any | None:
    if not importlib.util.find_spec(module_name):
        return None
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        log.warning("Optional dependency %s unavailable: %s", module_name, exc)
        return None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        log.warning("Invalid integer for %s=%r; using %d", name, value, default)
        return default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_data_dir(base_dir: Path) -> Path:
    candidates: list[Path] = []
    for env_name in (
        "MOVIEBUZZ_DATA_DIR",
        "MOVIELENS_DATA_DIR",
        "TFDS_MOVIELENS_DATA_DIR",
    ):
        raw_value = (os.getenv(env_name) or "").strip()
        if raw_value:
            candidates.append(Path(raw_value).expanduser())

    candidates.extend([
        base_dir / "data",
        base_dir.parent / "data",
    ])

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if (candidate / "movies.csv").exists():
            if candidate != (base_dir / "data").resolve(strict=False):
                log.info("Using MovieLens data from %s", candidate)
            return candidate

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    return (base_dir / "data").resolve(strict=False)


# ── OMDB ──────────────────────────────────────────────────────────────────────
OMDB_API_KEY  = (
    env("OMDB_API_KEY", "MOVIEBUZZ_OMDB_API_KEY", default="")
).strip()
# NOTE: If OMDB rejects the key, go to https://www.omdbapi.com/apikey.aspx
# and check your email to activate the key. Free tier = 1000 requests/day.
_OMDB_KEY_INVALID = False  # reset on each server start so new key is retried
OMDB_BASE_URL = "https://www.omdbapi.com/"
OMDB_TIMEOUT  = _env_int("MOVIEBUZZ_OMDB_TIMEOUT", 5)

# ── TMDB (free public search, no key needed for search endpoint) ───────────────
TMDB_API_KEY = (
    env("TMDB_API_KEY", "MOVIEBUZZ_TMDB_API_KEY", default="")
).strip()
TMDB_SEARCH_URL  = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE  = "https://image.tmdb.org/t/p/w500"
TMDB_TIMEOUT     = _env_int("MOVIEBUZZ_TMDB_TIMEOUT", 5)

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_TIMEOUT = _env_int("MOVIEBUZZ_ITUNES_TIMEOUT", 4)
OMDB_MEMORY_CACHE_TTL_SECONDS = _env_int(
    "MOVIEBUZZ_OMDB_MEMORY_CACHE_TTL_SECONDS",
    60 * 60 * 6,
)
OMDB_MEMORY_CACHE_MAX_ITEMS = _env_int(
    "MOVIEBUZZ_OMDB_MEMORY_CACHE_MAX_ITEMS",
    2048,
)
OMDB_PREFETCH_BATCH_SIZE = _env_int("MOVIEBUZZ_OMDB_PREFETCH_BATCH_SIZE", 25)
OMDB_PREFETCH_WORKERS = _env_int("MOVIEBUZZ_OMDB_PREFETCH_WORKERS", 4)
_OMDB_MEMORY_CACHE: dict[str, tuple[float, dict[str, str]]] = {}

# ── Optional deps (graceful fallbacks) ───────────────────────────────────────
_sentence_transformers = _load_optional_module("sentence_transformers")
SentenceTransformer = (
    _sentence_transformers.SentenceTransformer
    if _sentence_transformers is not None
    else None
)
HAS_SBERT = SentenceTransformer is not None

fuzz_process = _load_optional_module("rapidfuzz.process")
HAS_RAPIDFUZZ = fuzz_process is not None

_textblob = _load_optional_module("textblob")
TextBlob = _textblob.TextBlob if _textblob is not None else None
HAS_TEXTBLOB = TextBlob is not None

implicit = _load_optional_module("implicit")
HAS_IMPLICIT = implicit is not None

tf = None
keras = None
HAS_TF = False
_TF_RUNTIME_LOADED = False


def _load_tensorflow_runtime() -> tuple[Any | None, Any | None]:
    global tf, keras, HAS_TF, _TF_RUNTIME_LOADED
    if _TF_RUNTIME_LOADED:
        return tf, keras

    _TF_RUNTIME_LOADED = True
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    tf = _load_optional_module("tensorflow")
    keras = getattr(tf, "keras", None) if tf is not None else None
    HAS_TF = tf is not None and keras is not None
    return tf, keras

xgb = _load_optional_module("xgboost")
HAS_XGB = xgb is not None

_matplotlib = _load_optional_module("matplotlib")
plt = None
HAS_MATPLOTLIB = False
if _matplotlib is not None:
    try:
        _matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        HAS_MATPLOTLIB = True
    except Exception as exc:
        log.warning("Optional dependency matplotlib unavailable: %s", exc)
        plt = None

surprise = _load_optional_module("surprise")
Dataset = getattr(surprise, "Dataset", None)
Reader = getattr(surprise, "Reader", None)
SVD = getattr(surprise, "SVD", None)
HAS_SURPRISE = surprise is not None

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = _resolve_data_dir(BASE_DIR)
MODEL_DIR = BASE_DIR / "models"
DB_PATH   = format_db_target()
MODEL_DIR.mkdir(exist_ok=True)

RATING_CHUNK = 500_000
ALS_TRAIN_ROWS = _env_int("ALS_TRAIN_ROWS", 5_000_000)
SVD_TRAIN_ROWS = _env_int("SVD_TRAIN_ROWS", 2_000_000)
NCF_TRAIN_ROWS = _env_int("NCF_TRAIN_ROWS", 2_000_000)
XGB_TRAIN_ROWS = _env_int("XGB_TRAIN_ROWS", 500_000)
SBERT_MODEL  = "all-MiniLM-L6-v2"
SBERT_CACHE  = MODEL_DIR / "sbert_embeddings.npy"
SBERT_IDX    = MODEL_DIR / "sbert_index.pkl"
MOVIE_SBERT_EMBEDDINGS_PATH = MODEL_DIR / "movie_sbert_embeddings.pkl"
USER_TASTE_VECTORS_PATH = MODEL_DIR / "user_taste_vectors.npy"
USER_TASTE_ID_MAP_PATH = MODEL_DIR / "user_taste_id_map.pkl"
SVD_PATH     = MODEL_DIR / "svd_model.pkl"
ALS_PATH     = MODEL_DIR / "als_model.pkl"
NCF_PATH     = MODEL_DIR / "ncf_model.keras"
NCF_WEIGHTS_PATH = MODEL_DIR / "ncf_model.weights.h5"
NCF_META_PATH = MODEL_DIR / "ncf_model_meta.pkl"
XGB_PATH     = MODEL_DIR / "xgb_ranker.pkl"
XGB_JSON_PATH = MODEL_DIR / "xgb_ranker.json"
XGB_META_PATH = MODEL_DIR / "xgb_ranker_meta.pkl"
XGB_FEATURE_CONTEXT_PATH = MODEL_DIR / "xgb_feature_context.pkl"
NCF_ENC_PATH = MODEL_DIR / "ncf_encoders.pkl"
EVAL_REPORT_PATH = MODEL_DIR / "eval_report.json"
HOME_CACHE_TTL_SECONDS = 60 * 30
HOME_POSTER_HYDRATE_LIMIT = _env_int("MOVIEBUZZ_HOME_POSTER_HYDRATE_LIMIT", 24)
_HOME_MOVIES_CACHE: dict[str, dict[str, Any]] = {}
_CURATED_CATALOG_CACHE: list[dict[str, Any]] = []
_LIGHTWEIGHT_TITLE_CATALOG_CACHE: list[dict[str, Any]] = []
LEGACY_XGB_FEATURE_COLUMNS = [
    "avg_rating",
    "svd_score",
    "log_num_ratings",
    "sentiment_signal",
    "trending_score",
]


class XGBModelBundle:
    def __init__(
        self,
        model: Any,
        scaler: MinMaxScaler,
        calibrator: Any | None,
        model_kind: str,
        calibration_kind: str,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.calibrator = calibrator
        self.model_kind = model_kind
        self.calibration_kind = calibration_kind
        self.threshold = float(threshold)

    def _scale(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self.scaler.transform(arr).astype(np.float32)

    def _base_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.model_kind == "rank:ndcg":
            return np.asarray(self.model.predict(X_scaled), dtype=np.float32)
        scores = np.asarray(self.model.predict_proba(X_scaled)[:, 1], dtype=np.float32)
        return np.clip(scores, 1e-6, 1 - 1e-6)

    def _calibrate(self, base_scores: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            calibrated = base_scores
        elif self.calibration_kind == "platt":
            calibrated = self.calibrator.predict_proba(base_scores.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.predict(base_scores)
        return np.clip(np.asarray(calibrated, dtype=np.float32), 1e-6, 1 - 1e-6)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_scaled = self._scale(X)
        calibrated = self._calibrate(self._base_scores(X_scaled))
        return np.column_stack([1.0 - calibrated, calibrated]).astype(np.float32)

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(np.int32)


def _new_xgb_estimator(model_kind: str) -> Any | None:
    if not HAS_XGB or xgb is None:
        return None
    if model_kind == "rank:ndcg":
        return xgb.XGBRanker()
    return xgb.XGBClassifier()


def _load_xgb_artifacts() -> Any | None:
    if XGB_PATH.exists():
        try:
            with open(XGB_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            log.warning("Could not load legacy XGBoost pickle: %s", exc)

    if XGB_JSON_PATH.exists() and XGB_META_PATH.exists() and HAS_XGB and xgb is not None:
        try:
            with open(XGB_META_PATH, "rb") as f:
                payload = pickle.load(f)
            model_kind = str(payload.get("model_kind") or "binary:logistic")
            model = _new_xgb_estimator(model_kind)
            if model is None:
                return None
            model.load_model(str(XGB_JSON_PATH))
            return XGBModelBundle(
                model=model,
                scaler=payload["scaler"],
                calibrator=payload.get("calibrator"),
                model_kind=model_kind,
                calibration_kind=str(payload.get("calibration_kind") or "none"),
                threshold=float(payload.get("threshold", 0.5)),
            )
        except Exception as exc:
            log.warning("Could not load XGBoost JSON artifacts: %s", exc)
    return None

CURATED_HOME_MOVIES: list[tuple[str, str]] = [
    ("The Shawshank Redemption", "1994"),
    ("The Godfather", "1972"),
    ("The Dark Knight", "2008"),
    ("Pulp Fiction", "1994"),
    ("Fight Club", "1999"),
    ("Alien", "1979"),
    ("The Shining", "1980"),
    ("Get Out", "2017"),
    ("A Quiet Place", "2018"),
    ("Inception", "2010"),
    ("Interstellar", "2014"),
    ("The Matrix", "1999"),
    ("Forrest Gump", "1994"),
    ("The Lord of the Rings: The Fellowship of the Ring", "2001"),
    ("The Lord of the Rings: The Two Towers", "2002"),
    ("The Lord of the Rings: The Return of the King", "2003"),
    ("The Empire Strikes Back", "1980"),
    ("The Silence of the Lambs", "1991"),
    ("Se7en", "1995"),
    ("Gladiator", "2000"),
    ("The Green Mile", "1999"),
    ("Saving Private Ryan", "1998"),
    ("The Departed", "2006"),
    ("Whiplash", "2014"),
    ("Parasite", "2019"),
    ("Joker", "2019"),
    ("Avengers: Endgame", "2019"),
    ("Spider-Man", "2002"),
    ("Spider-Man 2", "2004"),
    ("Spider-Man: Homecoming", "2017"),
    ("Spider-Man: Into the Spider-Verse", "2018"),
    ("Spider-Man: No Way Home", "2021"),
    ("Spider-Man: Across the Spider-Verse", "2023"),
    ("Mad Max: Fury Road", "2015"),
    ("La La Land", "2016"),
    ("The Prestige", "2006"),
    ("Django Unchained", "2012"),
    ("The Social Network", "2010"),
    ("Blade Runner 2049", "2017"),
    ("The Grand Budapest Hotel", "2014"),
    ("Toy Story", "1995"),
    ("Toy Story 3", "2010"),
    ("Finding Nemo", "2003"),
    ("Up", "2009"),
    ("Coco", "2017"),
    ("Inside Out", "2015"),
    ("Soul", "2020"),
    ("Moana", "2016"),
    ("Black Panther", "2018"),
    ("Iron Man", "2008"),
    ("Captain America: The Winter Soldier", "2014"),
    ("Doctor Strange", "2016"),
    ("Guardians of the Galaxy", "2014"),
    ("Top Gun: Maverick", "2022"),
    ("Dune", "2021"),
    ("Dune: Part Two", "2024"),
    ("Oppenheimer", "2023"),
    ("Barbie", "2023"),
    ("The Conjuring", "2013"),
    ("Scream", "1996"),
    ("Psycho", "1960"),
    ("The Exorcist", "1973"),
    ("Hereditary", "2018"),
    ("The Sixth Sense", "1999"),
]

CURATED_HOME_METADATA: dict[tuple[str, str], dict[str, Any]] = {
    ("The Shawshank Redemption", "1994"): {"genres": "Drama Crime", "rating": 9.3},
    ("The Godfather", "1972"): {"genres": "Crime Drama", "rating": 9.2},
    ("The Dark Knight", "2008"): {"genres": "Action Crime Drama", "rating": 9.0},
    ("Pulp Fiction", "1994"): {"genres": "Crime Drama", "rating": 8.9},
    ("Fight Club", "1999"): {"genres": "Drama Thriller", "rating": 8.8},
    ("Alien", "1979"): {"genres": "Horror Sci-Fi", "rating": 8.5},
    ("The Shining", "1980"): {"genres": "Drama Horror", "rating": 8.4},
    ("Get Out", "2017"): {"genres": "Horror Mystery Thriller", "rating": 7.8},
    ("A Quiet Place", "2018"): {"genres": "Drama Horror Sci-Fi", "rating": 7.5},
    ("Inception", "2010"): {"genres": "Action Adventure Sci-Fi", "rating": 8.8},
    ("Interstellar", "2014"): {"genres": "Adventure Drama Sci-Fi", "rating": 8.7},
    ("The Matrix", "1999"): {"genres": "Action Sci-Fi", "rating": 8.7},
    ("Forrest Gump", "1994"): {"genres": "Drama Romance", "rating": 8.8},
    ("The Lord of the Rings: The Fellowship of the Ring", "2001"): {"genres": "Adventure Fantasy Action", "rating": 8.8},
    ("The Lord of the Rings: The Two Towers", "2002"): {"genres": "Adventure Fantasy Action", "rating": 8.8},
    ("The Lord of the Rings: The Return of the King", "2003"): {"genres": "Adventure Fantasy Action", "rating": 9.0},
    ("The Empire Strikes Back", "1980"): {"genres": "Action Adventure Fantasy Sci-Fi", "rating": 8.7},
    ("The Silence of the Lambs", "1991"): {"genres": "Crime Drama Thriller", "rating": 8.6},
    ("Se7en", "1995"): {"genres": "Crime Drama Thriller", "rating": 8.6},
    ("Gladiator", "2000"): {"genres": "Action Adventure Drama", "rating": 8.5},
    ("The Green Mile", "1999"): {"genres": "Crime Drama Fantasy", "rating": 8.6},
    ("Saving Private Ryan", "1998"): {"genres": "Drama War", "rating": 8.6},
    ("The Departed", "2006"): {"genres": "Crime Drama Thriller", "rating": 8.5},
    ("Whiplash", "2014"): {"genres": "Drama Music", "rating": 8.5},
    ("Parasite", "2019"): {"genres": "Drama Thriller", "rating": 8.5},
    ("Joker", "2019"): {"genres": "Crime Drama Thriller", "rating": 8.4},
    ("Avengers: Endgame", "2019"): {"genres": "Action Adventure Sci-Fi", "rating": 8.4},
    ("Spider-Man", "2002"): {"genres": "Action Adventure Sci-Fi", "rating": 7.4},
    ("Spider-Man 2", "2004"): {"genres": "Action Adventure Sci-Fi", "rating": 7.5},
    ("Spider-Man: Homecoming", "2017"): {"genres": "Action Adventure Sci-Fi", "rating": 7.4},
    ("Spider-Man: Into the Spider-Verse", "2018"): {"genres": "Animation Action Adventure", "rating": 8.4},
    ("Spider-Man: No Way Home", "2021"): {"genres": "Action Adventure Fantasy", "rating": 8.2},
    ("Spider-Man: Across the Spider-Verse", "2023"): {"genres": "Animation Action Adventure", "rating": 8.6},
    ("Mad Max: Fury Road", "2015"): {"genres": "Action Adventure Sci-Fi", "rating": 8.1},
    ("La La Land", "2016"): {"genres": "Comedy Drama Romance Musical", "rating": 8.0},
    ("The Prestige", "2006"): {"genres": "Drama Mystery Sci-Fi", "rating": 8.5},
    ("Django Unchained", "2012"): {"genres": "Drama Western", "rating": 8.5},
    ("The Social Network", "2010"): {"genres": "Drama Biography", "rating": 7.8},
    ("Blade Runner 2049", "2017"): {"genres": "Drama Mystery Sci-Fi", "rating": 8.0},
    ("The Grand Budapest Hotel", "2014"): {"genres": "Comedy Adventure Crime", "rating": 8.1},
    ("Toy Story", "1995"): {"genres": "Animation Adventure Comedy", "rating": 8.3},
    ("Toy Story 3", "2010"): {"genres": "Animation Adventure Comedy", "rating": 8.3},
    ("Finding Nemo", "2003"): {"genres": "Animation Adventure Comedy", "rating": 8.2},
    ("Up", "2009"): {"genres": "Animation Adventure Comedy", "rating": 8.3},
    ("Coco", "2017"): {"genres": "Animation Adventure Family", "rating": 8.4},
    ("Inside Out", "2015"): {"genres": "Animation Adventure Comedy", "rating": 8.1},
    ("Soul", "2020"): {"genres": "Animation Adventure Drama", "rating": 8.0},
    ("Moana", "2016"): {"genres": "Animation Adventure Comedy", "rating": 7.6},
    ("Black Panther", "2018"): {"genres": "Action Adventure Sci-Fi", "rating": 7.3},
    ("Iron Man", "2008"): {"genres": "Action Adventure Sci-Fi", "rating": 7.9},
    ("Captain America: The Winter Soldier", "2014"): {"genres": "Action Adventure Sci-Fi", "rating": 7.8},
    ("Doctor Strange", "2016"): {"genres": "Action Adventure Fantasy", "rating": 7.5},
    ("Guardians of the Galaxy", "2014"): {"genres": "Action Adventure Comedy", "rating": 8.0},
    ("Top Gun: Maverick", "2022"): {"genres": "Action Drama", "rating": 8.2},
    ("Dune", "2021"): {"genres": "Adventure Drama Sci-Fi", "rating": 8.0},
    ("Dune: Part Two", "2024"): {"genres": "Adventure Drama Sci-Fi", "rating": 8.6},
    ("Oppenheimer", "2023"): {"genres": "Drama Thriller", "rating": 8.3},
    ("Barbie", "2023"): {"genres": "Comedy Fantasy Adventure", "rating": 6.8},
    ("The Conjuring", "2013"): {"genres": "Horror Mystery Thriller", "rating": 7.5},
    ("Scream", "1996"): {"genres": "Horror Mystery", "rating": 7.4},
    ("Psycho", "1960"): {"genres": "Horror Mystery Thriller", "rating": 8.5},
    ("The Exorcist", "1973"): {"genres": "Horror", "rating": 8.1},
    ("Hereditary", "2018"): {"genres": "Drama Horror Mystery", "rating": 7.3},
    ("The Sixth Sense", "1999"): {"genres": "Drama Mystery Thriller", "rating": 8.2},
}

POSTER_PALETTES: list[tuple[str, str, str]] = [
    ("#111827", "#1f2937", "#475569"),
    ("#0f172a", "#1e293b", "#64748b"),
    ("#18181b", "#27272a", "#71717a"),
    ("#172033", "#24314a", "#6b7280"),
    ("#1c1917", "#292524", "#78716c"),
]

# ── Mood → genre map ──────────────────────────────────────────────────────────
MOOD_GENRE_MAP: Dict[str, List[str]] = {
    "happy":      ["Comedy", "Animation", "Family", "Musical"],
    "sad":        ["Drama", "Romance"],
    "excited":    ["Action", "Adventure", "Thriller"],
    "scared":     ["Horror", "Thriller", "Mystery"],
    "romantic":   ["Romance", "Drama"],
    "thoughtful": ["Documentary", "Drama", "Sci-Fi"],
    "adventurous":["Adventure", "Action", "Fantasy"],
    "relaxed":    ["Comedy", "Family", "Animation"],
}

TITLE_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "movie",
    "of",
    "on",
    "or",
    "part",
    "the",
    "to",
    "vs",
    "with",
}


def _curated_seed_metadata(clean_title: str, year: str) -> dict[str, Any]:
    return dict(CURATED_HOME_METADATA.get((clean_title, year), {}))


def _normalize_lookup_title(clean_title: str) -> str:
    normalized_title = clean_title.strip()
    trailing_article = re.match(r"^(?P<base>.+),\s*(?P<article>The|A|An)$", normalized_title, re.I)
    if trailing_article:
        return (
            f"{trailing_article.group('article')} {trailing_article.group('base')}"
        ).strip()
    return normalized_title


def _meaningful_title_tokens(value: str) -> set[str]:
    normalized = _normalize_search_text(value)
    tokens = {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in TITLE_TOKEN_STOPWORDS
    }
    if tokens:
        return tokens
    return {token for token in normalized.split() if len(token) > 2}


def _is_missing_poster(value: str) -> bool:
    poster = (value or "").strip().lower()
    if not poster:
        return True
    # Catch all known bad/placeholder values
    bad_patterns = (
        "placehold.co",
        "via.placeholder.com",
        "placeholder",
        "n/a",
        "no+poster",
        "no_poster",
        "noposter",
        "default_poster",
        "image_not_found",
        "notfound",
        "missing",
    )
    return any(p in poster for p in bad_patterns)


def _is_generated_poster(value: str) -> bool:
    poster = (value or "").strip().lower()
    return poster.startswith("data:image/svg+xml")


def _wrap_poster_lines(text: str, max_chars: int = 16, max_lines: int = 3) -> list[str]:
    words = text.split()
    if not words:
        return ["MovieBuzz"]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines - 1:
            break

    if len(lines) < max_lines:
        lines.append(current)

    remaining_words = words[len(" ".join(lines).split()):]
    if remaining_words and lines:
        suffix = " ..."
        trimmed = lines[-1][: max(1, max_chars - len(suffix))]
        lines[-1] = f"{trimmed}{suffix}"

    return lines[:max_lines]


def _generated_poster_url(clean_title: str, year: str, genres: str = "") -> str:
    seed = f"{clean_title}|{year}|{genres}"
    dark, primary, accent = POSTER_PALETTES[
        sum(ord(ch) for ch in seed) % len(POSTER_PALETTES)
    ]
    title_lines = _wrap_poster_lines(clean_title, max_chars=18, max_lines=3)
    genre_line = genres.replace("|", " ").strip() or "MovieBuzz Selection"
    genre_line = genre_line[:32]

    title_svg = []
    for index, line in enumerate(title_lines):
        y = 180 + index * 34
        title_svg.append(
            f"<text x='28' y='{y}' fill='white' font-size='28' font-weight='800' "
            f"font-family='Segoe UI, Arial, sans-serif'>{escape(line)}</text>"
        )

    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 450'>
  <defs>
    <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='{dark}' />
      <stop offset='65%' stop-color='{primary}' />
      <stop offset='100%' stop-color='{accent}' />
    </linearGradient>
    <radialGradient id='glow' cx='50%' cy='18%' r='72%'>
      <stop offset='0%' stop-color='rgba(255,255,255,0.18)' />
      <stop offset='100%' stop-color='rgba(255,255,255,0)' />
    </radialGradient>
  </defs>
  <rect width='300' height='450' rx='28' fill='url(#bg)' />
  <rect width='300' height='450' rx='28' fill='url(#glow)' />
  <rect x='18' y='18' width='264' height='414' rx='22'
        fill='rgba(255,255,255,0.06)' stroke='rgba(255,255,255,0.18)' />
  <rect x='28' y='118' width='244' height='5' rx='2.5' fill='rgba(248,113,113,0.78)' />
  <text x='28' y='52' fill='rgba(255,255,255,0.88)' font-size='13' font-weight='700'
        font-family='Segoe UI, Arial, sans-serif'>MOVIEBUZZ</text>
  <text x='28' y='92' fill='rgba(255,255,255,0.72)' font-size='12' font-weight='700'
        font-family='Segoe UI, Arial, sans-serif'>{escape(genre_line)}</text>
  {''.join(title_svg)}
  <rect x='28' y='352' width='244' height='1' fill='rgba(255,255,255,0.24)' />
  <text x='28' y='388' fill='rgba(255,255,255,0.9)' font-size='18' font-weight='700'
        font-family='Segoe UI, Arial, sans-serif'>{escape(year or 'Movie')}</text>
  <text x='28' y='416' fill='rgba(255,255,255,0.74)' font-size='12'
        font-family='Segoe UI, Arial, sans-serif'>Trailer, wishlist, and details ready</text>
</svg>
""".strip()
    return f"data:image/svg+xml;charset=UTF-8,{quote(svg)}"


def _fallback_movie_description(clean_title: str, year: str, genres: str = "") -> str:
    genre_label = genres.replace("|", " ").strip()
    genre_text = genre_label.lower() if genre_label else "movie"
    year_text = f" from {year}" if year else ""
    return (
        f"{clean_title} is a {genre_text} title{year_text}. "
        "Open the trailer to preview it and save it to your MovieBuzz wishlist."
    )


def _normalize_search_text(value: str) -> str:
    normalized = value.strip().lower().replace("&", " and ").replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _compact_search_text(value: str) -> str:
    return _normalize_search_text(value).replace(" ", "")


def _normalized_preference_phrases(values: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = _normalize_search_text(str(value or ""))
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        normalized.append(phrase)
    return normalized


def _age_affinity_genres(age: int | None) -> list[str]:
    if age is None:
        return []
    if age <= 12:
        return ["Animation", "Fantasy", "Comedy"]
    if age <= 17:
        return ["Animation", "Fantasy", "Action", "Sci-Fi"]
    if age <= 24:
        return ["Action", "Comedy", "Sci-Fi", "Thriller", "Romance"]
    if age <= 34:
        return ["Drama", "Thriller", "Sci-Fi", "Crime", "Romance"]
    if age <= 49:
        return ["Drama", "Crime", "Thriller", "Romance"]
    return ["Drama", "Crime", "Romance", "Fantasy"]


def _load_user_preference_context(user_email: Optional[str]) -> dict[str, Any] | None:
    normalized_email = str(user_email or "").strip().lower()
    if not normalized_email:
        return None

    try:
        preferences = get_user_preferences(normalized_email)
    except Exception as exc:
        log.debug("Preference lookup failed for %s: %s", normalized_email, exc)
        return None

    preferred_genres = [
        genre
        for genre in cast(list[str], preferences.get("preferred_genres") or [])
        if _normalize_search_text(genre) != "all"
    ]
    preferred_moods = [
        mood
        for mood in cast(list[str], preferences.get("preferred_moods") or [])
        if mood
    ]
    age = preferences.get("age")
    age_value = int(age) if isinstance(age, int) else None

    mood_genres: list[str] = []
    for mood in preferred_moods:
        mood_genres.extend(MOOD_GENRE_MAP.get(str(mood).lower(), []))

    context = {
        "user_email": normalized_email,
        "age": age_value,
        "preferred_genres_display": preferred_genres,
        "preferred_moods_display": preferred_moods,
        "preferred_genres": _normalized_preference_phrases(preferred_genres),
        "mood_genres": _normalized_preference_phrases(mood_genres),
        "age_genres": _normalized_preference_phrases(_age_affinity_genres(age_value)),
    }

    if any(context[key] for key in ("preferred_genres", "mood_genres", "age_genres")):
        return context
    return None


def _phrase_match_count(text: str, phrases: list[str]) -> int:
    return sum(1 for phrase in phrases if phrase and phrase in text)


def _preference_match_details(
    genres: str,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_genres = _normalize_search_text(str(genres or ""))
    if not context or not normalized_genres:
        return {
            "normalized_genres": normalized_genres,
            "explicit_hits": 0,
            "mood_hits": 0,
            "age_hits": 0,
            "matched_preferred_genres": [],
        }

    preferred_genres = cast(list[str], context.get("preferred_genres") or [])
    mood_genres = cast(list[str], context.get("mood_genres") or [])
    age_genres = cast(list[str], context.get("age_genres") or [])
    preferred_display = cast(list[str], context.get("preferred_genres_display") or [])

    matched_preferred_genres = [
        genre
        for genre in preferred_display
        if _normalize_search_text(genre) in normalized_genres
    ]
    explicit_hits = min(_phrase_match_count(normalized_genres, preferred_genres), 3)
    mood_hits = min(_phrase_match_count(normalized_genres, mood_genres), 2)
    age_hits = min(_phrase_match_count(normalized_genres, age_genres), 2)

    return {
        "normalized_genres": normalized_genres,
        "explicit_hits": explicit_hits,
        "mood_hits": mood_hits,
        "age_hits": age_hits,
        "matched_preferred_genres": matched_preferred_genres[:3],
    }


def _preference_boost_for_genres(
    genres: str,
    context: dict[str, Any] | None,
) -> float:
    details = _preference_match_details(genres, context)
    explicit_hits = int(details["explicit_hits"])
    mood_hits = int(details["mood_hits"])
    age_hits = int(details["age_hits"])
    synergy_bonus = 0.12 if explicit_hits and mood_hits else 0.0
    depth_bonus = 0.08 if explicit_hits >= 2 else 0.0
    return min(explicit_hits * 0.18 + mood_hits * 0.09 + age_hits * 0.05 + synergy_bonus + depth_bonus, 0.83)


def _preference_multiplier_for_genres(
    genres: str,
    context: dict[str, Any] | None,
) -> float:
    details = _preference_match_details(genres, context)
    explicit_hits = int(details["explicit_hits"])
    mood_hits = int(details["mood_hits"])
    age_hits = int(details["age_hits"])
    if explicit_hits <= 0 and mood_hits <= 0 and age_hits <= 0:
        return 1.0
    depth_bonus = 0.15 if explicit_hits >= 2 else 0.0
    return min(1.0 + explicit_hits * 0.36 + mood_hits * 0.14 + age_hits * 0.08 + depth_bonus, 2.25)


def _recommendation_reason_for_genres(
    genres: str,
    context: dict[str, Any] | None,
) -> str:
    details = _preference_match_details(genres, context)
    matched_preferred = cast(list[str], details.get("matched_preferred_genres") or [])
    preferred_moods = cast(list[str], context.get("preferred_moods_display") or []) if context else []
    mood_hits = int(details.get("mood_hits") or 0)
    age_hits = int(details.get("age_hits") or 0)

    if matched_preferred:
        if len(matched_preferred) == 1:
            return f"Because you like {matched_preferred[0]}"
        if len(matched_preferred) == 2:
            return f"Because you like {matched_preferred[0]} and {matched_preferred[1]}"
        return f"Because it matches your {matched_preferred[0]}, {matched_preferred[1]}, and {matched_preferred[2]} taste"
    if mood_hits and preferred_moods:
        return f"Fits your {preferred_moods[0]} mood"
    if age_hits:
        return "Matches your viewing profile"
    return "Trending for you"


def _apply_preference_ranking(
    movies: list[dict[str, Any]],
    user_email: Optional[str],
    *,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    context = _load_user_preference_context(user_email)
    if not movies:
        return []
    if context is None:
        return [dict(movie) for movie in (movies[:limit] if limit is not None else movies)]

    ranked: list[tuple[float, dict[str, Any]]] = []
    total = len(movies)
    for index, movie in enumerate(movies):
        item = dict(movie)
        movie_genres = str(item.get("genres") or "")
        preference_boost = _preference_boost_for_genres(movie_genres, context)
        preference_multiplier = _preference_multiplier_for_genres(movie_genres, context)
        try:
            rating_value = float(item.get("rating") or item.get("imdb_rating") or 0)
        except Exception:
            rating_value = 0.0
        try:
            trending_value = float(item.get("trending_score") or 0)
        except Exception:
            trending_value = 0.0

        base_rank_score = float(total - index)
        ranking_score = (
            base_rank_score * preference_multiplier
            + preference_boost * 28.0
            + rating_value * 0.10
            + trending_value * 1.8
        )
        item["preference_boost"] = round(preference_boost, 4)
        item["reason"] = str(item.get("reason") or _recommendation_reason_for_genres(movie_genres, context))
        ranked.append((ranking_score, item))

    ranked.sort(key=lambda entry: entry[0], reverse=True)
    results = [item for _, item in ranked]
    if limit is not None:
        return results[:limit]
    return results


def _sequence_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def _fetch_itunes_poster(clean_title: str, year: str) -> str:
    """Fetch movie poster from iTunes Search API.
    Tries title+year first, then title-only if no good match found.
    Returns high-res artwork URL or empty string.
    """
    normalized_title = _normalize_search_text(clean_title)

    def _query_itunes(term: str) -> list:
        try:
            r = requests.get(
                ITUNES_SEARCH_URL,
                params={"term": term, "media": "movie", "entity": "movie", "limit": 8},
                timeout=ITUNES_TIMEOUT,
            )
            payload = r.json()
            results = payload.get("results")
            return results if isinstance(results, list) else []
        except Exception:
            return []

    def _best_from_results(results: list) -> tuple:
        best_score = 0.0
        best_artwork = ""
        for result in results:
            if not isinstance(result, dict):
                continue
            candidate_title = str(result.get("trackName") or "").strip()
            if not candidate_title:
                continue
            candidate_year = str(result.get("releaseDate") or "")[:4]
            candidate_norm = _normalize_search_text(candidate_title)
            score = _sequence_similarity(normalized_title, candidate_norm)
            if year and candidate_year == year:
                score += 0.25
            if normalized_title and normalized_title in candidate_norm:
                score += 0.18
            artwork = str(
                result.get("artworkUrl600")
                or result.get("artworkUrl100")
                or result.get("artworkUrl60")
                or ""
            ).strip()
            if not artwork or score <= best_score:
                continue
            best_score = score
            best_artwork = re.sub(r"\d+x\d+bb", "600x900bb", artwork)
        return best_score, best_artwork

    # First attempt: title + year
    results = _query_itunes(f"{clean_title} {year}".strip())
    best_score, best_artwork = _best_from_results(results)
    if best_score >= 0.35 and best_artwork:
        return best_artwork

    # Retry: title only (handles older films & year mismatches)
    if year:
        results = _query_itunes(clean_title)
        best_score, best_artwork = _best_from_results(results)
        if best_score >= 0.40 and best_artwork:
            return best_artwork

    return ""
def _fetch_tmdb_poster(clean_title: str, year: str) -> str:
    """Fetch poster from TMDB (requires TMDB_API_KEY env var).
    Returns image URL or empty string.
    """
    if not TMDB_API_KEY:
        return ""
    try:
        params: dict = {"api_key": TMDB_API_KEY, "query": clean_title, "language": "en-US"}
        if year:
            params["year"] = year
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=TMDB_TIMEOUT)
        data = r.json()
        results = data.get("results") or []
        normalized = _normalize_search_text(clean_title)
        best_score = 0.0
        best_path = ""
        for item in results[:5]:
            title_raw = str(item.get("title") or item.get("original_title") or "")
            candidate_norm = _normalize_search_text(title_raw)
            score = _sequence_similarity(normalized, candidate_norm)
            release = str(item.get("release_date") or "")[:4]
            if year and release == year:
                score += 0.25
            path = str(item.get("poster_path") or "").strip()
            if path and score > best_score:
                best_score = score
                best_path = path
        if best_score >= 0.35 and best_path:
            return f"{TMDB_IMAGE_BASE}{best_path}"
    except Exception:
        pass
    return ""




def sigmoid(x: np.ndarray) -> np.ndarray:
    """σ(x) = 1 / (1 + e^-x)  — squash scores to [0, 1]"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU(x) = max(0, x)"""
    return np.maximum(0, x)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """LeakyReLU(x) = x if x>0 else α·x"""
    return np.where(x > 0, x, alpha * x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax for converting raw scores to probability distribution."""
    e = np.exp(x - x.max())
    return e / e.sum()

def bce_loss(y_true: np.ndarray, y_pred: np.ndarray,
             eps: float = 1e-7) -> float:
    """Binary Cross-Entropy: -[y·log(p) + (1-y)·log(1-p)]"""
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

def bpr_loss(pos_score: np.ndarray, neg_score: np.ndarray) -> float:
    """
    Bayesian Personalised Ranking loss.
    BPR = -Σ log σ(pos - neg)
    Encourages pos items to rank above neg items.
    """
    return float(-np.mean(np.log(sigmoid(pos_score - neg_score) + 1e-7)))

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error — used for rating prediction."""
    return float(np.mean((y_true - y_pred) ** 2))

def ndcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain @k.
    Measures ranking quality — higher = better ordering.
    """
    r = relevance[:k]
    dcg  = np.sum(r / np.log2(np.arange(2, len(r) + 2)))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  NCF  (Neural Collaborative Filtering)
#  Architecture: GMF branch + deep MLP branch → NeuMF concat → sigmoid output
# ═══════════════════════════════════════════════════════════════════════════════

def build_ncf(
    n_users: int,
    n_items: int,
    mf_dim: int = 96,
    mlp_layers: tuple[int, ...] | list[int] = (256, 128, 64, 32),
    dropout: float = 0.35,
    learning_rate: float = 1e-3,
    embed_dim: int | None = None,
) -> Any:
    """
    NeuMF = GMF (element-wise product) + MLP (deep layers)
    Activations : ReLU (hidden layers), Sigmoid (output)
    Loss        : Binary Cross-Entropy
    Optimiser   : Adam
    """
    _load_tensorflow_runtime()
    if not HAS_TF:
        raise RuntimeError("TensorFlow not installed")
    assert keras is not None
    if embed_dim is not None:
        mf_dim = int(embed_dim)
    mlp_units = [int(units) for units in mlp_layers] or [256, 128, 64, 32]
    mlp_embed_dim = max(16, mlp_units[0] // 2)

    # ── Inputs ────────────────────────────────────────────────────────────────
    user_in = keras.Input(shape=(1,), name="user_id")
    item_in = keras.Input(shape=(1,), name="item_id")

    # ── GMF branch ────────────────────────────────────────────────────────────
    gmf_u = keras.layers.Embedding(
        n_users,
        mf_dim,
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="gmf_user_emb",
    )(user_in)
    gmf_i = keras.layers.Embedding(
        n_items,
        mf_dim,
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="gmf_item_emb",
    )(item_in)
    gmf_out = keras.layers.Multiply(name="gmf_product")([
        keras.layers.Flatten()(gmf_u),
        keras.layers.Flatten()(gmf_i),
    ])

    # ── MLP branch ────────────────────────────────────────────────────────────
    mlp_u = keras.layers.Embedding(
        n_users,
        mlp_embed_dim,
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="mlp_user_emb",
    )(user_in)
    mlp_i = keras.layers.Embedding(
        n_items,
        mlp_embed_dim,
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="mlp_item_emb",
    )(item_in)
    mlp_vec = keras.layers.Concatenate(name="mlp_concat")([
        keras.layers.Flatten()(mlp_u),
        keras.layers.Flatten()(mlp_i),
    ])
    for layer_index, units in enumerate(mlp_units, start=1):
        mlp_vec = keras.layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-6),
            name=f"mlp_fc{layer_index}",
        )(mlp_vec)
        mlp_vec = keras.layers.BatchNormalization(name=f"mlp_bn{layer_index}")(mlp_vec)
        mlp_vec = keras.layers.Dropout(dropout, name=f"mlp_dropout{layer_index}")(mlp_vec)

    # ── NeuMF concat ──────────────────────────────────────────────────────────
    neumf = keras.layers.Concatenate(name="neumf_concat")([gmf_out, mlp_vec])
    neumf = keras.layers.Dropout(dropout, name="neumf_dropout")(neumf)

    # ── Output: Sigmoid → probability of interaction ──────────────────────────
    output = keras.layers.Dense(1, activation="sigmoid", name="output")(neumf)

    model = keras.Model(inputs=[user_in, item_in], outputs=output,
                        name="NeuMF")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _save_ncf_artifacts(
    model: Any,
    user_enc: dict[int, int],
    item_enc: dict[int, int],
) -> None:
    model.save_weights(str(NCF_WEIGHTS_PATH))
    with open(NCF_META_PATH, "wb") as f:
        pickle.dump(
            {
                "n_users": len(user_enc),
                "n_items": len(item_enc),
                "mf_dim": 96,
                "mlp_layers": [256, 128, 64, 32],
                "dropout": 0.35,
                "learning_rate": 1e-3,
            },
            f,
        )
    with open(NCF_ENC_PATH, "wb") as f:
        pickle.dump({"user": user_enc, "item": item_enc}, f)


def _load_ncf_artifacts() -> tuple[Any, dict[int, int], dict[int, int]] | None:
    _load_tensorflow_runtime()
    if not HAS_TF or keras is None:
        return None

    if NCF_WEIGHTS_PATH.exists() and NCF_META_PATH.exists() and NCF_ENC_PATH.exists():
        with open(NCF_META_PATH, "rb") as f:
            meta = pickle.load(f)
        with open(NCF_ENC_PATH, "rb") as f:
            enc = pickle.load(f)
        model = build_ncf(
            int(meta.get("n_users") or len(enc.get("user", {}))),
            int(meta.get("n_items") or len(enc.get("item", {}))),
            mf_dim=int(meta.get("mf_dim") or 96),
            mlp_layers=tuple(meta.get("mlp_layers") or [256, 128, 64, 32]),
            dropout=float(meta.get("dropout") or 0.35),
            learning_rate=float(meta.get("learning_rate") or 1e-3),
        )
        model.load_weights(str(NCF_WEIGHTS_PATH))
        return model, enc["user"], enc["item"]

    if NCF_PATH.exists() and NCF_ENC_PATH.exists():
        model = keras.models.load_model(str(NCF_PATH))
        with open(NCF_ENC_PATH, "rb") as f:
            enc = pickle.load(f)
        _save_ncf_artifacts(model, enc["user"], enc["item"])
        return model, enc["user"], enc["item"]

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def get_db():
    return open_db()


def _recommender_schema_sql() -> str:
    return """
        CREATE TABLE IF NOT EXISTS movies (
            movieId         INTEGER PRIMARY KEY,
            title           TEXT NOT NULL,
            genres          TEXT DEFAULT '',
            avg_rating      REAL DEFAULT 0,
            num_ratings     INTEGER DEFAULT 0,
            sentiment_score REAL DEFAULT 0,
            trending_score  REAL DEFAULT 0,
            poster          TEXT DEFAULT '',
            source          TEXT DEFAULT 'ml25m'
        );
        CREATE TABLE IF NOT EXISTS tags (
            movieId INTEGER,
            tag TEXT,
            FOREIGN KEY(movieId) REFERENCES movies(movieId)
        );
        CREATE TABLE IF NOT EXISTS genome_scores (
            movieId INTEGER,
            tagId INTEGER,
            relevance REAL,
            PRIMARY KEY (movieId, tagId)
        );
        CREATE TABLE IF NOT EXISTS omdb_cache (
            title TEXT PRIMARY KEY,
            poster TEXT DEFAULT '',
            plot TEXT DEFAULT '',
            "cast" TEXT DEFAULT '',
            director TEXT DEFAULT '',
            imdb_rating TEXT DEFAULT '',
            runtime TEXT DEFAULT '',
            fetched_at TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS user_feedback (
            user_id TEXT NOT NULL,
            movieId INTEGER NOT NULL,
            feedback TEXT CHECK(feedback IN ('like','dislike','neutral')),
            ts TEXT DEFAULT '',
            PRIMARY KEY (user_id, movieId)
        );
        CREATE TABLE IF NOT EXISTS rating_timestamps (
            movieId INTEGER PRIMARY KEY,
            latest_ts INTEGER DEFAULT 0,
            num_recent INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS model_metrics (
            run_id TEXT PRIMARY KEY,
            model TEXT,
            bce_loss REAL,
            bpr_loss REAL,
            mse REAL,
            ndcg_10 REAL,
            auc REAL,
            ts TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title);
        CREATE INDEX IF NOT EXISTS idx_movies_title_lower ON movies(lower(title));
        CREATE INDEX IF NOT EXISTS idx_movies_genres_lower ON movies(lower(genres));
        CREATE INDEX IF NOT EXISTS idx_movies_home_sort
            ON movies(num_ratings DESC, trending_score DESC, avg_rating DESC, title ASC);
        CREATE INDEX IF NOT EXISTS idx_tags_mid ON tags(movieId);
        CREATE INDEX IF NOT EXISTS idx_genome_mid ON genome_scores(movieId);
    """


def init_db():
    with get_db() as conn:
        conn.executescript(_recommender_schema_sql())
    log.info("DB ready: %s", DB_PATH)
    # Migrate: clear old hardcoded placeholder poster strings so real ones get fetched
    try:
        with get_db() as conn:
            conn.execute("""
                UPDATE movies SET poster = ''
                WHERE poster LIKE '%placehold.co%'
                   OR poster LIKE '%via.placeholder.com%'
                   OR poster = 'https://placehold.co/300x450?text=No+Poster'
            """)
            # Clear omdb_cache entries that have empty/bad posters
            # so the new OMDB key re-fetches them fresh
            conn.execute("""
                DELETE FROM omdb_cache
                WHERE poster = ''
                   OR poster IS NULL
                   OR poster LIKE '%placehold.co%'
                   OR poster LIKE '%placeholder%'
            """)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  ML25M CHUNKED LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_ml25m_to_db():
    with get_db() as conn:
        count_result = conn.execute("SELECT COUNT(*) FROM movies").fetchone()
        if (count_result and count_result[0] > 0):
            log.info("DB already populated – skipping ML25M import")
            return

    if is_postgres():
        raise RuntimeError(
            "Postgres database is empty. Import or migrate the MovieBuzz data "
            "into Supabase before starting the API."
        )

    movies_path        = DATA_DIR / "movies.csv"
    ratings_path       = DATA_DIR / "ratings.csv"
    tags_path          = DATA_DIR / "tags.csv"
    genome_scores_path = DATA_DIR / "genome-scores.csv"

    if not movies_path.exists():
        log.warning("movies.csv not found – DB empty")
        return

    log.info("Loading movies.csv …")
    movies_df = pd.read_csv(movies_path)
    movies_df["genres"] = movies_df["genres"].str.replace("|", " ", regex=False)
    # Don't set a placeholder — leave poster empty; _movie_payload will fetch real ones
    movies_df["poster"] = ""
    movies_df["source"] = "ml25m"

    # ── aggregate ratings in chunks ───────────────────────────────────────────
    log.info("Aggregating 25M ratings in chunks …")
    sum_d, cnt_d, latest_ts, recent_cnt = {}, {}, {}, {}
    cutoff = int(datetime(2019, 1, 1).timestamp())

    for chunk in pd.read_csv(ratings_path, chunksize=RATING_CHUNK,
                              usecols=["userId", "movieId", "rating", "timestamp"]):
        for mid, grp in chunk.groupby("movieId"):
            sum_d[mid]      = sum_d.get(mid, 0) + grp["rating"].sum()
            cnt_d[mid]      = cnt_d.get(mid, 0) + len(grp)
            latest_ts[mid]  = max(latest_ts.get(mid, 0), int(grp["timestamp"].max()))
            recent_cnt[mid] = recent_cnt.get(mid, 0) + int((grp["timestamp"] >= cutoff).sum())

    avg_df = pd.DataFrame({
        "movieId":     list(sum_d.keys()),
        "avg_rating":  [sum_d[k] / cnt_d[k] for k in sum_d],
        "num_ratings": list(cnt_d.values()),
    })
    movies_df = movies_df.merge(avg_df, on="movieId", how="left")
    movies_df["avg_rating"]  = movies_df["avg_rating"].fillna(0)
    movies_df["num_ratings"] = movies_df["num_ratings"].fillna(0).astype(int)

    # trending score (MinMax scaled recent rating count)
    ts_df = pd.DataFrame({
        "movieId":    list(latest_ts.keys()),
        "latest_ts":  list(latest_ts.values()),
        "num_recent": [recent_cnt.get(k, 0) for k in latest_ts],
    })
    ts_df["trending_score"] = MinMaxScaler().fit_transform(ts_df[["num_recent"]])
    movies_df = movies_df.merge(ts_df[["movieId", "trending_score"]], on="movieId", how="left")
    movies_df["trending_score"] = movies_df["trending_score"].fillna(0)

    with get_db() as conn:
        ts_df.to_sql("rating_timestamps", conn, if_exists="replace",
                     index=False, chunksize=5_000)

    # ── tags + sentiment ──────────────────────────────────────────────────────
    movies_df["sentiment_score"] = 0.0
    if tags_path.exists():
        log.info("Loading tags …")
        sentiment_sum: Dict[int, float] = {}
        sentiment_count: Dict[int, int] = {}
        first_chunk = True

        with get_db() as conn:
            for chunk in pd.read_csv(
                tags_path,
                chunksize=RATING_CHUNK,
                usecols=["movieId", "tag"],
            ):
                chunk["tag"] = chunk["tag"].fillna("").astype(str)
                chunk.to_sql(
                    "tags",
                    conn,
                    if_exists="replace" if first_chunk else "append",
                    index=False,
                    chunksize=10_000,
                )
                first_chunk = False

                if HAS_TEXTBLOB:
                    assert TextBlob is not None
                    grouped = chunk.groupby("movieId")["tag"].agg(list)
                    for mid, tags in grouped.items():
                        text = " ".join(tags)
                        weight = len(tags)
                        polarity = TextBlob(text).sentiment.polarity
                        sentiment_sum[mid] = sentiment_sum.get(int(mid), 0.0) + polarity * weight
                        sentiment_count[mid] = sentiment_count.get(int(mid), 0) + weight

        if HAS_TEXTBLOB and sentiment_sum:
            sentiment_df = pd.DataFrame(
                {
                    "movieId": list(sentiment_sum.keys()),
                    "sentiment_score": [
                        sentiment_sum[mid] / max(sentiment_count[mid], 1)
                        for mid in sentiment_sum
                    ],
                }
            )
            movies_df = movies_df.merge(
                sentiment_df,
                on="movieId",
                how="left",
                suffixes=("", "_chunked"),
            )
            if "sentiment_score_chunked" in movies_df.columns:
                movies_df["sentiment_score"] = (
                    movies_df["sentiment_score_chunked"]
                    .fillna(movies_df["sentiment_score"])
                    .fillna(0)
                )
                movies_df = movies_df.drop(columns=["sentiment_score_chunked"])

    # ── genome scores ─────────────────────────────────────────────────────────
    if genome_scores_path.exists():
        log.info("Loading genome scores …")
        for chunk in pd.read_csv(genome_scores_path, chunksize=RATING_CHUNK):
            with get_db() as conn:
                chunk.to_sql("genome_scores", conn, if_exists="append",
                             index=False, chunksize=10_000)

    # ── write movies ──────────────────────────────────────────────────────────
    cols = ["movieId", "title", "genres", "avg_rating", "num_ratings",
            "sentiment_score", "trending_score", "poster", "source"]
    movies_df[[c for c in cols if c in movies_df.columns]].to_sql(
        "movies", get_db(), if_exists="replace", index=False, chunksize=5_000
    )
    log.info("ML25M import done – %d movies", len(movies_df))


def _sample_ratings_csv(
    csv_path: Path,
    sample_rows: int,
    usecols: List[str],
) -> pd.DataFrame:
    dtype_map = {
        "userId": np.int32,
        "movieId": np.int32,
        "rating": np.float32,
    }
    total_rows = 0
    for chunk in pd.read_csv(
        csv_path,
        chunksize=RATING_CHUNK,
        usecols=usecols,
        dtype={key: value for key, value in dtype_map.items() if key in usecols},
    ):
        total_rows += len(chunk)

    if total_rows <= sample_rows:
        return pd.read_csv(
            csv_path,
            usecols=usecols,
            dtype={key: value for key, value in dtype_map.items() if key in usecols},
        )

    ratio = sample_rows / total_rows
    sampled_chunks = []
    for index, chunk in enumerate(
        pd.read_csv(
            csv_path,
            chunksize=RATING_CHUNK,
            usecols=usecols,
            dtype={key: value for key, value in dtype_map.items() if key in usecols},
        )
    ):
        take = max(1, int(round(len(chunk) * ratio)))
        take = min(take, len(chunk))
        sampled_chunks.append(chunk.sample(n=take, random_state=42 + index))

    sampled = pd.concat(sampled_chunks, ignore_index=True)
    if len(sampled) > sample_rows:
        sampled = sampled.sample(n=sample_rows, random_state=42)
    return sampled.reset_index(drop=True)


def _omdb_cache_get(cache_key: str) -> Optional[dict[str, str]]:
    entry = _OMDB_MEMORY_CACHE.get(cache_key)
    if entry is None:
        return None
    expires_at, payload = entry
    if expires_at <= _utc_now().timestamp():
        _OMDB_MEMORY_CACHE.pop(cache_key, None)
        return None
    return dict(payload)


def _omdb_cache_set(cache_key: str, payload: dict[str, str]):
    if len(_OMDB_MEMORY_CACHE) >= OMDB_MEMORY_CACHE_MAX_ITEMS:
        expired = [
            key
            for key, (expires_at, _) in _OMDB_MEMORY_CACHE.items()
            if expires_at <= _utc_now().timestamp()
        ]
        for key in expired:
            _OMDB_MEMORY_CACHE.pop(key, None)
        if len(_OMDB_MEMORY_CACHE) >= OMDB_MEMORY_CACHE_MAX_ITEMS:
            oldest_key = min(
                _OMDB_MEMORY_CACHE.items(),
                key=lambda item: item[1][0],
            )[0]
            _OMDB_MEMORY_CACHE.pop(oldest_key, None)

    _OMDB_MEMORY_CACHE[cache_key] = (
        _utc_now().timestamp() + OMDB_MEMORY_CACHE_TTL_SECONDS,
        dict(payload),
    )


def _read_omdb_cache_entry(cache_key: str) -> Optional[dict[str, str]]:
    cached = _omdb_cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT poster, plot, cast, director, imdb_rating, runtime
                FROM omdb_cache
                WHERE title = ?
                """,
                (cache_key,),
            ).fetchone()
    except Exception as exc:
        log.debug("OMDb cache read error for %s: %s", cache_key, exc)
        return None

    if row is None:
        return None

    payload = {
        field: str(row[field] or "")
        for field in ("poster", "plot", "cast", "director", "imdb_rating", "runtime")
    }
    _omdb_cache_set(cache_key, payload)
    return dict(payload)


def _latest_model_metrics_from_db() -> dict[str, Any]:
    try:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT run_id, model, bce_loss, bpr_loss, mse, ndcg_10, auc, ts
                FROM model_metrics
                ORDER BY ts DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
    except Exception as exc:
        log.debug("Metrics DB read error: %s", exc)
        return {}

    if row is None:
        return {}

    metrics: dict[str, Any] = {
        "run_id": str(row["run_id"] or ""),
        "model": str(row["model"] or ""),
        "updated_at": str(row["ts"] or ""),
    }

    numeric_fields = {
        "ncf_bce": row["bce_loss"],
        "ncf_bpr": row["bpr_loss"],
        "svd_mse": row["mse"],
        "ndcg_10": row["ndcg_10"],
        "ncf_auc": row["auc"],
    }
    for field, value in numeric_fields.items():
        if value is None:
            continue
        try:
            metrics[field] = round(float(value), 4)
        except Exception:
            continue

    return metrics


def _read_eval_report() -> dict[str, Any]:
    if not EVAL_REPORT_PATH.exists():
        return {}

    try:
        raw_payload = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.debug("Evaluation report read error: %s", exc)
        return {}

    return raw_payload if isinstance(raw_payload, dict) else {}


def _metric_float(value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except Exception:
        return None

    if not np.isfinite(numeric_value):
        return None

    return round(numeric_value, 4)


def _extract_genre_tokens(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    tokens = [
        token.strip()
        for token in raw.replace("|", " ").split()
        if token.strip()
    ]
    return [
        token
        for token in tokens
        if token.lower() not in {"no", "genres", "listed"}
    ]


def _genre_feature_name(token: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in token.lower()).strip("_")
    return f"genre_{normalized}"


def _available_model_artifacts() -> dict[str, list[str]]:
    model_files = {
        "SVD": SVD_PATH.exists(),
        "ALS": ALS_PATH.exists(),
        "NCF": NCF_PATH.exists() and NCF_ENC_PATH.exists(),
        "SBERT": SBERT_CACHE.exists() and SBERT_IDX.exists(),
        "XGB": XGB_PATH.exists(),
    }
    return {
        "available_models": [name for name, present in model_files.items() if present],
        "missing_models": [name for name, present in model_files.items() if not present],
    }


def _metrics_section(metrics_block: dict[str, Any], key: str) -> dict[str, Any]:
    section = metrics_block.get(key)
    return cast(dict[str, Any], section) if isinstance(section, dict) else {}


def _metrics_from_eval_report() -> dict[str, Any]:
    report = _read_eval_report()
    if not report:
        return {}

    metrics_block = report.get("metrics")
    if not isinstance(metrics_block, dict):
        metrics_block = {}

    ncf_metrics = _metrics_section(metrics_block, "NCF")
    svd_metrics = _metrics_section(metrics_block, "SVD")
    xgb_metrics = _metrics_section(metrics_block, "XGB")
    artifact_status = _available_model_artifacts()

    comparison: list[dict[str, Any]] = []
    for label, values, loss_key, loss_label in (
        ("NCF", ncf_metrics, "BCE", "BCE"),
        ("XGB", xgb_metrics, "LogLoss", "LogLoss"),
        ("SVD", svd_metrics, "RMSE", "RMSE"),
    ):
        if not values:
            continue

        comparison.append({
            "model": label,
            "auc": _metric_float(values.get("AUC")),
            "f1": _metric_float(values.get("F1")),
            "precision": _metric_float(values.get("Precision")),
            "recall": _metric_float(values.get("Recall")),
            "loss": _metric_float(values.get(loss_key)),
            "loss_label": loss_label,
        })

    payload: dict[str, Any] = {
        "report_generated_at": str(report.get("generated_at") or ""),
        "test_ratio": _metric_float(report.get("test_ratio")),
        "report_metrics": metrics_block,
        "comparison": comparison,
        **artifact_status,
    }

    field_mapping = {
        "ncf_bce": ncf_metrics.get("BCE"),
        "ncf_auc": ncf_metrics.get("AUC"),
        "ncf_f1": ncf_metrics.get("F1"),
        "ncf_precision": ncf_metrics.get("Precision"),
        "ncf_recall": ncf_metrics.get("Recall"),
        "svd_mse": svd_metrics.get("RMSE"),
        "xgb_auc": xgb_metrics.get("AUC"),
        "xgb_f1": xgb_metrics.get("F1"),
        "xgb_logloss": xgb_metrics.get("LogLoss"),
    }

    for field_name, value in field_mapping.items():
        numeric_value = _metric_float(value)
        if numeric_value is not None:
            payload[field_name] = numeric_value

    return payload


# ═══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RecommenderEngine:
    _instance: Optional["RecommenderEngine"] = None

    def __init__(self):
        self.movies_df           = pd.DataFrame()
        self.tfidf_matrix        = None
        self.genome_matrix       = None
        self.sbert_embeddings    = None
        self.movie_sbert_embeddings: Dict[int, np.ndarray] = {}
        self.user_taste_vectors   = None
        self.user_taste_id_map: Dict[int, int] = {}
        self.svd_model           = None
        self.als_model           = None
        self.ncf_model           = None
        self.xgb_ranker          = None
        self.xgb_feature_context: Dict[str, Any] = {
            "feature_columns": list(LEGACY_XGB_FEATURE_COLUMNS),
            "user_stats": {},
            "movie_stats": {},
        }
        self.ncf_user_enc: Dict  = {}
        self.ncf_item_enc: Dict  = {}
        self.title_index         = pd.Series(dtype=int)
        self.normalized_title_index = pd.Series(dtype=int)
        self._ready              = False
        self._metrics: Dict      = {}   # store latest loss/metric values

    @classmethod
    def get(cls) -> "RecommenderEngine":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._build()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    # ── BUILD ─────────────────────────────────────────────────────────────────
    def _build(self):
        log.info("Building engine …")
        with get_db() as conn:
            self.movies_df = read_sql_df(conn, "SELECT * FROM movies")
        if self.movies_df.empty:
            log.warning("No movies in DB")
            return

        self._build_tfidf()
        self._build_genome()
        self._build_sbert()
        self._load_sbert_taste_artifacts()
        self._load_or_train_svd()
        self._load_or_train_als()
        self._load_or_train_ncf()
        self._load_or_train_xgb()
        self._ensure_search_columns()

        self.title_index = pd.Series(
            self.movies_df.index,
            index=self.movies_df["title"].str.lower(),
        )
        self.normalized_title_index = pd.Series(
            self.movies_df.index,
            index=self.movies_df["_search_title"],
        )
        self._ready = True
        log.info("Engine ready – %d movies", len(self.movies_df))

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    def _build_tfidf(self):
        log.info("Building TF-IDF …")
        with get_db() as conn:
            tag_agg_sql = (
                "SELECT movieId, string_agg(tag, ' ') AS tag_text "
                "FROM tags GROUP BY movieId"
                if DB_BACKEND == "postgres"
                else
                "SELECT movieId, GROUP_CONCAT(tag,' ') AS tag_text "
                "FROM tags GROUP BY movieId"
            )
            tags_count_result = conn.execute("SELECT COUNT(*) FROM tags").fetchone()
            if tags_count_result and tags_count_result[0] > 0:
                tags_agg = read_sql_df(conn, tag_agg_sql)
                self.movies_df = self.movies_df.merge(tags_agg, on="movieId", how="left")
            self.movies_df["tag_text"] = self.movies_df.get("tag_text",
                pd.Series("", index=self.movies_df.index)).fillna("")

            plots = read_sql_df(
                conn,
                'SELECT title as ct, plot, "cast", director FROM omdb_cache',
            )

        def _omdb_text(row):
            clean, _ = _clean_title(row["title"])
            m = plots[plots["ct"].str.lower() == clean.lower()]
            if not m.empty:
                r = m.iloc[0]
                return f"{r['plot']} {r['cast']} {r['director']}"
            return ""

        self.movies_df["omdb_text"] = self.movies_df.apply(_omdb_text, axis=1)
        self.movies_df["feature_text"] = (
            self.movies_df["genres"].fillna("") + " "
            + self.movies_df["tag_text"] + " "
            + self.movies_df["omdb_text"]
        ).str.strip()

        tfidf = TfidfVectorizer(stop_words="english", max_features=20_000,
                                ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df["feature_text"])
        log.info("TF-IDF: %s", self.tfidf_matrix.shape)

    # ── Genome ────────────────────────────────────────────────────────────────
    def _build_genome(self):
        with get_db() as conn:
            genome_count_result = conn.execute("SELECT COUNT(*) FROM genome_scores").fetchone()
            if not genome_count_result or genome_count_result[0] == 0:
                return
            genome_df = read_sql_df(conn, "SELECT * FROM genome_scores")
        pivot = genome_df.pivot_table(index="movieId", columns="tagId",
                                      values="relevance", fill_value=0)
        pivot = pivot.reindex(self.movies_df["movieId"].values, fill_value=0)
        self.genome_matrix = pivot.values.astype(np.float32)
        log.info("Genome matrix: %s", self.genome_matrix.shape)

    # ── SBERT ─────────────────────────────────────────────────────────────────
    def _build_sbert(self):
        if not HAS_SBERT:
            return
        if SBERT_CACHE.exists() and SBERT_IDX.exists():
            self.sbert_embeddings = np.load(str(SBERT_CACHE))
            with open(SBERT_IDX, "rb") as f:
                if len(pickle.load(f)) == len(self.movies_df):
                    log.info("SBERT loaded from cache")
                    return
        log.info("Encoding SBERT embeddings …")
        assert SentenceTransformer is not None
        model = SentenceTransformer(SBERT_MODEL)
        texts = (
            self.movies_df["genres"].fillna("") + ". "
            + self.movies_df["tag_text"].fillna("") + ". "
            + self.movies_df["omdb_text"].fillna("")
        ).tolist()
        self.sbert_embeddings = model.encode(
            texts, batch_size=256, show_progress_bar=True,
            normalize_embeddings=True).astype(np.float32)
        np.save(str(SBERT_CACHE), self.sbert_embeddings)
        with open(SBERT_IDX, "wb") as f:
            pickle.dump(self.movies_df["movieId"].tolist(), f)

    def _load_sbert_taste_artifacts(self):
        if not (
            MOVIE_SBERT_EMBEDDINGS_PATH.exists()
            and USER_TASTE_VECTORS_PATH.exists()
            and USER_TASTE_ID_MAP_PATH.exists()
        ):
            return
        try:
            with open(MOVIE_SBERT_EMBEDDINGS_PATH, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                self.movie_sbert_embeddings = {
                    int(movie_id): np.asarray(embedding, dtype=np.float32)
                    for movie_id, embedding in payload.items()
                }
            self.user_taste_vectors = np.asarray(
                np.load(str(USER_TASTE_VECTORS_PATH)),
                dtype=np.float32,
            )
            with open(USER_TASTE_ID_MAP_PATH, "rb") as f:
                loaded_id_map = pickle.load(f)
            if isinstance(loaded_id_map, dict):
                self.user_taste_id_map = {
                    int(user_id): int(row_idx)
                    for user_id, row_idx in loaded_id_map.items()
                }
            if self.movie_sbert_embeddings and self.user_taste_id_map:
                log.info("SBERT taste artefacts loaded")
        except Exception as exc:
            log.warning("Could not load SBERT taste artefacts: %s", exc)
            self.movie_sbert_embeddings = {}
            self.user_taste_vectors = None
            self.user_taste_id_map = {}

    # ── SVD ───────────────────────────────────────────────────────────────────
    def _load_or_train_svd(self):
        if not HAS_SURPRISE:
            return
        assert Reader is not None and Dataset is not None and SVD is not None
        if SVD_PATH.exists():
            with open(SVD_PATH, "rb") as f:
                self.svd_model = pickle.load(f)
            log.info("SVD loaded")
            return
        rp = DATA_DIR / "ratings.csv"
        if not rp.exists():
            return
        log.info("Training SVD on a streamed sample of up to %d ratings …", SVD_TRAIN_ROWS)
        sampled_ratings = cast(pd.DataFrame, _sample_ratings_csv(
            rp,
            SVD_TRAIN_ROWS,
            usecols=["userId", "movieId", "rating"],
        ))
        reader = Reader(rating_scale=(0.5, 5.0))
        dataset = Dataset.load_from_df(
            sampled_ratings[["userId", "movieId", "rating"]],
            reader,
        )
        trainset = dataset.build_full_trainset()
        self.svd_model = SVD(n_factors=100, n_epochs=25, lr_all=0.005,
                             reg_all=0.02, random_state=42)
        self.svd_model.fit(trainset)

        # compute & log MSE on full trainset
        assert self.svd_model is not None
        svd_model = cast(Any, self.svd_model)
        preds = [svd_model.predict(uid, iid).est
                 for uid, iid, r in trainset.all_ratings()]
        actuals = [r for _, _, r in trainset.all_ratings()]
        mse = mse_loss(np.array(actuals), np.array(preds))
        log.info("SVD MSE=%.4f", mse)
        self._metrics["svd_mse"] = mse

        with open(SVD_PATH, "wb") as f:
            pickle.dump(self.svd_model, f)

    # ── ALS ───────────────────────────────────────────────────────────────────
    def _load_or_train_als(self):
        if not HAS_IMPLICIT:
            return
        assert implicit is not None
        if ALS_PATH.exists():
            with open(ALS_PATH, "rb") as f:
                d = pickle.load(f)
                self.als_model     = d["model"]
                self._als_item_ids = d["item_ids"]
                self._als_user_ids = d["user_ids"]
            log.info("ALS loaded")
            return
        rp = DATA_DIR / "ratings.csv"
        if not rp.exists():
            return
        import scipy.sparse as sp
        log.info("Training ALS on a streamed sample of up to %d ratings …", ALS_TRAIN_ROWS)
        df = cast(pd.DataFrame, _sample_ratings_csv(
            rp,
            ALS_TRAIN_ROWS,
            usecols=["userId", "movieId", "rating"],
        ))
        users_ = pd.Categorical(df["userId"])
        items_ = pd.Categorical(df["movieId"])
        mat = sp.csr_matrix(
            (df["rating"].astype(np.float32), (items_.codes, users_.codes))
        )
        model = implicit.als.AlternatingLeastSquares(
            factors=64, iterations=20, regularization=0.1, random_state=42)
        model.fit(mat)
        self._als_item_ids = list(items_.categories)
        self._als_user_ids = list(users_.categories)
        self.als_model     = model
        with open(ALS_PATH, "wb") as f:
            pickle.dump({"model": model,
                         "item_ids": self._als_item_ids,
                         "user_ids": self._als_user_ids}, f)
        log.info("ALS saved")

    # ── NCF ───────────────────────────────────────────────────────────────────
    def _load_or_train_ncf(self):
        _load_tensorflow_runtime()
        if not HAS_TF:
            return
        assert keras is not None
        assert tf is not None
        try:
            loaded = _load_ncf_artifacts()
        except Exception as exc:
            loaded = None
            log.warning("Existing NCF artifacts could not be loaded; retraining NCF")
            log.warning("  Reason: %s", exc)
        if loaded is not None:
            self.ncf_model, self.ncf_user_enc, self.ncf_item_enc = loaded
            log.info("NCF loaded")
            return
        rp = DATA_DIR / "ratings.csv"
        if not rp.exists():
            return
        log.info("Training NCF (NeuMF) on a streamed sample of up to %d ratings …", NCF_TRAIN_ROWS)
        df = cast(pd.DataFrame, _sample_ratings_csv(
            rp,
            NCF_TRAIN_ROWS,
            usecols=["userId", "movieId", "rating"],
        ))

        users_u = df["userId"].unique()
        items_u = df["movieId"].unique()
        self.ncf_user_enc = {u: i for i, u in enumerate(users_u)}
        self.ncf_item_enc = {m: i for i, m in enumerate(items_u)}

        df["u_enc"] = df["userId"].map(self.ncf_user_enc)
        df["i_enc"] = df["movieId"].map(self.ncf_item_enc)
        df["label"] = (df["rating"] >= 4.0).astype(np.float32)

        # Compute initial BCE before training
        init_preds = np.full(len(df), 0.5)
        init_bce   = bce_loss(np.asarray(df["label"]), init_preds)
        log.info("NCF initial BCE (random) = %.4f", init_bce)

        self.ncf_model = build_ncf(len(users_u), len(items_u))

        callbacks = [
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True,
                                          monitor="val_auc", mode="max"),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                              factor=0.5, patience=1),
        ]
        
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(
            df[["u_enc", "i_enc"]].values, df["label"].values, test_size=0.05, random_state=42
        )

        # ── tf.data Optimizations: Mapping, Caching, Sharding, Batching, Prefetching ──
        assert tf is not None  # already checked HAS_TF above
        _tf = tf  # local alias so Pylance knows it's not None inside closure
        def make_dataset(X, y, is_training=True, num_shards=1, shard_index=0):
            dataset = _tf.data.Dataset.from_tensor_slices((X, y))
            if num_shards > 1:
                dataset = dataset.shard(num_shards, shard_index)
            def preprocess_func(features, label):
                return (features[0], features[1]), label
            dataset = dataset.map(preprocess_func, num_parallel_calls=_tf.data.AUTOTUNE)
            dataset = dataset.cache()
            if is_training:
                dataset = dataset.shuffle(buffer_size=100_000)
            dataset = dataset.batch(2048)
            dataset = dataset.prefetch(_tf.data.AUTOTUNE)
            return dataset

        history = self.ncf_model.fit(
            make_dataset(X_tr, y_tr, is_training=True),
            epochs=10,
            validation_data=make_dataset(X_va, y_va, is_training=False),
            callbacks=callbacks, verbose=1,
        )

        # Log final metrics
        final_bce = history.history["loss"][-1]
        final_auc = history.history.get("auc", [0])[-1]
        log.info("NCF final BCE=%.4f  AUC=%.4f", final_bce, final_auc)
        self._metrics["ncf_bce"] = final_bce
        self._metrics["ncf_auc"] = final_auc

        # BPR loss on validation sample
        val_mask = np.random.rand(len(df)) < 0.02
        val_df   = df[val_mask]
        pos_mask = val_df["label"] == 1
        neg_mask = val_df["label"] == 0
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_s = self.ncf_model.predict(
                [val_df[pos_mask]["u_enc"].values,
                 val_df[pos_mask]["i_enc"].values], verbose=0).flatten()
            neg_s = self.ncf_model.predict(
                [val_df[neg_mask]["u_enc"].values[:len(pos_s)],
                 val_df[neg_mask]["i_enc"].values[:len(pos_s)]], verbose=0).flatten()
            b_loss = bpr_loss(pos_s, neg_s[:len(pos_s)])
            log.info("NCF BPR loss=%.4f", b_loss)
            self._metrics["ncf_bpr"] = b_loss

        _save_ncf_artifacts(self.ncf_model, self.ncf_user_enc, self.ncf_item_enc)

        self._save_metrics("ncf")

    # ── XGBoost re-ranker ─────────────────────────────────────────────────────
    def _load_or_train_xgb(self):
        if not HAS_XGB:
            log.info("XGBoost not installed – re-ranker skipped")
            return
        assert xgb is not None
        existing_xgb = _load_xgb_artifacts()
        if existing_xgb is not None:
            self.xgb_ranker = existing_xgb
            self._load_xgb_feature_context()
            log.info("XGB ranker loaded")
            return

        rp = DATA_DIR / "ratings.csv"
        if not rp.exists() or self.svd_model is None:
            return

        log.info("Training XGBoost re-ranker on a streamed sample of up to %d ratings …", XGB_TRAIN_ROWS)
        assert self.svd_model is not None
        svd = self.svd_model
        df = cast(pd.DataFrame, _sample_ratings_csv(
            rp,
            XGB_TRAIN_ROWS,
            usecols=["userId", "movieId", "rating"],
        ))
        df = df.merge(
            self.movies_df[["movieId", "avg_rating", "num_ratings",
                            "sentiment_score", "trending_score"]],
            on="movieId", how="left"
        ).fillna(0)

        # SVD predicted rating as feature
        df["svd_pred"] = df.apply(
            lambda r: svd.predict(int(r["userId"]), int(r["movieId"])).est,
            axis=1
        )

        # log(num_ratings) popularity feature
        df["log_num_ratings"] = np.log1p(df["num_ratings"])

        feature_cols = ["avg_rating", "svd_pred", "log_num_ratings",
                        "sentiment_score", "trending_score"]
        X = df[feature_cols].values.astype(np.float32)
        y = (df["rating"] >= 4.0).astype(int).values

        self.xgb_ranker = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.xgb_ranker.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=50,
        )
        with open(XGB_PATH, "wb") as f:
            pickle.dump(self.xgb_ranker, f)
        log.info("XGBoost re-ranker saved")

    def _load_xgb_feature_context(self):
        self.xgb_feature_context = {
            "feature_columns": list(LEGACY_XGB_FEATURE_COLUMNS),
            "genre_tokens": [],
            "tag_feature_columns": [],
            "user_feature_columns": [],
            "movie_feature_columns": [],
            "user_stats": {},
            "movie_stats": {},
            "user_genre_affinity": {},
            "user_tag_profiles": {},
        }
        if not XGB_FEATURE_CONTEXT_PATH.exists():
            return
        try:
            with open(XGB_FEATURE_CONTEXT_PATH, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                feature_columns = payload.get("feature_columns") or LEGACY_XGB_FEATURE_COLUMNS
                self.xgb_feature_context = {
                    "feature_columns": list(feature_columns),
                    "genre_tokens": list(payload.get("genre_tokens") or []),
                    "tag_feature_columns": list(payload.get("tag_feature_columns") or []),
                    "user_feature_columns": list(payload.get("user_feature_columns") or []),
                    "movie_feature_columns": list(payload.get("movie_feature_columns") or []),
                    "user_stats": dict(payload.get("user_stats") or {}),
                    "movie_stats": dict(payload.get("movie_stats") or {}),
                    "user_genre_affinity": dict(payload.get("user_genre_affinity") or {}),
                    "user_tag_profiles": {
                        int(user_id): np.asarray(tag_profile, dtype=np.float32)
                        for user_id, tag_profile in dict(payload.get("user_tag_profiles") or {}).items()
                    },
                }
        except Exception as exc:
            log.warning("Could not load XGB feature context: %s", exc)

    def _als_predict_scores(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        if (
            not self.als_model
            or getattr(self, "_als_item_ids", None) is None
            or getattr(self, "_als_user_ids", None) is None
        ):
            return np.zeros(len(movie_ids), dtype=np.float32)

        item_index = {
            int(mid): idx
            for idx, mid in enumerate(cast(list[int], self._als_item_ids))
        }
        user_index = {
            int(uid): idx
            for idx, uid in enumerate(cast(list[int], self._als_user_ids))
        }
        u_idx = user_index.get(int(user_id))
        if u_idx is None:
            return np.zeros(len(movie_ids), dtype=np.float32)

        als_model = cast(Any, self.als_model)
        item_factors = getattr(als_model, "item_factors", None)
        user_factors = getattr(als_model, "user_factors", None)
        if item_factors is None or user_factors is None:
            return np.zeros(len(movie_ids), dtype=np.float32)

        raw_scores = np.zeros(len(movie_ids), dtype=np.float32)
        for idx, movie_id in enumerate(movie_ids):
            i_idx = item_index.get(int(movie_id))
            if i_idx is None:
                continue
            if u_idx >= len(user_factors) or i_idx >= len(item_factors):
                continue
            raw_scores[idx] = float(np.dot(user_factors[u_idx], item_factors[i_idx]))
        return sigmoid(raw_scores)

    def _build_xgb_feature_frame(
        self,
        rows: pd.DataFrame,
        user_id: int,
    ) -> pd.DataFrame:
        feature_rows = rows.copy()
        feature_rows["avg_rating"] = feature_rows.get(
            "avg_rating",
            pd.Series(0.0, index=feature_rows.index),
        ).fillna(0).astype(np.float32)
        feature_rows["rating_stddev"] = feature_rows.get(
            "rating_stddev",
            pd.Series(0.0, index=feature_rows.index),
        ).fillna(0).astype(np.float32)
        feature_rows["log_num_ratings"] = np.log1p(feature_rows["num_ratings"].fillna(0))
        feature_rows["sentiment_signal"] = (
            (feature_rows["sentiment_score"].fillna(0) + 1.0) / 2.0
        ).clip(0.0, 1.0)

        user_stats = dict(self.xgb_feature_context.get("user_stats") or {}).get(int(user_id), {})
        user_feature_columns = list(
            self.xgb_feature_context.get("user_feature_columns")
            or [
                "user_activity_level",
                "user_avg_rating",
                "user_rating_std",
                "user_recency_days_log",
                "user_recency_score",
                "user_rating_freq_trend",
            ]
        )
        for column_name in user_feature_columns:
            feature_rows[column_name] = float(user_stats.get(column_name) or 0.0)
        if "user_avg_rating" not in feature_rows.columns:
            feature_rows["user_avg_rating"] = 0.0

        movie_stats_map = dict(self.xgb_feature_context.get("movie_stats") or {})
        movie_feature_columns = list(
            self.xgb_feature_context.get("movie_feature_columns")
            or [
                "genome_mean_relevance",
                "genome_max_relevance",
                "genome_high_relevance_log",
                "item_popularity_decay",
                "item_recent_rating_velocity",
            ]
        )
        for column_name in movie_feature_columns:
            feature_rows[column_name] = feature_rows["movieId"].apply(
                lambda movie_id: float(
                    dict(movie_stats_map.get(int(movie_id), {})).get(column_name) or 0.0
                )
            )

        genre_tokens = list(self.xgb_feature_context.get("genre_tokens") or [])
        genre_sets = feature_rows.get("genres", pd.Series("", index=feature_rows.index)).fillna("").map(
            lambda value: set(_extract_genre_tokens(value))
        )
        if genre_tokens:
            for token in genre_tokens:
                feature_rows[_genre_feature_name(token)] = genre_sets.map(
                    lambda genres: 1.0 if token in genres else 0.0
                ).astype(np.float32)
        user_affinity = dict(self.xgb_feature_context.get("user_genre_affinity") or {}).get(int(user_id), {})
        feature_rows["genre_affinity_score"] = genre_sets.map(
            lambda genres: float(
                np.mean(
                    [
                        float(user_affinity.get(token, 0.0))
                        for token in genres
                        if token in user_affinity
                    ]
                )
            )
            if user_affinity and any(token in user_affinity for token in genres)
            else 0.0
        ).astype(np.float32)

        score_columns = [
            column_name
            for column_name in ("svd_score", "als_score", "ncf_score")
            if column_name in feature_rows.columns
        ]
        if score_columns:
            score_frame = feature_rows[score_columns].astype(np.float32)
            feature_rows["ensemble_mean"] = score_frame.mean(axis=1).astype(np.float32)
            feature_rows["ensemble_std"] = score_frame.std(axis=1, ddof=0).astype(np.float32)
            feature_rows["score_range"] = (
                score_frame.max(axis=1) - score_frame.min(axis=1)
            ).astype(np.float32)
        else:
            feature_rows["ensemble_mean"] = 0.0
            feature_rows["ensemble_std"] = 0.0
            feature_rows["score_range"] = 0.0

        feature_rows["user_movie_rating_diff"] = (
            feature_rows.get("svd_score", pd.Series(0.0, index=feature_rows.index)).astype(np.float32)
            - feature_rows["user_avg_rating"].astype(np.float32)
        ).astype(np.float32)
        tag_feature_columns = list(self.xgb_feature_context.get("tag_feature_columns") or [])
        user_tag_profile = dict(self.xgb_feature_context.get("user_tag_profiles") or {}).get(int(user_id))
        if tag_feature_columns and user_tag_profile is not None:
            movie_tag_matrix = feature_rows[tag_feature_columns].fillna(0).to_numpy(dtype=np.float32)
            movie_norms = np.linalg.norm(movie_tag_matrix, axis=1)
            tag_scores = np.zeros(len(feature_rows), dtype=np.float32)
            profile_vector = np.asarray(user_tag_profile, dtype=np.float32)
            for idx, movie_vector in enumerate(movie_tag_matrix):
                if movie_norms[idx] <= 0:
                    continue
                common_len = min(len(profile_vector), len(movie_vector))
                if common_len <= 0:
                    continue
                tag_scores[idx] = float(
                    np.dot(
                        profile_vector[:common_len],
                        movie_vector[:common_len] / max(float(np.linalg.norm(movie_vector[:common_len])), 1e-6),
                    )
                )
            feature_rows["tag_cooccurrence_strength"] = tag_scores.astype(np.float32)
        else:
            feature_rows["tag_cooccurrence_strength"] = 0.0
        if (
            "sbert_sim" in set(self.xgb_feature_context.get("feature_columns") or [])
            and self.user_taste_vectors is not None
            and self.movie_sbert_embeddings
        ):
            user_row_idx = self.user_taste_id_map.get(int(user_id))
            if user_row_idx is not None and user_row_idx < len(self.user_taste_vectors):
                taste_vector = np.asarray(
                    self.user_taste_vectors[user_row_idx],
                    dtype=np.float32,
                )
                feature_rows["sbert_sim"] = feature_rows["movieId"].apply(
                    lambda movie_id: float(
                        np.dot(
                            taste_vector,
                            self.movie_sbert_embeddings.get(
                                int(movie_id),
                                np.zeros_like(taste_vector),
                            ),
                        )
                    )
                ).astype(np.float32)
            else:
                feature_rows["sbert_sim"] = 0.0
        else:
            feature_rows["sbert_sim"] = 0.0
        return feature_rows

    # ── Save metrics to DB ────────────────────────────────────────────────────
    def _save_metrics(self, model_name: str):
        try:
            run_id = f"{model_name}_{_utc_now().strftime('%Y%m%d_%H%M%S')}"
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO model_metrics "
                    "(run_id,model,bce_loss,bpr_loss,mse,ndcg_10,auc,ts) "
                    "VALUES (?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(run_id) DO UPDATE SET "
                    "model = excluded.model, "
                    "bce_loss = excluded.bce_loss, "
                    "bpr_loss = excluded.bpr_loss, "
                    "mse = excluded.mse, "
                    "ndcg_10 = excluded.ndcg_10, "
                    "auc = excluded.auc, "
                    "ts = excluded.ts",
                    (run_id, model_name,
                     self._metrics.get("ncf_bce"),
                     self._metrics.get("ncf_bpr"),
                     self._metrics.get("svd_mse"),
                     self._metrics.get("ndcg_10"),
                     self._metrics.get("ncf_auc"),
                     _utc_now().isoformat())
                )
        except Exception as e:
            log.debug("Metrics save error: %s", e)

    # ═══════════════════════════════════════════════════════════════════════════
    #  INFERENCE
    # ═══════════════════════════════════════════════════════════════════════════

    def _ensure_search_columns(self):
        if self.movies_df.empty or "_search_title" in self.movies_df.columns:
            return

        raw_titles = self.movies_df["title"].fillna("").astype(str)
        clean_titles = raw_titles.map(lambda value: _clean_title(value)[0])
        normalized_titles = clean_titles.map(_normalize_search_text)
        normalized_genres = (
            self.movies_df.get("genres", pd.Series("", index=self.movies_df.index))
            .fillna("")
            .astype(str)
            .map(_normalize_search_text)
        )
        normalized_features = (
            self.movies_df.get("feature_text", pd.Series("", index=self.movies_df.index))
            .fillna("")
            .astype(str)
            .map(_normalize_search_text)
        )

        self.movies_df["_clean_title"] = clean_titles
        self.movies_df["_search_title"] = normalized_titles
        self.movies_df["_search_compact"] = normalized_titles.str.replace(" ", "", regex=False)
        self.movies_df["_search_blob"] = (
            normalized_titles + " " + normalized_genres + " " + normalized_features
        ).str.strip()

    def resolve_title(self, query: str) -> Optional[str]:
        q = _normalize_search_text(query)
        if not q:
            return None
        if self.movies_df.empty:
            resolved = _resolve_curated_title(query)
            return resolved["title"] if resolved else None

        self._ensure_search_columns()
        compact_query = q.replace(" ", "")

        if q in self.normalized_title_index:
            idx = self.normalized_title_index[q]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            return str(self.movies_df.loc[idx, "title"])
        if query.strip().lower() in self.title_index:
            idx = self.title_index[query.strip().lower()]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            return str(self.movies_df.loc[idx, "title"])
        mask = self.movies_df["_search_title"].str.contains(re.escape(q), na=False)
        if compact_query:
            mask = mask | self.movies_df["_search_compact"].str.contains(
                re.escape(compact_query),
                na=False,
            )
        if mask.any():
            return str(self.movies_df[mask]
                    .sort_values("avg_rating", ascending=False)
                    .iloc[0]["title"])
        all_titles = self.movies_df["_search_title"].tolist()
        if HAS_RAPIDFUZZ and fuzz_process is not None:
            m = fuzz_process.extractOne(q, all_titles, score_cutoff=62)
            matched_lower = m[0] if m else None
        else:
            ms = get_close_matches(q, all_titles, n=1, cutoff=0.58)
            matched_lower = ms[0] if ms else None
        if not matched_lower:
            scores = self.movies_df["_search_title"].map(
                lambda value: _sequence_similarity(q, value)
            )
            if scores.empty or float(scores.max()) < 0.48:
                return None
            matched_idx = int(scores.idxmax())
            return str(self.movies_df.loc[matched_idx, "title"])
        row = self.movies_df[self.movies_df["_search_title"] == matched_lower]
        return str(row.iloc[0]["title"]) if not row.empty else None

    def search(self, q: str, limit: int = 50) -> list:
        if self.movies_df.empty:
            return _search_curated_catalog(q, limit)

        normalized_query = _normalize_search_text(q)
        if not normalized_query:
            return []

        self._ensure_search_columns()
        compact_query = normalized_query.replace(" ", "")
        tokens = normalized_query.split()
        scores = pd.Series(0.0, index=self.movies_df.index, dtype=float)

        title_series = self.movies_df["_search_title"]
        compact_series = self.movies_df["_search_compact"]
        blob_series = self.movies_df["_search_blob"]

        scores += title_series.eq(normalized_query).astype(float) * 180
        if compact_query:
            scores += compact_series.eq(compact_query).astype(float) * 170
        scores += title_series.str.contains(re.escape(normalized_query), na=False).astype(float) * 125
        if compact_query:
            scores += compact_series.str.contains(re.escape(compact_query), na=False).astype(float) * 110
        scores += title_series.str.startswith(normalized_query, na=False).astype(float) * 45

        if tokens:
            all_title_tokens = pd.Series(True, index=self.movies_df.index)
            for token in tokens:
                token_pattern = re.escape(token)
                in_title = title_series.str.contains(fr"\b{token_pattern}\b", na=False)
                in_blob = blob_series.str.contains(fr"\b{token_pattern}\b", na=False)
                scores += in_title.astype(float) * 18
                scores += in_blob.astype(float) * 6
                all_title_tokens &= in_title
            scores += all_title_tokens.astype(float) * 36

        candidate_index = scores[scores > 0].index
        if len(candidate_index) == 0:
            similarity = title_series.map(
                lambda value: _sequence_similarity(normalized_query, value)
            )
            candidate_index = similarity[similarity >= 0.46].index
            if len(candidate_index) == 0:
                return []
            candidate_scores = similarity.loc[candidate_index] * 100
        else:
            similarity = title_series.loc[candidate_index].map(
                lambda value: _sequence_similarity(normalized_query, value)
            )
            candidate_scores = scores.loc[candidate_index] + similarity * 42

        rows = self.movies_df.loc[candidate_index].copy()
        rows["_search_score"] = (
            candidate_scores.values
            + rows["avg_rating"].fillna(0).astype(float) / 5.0
            + rows.get("trending_score", pd.Series(0, index=rows.index)).fillna(0).astype(float) * 6
        )

        ranked = rows.sort_values(
            ["_search_score", "avg_rating", "title"],
            ascending=[False, False, True],
        ).head(limit)
        return self._format(ranked)

    def recommend(
        self,
        title: str,
        user_id: int = 1,
        top_n: int = 50,
        mood: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> dict:
        if not self._ready or self.movies_df.empty:
            return _recommend_from_curated_catalog(title, top_n, mood, user_email=user_email)

        resolved = self.resolve_title(title)
        if resolved is None:
            fallback = _recommend_from_curated_catalog(
                title,
                top_n,
                mood,
                user_email=user_email,
            )
            if fallback["results"]:
                return fallback
            return {"resolved_title": title, "results": []}

        mask = self.movies_df["title"].str.lower() == resolved.lower()
        idx  = int(self.movies_df[mask].index[0])

        # ── TF-IDF candidate pool ─────────────────────────────────────────────
        assert self.tfidf_matrix is not None
        tfidf_mat = cast(Any, self.tfidf_matrix)
        tfidf_sim = cosine_similarity(
            tfidf_mat[idx], tfidf_mat).flatten()
        tfidf_sim[idx] = 0
        preference_context = _load_user_preference_context(user_email)
        # Use larger pool when ML models are absent for better recall
        pool_size = 200 if (self.svd_model is None and self.ncf_model is None) else 100
        if preference_context is not None:
            pool_size = max(pool_size, min(max(top_n * 6, 180), len(tfidf_sim) - 1))
        pool_idx = np.argsort(tfidf_sim)[::-1][:pool_size]

        rows = self.movies_df.iloc[pool_idx].copy()
        rows["tfidf_score"] = tfidf_sim[pool_idx]

        # ── SBERT ────────────────────────────────────────────────────────────
        if self.sbert_embeddings is not None:
            rows["sbert_score"] = (self.sbert_embeddings[pool_idx]
                                   @ self.sbert_embeddings[idx])
        else:
            rows["sbert_score"] = rows["tfidf_score"]

        # ── Genome ────────────────────────────────────────────────────────────
        if self.genome_matrix is not None:
            try:
                rows["genome_score"] = cosine_similarity(
                    self.genome_matrix[pool_idx],
                    self.genome_matrix[[idx]]
                ).flatten()
            except Exception:
                rows["genome_score"] = 0.0
        else:
            rows["genome_score"] = 0.0

        # ── SVD ───────────────────────────────────────────────────────────────
        if self.svd_model:
            svd = self.svd_model
            rows["svd_score"] = rows["movieId"].apply(
                lambda m: svd.predict(user_id, int(m)).est / 5.0)
        else:
            rows["svd_score"] = rows["avg_rating"] / 5.0

        # ── ALS ───────────────────────────────────────────────────────────────
        if self.als_model and getattr(self, "_als_item_ids", None) is not None and getattr(self, "_als_user_ids", None) is not None:
            rows["als_score"] = self._als_predict_scores(
                user_id,
                [int(movie_id) for movie_id in rows["movieId"].tolist()],
            )
        else:
            rows["als_score"] = rows["svd_score"]

        # ── NCF ───────────────────────────────────────────────────────────────
        if self.ncf_model and self.ncf_item_enc:
            assert self.ncf_model is not None
            u_enc  = self.ncf_user_enc.get(user_id, 0)
            i_encs = np.array([self.ncf_item_enc.get(int(m), 0)
                               for m in rows["movieId"]])
            raw_preds = self.ncf_model.predict(
                [np.full_like(i_encs, u_enc), i_encs], verbose=0).flatten()
            rows["ncf_score"] = np.clip(raw_preds.astype(np.float32), 0.0, 1.0)
        else:
            rows["ncf_score"] = rows["svd_score"]

        # ── Signals ───────────────────────────────────────────────────────────
        rows["trending"]  = rows.get("trending_score",
                                     pd.Series(0, index=rows.index)).fillna(0)
        rows["sentiment"] = ((rows.get("sentiment_score",
                              pd.Series(0, index=rows.index)).fillna(0) + 1) / 2)

        liked, disliked = self._get_user_feedback(str(user_id))
        rows["feedback_boost"] = rows["movieId"].apply(
            lambda m: 0.3 if int(m) in liked else (-0.5 if int(m) in disliked else 0.0))

        if mood and mood.lower() in MOOD_GENRE_MAP:
            mg = MOOD_GENRE_MAP[mood.lower()]
            rows["mood_boost"] = rows["genres"].apply(
                lambda g: 0.2 if any(x.lower() in str(g).lower() for x in mg) else 0.0)
        else:
            rows["mood_boost"] = 0.0

        rows["preference_boost"] = rows["genres"].apply(
            lambda g: _preference_boost_for_genres(str(g or ""), preference_context)
        )
        rows["preference_multiplier"] = rows["genres"].apply(
            lambda g: _preference_multiplier_for_genres(str(g or ""), preference_context)
        )

        # ── XGBoost re-rank ───────────────────────────────────────────────────
        if self.xgb_ranker is not None:
            assert self.xgb_ranker is not None
            xgb_rows = self._build_xgb_feature_frame(rows, user_id)
            feature_columns = list(
                self.xgb_feature_context.get("feature_columns") or LEGACY_XGB_FEATURE_COLUMNS
            )
            feat = xgb_rows.reindex(columns=feature_columns, fill_value=0).to_numpy(
                dtype=np.float32
            )
            rows["xgb_score"] = self.xgb_ranker.predict_proba(feat)[:, 1]
        else:
            rows["xgb_score"] = rows["svd_score"]

        # ── ENSEMBLE (dynamic weighted rank fusion) ──────────────────────────
        # Detect which models are genuinely trained (not just avg_rating proxies)
        has_svd  = self.svd_model is not None
        has_als  = self.als_model is not None and getattr(self, "_als_item_ids", None) is not None
        has_ncf  = self.ncf_model is not None and bool(self.ncf_item_enc)
        has_xgb  = self.xgb_ranker is not None
        has_genome = self.genome_matrix is not None
        has_sbert  = self.sbert_embeddings is not None

        if has_svd and has_ncf and has_xgb:
            # Full stack: balanced weights across all signals
            w_tfidf   = 0.18
            w_sbert   = 0.18
            w_genome  = 0.08 if has_genome else 0.0
            w_svd     = 0.12
            w_als     = 0.08 if has_als else 0.0
            w_ncf     = 0.10
            w_xgb     = 0.12
            w_trend   = 0.05
            w_sent    = 0.04
            w_mood    = 0.03
        elif has_svd:
            # Partial: SVD trained but not NCF/XGB
            w_tfidf   = 0.28
            w_sbert   = 0.28 if has_sbert else 0.0
            w_genome  = 0.10 if has_genome else 0.0
            w_svd     = 0.20
            w_als     = 0.08 if has_als else 0.0
            w_ncf     = 0.0
            w_xgb     = 0.0
            w_trend   = 0.04
            w_sent    = 0.02
            w_mood    = 0.02
        else:
            # No ML models trained — lean entirely on content similarity + popularity
            w_tfidf   = 0.40
            w_sbert   = 0.35 if has_sbert else 0.0
            w_genome  = 0.12 if has_genome else 0.0
            w_svd     = 0.0
            w_als     = 0.0
            w_ncf     = 0.0
            w_xgb     = 0.0
            w_trend   = 0.06
            w_sent    = 0.04
            w_mood    = 0.03

        # Normalise weights to sum to 1.0
        total_w = (w_tfidf + w_sbert + w_genome + w_svd + w_als +
                   w_ncf + w_xgb + w_trend + w_sent + w_mood)
        if total_w > 0:
            norm = 1.0 / total_w
            w_tfidf  *= norm; w_sbert *= norm; w_genome *= norm
            w_svd    *= norm; w_als   *= norm; w_ncf    *= norm
            w_xgb    *= norm; w_trend *= norm; w_sent   *= norm
            w_mood   *= norm

        rows["raw_score"] = (
            w_tfidf  * rows["tfidf_score"]
            + w_sbert  * rows["sbert_score"]
            + w_genome * rows["genome_score"]
            + w_svd    * rows["svd_score"]
            + w_als    * rows["als_score"]
            + w_ncf    * rows["ncf_score"]
            + w_xgb    * rows["xgb_score"]
            + w_trend  * rows["trending"]
            + w_sent   * rows["sentiment"]
            + w_mood   * rows["mood_boost"]
        )
        log.debug(
            "Ensemble weights — tfidf=%.2f sbert=%.2f svd=%.2f als=%.2f "
            "ncf=%.2f xgb=%.2f trend=%.2f",
            w_tfidf, w_sbert, w_svd, w_als, w_ncf, w_xgb, w_trend,
        )

        # Apply sigmoid activation to compress final score → [0,1]
        base_score = sigmoid(
            leaky_relu(np.asarray(rows["raw_score"]) * 6 - 3)
        )
        rows["score"] = (
            base_score * np.asarray(rows["preference_multiplier"])
            + np.asarray(rows["feedback_boost"])
            + np.asarray(rows["preference_boost"])
        )

        # Compute NDCG on candidate pool
        relevance = np.asarray(rows["score"])
        ndcg = ndcg_at_k(relevance, k=10)
        self._metrics["ndcg_10"] = ndcg

        rerank_pool_size = top_n
        if preference_context is not None:
            rerank_pool_size = min(len(rows.index), max(top_n * 6, 120))

        top_rows = rows.nlargest(rerank_pool_size, "score")
        results = _apply_preference_ranking(
            self._format(top_rows),
            user_email,
            limit=top_n,
        )
        return {
            "resolved_title": resolved,
            "mood_applied":   mood,
            "ndcg_10":        round(ndcg, 4),
            "results":        results[:top_n],
        }

    def browse_by_mood(self, mood: str, top_n: int = 20) -> list:
        if mood.lower() not in MOOD_GENRE_MAP:
            return []
        pattern = "|".join(MOOD_GENRE_MAP[mood.lower()])
        mask = self.movies_df["genres"].str.contains(pattern, case=False, na=False)
        result = (self.movies_df[mask]
                  .sort_values(["trending_score", "avg_rating"], ascending=False)
                  .head(top_n))
        return self._format(result)

    def get_metrics(self) -> dict:
        return self._metrics

    def _get_user_feedback(self, user_id: str):
        liked, disliked = set(), set()
        try:
            with get_db() as conn:
                for r in conn.execute(
                    "SELECT movieId, feedback FROM user_feedback WHERE user_id=?",
                    (user_id,)
                ).fetchall():
                    (liked if r["feedback"] == "like" else disliked).add(r["movieId"])
        except Exception:
            pass
        return liked, disliked

    # ── OMDB ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _fetch_omdb(clean_title: str, year: str) -> dict:
        global _OMDB_KEY_INVALID
        # Build the best possible fallback poster: TMDB > iTunes > generated SVG
        tmdb_poster = _fetch_tmdb_poster(clean_title, year)
        itunes_poster = _fetch_itunes_poster(clean_title, year) if not tmdb_poster else ""
        fallback_poster = tmdb_poster or itunes_poster or _generated_poster_url(clean_title, year)
        FALLBACK = {
            "poster": fallback_poster,
            "plot": _fallback_movie_description(clean_title, year),
            "cast": "",
            "director": "",
            "imdb_rating": "",
            "runtime": "",
        }
        cache_key = f"{clean_title}|{year}"
        cached = _omdb_cache_get(cache_key)
        if cached is not None:
            # Upgrade stale placeholder poster in cached entry
            if _is_missing_poster(cached.get("poster", "")):
                cached["poster"] = fallback_poster
            return cached
        try:
            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM omdb_cache WHERE title=?", (cache_key,)
                ).fetchone()
                if row:
                    result = {k: row[k] or FALLBACK.get(k, "")
                              for k in FALLBACK}
                    # Upgrade stale placeholder poster stored in DB
                    if _is_missing_poster(result.get("poster", "")):
                        result["poster"] = fallback_poster
                    _omdb_cache_set(cache_key, result)
                    return result
        except Exception:
            pass
        if not OMDB_API_KEY or _OMDB_KEY_INVALID:
            _omdb_cache_set(cache_key, FALLBACK)
            return dict(FALLBACK)
        params = {"apikey": OMDB_API_KEY, "t": clean_title, "type": "movie"}
        if year:
            params["y"] = year
        try:
            data = requests.get(OMDB_BASE_URL, params=params,
                                timeout=OMDB_TIMEOUT).json()
            if data.get("Response") != "True":
                if data.get("Error") == "Invalid API key!":
                    _OMDB_KEY_INVALID = True
                    log.warning(
                        "OMDb rejected the configured API key. Poster fetches will use the iTunes/neutral fallback until the key is updated."
                    )
                    _omdb_cache_set(cache_key, FALLBACK)
                    return dict(FALLBACK)
                data = requests.get(OMDB_BASE_URL,
                    params={"apikey": OMDB_API_KEY, "t": clean_title,
                            "type": "movie"}, timeout=OMDB_TIMEOUT).json()
        except Exception:
            _omdb_cache_set(cache_key, FALLBACK)
            return dict(FALLBACK)
        if data.get("Response") != "True":
            _omdb_cache_set(cache_key, FALLBACK)
            return dict(FALLBACK)

        def v(k):
            val = data.get(k, "")
            return "" if val == "N/A" else val

        result = {"poster":      v("Poster") or fallback_poster,
                  "plot":        v("Plot"),
                  "cast":        v("Actors"),
                  "director":    v("Director"),
                  "imdb_rating": v("imdbRating"),
                  "runtime":     v("Runtime")}
        # OMDB sometimes returns "N/A" as Poster — upgrade with TMDB/iTunes
        if _is_missing_poster(result["poster"]):
            result["poster"] = fallback_poster
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO omdb_cache "
                    '(title,poster,plot,"cast",director,imdb_rating,runtime,fetched_at) '
                    "VALUES (?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(title) DO UPDATE SET "
                    "poster = excluded.poster, "
                    "plot = excluded.plot, "
                    '"cast" = excluded."cast", '
                    "director = excluded.director, "
                    "imdb_rating = excluded.imdb_rating, "
                    "runtime = excluded.runtime, "
                    "fetched_at = excluded.fetched_at",
                    (cache_key, result["poster"], result["plot"], result["cast"],
                     result["director"], result["imdb_rating"], result["runtime"],
                     _utc_now().isoformat()))
        except Exception:
            pass
        _omdb_cache_set(cache_key, result)
        return result

    def _format(self, df: pd.DataFrame) -> list:
        out = []
        for _, row in df.iterrows():
            title    = row["title"]
            clean, yr = _clean_title(title)
            out.append(
                _movie_payload(
                    title=title,
                    clean_title=clean,
                    year=yr,
                    genres=str(row.get("genres", "")),
                    avg_rating=row.get("avg_rating", 0),
                    trending_score=float(row.get("trending_score", 0) or 0),
                    movie_id=row.get("movieId"),
                )
            )
        return out


# ── helpers ───────────────────────────────────────────────────────────────────
def _clean_title(title: str):
    m = re.search(r"^(.+?)\s*\((\d{4})\)\s*$", title.strip())
    return (m.group(1).strip(), m.group(2)) if m else (title.strip(), "")


def _movie_key(clean_title: str, year: str) -> str:
    raw = f"{clean_title}-{year}".strip("-").lower()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-") or "moviebuzz-item"


def _youtube_link(clean_title: str, year: str) -> str:
    query = f"{clean_title} {year} official trailer".strip()
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


def _movie_payload(
    title: str,
    clean_title: str,
    year: str,
    genres: str = "",
    avg_rating: Any = 0,
    trending_score: float = 0.0,
    movie_id: Any = None,
    poster_override: str = "",
) -> dict[str, Any]:
    lookup_title = _normalize_lookup_title(clean_title)
    omdb = RecommenderEngine._fetch_omdb(lookup_title, year)
    resolved_movie_id: int | None = None
    try:
        if movie_id is not None and not pd.isna(movie_id):
            resolved_movie_id = int(movie_id)
    except Exception:
        resolved_movie_id = None

    rating: float | str = "N/A"
    try:
        numeric_rating = float(avg_rating)
        if pd.notna(numeric_rating) and numeric_rating > 0:
            rating = round(numeric_rating, 1)
    except Exception:
        rating = "N/A"

    # Poster priority: non-empty override → OMDB/iTunes result → generated SVG
    if poster_override and not _is_missing_poster(poster_override):
        resolved_poster = poster_override.strip()
    elif omdb["poster"] and not _is_missing_poster(omdb["poster"]):
        resolved_poster = omdb["poster"]
    else:
        resolved_poster = _generated_poster_url(clean_title, year, genres)

    generic_plot = _fallback_movie_description(clean_title, year)
    if not omdb["plot"] or omdb["plot"] == generic_plot:
        resolved_plot = _fallback_movie_description(clean_title, year, genres)
    else:
        resolved_plot = omdb["plot"]
    resolved_imdb_rating = omdb["imdb_rating"]
    if not resolved_imdb_rating and rating != "N/A":
        resolved_imdb_rating = str(rating)

    return {
        "movie_key": _movie_key(clean_title, year),
        "movie_id": resolved_movie_id,
        "title": title,
        "clean_title": clean_title,
        "year": year,
        "genres": genres,
        "poster": resolved_poster,
        "plot": resolved_plot,
        "description": resolved_plot,
        "cast": omdb["cast"],
        "director": omdb["director"],
        "imdb_rating": resolved_imdb_rating,
        "runtime": omdb["runtime"],
        "rating": rating,
        "trending_score": round(float(trending_score or 0), 3),
        "youtube_link": _youtube_link(clean_title, year),
    }


def _movie_payload_light(
    title: str,
    clean_title: str,
    year: str,
    genres: str = "",
    avg_rating: Any = 0,
    trending_score: float = 0.0,
    movie_id: Any = None,
    poster_override: str = "",
) -> dict[str, Any]:
    lookup_title = _normalize_lookup_title(clean_title)
    resolved_movie_id: int | None = None
    try:
        if movie_id is not None and not pd.isna(movie_id):
            resolved_movie_id = int(movie_id)
    except Exception:
        resolved_movie_id = None

    rating: float | str = "N/A"
    try:
        numeric_rating = float(avg_rating)
        if pd.notna(numeric_rating) and numeric_rating > 0:
            rating = round(numeric_rating, 1)
    except Exception:
        rating = "N/A"

    cached_omdb = _read_omdb_cache_entry(f"{lookup_title}|{year}")
    if cached_omdb is None and lookup_title != clean_title:
        cached_omdb = _read_omdb_cache_entry(f"{clean_title}|{year}")

    if poster_override and not _is_missing_poster(poster_override):
        resolved_poster = poster_override.strip()
    elif cached_omdb and not _is_missing_poster(str(cached_omdb.get("poster") or "")):
        resolved_poster = str(cached_omdb.get("poster") or "").strip()
    else:
        resolved_poster = _generated_poster_url(clean_title, year, genres)

    generic_plot = _fallback_movie_description(clean_title, year)
    cached_plot = str(cached_omdb.get("plot") or "").strip() if cached_omdb else ""
    if not cached_plot or cached_plot == generic_plot:
        resolved_plot = _fallback_movie_description(clean_title, year, genres)
    else:
        resolved_plot = cached_plot

    resolved_imdb_rating = (
        str(cached_omdb.get("imdb_rating") or "").strip() if cached_omdb else ""
    )
    if not resolved_imdb_rating and rating != "N/A":
        resolved_imdb_rating = str(rating)

    return {
        "movie_key": _movie_key(clean_title, year),
        "movie_id": resolved_movie_id,
        "title": title,
        "clean_title": clean_title,
        "year": year,
        "genres": genres,
        "poster": resolved_poster,
        "plot": resolved_plot,
        "description": resolved_plot,
        "cast": str(cached_omdb.get("cast") or "").strip() if cached_omdb else "",
        "director": str(cached_omdb.get("director") or "").strip() if cached_omdb else "",
        "imdb_rating": resolved_imdb_rating,
        "runtime": str(cached_omdb.get("runtime") or "").strip() if cached_omdb else "",
        "rating": rating,
        "trending_score": round(float(trending_score or 0), 3),
        "youtube_link": _youtube_link(clean_title, year),
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _movie_search_candidate(
    title: str,
    genres: str = "",
    avg_rating: Any = 0,
    trending_score: Any = 0,
    movie_id: Any = None,
    poster: str = "",
    source: str = "",
    description: str = "",
    plot: str = "",
) -> dict[str, Any]:
    clean_title, year = _clean_title(title)
    resolved_movie_id: int | None = None
    try:
        if movie_id is not None and not pd.isna(movie_id):
            resolved_movie_id = int(movie_id)
    except Exception:
        resolved_movie_id = None

    search_blob = _normalize_search_text(
        " ".join(
            [
                clean_title,
                genres,
                str(description or plot or ""),
            ]
        )
    )
    search_title = _normalize_search_text(clean_title or title)

    return {
        "movie_key": _movie_key(clean_title or title, year),
        "movie_id": resolved_movie_id,
        "title": title,
        "clean_title": clean_title or title,
        "year": year,
        "genres": genres,
        "poster": str(poster or "").strip(),
        "description": str(description or "").strip(),
        "plot": str(plot or description or "").strip(),
        "rating": _safe_float(avg_rating),
        "trending_score": _safe_float(trending_score),
        "source": str(source or "").strip(),
        "_search_title": search_title,
        "_compact_title": search_title.replace(" ", ""),
        "_search_blob": search_blob,
        "_compact_blob": search_blob.replace(" ", ""),
    }


def _candidate_to_movie_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = _movie_payload_light(
        title=str(candidate.get("title") or ""),
        clean_title=str(candidate.get("clean_title") or candidate.get("title") or ""),
        year=str(candidate.get("year") or ""),
        genres=str(candidate.get("genres") or ""),
        avg_rating=candidate.get("rating") or 0,
        trending_score=candidate.get("trending_score") or 0,
        movie_id=candidate.get("movie_id"),
        poster_override=str(candidate.get("poster") or ""),
    )
    source = str(candidate.get("source") or "").strip()
    if source:
        payload["source"] = source
    return payload


def _row_to_movie_candidate(row: Any) -> dict[str, Any]:
    return _movie_search_candidate(
        title=str(row["title"]),
        genres=str(row["genres"] or ""),
        avg_rating=row["avg_rating"],
        trending_score=row["trending_score"] if "trending_score" in row.keys() else 0,
        movie_id=row["movieId"],
        poster=str(row["poster"] or ""),
        source=str(row["source"] or "") if "source" in row.keys() else "",
    )


def _rank_search_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    normalized_query = _normalize_search_text(query)
    if not normalized_query or limit <= 0 or not candidates:
        return []

    ranked = sorted(
        (
            (
                _score_catalog_movie_search(candidate, normalized_query, details),
                candidate,
                details,
            )
            for candidate in candidates
            for details in [_catalog_search_details(candidate, normalized_query)]
        ),
        key=lambda item: (
            item[0],
            _safe_float(item[1].get("rating")),
            _safe_float(item[1].get("trending_score")),
        ),
        reverse=True,
    )

    preferred_results: list[dict[str, Any]] = []
    secondary_results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for score, candidate, details in ranked:
        if score <= 20:
            continue

        movie_key = str(candidate.get("movie_key") or "")
        if movie_key and movie_key in seen_keys:
            continue
        if movie_key:
            seen_keys.add(movie_key)

        is_preferred = bool(
            details["direct_title_match"]
            or details["direct_compact_match"]
            or details["all_significant_tokens_matched"]
        )
        target = preferred_results if is_preferred else secondary_results
        target.append(dict(candidate))

        if len(preferred_results) >= limit and len(preferred_results) + len(secondary_results) >= limit * 2:
            break

    results = preferred_results[:limit]
    if len(results) < limit:
        results.extend(secondary_results[: max(0, limit - len(results))])
    return results[:limit]


def _curated_home_movie_payload(seed: tuple[str, str]) -> dict[str, Any]:
    clean_title, year = seed
    metadata = _curated_seed_metadata(clean_title, year)
    return _movie_payload(
        title=f"{clean_title} ({year})",
        clean_title=clean_title,
        year=year,
        genres=str(metadata.get("genres", "")),
        avg_rating=metadata.get("rating", 0),
        trending_score=0.0,
    )


def _curated_catalog_movies() -> list[dict[str, Any]]:
    global _CURATED_CATALOG_CACHE
    if _CURATED_CATALOG_CACHE:
        return [dict(movie) for movie in _CURATED_CATALOG_CACHE]

    with ThreadPoolExecutor(max_workers=min(8, max(len(CURATED_HOME_MOVIES), 1))) as executor:
        _CURATED_CATALOG_CACHE = list(executor.map(_curated_home_movie_payload, CURATED_HOME_MOVIES))
    return [dict(movie) for movie in _CURATED_CATALOG_CACHE]


def _catalog_search_details(
    movie: dict[str, Any],
    normalized_query: str,
) -> dict[str, Any]:
    compact_query = normalized_query.replace(" ", "")
    tokens = [token for token in normalized_query.split() if token]
    significant_tokens = [token for token in tokens if len(token) > 2] or tokens
    clean_title = str(movie.get("clean_title") or movie.get("title") or "")
    title_search = str(movie.get("_search_title") or _normalize_search_text(clean_title))
    compact_title = str(movie.get("_compact_title") or title_search.replace(" ", ""))
    blob = str(
        movie.get("_search_blob")
        or _normalize_search_text(
            " ".join(
                [
                    clean_title,
                    str(movie.get("genres") or ""),
                    str(movie.get("description") or movie.get("plot") or ""),
                ]
            )
        )
    )
    compact_blob = str(movie.get("_compact_blob") or blob.replace(" ", ""))

    title_token_hits = 0
    significant_title_hits = 0
    blob_token_hits = 0

    for token in tokens:
        token_pattern = fr"\b{re.escape(token)}\b"
        title_match = bool(re.search(token_pattern, title_search))
        if not title_match and len(token) > 2 and token in compact_title:
            title_match = True
        if title_match:
            title_token_hits += 1
            if token in significant_tokens:
                significant_title_hits += 1

        blob_match = bool(re.search(token_pattern, blob))
        if not blob_match and len(token) > 2 and token in compact_blob:
            blob_match = True
        if blob_match:
            blob_token_hits += 1

    return {
        "compact_query": compact_query,
        "tokens": tokens,
        "significant_tokens": significant_tokens,
        "clean_title": clean_title,
        "title_search": title_search,
        "compact_title": compact_title,
        "blob": blob,
        "title_token_hits": title_token_hits,
        "significant_title_hits": significant_title_hits,
        "blob_token_hits": blob_token_hits,
        "direct_title_match": bool(normalized_query and normalized_query in title_search),
        "direct_compact_match": bool(compact_query and compact_query in compact_title),
        "all_title_tokens_matched": bool(tokens) and title_token_hits == len(tokens),
        "all_significant_tokens_matched": bool(significant_tokens)
        and significant_title_hits == len(significant_tokens),
    }


def _score_catalog_movie_search(
    movie: dict[str, Any],
    normalized_query: str,
    details: Optional[dict[str, Any]] = None,
) -> float:
    search_details = details or _catalog_search_details(movie, normalized_query)
    compact_query = str(search_details["compact_query"])
    tokens = cast(list[str], search_details["tokens"])
    significant_tokens = cast(list[str], search_details["significant_tokens"])
    title_search = str(search_details["title_search"])
    compact_title = str(search_details["compact_title"])
    title_token_hits = int(search_details["title_token_hits"])
    significant_title_hits = int(search_details["significant_title_hits"])
    blob_token_hits = int(search_details["blob_token_hits"])
    direct_title_match = bool(search_details["direct_title_match"])
    direct_compact_match = bool(search_details["direct_compact_match"])
    all_title_tokens_matched = bool(search_details["all_title_tokens_matched"])
    all_significant_tokens_matched = bool(search_details["all_significant_tokens_matched"])

    score = 0.0
    if title_search == normalized_query:
        score += 180
    if compact_query and compact_title == compact_query:
        score += 170
    if direct_title_match:
        score += 125
    if direct_compact_match:
        score += 110
    if title_search.startswith(normalized_query):
        score += 45

    if tokens:
        score += title_token_hits * 18
        score += blob_token_hits * 6

        if all_title_tokens_matched:
            score += 70
        elif len(tokens) > 1:
            missing_title_tokens = len(tokens) - title_token_hits
            score -= missing_title_tokens * 52
            if title_token_hits == 0:
                score -= 24

        if all_significant_tokens_matched and len(significant_tokens) >= 1:
            score += 24
        elif len(tokens) > 1 and significant_tokens:
            missing_significant_tokens = len(significant_tokens) - significant_title_hits
            score -= missing_significant_tokens * 36

    score += _sequence_similarity(normalized_query, title_search) * 42
    try:
        score += float(movie.get("rating") or 0) / 5.0
    except Exception:
        pass
    return score


def _resolve_curated_title(query: str) -> Optional[dict[str, Any]]:
    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        return None

    ranked = sorted(
        (
            (_score_catalog_movie_search(movie, normalized_query), movie)
            for movie in _curated_catalog_movies()
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] < 40:
        return None
    return dict(ranked[0][1])


def _search_curated_catalog(query: str, limit: int = 50) -> list[dict[str, Any]]:
    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        return []

    ranked = sorted(
        (
            (_score_catalog_movie_search(movie, normalized_query), movie)
            for movie in _curated_catalog_movies()
        ),
        key=lambda item: (item[0], float(item[1].get("rating") or 0)),
        reverse=True,
    )
    return [dict(movie) for score, movie in ranked if score > 20][:limit]


def _lightweight_catalog_movies() -> list[dict[str, Any]]:
    global _LIGHTWEIGHT_TITLE_CATALOG_CACHE
    if _LIGHTWEIGHT_TITLE_CATALOG_CACHE:
        return [dict(movie) for movie in _LIGHTWEIGHT_TITLE_CATALOG_CACHE]

    movies_path = DATA_DIR / "movies.csv"
    if not movies_path.exists():
        return []

    try:
        movies_df = pd.read_csv(
            movies_path,
            usecols=["movieId", "title", "genres"],
        )
    except Exception as exc:
        log.warning("Lightweight movies.csv load failed: %s", exc)
        return []

    movies_df["genres"] = movies_df["genres"].fillna("").str.replace("|", " ", regex=False)
    _LIGHTWEIGHT_TITLE_CATALOG_CACHE = [
        _movie_search_candidate(
            title=str(row.title),
            genres=str(row.genres or ""),
            movie_id=row.movieId,
            source="ml25m_csv",
        )
        for row in movies_df.itertuples(index=False)
    ]
    return [dict(movie) for movie in _LIGHTWEIGHT_TITLE_CATALOG_CACHE]


def _row_to_movie_payload(row: Any) -> dict[str, Any]:
    return _candidate_to_movie_payload(_row_to_movie_candidate(row))


def _search_movies_from_lightweight_catalog(
    query: str,
    limit: int = 50,
    exclude_keys: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    normalized_query = _normalize_search_text(query)
    if not normalized_query or limit <= 0:
        return []

    compact_query = normalized_query.replace(" ", "")
    tokens = [token for token in normalized_query.split() if len(token) > 1][:4]
    exclude_keys = exclude_keys or set()
    catalog = _lightweight_catalog_movies()

    candidates = [
        movie
        for movie in catalog
        if str(movie.get("movie_key") or "") not in exclude_keys
        and (
            normalized_query in str(movie.get("_search_title") or "")
            or normalized_query in str(movie.get("_search_blob") or "")
            or (compact_query and compact_query in str(movie.get("_compact_title") or ""))
            or any(
                token in str(movie.get("_search_title") or "")
                or token in str(movie.get("_search_blob") or "")
                for token in tokens
            )
        )
    ]

    if not candidates:
        candidates = [
            movie
            for movie in catalog
            if str(movie.get("movie_key") or "") not in exclude_keys
            and _sequence_similarity(
                normalized_query,
                str(movie.get("_search_title") or ""),
            )
            >= 0.45
        ]

    ranked_candidates = _rank_search_candidates(candidates, query, limit=limit)
    return [_candidate_to_movie_payload(candidate) for candidate in ranked_candidates]


def _hydrate_movie_payload(movie: dict[str, Any]) -> dict[str, Any]:
    title = str(movie.get("title") or movie.get("clean_title") or "").strip()
    clean_title = str(movie.get("clean_title") or title).strip()
    year = str(movie.get("year") or "").strip()
    genres = str(movie.get("genres") or "").strip()
    poster = str(movie.get("poster") or "").strip()

    try:
        trending_score = float(movie.get("trending_score") or 0)
    except Exception:
        trending_score = 0.0

    poster_override = poster
    if _is_generated_poster(poster) or _is_missing_poster(poster):
        poster_override = ""

    return _movie_payload(
        title=title or clean_title,
        clean_title=clean_title or title,
        year=year,
        genres=genres,
        avg_rating=movie.get("rating") or movie.get("imdb_rating") or 0,
        trending_score=trending_score,
        movie_id=movie.get("movie_id"),
        poster_override=poster_override,
    )


def _hydrate_visible_movie_posters(
    movies: list[dict[str, Any]],
    max_items: int = HOME_POSTER_HYDRATE_LIMIT,
) -> list[dict[str, Any]]:
    if not movies or max_items <= 0:
        return movies

    hydrated_movies = [dict(movie) for movie in movies]
    pending_indices = [
        index
        for index, movie in enumerate(hydrated_movies[:max_items])
        if _is_generated_poster(str(movie.get("poster") or ""))
        or _is_missing_poster(str(movie.get("poster") or ""))
    ]

    if not pending_indices:
        return hydrated_movies

    worker_count = min(max(1, OMDB_PREFETCH_WORKERS), len(pending_indices))

    def _resolve(index: int) -> tuple[int, dict[str, Any]]:
        try:
            return index, _hydrate_movie_payload(hydrated_movies[index])
        except Exception as exc:
            log.debug("Poster hydration failed for %s: %s", hydrated_movies[index], exc)
            return index, hydrated_movies[index]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index, hydrated_movie in executor.map(_resolve, pending_indices):
            hydrated_movies[index] = hydrated_movie

    return hydrated_movies


def _curated_movies_for_genre(
    genre: Optional[str],
    exclude_keys: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    normalized_genre = _normalize_search_text(genre or "")
    curated_results: list[dict[str, Any]] = []

    for movie in _curated_catalog_movies():
        movie_key = str(movie.get("movie_key") or "")
        if movie_key in exclude_keys:
            continue
        if normalized_genre and normalized_genre not in _normalize_search_text(str(movie.get("genres") or "")):
            continue
        curated_results.append(dict(movie))
        if len(curated_results) >= limit:
            break

    return curated_results


def _search_movies_from_database(query: str, limit: int = 50) -> list[dict[str, Any]]:
    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        return []

    like_query = f"%{normalized_query.replace(' ', '%')}%"
    tokens = [token for token in normalized_query.split() if len(token) > 1][:4]
    where_clauses = ["lower(title) LIKE ?", "lower(genres) LIKE ?"]
    params: list[Any] = [like_query, like_query]

    for token in tokens:
        token_like = f"%{token}%"
        where_clauses.extend(["lower(title) LIKE ?", "lower(genres) LIKE ?"])
        params.extend([token_like, token_like])

    candidate_limit = min(max(limit * 8, 250), 500)
    rows: list[Any] = []

    try:
        with get_db() as conn:
            rows = conn.execute(
                f"""
                SELECT movieId, title, genres, avg_rating, num_ratings, trending_score, poster, source
                FROM movies
                WHERE {' OR '.join(where_clauses)}
                ORDER BY
                    CASE WHEN lower(coalesce(source, '')) = 'admin' THEN 0 ELSE 1 END,
                    CASE WHEN num_ratings > 0 THEN 0 ELSE 1 END,
                    trending_score DESC,
                    avg_rating DESC,
                    title ASC
                LIMIT ?
                """,
                tuple(params + [candidate_limit]),
            ).fetchall()
    except Exception as exc:
        log.warning("Movie search DB load failed: %s", exc)

    ranked_candidates = _rank_search_candidates(
        [_row_to_movie_candidate(row) for row in rows],
        query,
        limit=limit,
    )
    results = [_candidate_to_movie_payload(candidate) for candidate in ranked_candidates]

    if len(results) < min(limit, 5):
        known_keys = {
            str(movie.get("movie_key") or "")
            for movie in results
            if str(movie.get("movie_key") or "")
        }
        fallback_results = _search_movies_from_lightweight_catalog(
            query,
            limit=max(0, limit - len(results)),
            exclude_keys=known_keys,
        )
        results.extend(fallback_results[: max(0, limit - len(results))])

    if len(results) < min(limit, 5):
        known_keys = {str(movie.get("movie_key") or "") for movie in results}
        curated_results = [
            movie
            for movie in _search_curated_catalog(query, limit=max(limit * 2, 50))
            if str(movie.get("movie_key") or "") not in known_keys
        ]
        results.extend(curated_results[: max(0, limit - len(results))])

    return results[:limit]


def _recommend_from_curated_catalog(
    title: str,
    top_n: int = 50,
    mood: Optional[str] = None,
    user_email: Optional[str] = None,
) -> dict[str, Any]:
    anchor = _resolve_curated_title(title)
    if anchor is None:
        return {"resolved_title": title, "mood_applied": mood, "results": []}

    anchor_genres = set(_normalize_search_text(str(anchor.get("genres") or "")).split())
    anchor_tokens = _meaningful_title_tokens(
        str(anchor.get("clean_title") or anchor.get("title") or "")
    )
    mood_genres = {
        _normalize_search_text(genre)
        for genre in MOOD_GENRE_MAP.get((mood or "").lower(), [])
    }

    ranked: list[tuple[float, dict[str, Any]]] = []
    for movie in _curated_catalog_movies():
        if movie.get("movie_key") == anchor.get("movie_key"):
            continue

        movie_genres = set(_normalize_search_text(str(movie.get("genres") or "")).split())
        movie_tokens = _meaningful_title_tokens(
            str(movie.get("clean_title") or movie.get("title") or "")
        )

        shared_genres = len(anchor_genres & movie_genres)
        shared_title_tokens = len(anchor_tokens & movie_tokens)
        full_title_match_bonus = (
            40 if anchor_tokens and shared_title_tokens == len(anchor_tokens) else 0
        )
        mood_bonus = 10 if mood_genres & movie_genres else 0

        same_era = 0
        anchor_year = str(anchor.get("year") or "")
        movie_year = str(movie.get("year") or "")
        if anchor_year.isdigit() and movie_year.isdigit():
            same_era = max(0, 8 - min(abs(int(anchor_year) - int(movie_year)), 8))

        try:
            numeric_rating = float(movie.get("rating") or 0)
        except Exception:
            numeric_rating = 0.0

        score = (
            shared_genres * 22
            + shared_title_tokens * 24
            + full_title_match_bonus
            + mood_bonus
            + same_era
            + numeric_rating
        )
        ranked.append((score, movie))

    ranked.sort(key=lambda item: item[0], reverse=True)
    results = _apply_preference_ranking(
        [dict(movie) for score, movie in ranked if score > 0],
        user_email,
        limit=top_n,
    )
    return {
        "resolved_title": anchor.get("title") or title,
        "mood_applied": mood,
        "results": results,
    }


def _recommend_from_database(
    title: str,
    top_n: int = 50,
    mood: Optional[str] = None,
    user_email: Optional[str] = None,
) -> dict[str, Any]:
    anchor = next(iter(_search_movies_from_database(title, limit=1)), None)
    if anchor is None:
        return _recommend_from_curated_catalog(title, top_n, mood, user_email=user_email)

    preference_context = _load_user_preference_context(user_email)
    anchor_genres = {
        token
        for token in _normalize_search_text(str(anchor.get("genres") or "")).split()
        if len(token) > 2
    }
    anchor_tokens = _meaningful_title_tokens(
        str(anchor.get("clean_title") or anchor.get("title") or "")
    )
    mood_genres = {
        _normalize_search_text(genre)
        for genre in MOOD_GENRE_MAP.get((mood or "").lower(), [])
    }
    genre_terms = list(anchor_genres)[:4]
    candidate_limit = min(max(top_n * 8, 250), 500)
    if preference_context is not None:
        candidate_limit = min(max(top_n * 10, 400), 800)

    query = """
        SELECT movieId, title, genres, avg_rating, num_ratings, trending_score, poster
        FROM movies
    """
    params: list[Any] = []

    if genre_terms:
        query += " WHERE " + " OR ".join("lower(genres) LIKE ?" for _ in genre_terms)
        params.extend([f"%{term}%" for term in genre_terms])

    query += """
        ORDER BY
            CASE WHEN num_ratings > 0 THEN 0 ELSE 1 END,
            trending_score DESC,
            avg_rating DESC,
            title ASC
        LIMIT ?
    """
    params.append(candidate_limit)

    rows: list[Any] = []
    try:
        with get_db() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
    except Exception as exc:
        log.warning("Recommendation DB load failed: %s", exc)

    anchor_key = str(anchor.get("movie_key") or "")
    anchor_year = str(anchor.get("year") or "")
    anchor_title = str(anchor.get("clean_title") or anchor.get("title") or title)
    ranked: list[tuple[float, dict[str, Any]]] = []
    candidate_movies: list[dict[str, Any]] = [
        movie
        for movie in _search_movies_from_database(anchor_title, limit=max(top_n * 2, 20))
        if str(movie.get("movie_key") or "") != anchor_key
    ]
    candidate_movies.extend(_row_to_movie_payload(row) for row in rows)
    seen_candidate_keys: set[str] = set()

    for movie in candidate_movies:
        movie_key = str(movie.get("movie_key") or "")
        if not movie_key or movie_key == anchor_key or movie_key in seen_candidate_keys:
            continue
        seen_candidate_keys.add(movie_key)

        movie_genres = {
            token
            for token in _normalize_search_text(str(movie.get("genres") or "")).split()
            if len(token) > 2
        }
        movie_tokens = _meaningful_title_tokens(
            str(movie.get("clean_title") or movie.get("title") or "")
        )

        shared_genres = len(anchor_genres & movie_genres)
        shared_title_tokens = len(anchor_tokens & movie_tokens)
        full_title_match_bonus = (
            40 if anchor_tokens and shared_title_tokens == len(anchor_tokens) else 0
        )
        mood_bonus = 10 if mood_genres & movie_genres else 0

        same_era = 0
        movie_year = str(movie.get("year") or "")
        if anchor_year.isdigit() and movie_year.isdigit():
            same_era = max(0, 8 - min(abs(int(anchor_year) - int(movie_year)), 8))

        try:
            numeric_rating = float(movie.get("rating") or 0)
        except Exception:
            numeric_rating = 0.0

        try:
            trending_bonus = float(movie.get("trending_score") or 0) * 8.0
        except Exception:
            trending_bonus = 0.0

        score = (
            shared_genres * 22
            + shared_title_tokens * 24
            + full_title_match_bonus
            + mood_bonus
            + same_era
            + numeric_rating
            + trending_bonus
        )
        ranked.append((score, movie))

    ranked.sort(
        key=lambda item: (
            item[0],
            float(item[1].get("rating") or 0),
            float(item[1].get("trending_score") or 0),
        ),
        reverse=True,
    )

    preference_context = _load_user_preference_context(user_email)
    rerank_pool_size = top_n
    if preference_context is not None:
        rerank_pool_size = min(len(ranked), max(top_n * 3, 75))

    results = _apply_preference_ranking(
        [dict(movie) for _, movie in ranked[:rerank_pool_size]],
        user_email,
        limit=top_n,
    )
    if len(results) < top_n:
        known_keys = {str(movie.get("movie_key") or "") for movie in results}
        known_keys.add(anchor_key)
        curated_results = [
            movie
            for movie in _recommend_from_curated_catalog(
                title,
                max(top_n * 2, 50),
                mood,
                user_email=user_email,
            )["results"]
            if str(movie.get("movie_key") or "") not in known_keys
        ]
        results.extend(curated_results[: max(0, top_n - len(results))])

    results = _apply_preference_ranking(results, user_email, limit=top_n)
    return {
        "resolved_title": str(anchor.get("clean_title") or anchor.get("title") or title),
        "mood_applied": mood,
        "results": results[:top_n],
    }


def prefetch_omdb_cache(
    limit: int | None = None,
    batch_size: int = OMDB_PREFETCH_BATCH_SIZE,
) -> dict[str, int]:
    if batch_size <= 0:
        batch_size = OMDB_PREFETCH_BATCH_SIZE

    init_db()
    processed = 0
    missing = 0
    cached = 0

    with get_db() as conn:
        cached_keys = {
            str(row["title"])
            for row in conn.execute("SELECT title FROM omdb_cache").fetchall()
        }
        query = """
            SELECT title
            FROM movies
            ORDER BY num_ratings DESC, avg_rating DESC, title ASC
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(query, params).fetchall()

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        pending: list[tuple[str, str]] = []
        for row in batch:
            title = str(row["title"])
            clean_title, year = _clean_title(title)
            cache_key = f"{clean_title}|{year}"
            processed += 1
            if cache_key in cached_keys:
                cached += 1
                continue
            pending.append((clean_title, year))

        if not pending:
            continue

        missing += len(pending)
        workers = min(OMDB_PREFETCH_WORKERS, len(pending))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            list(executor.map(lambda movie: RecommenderEngine._fetch_omdb(*movie), pending))

    return {
        "processed": processed,
        "missing": missing,
        "cached": cached,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def search_movies(q: str, limit: int = 50) -> list:
    return _search_movies_from_database(q, limit=limit)

def recommend_movies(
    title: str,
    user_id: int = 1,
    mood: Optional[str] = None,
    top_n: int = 50,
    user_email: Optional[str] = None,
) -> dict:
    engine = RecommenderEngine._instance
    if engine is not None and getattr(engine, "_ready", False):
        return engine.recommend(
            title,
            user_id,
            top_n=top_n,
            mood=mood,
            user_email=user_email,
        )
    return _recommend_from_database(
        title,
        top_n=top_n,
        mood=mood,
        user_email=user_email,
    )

def browse_mood(mood: str) -> list:
    return RecommenderEngine.get().browse_by_mood(mood)

def get_model_metrics() -> dict:
    report_metrics = _metrics_from_eval_report()
    engine = RecommenderEngine._instance
    if engine is not None and getattr(engine, "_ready", False):
        engine_metrics = engine.get_metrics()
        if engine_metrics:
            combined_metrics = dict(report_metrics)
            combined_metrics.update(dict(engine_metrics))
            return combined_metrics

    latest_db_metrics = _latest_model_metrics_from_db()
    combined_metrics = dict(report_metrics)
    combined_metrics.update(latest_db_metrics)

    report_timestamp = str(combined_metrics.get("report_generated_at") or "").strip()
    if report_timestamp:
        combined_metrics.setdefault("updated_at", report_timestamp)
        combined_metrics.setdefault(
            "run_id",
            f"saved-report-{report_timestamp.replace(':', '-').replace(' ', '_')}",
        )

    if not combined_metrics.get("model"):
        available_models = cast(list[str], combined_metrics.get("available_models") or [])
        combined_metrics["model"] = (
            ", ".join(available_models)
            if available_models
            else "Saved evaluation report"
        )

    return combined_metrics


def _plot_style(theme: str) -> dict[str, str]:
    if str(theme or "").strip().lower() == "dark":
        return {
            "figure": "#09090b",
            "axes": "#18181b",
            "grid": "#3f3f46",
            "text": "#f4f4f5",
            "muted": "#a1a1aa",
            "ready": "#22c55e",
            "missing": "#ef4444",
        }
    return {
        "figure": "#ffffff",
        "axes": "#f8fafc",
        "grid": "#d4d4d8",
        "text": "#18181b",
        "muted": "#52525b",
        "ready": "#16a34a",
        "missing": "#dc2626",
    }


def _apply_plot_theme(ax: Any, style: dict[str, str]):
    ax.set_facecolor(style["axes"])
    ax.figure.patch.set_facecolor(style["figure"])
    ax.tick_params(colors=style["muted"])
    for spine in ax.spines.values():
        spine.set_color(style["grid"])
    ax.grid(axis="y", color=style["grid"], alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)


def _annotate_bars(ax: Any, bars: Any, values: list[float], style: dict[str, str], suffix: str = ""):
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + (bar.get_width() / 2),
            value + max(value * 0.02, 0.02),
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=style["text"],
        )


def render_model_metrics_plot(kind: str, theme: str = "light") -> bytes | None:
    normalized_kind = str(kind or "").strip().lower()
    if normalized_kind == "engine":
        normalized_kind = "availability"
    if normalized_kind not in {"comparison", "loss", "availability"}:
        return None
    if not HAS_MATPLOTLIB or plt is None:
        return None

    metrics = get_model_metrics()
    comparison = cast(list[dict[str, Any]], metrics.get("comparison") or [])
    style = _plot_style(theme)

    fig = None
    try:
        if normalized_kind == "comparison":
            metric_series = [
                ("auc", "AUC %", "#2563eb"),
                ("f1", "F1 %", "#16a34a"),
                ("precision", "Precision %", "#f59e0b"),
                ("recall", "Recall %", "#ef4444"),
            ]
            labels = [str(entry.get("model") or "").strip() for entry in comparison]
            if not labels:
                return None

            fig, ax = plt.subplots(figsize=(10.5, 5.25), constrained_layout=True)
            _apply_plot_theme(ax, style)
            x = np.arange(len(labels), dtype=float)
            width = 0.18
            offset_origin = (len(metric_series) - 1) / 2
            has_values = False

            for index, (metric_key, label, color) in enumerate(metric_series):
                values: list[float] = []
                for entry in comparison:
                    raw_value = entry.get(metric_key)
                    if isinstance(raw_value, (int, float)) and np.isfinite(raw_value):
                        has_values = True
                        values.append(float(raw_value) * 100.0)
                    else:
                        values.append(np.nan)
                bars = ax.bar(
                    x + ((index - offset_origin) * width),
                    values,
                    width=width,
                    label=label,
                    color=color,
                )
                _annotate_bars(ax, bars, values, style, suffix="%")

            if not has_values:
                return None

            ax.set_xticks(x, labels)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Score", color=style["text"])
            ax.set_xlabel("Model", color=style["text"])
            ax.set_title("Saved Evaluation Quality", color=style["text"], fontsize=14, fontweight="bold")
            legend = ax.legend(frameon=True, facecolor=style["figure"], edgecolor=style["grid"])
            for text in legend.get_texts():
                text.set_color(style["text"])

        elif normalized_kind == "loss":
            loss_entries = [
                (
                    str(entry.get("model") or "").strip(),
                    float(entry["loss"]),
                    str(entry.get("loss_label") or "Loss").strip(),
                )
                for entry in comparison
                if isinstance(entry.get("loss"), (int, float)) and np.isfinite(entry["loss"])
            ]
            if not loss_entries:
                return None

            labels = [f"{model}\n{loss_label}" for model, _, loss_label in loss_entries]
            values = [loss_value for _, loss_value, _ in loss_entries]
            colors = ["#f97316", "#fb7185", "#8b5cf6", "#14b8a6"][: len(values)]

            fig, ax = plt.subplots(figsize=(8.5, 5.25), constrained_layout=True)
            _apply_plot_theme(ax, style)
            bars = ax.bar(labels, values, color=colors, width=0.6)
            _annotate_bars(ax, bars, values, style)
            ax.set_ylabel("Loss", color=style["text"])
            ax.set_xlabel("Stage", color=style["text"])
            ax.set_title("Latest Ranking Loss Snapshot", color=style["text"], fontsize=14, fontweight="bold")

        else:
            available_models = {
                str(model).strip()
                for model in cast(list[str], metrics.get("available_models") or [])
                if str(model).strip()
            }
            missing_models = {
                str(model).strip()
                for model in cast(list[str], metrics.get("missing_models") or [])
                if str(model).strip()
            }
            labels = sorted(available_models | missing_models)
            if not labels:
                return None

            values = [1.0 if label in available_models else 0.0 for label in labels]
            colors = [
                style["ready"] if label in available_models else style["missing"]
                for label in labels
            ]

            fig, ax = plt.subplots(figsize=(8.5, 5.25), constrained_layout=True)
            _apply_plot_theme(ax, style)
            bars = ax.bar(labels, values, color=colors, width=0.6)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + (bar.get_width() / 2),
                    value + 0.04,
                    "Ready" if value >= 1 else "Missing",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=style["text"],
                )
            ax.set_ylim(0, 1.2)
            ax.set_yticks([0, 1], ["Missing", "Ready"])
            ax.set_ylabel("Live status", color=style["text"])
            ax.set_xlabel("Recommender component", color=style["text"])
            ax.set_title("Live Recommender Stack", color=style["text"], fontsize=14, fontweight="bold")

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=160,
            bbox_inches="tight",
            facecolor=style["figure"],
        )
        return buffer.getvalue()
    finally:
        if fig is not None and plt is not None:
            plt.close(fig)

def record_feedback(user_id: str, movie_id: int, feedback: str) -> bool:
    if feedback not in ("like", "dislike", "neutral"):
        return False
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO user_feedback "
                "(user_id,movieId,feedback,ts) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id,movieId) DO UPDATE SET "
                "feedback = excluded.feedback, "
                "ts = excluded.ts",
                (user_id, movie_id, feedback, _utc_now().isoformat()))
        return True
    except Exception as e:
        log.error("Feedback error: %s", e)
        return False

def add_movies_to_db(movies_list: list) -> int:
    if not movies_list:
        return 0
    rows = []
    for i, m in enumerate(movies_list):
        title = str(m.get("title", "")).strip()
        if not title:
            continue
        year = str(m.get("year", "")).strip()
        normalized_title = f"{title} ({year})" if year and "(" not in title else title
        clean_title, parsed_year = _clean_title(normalized_title)
        resolved_year = parsed_year or year
        genres = str(m.get("genres", "")).strip().replace("|", " ")
        poster_override = str(m.get("poster", "")).strip()
        try:
            avg_r = float(m.get("rating", 0) or 0)
        except Exception:
            avg_r = 0.0

        lookup_title = _normalize_lookup_title(clean_title or title)
        omdb_payload = RecommenderEngine._fetch_omdb(lookup_title, resolved_year)
        if poster_override and not _is_missing_poster(poster_override):
            resolved_poster = poster_override
        else:
            resolved_poster = str(omdb_payload.get("poster") or "").strip()
        if _is_missing_poster(resolved_poster):
            resolved_poster = _generated_poster_url(clean_title or title, resolved_year, genres)

        rows.append({
            "movieId":          -(i + 1 + int(pd.Timestamp.now().timestamp())),
            "title":            normalized_title,
            "genres":           genres,
            "avg_rating":       avg_r,
            "num_ratings":      1,
            "sentiment_score":  0.0,
            "trending_score":   0.0,
            "poster":           resolved_poster,
            "source":           "admin",
        })
    if not rows:
        return 0
    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO movies (
                movieId,
                title,
                genres,
                avg_rating,
                num_ratings,
                sentiment_score,
                trending_score,
                poster,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["movieId"],
                    row["title"],
                    row["genres"],
                    row["avg_rating"],
                    row["num_ratings"],
                    row["sentiment_score"],
                    row["trending_score"],
                    row["poster"],
                    row["source"],
                )
                for row in rows
            ],
        )
    _HOME_MOVIES_CACHE.clear()
    RecommenderEngine.reset()
    return len(rows)


def _admin_catalog_filters(
    search: Optional[str] = None,
    genre: Optional[str] = None,
) -> tuple[str, list[Any]]:
    normalized_search = _normalize_search_text(search or "")
    normalized_genre = _normalize_search_text(genre or "")
    title_expression = (
        "replace(replace(replace(replace(lower(title), '-', ' '), ':', ' '), ',', ' '), '|', ' ')"
    )
    genres_expression = "replace(replace(lower(genres), '-', ' '), '|', ' ')"

    clauses: list[str] = []
    params: list[Any] = []

    search_tokens = [token for token in normalized_search.split() if token][:6]
    if search_tokens:
        token_clauses: list[str] = []
        for token in search_tokens:
            token_like = f"%{token}%"
            token_clauses.append(
                f"({title_expression} LIKE ? OR {genres_expression} LIKE ?)"
            )
            params.extend([token_like, token_like])
        clauses.append("(" + " AND ".join(token_clauses) + ")")

    genre_tokens = [token for token in normalized_genre.split() if token][:4]
    if genre_tokens:
        clauses.append(
            "(" + " AND ".join(f"{genres_expression} LIKE ?" for _ in genre_tokens) + ")"
        )
        params.extend([f"%{token}%" for token in genre_tokens])

    if not clauses:
        return "", params

    return " WHERE " + " AND ".join(clauses), params


def _admin_catalog_genres() -> list[str]:
    genres: set[str] = set()
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT genres
                FROM movies
                WHERE trim(genres) != ''
                """
            ).fetchall()
    except Exception as exc:
        log.debug("Admin genre load failed: %s", exc)
        return []

    for row in rows:
        raw_value = str(row["genres"] or "")
        if "no genres listed" in raw_value.lower():
            continue
        for genre_value in raw_value.replace("|", " ").replace(",", " ").split():
            cleaned = genre_value.strip()
            if cleaned:
                genres.add(cleaned)

    return sorted(genres)


def list_admin_movies(
    limit: int | None = None,
    offset: int = 0,
    search: Optional[str] = None,
    genre: Optional[str] = None,
) -> dict[str, Any]:
    where_sql, params = _admin_catalog_filters(search=search, genre=genre)
    safe_offset = max(0, int(offset or 0))

    count_query = f"SELECT COUNT(*) AS total FROM movies{where_sql}"
    query = """
        SELECT movieId, title, genres, avg_rating, poster, source
        FROM movies
    """
    query += where_sql
    query += """
        ORDER BY
            CASE WHEN source = 'admin' THEN 0 ELSE 1 END,
            CASE WHEN source = 'admin' THEN -movieId ELSE movieId END DESC,
            avg_rating DESC,
            title ASC
    """

    with get_db() as conn:
        count_result = conn.execute(count_query, tuple(params)).fetchone()
        total = int(count_result["total"] or 0) if count_result else 0
        row_params: list[Any] = list(params)
        if limit is not None:
            safe_limit = max(1, int(limit))
            query += " LIMIT ? OFFSET ?"
            row_params.extend([safe_limit, safe_offset])
        rows = conn.execute(query, tuple(row_params)).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        source = str(row["source"] or "").strip().lower() or "catalog"
        movie = _row_to_movie_payload(row)
        movie.update({
            "source": source,
            "source_label": "Admin" if source == "admin" else "Catalog",
            "can_delete": source == "admin",
        })
        items.append(movie)

    return {
        "items": items,
        "total": total,
        "limit": limit if limit is not None else total,
        "offset": safe_offset,
        "genres": _admin_catalog_genres(),
        "has_more": (safe_offset + len(items)) < total,
    }


def delete_movie_from_db(movie_id: int) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT source FROM movies WHERE movieId = ?",
            (movie_id,),
        ).fetchone()
        if not row:
            return False
        if str(row["source"] or "").strip().lower() != "admin":
            return False

        conn.execute("DELETE FROM tags WHERE movieId = ?", (movie_id,))
        conn.execute("DELETE FROM genome_scores WHERE movieId = ?", (movie_id,))
        conn.execute("DELETE FROM rating_timestamps WHERE movieId = ?", (movie_id,))
        conn.execute("DELETE FROM user_feedback WHERE movieId = ?", (movie_id,))
        conn.execute("DELETE FROM movies WHERE movieId = ?", (movie_id,))

    _HOME_MOVIES_CACHE.clear()
    RecommenderEngine.reset()
    return True


def get_home_movies(
    limit: int = 50,
    genre: Optional[str] = None,
    user_email: Optional[str] = None,
) -> list[dict[str, Any]]:
    now = _utc_now().timestamp()
    normalized_genre = _normalize_search_text(genre or "")
    normalized_user = _normalize_search_text(user_email or "")
    cache_key = f"{normalized_genre or 'all'}::{limit}::{normalized_user or 'anon'}"
    cached_entry = _HOME_MOVIES_CACHE.get(cache_key, {})
    cached_results = cached_entry.get("results", [])
    if cached_results and now < float(cached_entry.get("expires_at", 0)):
        return cast(list[dict[str, Any]], cached_results[:limit])

    candidate_limit = limit
    if _load_user_preference_context(user_email) is not None:
        # Pull a wider pool so preference reranking can surface better matches.
        candidate_limit = min(max(limit * 8, 160), 500)

    query = """
        SELECT movieId, title, genres, avg_rating, num_ratings, trending_score, poster, source
        FROM movies
    """
    params: list[Any] = []
    if normalized_genre:
        genre_expression = "replace(replace(lower(genres), '-', ' '), '|', ' ')"
        genre_tokens = [token for token in normalized_genre.split() if token]
        query += " WHERE " + " AND ".join(f"{genre_expression} LIKE ?" for _ in genre_tokens)
        params.extend([f"%{token}%" for token in genre_tokens])
    query += """
        ORDER BY
            CASE WHEN source = 'admin' THEN 0 ELSE 1 END,
            CASE WHEN num_ratings > 0 THEN 0 ELSE 1 END,
            trending_score DESC,
            avg_rating DESC,
            title ASC
        LIMIT ?
    """
    params.append(candidate_limit)

    rows: list[Any] = []
    try:
        with get_db() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
    except Exception as exc:
        log.warning("Home movies DB load failed: %s", exc)

    results = [_row_to_movie_payload(row) for row in rows]

    if len(results) < candidate_limit:
        known_keys = {str(movie.get("movie_key") or "") for movie in results}
        results.extend(
            _curated_movies_for_genre(
                genre,
                known_keys,
                candidate_limit - len(results),
            )
        )

    if not results:
        results = _curated_movies_for_genre(genre, set(), candidate_limit)

    results = _hydrate_visible_movie_posters(
        results,
        max_items=min(limit, HOME_POSTER_HYDRATE_LIMIT),
    )
    results = _apply_preference_ranking(results, user_email, limit=limit)

    _HOME_MOVIES_CACHE[cache_key] = {
        "results": results[:limit],
        "expires_at": now + HOME_CACHE_TTL_SECONDS,
    }
    return results[:limit]


# ── bootstrap ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Initialising DB …")
    init_db()
    load_ml25m_to_db()
    log.info("Warming engine …")
    RecommenderEngine.get()
