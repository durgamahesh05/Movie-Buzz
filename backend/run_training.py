"""
run_training.py  -  MovieBuzz One-Click Training Script
========================================================
Run this ONCE before starting the server:

    python run_training.py

What it does (in order):
  1.  Checks all required data files exist
  2.  Initialises the SQLite database
  3.  Loads movies.csv + aggregates ratings in 500K chunks (RAM-safe)
  4.  Loads tags + sentiment scoring
  5.  Loads genome scores
  6.  Temporal train/test split (last 20% of each user's ratings = test)
  7.  Trains SVD   (scikit-surprise)
  8.  Trains ALS   (implicit)
  9.  Trains NCF   (TensorFlow / NeuMF)
  10. Trains XGBoost re-ranker
  11. Evaluates all models and saves models/eval_report.json
  12. Prints a full report

All models are saved to models/ and reused on next run (skip if already exist).
Set FORCE_RETRAIN=1 env var to retrain everything from scratch.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import logging
import pickle
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import env_path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("run_training")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DB_PATH   = env_path("DATABASE_URL", "DB_PATH", "MOVIEBUZZ_DB_PATH", default=BASE_DIR / "moviebuzz.db")
MODEL_DIR.mkdir(exist_ok=True)

# ── Tunable sample sizes (reduce for faster runs, increase for accuracy) ──────
CHUNK              = 500_000      # rows per CSV chunk
SVD_SAMPLE_ROWS    = int(os.getenv("SVD_SAMPLE_ROWS",  "10000000"))
ALS_SAMPLE_ROWS    = int(os.getenv("ALS_SAMPLE_ROWS",  "5000000"))
NCF_SAMPLE_ROWS    = int(os.getenv("NCF_SAMPLE_ROWS",  "600000"))
XGB_SAMPLE_ROWS    = int(os.getenv("XGB_SAMPLE_ROWS",  "500000"))
FORCE_RETRAIN      = os.getenv("FORCE_RETRAIN", "0") == "1"
FORCE_XGB_RETRAIN  = os.getenv("FORCE_XGB_RETRAIN", "0") == "1"
XGB_ENABLE_RANK_OBJECTIVE = os.getenv("XGB_ENABLE_RANK_OBJECTIVE", "0") == "1"
NEG_RATIO          = int(os.getenv("NCF_NEG_RATIO", "10"))
TEST_RATIO         = 0.20
NCF_POSITIVE_RATING = float(os.getenv("NCF_POSITIVE_RATING", "4.0"))
NCF_EPOCHS          = int(os.getenv("NCF_EPOCHS", "60"))
NCF_EMBED_DIM       = int(os.getenv("NCF_MF_DIM", os.getenv("NCF_EMBED_DIM", "96")))
NCF_MLP_LAYERS      = (256, 128, 64, 32)
NCF_DROPOUT         = float(os.getenv("NCF_DROPOUT", "0.35"))
NCF_BATCH_SIZE      = int(os.getenv("NCF_BATCH_SIZE", "4096"))
NCF_PATIENCE        = int(os.getenv("NCF_PATIENCE", "5"))
SVD_N_FACTORS       = int(os.getenv("SVD_N_FACTORS", "150"))
SVD_N_EPOCHS        = int(os.getenv("SVD_N_EPOCHS", "30"))
SVD_LR_ALL          = float(os.getenv("SVD_LR_ALL", "0.007"))
SVD_REG_ALL         = float(os.getenv("SVD_REG_ALL", "0.02"))
XGB_VALIDATION_RATIO = float(os.getenv("XGB_VALIDATION_RATIO", "0.2"))
XGB_EARLY_STOPPING_ROUNDS = int(os.getenv("XGB_EARLY_STOPPING_ROUNDS", "40"))
XGB_MAX_GENRE_FEATURES = int(os.getenv("XGB_MAX_GENRE_FEATURES", "24"))
GENOME_TOP_TAG_FEATURES = int(os.getenv("XGB_GENOME_TOP_TAGS", "10"))
EVAL_SAMPLE_ROWS = int(os.getenv("EVAL_SAMPLE_ROWS", "250000"))
XGB_USER_STATS_PATH = MODEL_DIR / "user_feature_stats.pkl"
XGB_GENOME_STATS_PATH = MODEL_DIR / "movie_genome_stats.pkl"
XGB_FEATURE_CONTEXT_PATH = MODEL_DIR / "xgb_feature_context.pkl"
XGB_MODEL_JSON_PATH = MODEL_DIR / "xgb_ranker.json"
XGB_MODEL_META_PATH = MODEL_DIR / "xgb_ranker_meta.pkl"
XGB_LEGACY_PATH = MODEL_DIR / "xgb_ranker.pkl"
NCF_WEIGHTS_PATH = MODEL_DIR / "ncf_model.weights.h5"
NCF_META_PATH = MODEL_DIR / "ncf_model_meta.pkl"
MOVIE_SBERT_EMBEDDINGS_PATH = MODEL_DIR / "movie_sbert_embeddings.pkl"
USER_TASTE_VECTORS_PATH = MODEL_DIR / "user_taste_vectors.npy"
USER_TASTE_ID_MAP_PATH = MODEL_DIR / "user_taste_id_map.pkl"
TRAIN_USE_GPU = os.getenv("TRAIN_USE_GPU", "1") == "1"
LEGACY_XGB_FEATURE_COLS = [
    "avg_rating",
    "svd_score",
    "log_num_ratings",
    "sentiment_signal",
    "trending_score",
]
XGB_FEATURE_COLS = [
    "avg_rating",
    "rating_stddev",
    "svd_score",
    "als_score",
    "ncf_score",
    "ensemble_mean",
    "ensemble_std",
    "score_range",
    "log_num_ratings",
    "sentiment_signal",
    "trending_score",
    "user_activity_level",
    "user_avg_rating",
    "user_rating_std",
    "user_recency_days_log",
    "user_recency_score",
    "user_rating_freq_trend",
    "user_movie_rating_diff",
    "genre_affinity_score",
    "item_popularity_decay",
    "item_recent_rating_velocity",
    "tag_cooccurrence_strength",
    "sbert_sim",
    "genome_mean_relevance",
    "genome_max_relevance",
    "genome_high_relevance_log",
]


class XGBModelBundle:
    """Keeps XGBoost inference backward compatible while adding scaling and calibration."""

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
        else:
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_int(value: Any) -> int:
    return int(value)


_GPU_CHECK_COMPLETE = False
_GPU_AVAILABLE = False


def _gpu_available() -> bool:
    global _GPU_CHECK_COMPLETE, _GPU_AVAILABLE
    if _GPU_CHECK_COMPLETE:
        return _GPU_AVAILABLE

    _GPU_CHECK_COMPLETE = True
    if not TRAIN_USE_GPU:
        _GPU_AVAILABLE = False
        return _GPU_AVAILABLE

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        _GPU_AVAILABLE = False
        return _GPU_AVAILABLE

    try:
        probe = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        _GPU_AVAILABLE = probe.returncode == 0 and "GPU" in (probe.stdout or "")
    except Exception:
        _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def _configure_tensorflow_training(tf_module: Any) -> bool:
    if not TRAIN_USE_GPU:
        log.info("NCF GPU training disabled via TRAIN_USE_GPU=0")
        return False

    if not _gpu_available():
        log.info("NCF GPU training unavailable; no GPU detected by nvidia-smi")
        return False

    try:
        gpus = tf_module.config.list_physical_devices("GPU")
    except Exception as exc:
        log.warning("Could not inspect TensorFlow GPU devices: %s", exc)
        return False

    if not gpus:
        log.info("TensorFlow GPU libraries are unavailable in this WSL environment; NCF will train on CPU")
        return False

    for gpu in gpus:
        try:
            tf_module.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    log.info("NCF will train on TensorFlow GPU: %s", ", ".join(device.name for device in gpus))
    return True


def _xgb_runtime_kwargs() -> dict[str, Any]:
    if TRAIN_USE_GPU and _gpu_available():
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "hist"}


def _new_xgb_estimator(xgb_module: Any, model_kind: str) -> Any:
    if model_kind == "rank:ndcg":
        return xgb_module.XGBRanker()
    return xgb_module.XGBClassifier()


def _scikit_surprise_install_guidance() -> list[str]:
    if sys.platform.startswith("win"):
        return [
            "Native Windows needs Microsoft C++ Build Tools for scikit-surprise.",
            "Recommended: run SVD training inside WSL2 or a conda-forge environment.",
            "WSL2: sudo apt-get install build-essential python3-dev && pip install scikit-surprise",
            "Conda: conda create -n moviebuzz-surprise python=3.10 scikit-surprise -c conda-forge",
        ]
    return ["Install with: pip install scikit-surprise"]


def _save_xgb_bundle(bundle: XGBModelBundle) -> None:
    with open(XGB_LEGACY_PATH, "wb") as f:
        pickle.dump(bundle, f)


def _load_xgb_bundle(xgb_module: Any) -> XGBModelBundle | None:
    if XGB_LEGACY_PATH.exists():
        with open(XGB_LEGACY_PATH, "rb") as f:
            legacy_bundle = pickle.load(f)
        if isinstance(legacy_bundle, XGBModelBundle):
            return legacy_bundle

    if XGB_MODEL_JSON_PATH.exists() and XGB_MODEL_META_PATH.exists():
        with open(XGB_MODEL_META_PATH, "rb") as f:
            payload = pickle.load(f)
        model_kind = str(payload.get("model_kind") or "binary:logistic")
        model = _new_xgb_estimator(xgb_module, model_kind)
        model.load_model(str(XGB_MODEL_JSON_PATH))
        return XGBModelBundle(
            model=model,
            scaler=payload["scaler"],
            calibrator=payload.get("calibrator"),
            model_kind=model_kind,
            calibration_kind=str(payload.get("calibration_kind") or "none"),
            threshold=float(payload.get("threshold", 0.5)),
        )
    return None

# ── Optional dependency loader ────────────────────────────────────────────────
def _try_import(name: str) -> Any:
    import importlib, importlib.util
    if not importlib.util.find_spec(name):
        return None
    try:
        return importlib.import_module(name)
    except Exception as exc:
        log.warning("Could not import %s: %s", name, exc)
        return None


# =============================================================================
#  STEP 0 — PRE-FLIGHT CHECKS
# =============================================================================
def preflight():
    log.info("=" * 60)
    log.info("  MovieBuzz Training Script")
    log.info("=" * 60)

    required = [
        DATA_DIR / "movies.csv",
        DATA_DIR / "ratings.csv",
    ]
    optional = [
        DATA_DIR / "tags.csv",
        DATA_DIR / "genome-scores.csv",
        DATA_DIR / "genome-tags.csv",
    ]

    ok = True
    for f in required:
        if f.exists():
            size_mb = f.stat().st_size / 1_048_576
            log.info("  [OK]  %s  (%.1f MB)", f.name, size_mb)
        else:
            log.error("  [MISSING]  %s  ← REQUIRED", f.name)
            ok = False

    for f in optional:
        if f.exists():
            size_mb = f.stat().st_size / 1_048_576
            log.info("  [OK]  %s  (%.1f MB)", f.name, size_mb)
        else:
            log.warning("  [SKIP] %s  (optional – richer recommendations if present)", f.name)

    if not ok:
        log.error("Required files missing. Download MovieLens 25M from:")
        log.error("  https://grouplens.org/datasets/movielens/25m/")
        log.error("Extract and place files in:  %s", DATA_DIR)
        sys.exit(1)

    # Dependency check
    deps = {
        "surprise":            "scikit-surprise (SVD)",
        "implicit":            "implicit (ALS)",
        "tensorflow":          "tensorflow (NCF)",
        "xgboost":             "xgboost (XGBoost re-ranker)",
        "sentence_transformers":"sentence-transformers (SBERT)",
        "textblob":            "textblob (sentiment)",
        "rapidfuzz":           "rapidfuzz (fuzzy search)",
    }
    log.info("-" * 60)
    log.info("  Dependency check:")
    for mod, label in deps.items():
        m = _try_import(mod)
        status = "OK  " if m is not None else "MISS"
        log.info("    [%s]  %s", status, label)
        if mod == "surprise" and m is None:
            for message in _scikit_surprise_install_guidance():
                log.info("           %s", message)
    log.info("-" * 60)


# =============================================================================
#  STEP 1 — DATABASE INIT
# =============================================================================
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-131072")   # 128 MB page cache
    return conn


def init_db():
    log.info("Initialising database: %s", DB_PATH)
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS movies (
                movieId         INTEGER PRIMARY KEY,
                title           TEXT NOT NULL,
                genres          TEXT DEFAULT '',
                avg_rating      REAL DEFAULT 0,
                rating_stddev   REAL DEFAULT 0,
                num_ratings     INTEGER DEFAULT 0,
                sentiment_score REAL DEFAULT 0,
                trending_score  REAL DEFAULT 0,
                poster          TEXT DEFAULT '',
                source          TEXT DEFAULT 'ml25m'
            );
            CREATE TABLE IF NOT EXISTS tags (
                movieId INTEGER,
                tag     TEXT,
                FOREIGN KEY(movieId) REFERENCES movies(movieId)
            );
            CREATE TABLE IF NOT EXISTS genome_scores (
                movieId   INTEGER,
                tagId     INTEGER,
                relevance REAL,
                PRIMARY KEY (movieId, tagId)
            );
            CREATE TABLE IF NOT EXISTS omdb_cache (
                title       TEXT PRIMARY KEY,
                poster      TEXT DEFAULT '',
                plot        TEXT DEFAULT '',
                cast        TEXT DEFAULT '',
                director    TEXT DEFAULT '',
                imdb_rating TEXT DEFAULT '',
                runtime     TEXT DEFAULT '',
                fetched_at  TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS user_feedback (
                user_id  TEXT NOT NULL,
                movieId  INTEGER NOT NULL,
                feedback TEXT CHECK(feedback IN ('like','dislike','neutral')),
                ts       TEXT DEFAULT '',
                PRIMARY KEY (user_id, movieId)
            );
            CREATE TABLE IF NOT EXISTS rating_timestamps (
                movieId    INTEGER PRIMARY KEY,
                latest_ts  INTEGER DEFAULT 0,
                num_recent INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS model_metrics (
                run_id   TEXT PRIMARY KEY,
                model    TEXT,
                bce_loss REAL,
                bpr_loss REAL,
                mse      REAL,
                ndcg_10  REAL,
                auc      REAL,
                ts       TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title);
            CREATE INDEX IF NOT EXISTS idx_tags_mid     ON tags(movieId);
            CREATE INDEX IF NOT EXISTS idx_genome_mid   ON genome_scores(movieId);
        """)
        # Clear old placeholder posters so real ones get fetched
        conn.execute("""
            UPDATE movies SET poster = ''
            WHERE poster LIKE '%placehold.co%'
               OR poster LIKE '%placeholder%'
        """)
        movie_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(movies)").fetchall()
        }
        if "rating_stddev" not in movie_columns:
            conn.execute("ALTER TABLE movies ADD COLUMN rating_stddev REAL DEFAULT 0")
    log.info("Database ready")


# =============================================================================
#  STEP 2 — LOAD MOVIES + AGGREGATE RATINGS IN CHUNKS
# =============================================================================
def load_and_aggregate() -> pd.DataFrame:
    """Load movies.csv and aggregate ratings.csv in 500K-row chunks."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        if count > 0 and not FORCE_RETRAIN:
            log.info("Movies already in DB (%d rows) – skipping import", count)
            return pd.read_sql("SELECT * FROM movies", conn)

    log.info("Loading movies.csv …")
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    movies_df["genres"] = movies_df["genres"].str.replace("|", " ", regex=False)
    movies_df["poster"] = ""
    movies_df["source"] = "ml25m"
    log.info("  %d movies loaded", len(movies_df))

    # ── Aggregate ratings in chunks (never loads all 25M rows at once) ────────
    ratings_path = DATA_DIR / "ratings.csv"
    log.info("Aggregating ratings.csv in chunks of %d …", CHUNK)
    sum_d: Dict[int, float] = {}
    sum_sq_d: Dict[int, float] = {}
    cnt_d: Dict[int, int]   = {}
    latest_ts: Dict[int, int] = {}
    recent_cnt: Dict[int, int] = {}
    cutoff = int(datetime(2019, 1, 1).timestamp())
    total_rows = 0
    chunk_num = 0

    for chunk in pd.read_csv(
        ratings_path, chunksize=CHUNK,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": np.int32, "movieId": np.int32,
               "rating": np.float32, "timestamp": np.int32},
    ):
        chunk_num += 1
        total_rows += len(chunk)
        chunk["rating_sq"] = np.square(chunk["rating"].to_numpy(dtype=np.float32))
        if chunk_num % 10 == 0:
            log.info("  … processed %d M rows", total_rows // 1_000_000)

        for mid, grp in chunk.groupby("movieId"):
            mid = _coerce_int(mid)
            sum_d[mid]       = sum_d.get(mid, 0.0)  + float(grp["rating"].sum())
            sum_sq_d[mid]    = sum_sq_d.get(mid, 0.0) + float(grp["rating_sq"].sum())
            cnt_d[mid]       = cnt_d.get(mid, 0)    + len(grp)
            latest_ts[mid]   = max(latest_ts.get(mid, 0), int(grp["timestamp"].max()))
            recent_cnt[mid]  = recent_cnt.get(mid, 0) + int((grp["timestamp"] >= cutoff).sum())

    log.info("  Total ratings: %d", total_rows)

    avg_df = pd.DataFrame({
        "movieId":     list(sum_d.keys()),
        "avg_rating":  [sum_d[k] / cnt_d[k] for k in sum_d],
        "rating_stddev": [
            float(np.sqrt(max((sum_sq_d[k] / cnt_d[k]) - ((sum_d[k] / cnt_d[k]) ** 2), 0.0)))
            for k in sum_d
        ],
        "num_ratings": list(cnt_d.values()),
    })
    movies_df = movies_df.merge(avg_df, on="movieId", how="left")
    movies_df["avg_rating"]  = movies_df["avg_rating"].fillna(0)
    movies_df["rating_stddev"] = movies_df["rating_stddev"].fillna(0)
    movies_df["num_ratings"] = movies_df["num_ratings"].fillna(0).astype(int)

    ts_df = pd.DataFrame({
        "movieId":    list(latest_ts.keys()),
        "latest_ts":  list(latest_ts.values()),
        "num_recent": [recent_cnt.get(k, 0) for k in latest_ts],
    })
    ts_df["trending_score"] = MinMaxScaler().fit_transform(ts_df[["num_recent"]])
    movies_df = movies_df.merge(
        ts_df[["movieId", "trending_score"]], on="movieId", how="left"
    )
    movies_df["trending_score"] = movies_df["trending_score"].fillna(0)

    with get_db() as conn:
        ts_df.to_sql("rating_timestamps", conn, if_exists="replace",
                     index=False, chunksize=5_000)

    log.info("Ratings aggregated successfully")
    return movies_df


# =============================================================================
#  STEP 3 — LOAD TAGS + SENTIMENT
# =============================================================================
def load_tags(movies_df: pd.DataFrame) -> pd.DataFrame:
    tags_path = DATA_DIR / "tags.csv"
    if not tags_path.exists():
        log.warning("tags.csv not found – skipping tag/sentiment loading")
        movies_df["sentiment_score"] = 0.0
        return movies_df

    textblob = _try_import("textblob")
    TextBlob = getattr(textblob, "TextBlob", None) if textblob else None

    log.info("Loading tags.csv in chunks …")
    sentiment_sum: Dict[int, float] = {}
    sentiment_cnt: Dict[int, int]   = {}
    first_chunk = True
    total_tags = 0

    with get_db() as conn:
        for chunk in pd.read_csv(
            tags_path, chunksize=CHUNK, usecols=["movieId", "tag"]
        ):
            chunk["tag"] = chunk["tag"].fillna("").astype(str)
            total_tags += len(chunk)
            chunk.to_sql(
                "tags", conn,
                if_exists="replace" if first_chunk else "append",
                index=False, chunksize=10_000,
            )
            first_chunk = False

            if TextBlob:
                for mid, grp in chunk.groupby("movieId"):
                    text   = " ".join(grp["tag"].tolist())
                    weight = len(grp)
                    polarity = TextBlob(text).sentiment.polarity
                    mid = _coerce_int(mid)
                    sentiment_sum[mid] = sentiment_sum.get(mid, 0.0) + polarity * weight
                    sentiment_cnt[mid] = sentiment_cnt.get(mid, 0)   + weight

    log.info("  %d tags loaded", total_tags)

    if TextBlob and sentiment_sum:
        sent_df = pd.DataFrame({
            "movieId":       list(sentiment_sum.keys()),
            "sentiment_score": [
                sentiment_sum[m] / max(sentiment_cnt[m], 1)
                for m in sentiment_sum
            ],
        })
        movies_df = movies_df.merge(sent_df, on="movieId", how="left",
                                    suffixes=("", "_new"))
        if "sentiment_score_new" in movies_df.columns:
            movies_df["sentiment_score"] = (
                movies_df["sentiment_score_new"]
                .fillna(movies_df.get("sentiment_score", 0))
                .fillna(0)
            )
            movies_df.drop(columns=["sentiment_score_new"], inplace=True)
    else:
        movies_df["sentiment_score"] = movies_df.get("sentiment_score", 0.0)

    return movies_df


# =============================================================================
#  STEP 4 — LOAD GENOME SCORES
# =============================================================================
def load_genome():
    genome_path = DATA_DIR / "genome-scores.csv"
    if not genome_path.exists():
        log.warning("genome-scores.csv not found – skipping genome loading")
        return

    with get_db() as conn:
        if conn.execute("SELECT COUNT(*) FROM genome_scores").fetchone()[0] > 0 \
                and not FORCE_RETRAIN:
            log.info("Genome scores already in DB – skipping")
            return

    log.info("Loading genome-scores.csv in chunks (%.0f MB) …",
             genome_path.stat().st_size / 1_048_576)
    first_chunk = True
    total = 0
    with get_db() as conn:
        for chunk in pd.read_csv(genome_path, chunksize=CHUNK):
            chunk.to_sql(
                "genome_scores", conn,
                if_exists="replace" if first_chunk else "append",
                index=False, chunksize=10_000,
            )
            first_chunk = False
            total += len(chunk)
            if total % 1_000_000 == 0:
                log.info("  … %d M genome rows", total // 1_000_000)

    log.info("  Genome scores loaded: %d rows", total)


# =============================================================================
#  STEP 5 — WRITE MOVIES TO DB
# =============================================================================
MOVIE_TABLE_COLUMNS = [
    "movieId",
    "title",
    "genres",
    "avg_rating",
    "rating_stddev",
    "num_ratings",
    "sentiment_score",
    "trending_score",
    "poster",
    "source",
]


def _load_existing_admin_movies() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=MOVIE_TABLE_COLUMNS)

    try:
        with get_db() as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'movies'"
            ).fetchone()
            if not table_exists:
                return pd.DataFrame(columns=MOVIE_TABLE_COLUMNS)

            admin_movies = pd.read_sql_query(
                f"""
                SELECT {", ".join(MOVIE_TABLE_COLUMNS)}
                FROM movies
                WHERE lower(coalesce(source, '')) = 'admin'
                """,
                conn,
            )
    except Exception as exc:
        log.warning("Could not load existing admin movies: %s", exc)
        return pd.DataFrame(columns=MOVIE_TABLE_COLUMNS)

    if admin_movies.empty:
        return pd.DataFrame(columns=MOVIE_TABLE_COLUMNS)

    return admin_movies[MOVIE_TABLE_COLUMNS]


def restore_admin_movies_to_db(admin_movies: pd.DataFrame):
    if admin_movies.empty:
        return

    with get_db() as conn:
        admin_movies[MOVIE_TABLE_COLUMNS].to_sql(
            "movies",
            conn,
            if_exists="append",
            index=False,
            chunksize=5_000,
        )
    log.info("Restored %d admin movies to DB", len(admin_movies))


def write_movies_to_db(movies_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["movieId", "title", "genres", "avg_rating", "rating_stddev", "num_ratings",
            "sentiment_score", "trending_score", "poster", "source"]
    present_cols = [c for c in cols if c in movies_df.columns]
    admin_movies = _load_existing_admin_movies()
    log.info("Writing %d movies to DB …", len(movies_df))
    with get_db() as conn:
        movies_df[present_cols].to_sql(
            "movies", conn, if_exists="replace", index=False, chunksize=5_000
        )
    log.info("Movies written to DB")
    if not admin_movies.empty:
        log.info("Preserving %d existing admin movies for post-training restore", len(admin_movies))
    return admin_movies


# =============================================================================
#  STEP 6 — TEMPORAL TRAIN/TEST SPLIT
# =============================================================================
def temporal_split() -> Tuple[Path, Path]:
    train_path = DATA_DIR / "train.csv"
    test_path  = DATA_DIR / "test.csv"

    if train_path.exists() and test_path.exists() and not FORCE_RETRAIN:
        log.info("train.csv / test.csv already exist – skipping split")
        return train_path, test_path

    log.info("Performing temporal train/test split (%.0f%% test) …", TEST_RATIO * 100)
    log.info("  Loading ratings.csv into SQLite for window function split …")

    ratings_path = DATA_DIR / "ratings.csv"
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA cache_size=-131072")
        conn.execute("DROP TABLE IF EXISTS ratings_raw")
        conn.execute("DROP TABLE IF EXISTS train_ratings")
        conn.execute("DROP TABLE IF EXISTS test_ratings")

        first = True
        total = 0
        for chunk in pd.read_csv(
            ratings_path, chunksize=CHUNK,
            usecols=["userId", "movieId", "rating", "timestamp"],
            dtype={"userId": np.int32, "movieId": np.int32,
                   "rating": np.float32, "timestamp": np.int32},
        ):
            chunk.to_sql(
                "ratings_raw", conn,
                if_exists="replace" if first else "append",
                index=False, chunksize=10_000,
            )
            first = False
            total += len(chunk)
            if total % 5_000_000 == 0:
                log.info("  … imported %d M rows", total // 1_000_000)

        log.info("  Creating indexes …")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rr_user_ts ON ratings_raw(userId, timestamp)"
        )

        log.info("  Running SQL window-function split …")
        n_test_expr = (
            f"CASE WHEN CAST(cnt * {TEST_RATIO} AS INTEGER) < 1 "
            f"THEN 1 ELSE CAST(cnt * {TEST_RATIO} AS INTEGER) END"
        )
        conn.executescript(f"""
            CREATE TABLE train_ratings AS
            WITH ranked AS (
                SELECT userId, movieId, rating, timestamp,
                    ROW_NUMBER() OVER (PARTITION BY userId ORDER BY timestamp) AS rn,
                    COUNT(*)     OVER (PARTITION BY userId)                    AS cnt
                FROM ratings_raw
            )
            SELECT userId, movieId, rating, timestamp
            FROM ranked WHERE rn <= cnt - {n_test_expr};

            CREATE TABLE test_ratings AS
            WITH ranked AS (
                SELECT userId, movieId, rating, timestamp,
                    ROW_NUMBER() OVER (PARTITION BY userId ORDER BY timestamp) AS rn,
                    COUNT(*)     OVER (PARTITION BY userId)                    AS cnt
                FROM ratings_raw
            )
            SELECT userId, movieId, rating, timestamp
            FROM ranked WHERE rn > cnt - {n_test_expr};
        """)

        tr = conn.execute("SELECT COUNT(*) FROM train_ratings").fetchone()[0]
        te = conn.execute("SELECT COUNT(*) FROM test_ratings").fetchone()[0]
        log.info("  Train: %d  |  Test: %d", tr, te)

        # Export to CSV in chunks
        log.info("  Exporting train.csv …")
        _export_table_to_csv(conn, "train_ratings", train_path)
        log.info("  Exporting test.csv …")
        _export_table_to_csv(conn, "test_ratings", test_path)

        conn.execute("DROP TABLE IF EXISTS ratings_raw")

    log.info("Split complete -> data/train.csv  data/test.csv")
    return train_path, test_path


def _export_table_to_csv(conn, table: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for chunk in pd.read_sql_query(
        f"SELECT userId, movieId, rating, timestamp FROM {table}",
        conn, chunksize=CHUNK
    ):
        chunk.to_csv(path, mode="w" if first else "a", header=first, index=False)
        first = False


# =============================================================================
#  STEP 7 — SAMPLE RATINGS HELPER
# =============================================================================
def _sample_csv(csv_path: Path, sample_rows: int,
                usecols: List[str]) -> pd.DataFrame:
    """Sample up to sample_rows rows from a CSV file using chunked proportional sampling."""
    from typing import cast as _cast, Any as _Any
    dtype_map: dict[str, _Any] = {
        "userId": np.int32, "movieId": np.int32,
        "rating": np.float32, "timestamp": np.int32,
    }
    dtypes: _Any = {k: v for k, v in dtype_map.items() if k in usecols}

    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK, usecols=usecols, dtype=dtypes):
        total += len(chunk)

    if total <= sample_rows:
        return pd.read_csv(csv_path, usecols=usecols, dtype=dtypes)

    ratio = sample_rows / total
    parts = []
    for i, chunk in enumerate(
        pd.read_csv(csv_path, chunksize=CHUNK, usecols=usecols, dtype=dtypes)
    ):
        take = max(1, min(len(chunk), int(round(len(chunk) * ratio))))
        parts.append(chunk.sample(n=take, random_state=42 + i))

    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > sample_rows:
        sampled = sampled.sample(n=sample_rows, random_state=42)
    return sampled.reset_index(drop=True)


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _ndcg(relevance: np.ndarray, k: int = 10) -> float:
    if len(relevance) == 0:
        return 0.0
    r = relevance[:k]
    dcg = np.sum(r / np.log2(np.arange(2, len(r) + 2)))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


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


def _extract_genre_token_set(value: Any) -> set[str]:
    return set(_extract_genre_tokens(value))


def _genre_feature_name(token: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in token.lower()).strip("_")
    return f"genre_{normalized}"


def _select_genre_tokens(movie_meta_df: pd.DataFrame) -> list[str]:
    if movie_meta_df.empty or "genres" not in movie_meta_df.columns:
        return []

    genre_series = (
        movie_meta_df[["movieId", "genres"]]
        .drop_duplicates(subset=["movieId"])
        .get("genres", pd.Series(dtype=object))
        .fillna("")
        .map(_extract_genre_tokens)
    )
    exploded = genre_series.explode()
    if exploded.empty:
        return []

    counts = exploded.value_counts()
    if XGB_MAX_GENRE_FEATURES > 0:
        counts = counts.head(XGB_MAX_GENRE_FEATURES)
    return [str(token) for token in counts.index.tolist()]


def _add_genre_indicator_features(df: pd.DataFrame, genre_tokens: list[str]) -> pd.DataFrame:
    if not genre_tokens:
        return df

    genre_sets = cast(
        pd.Series,
        df.get("genres", pd.Series("", index=df.index)).fillna("").map(
            _extract_genre_token_set
        ),
    )
    for token in genre_tokens:
        feature_name = _genre_feature_name(token)
        df[feature_name] = genre_sets.map(
            lambda tokens: 1.0 if token in cast(set[str], tokens) else 0.0
        ).astype(np.float32)
    return df


def _compose_xgb_feature_columns(
    genre_tokens: list[str],
    genome_feature_df: pd.DataFrame,
) -> list[str]:
    feature_cols = list(XGB_FEATURE_COLS)
    for token in genre_tokens:
        column_name = _genre_feature_name(token)
        if column_name not in feature_cols:
            feature_cols.append(column_name)

    for column_name in genome_feature_df.columns.tolist():
        if column_name == "movieId" or column_name in feature_cols:
            continue
        feature_cols.append(str(column_name))

    return feature_cols


def _tag_feature_columns(genome_feature_df: pd.DataFrame) -> list[str]:
    return [
        str(column_name)
        for column_name in genome_feature_df.columns.tolist()
        if str(column_name).startswith("genome_tag_")
    ]


def _global_rating_reference_timestamp() -> int:
    with get_db() as conn:
        try:
            row = conn.execute("SELECT MAX(latest_ts) FROM rating_timestamps").fetchone()
        except Exception:
            row = None
    if row and row[0]:
        return _coerce_int(row[0])

    max_ts = 0
    for candidate_path in (DATA_DIR / "train.csv", DATA_DIR / "ratings.csv"):
        if not candidate_path.exists():
            continue
        for chunk in pd.read_csv(
            candidate_path,
            chunksize=CHUNK,
            usecols=["timestamp"],
            dtype={"timestamp": np.int32},
        ):
            chunk_max = int(chunk["timestamp"].max() or 0)
            if chunk_max > max_ts:
                max_ts = chunk_max
        if max_ts > 0:
            break
    return max_ts


def _load_user_feature_frame(
    train_path: Path,
    genre_tokens: list[str],
    genome_feature_df: pd.DataFrame,
    movie_meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict[str, float]], dict[int, np.ndarray]]:
    required_columns = {
        "userId",
        "user_activity_level",
        "user_avg_rating",
        "user_rating_std",
        "user_recency_days_log",
        "user_recency_score",
        "user_rating_freq_trend",
    }
    tag_feature_columns = _tag_feature_columns(genome_feature_df)
    if XGB_USER_STATS_PATH.exists() and not FORCE_RETRAIN:
        with open(XGB_USER_STATS_PATH, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict):
            cached_df = cached.get("feature_df")
            cached_genre_tokens = list(cached.get("genre_tokens") or [])
            cached_tag_columns = list(cached.get("tag_feature_columns") or [])
            if (
                isinstance(cached_df, pd.DataFrame)
                and required_columns.issubset(set(cached_df.columns))
                and cached_genre_tokens == list(genre_tokens)
                and cached_tag_columns == tag_feature_columns
            ):
                genre_affinity_map = {
                    _coerce_int(user_id): {
                        str(token): float(score)
                        for token, score in dict(scores).items()
                    }
                    for user_id, scores in dict(cached.get("genre_affinity_map") or {}).items()
                }
                tag_profile_map = {
                    _coerce_int(user_id): np.asarray(vector, dtype=np.float32)
                    for user_id, vector in dict(cached.get("tag_profile_map") or {}).items()
                }
                return cached_df, genre_affinity_map, tag_profile_map
        log.info("User feature cache missing newer affinity columns – rebuilding")

    log.info("Building user activity + affinity features from train.csv …")
    max_ts = _global_rating_reference_timestamp()
    rating_sum: Dict[int, float] = {}
    rating_sq_sum: Dict[int, float] = {}
    rating_count: Dict[int, int] = {}
    latest_ts: Dict[int, int] = {}
    recent_30_count: Dict[int, int] = defaultdict(int)
    prior_90_count: Dict[int, int] = defaultdict(int)
    user_genre_sum: dict[int, dict[str, float]] = defaultdict(dict)
    user_genre_count: dict[int, dict[str, int]] = defaultdict(dict)
    global_genre_sum: dict[str, float] = defaultdict(float)
    global_genre_count: dict[str, int] = defaultdict(int)
    user_tag_sum: dict[int, np.ndarray] = {}
    user_tag_weight: dict[int, float] = defaultdict(float)

    genre_token_set = set(genre_tokens)
    genre_lookup = (
        movie_meta_df[["movieId", "genres"]]
        .drop_duplicates(subset=["movieId"])
        .copy()
    )
    genre_lookup["selected_genres"] = genre_lookup["genres"].fillna("").map(
        lambda value: [
            token
            for token in _extract_genre_tokens(value)
            if token in genre_token_set
        ]
    )
    genre_lookup = genre_lookup[["movieId", "selected_genres"]]

    movie_tag_df = (
        genome_feature_df[["movieId", *tag_feature_columns]].copy()
        if tag_feature_columns
        else pd.DataFrame(columns=["movieId"])
    )

    for chunk in pd.read_csv(
        train_path,
        chunksize=CHUNK,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
            "timestamp": np.int32,
        },
    ):
        chunk["rating_sq"] = np.square(chunk["rating"].to_numpy(dtype=np.float32))
        grouped = chunk.groupby("userId").agg(
            rating_sum=("rating", "sum"),
            rating_sq_sum=("rating_sq", "sum"),
            rating_count=("rating", "size"),
            latest_ts=("timestamp", "max"),
        )
        for user_id, row in grouped.iterrows():
            uid = _coerce_int(user_id)
            rating_sum[uid] = rating_sum.get(uid, 0.0) + float(row["rating_sum"])
            rating_sq_sum[uid] = rating_sq_sum.get(uid, 0.0) + float(row["rating_sq_sum"])
            rating_count[uid] = rating_count.get(uid, 0) + int(row["rating_count"])
            latest_ts[uid] = max(latest_ts.get(uid, 0), int(row["latest_ts"]))

        if max_ts > 0:
            age_days = np.maximum(
                0.0,
                (max_ts - chunk["timestamp"].to_numpy(dtype=np.int64)) / 86400.0,
            )
            chunk["recent_30"] = (age_days <= 30.0).astype(np.int16)
            chunk["prior_90"] = ((age_days > 30.0) & (age_days <= 120.0)).astype(np.int16)
            trend_grouped = chunk.groupby("userId").agg(
                recent_30=("recent_30", "sum"),
                prior_90=("prior_90", "sum"),
            )
            for user_id, row in trend_grouped.iterrows():
                uid = _coerce_int(user_id)
                recent_30_count[uid] += int(row["recent_30"])
                prior_90_count[uid] += int(row["prior_90"])

        if genre_tokens:
            genre_chunk = chunk[["userId", "movieId", "rating"]].merge(
                genre_lookup,
                on="movieId",
                how="left",
            )
            genre_chunk = genre_chunk.explode("selected_genres")
            genre_chunk = genre_chunk[genre_chunk["selected_genres"].notna()]
            if not genre_chunk.empty:
                user_genre_grouped = genre_chunk.groupby(["userId", "selected_genres"]).agg(
                    rating_sum=("rating", "sum"),
                    rating_count=("rating", "size"),
                )
                for (user_id, genre_name), row in user_genre_grouped.iterrows():
                    uid = _coerce_int(user_id)
                    genre_label = str(genre_name)
                    user_genre_sum[uid][genre_label] = (
                        user_genre_sum[uid].get(genre_label, 0.0) + float(row["rating_sum"])
                    )
                    user_genre_count[uid][genre_label] = (
                        user_genre_count[uid].get(genre_label, 0) + int(row["rating_count"])
                    )
                global_genre_grouped = genre_chunk.groupby("selected_genres").agg(
                    rating_sum=("rating", "sum"),
                    rating_count=("rating", "size"),
                )
                for genre_name, row in global_genre_grouped.iterrows():
                    genre_label = str(genre_name)
                    global_genre_sum[genre_label] += float(row["rating_sum"])
                    global_genre_count[genre_label] += int(row["rating_count"])

        if tag_feature_columns:
            liked_chunk = chunk.loc[chunk["rating"] >= 4.0, ["userId", "movieId", "rating"]].merge(
                movie_tag_df,
                on="movieId",
                how="left",
            )
            if not liked_chunk.empty:
                weights = np.clip(
                    liked_chunk["rating"].to_numpy(dtype=np.float32) / 5.0,
                    0.1,
                    None,
                )
                weighted_tag_frame = liked_chunk[tag_feature_columns].fillna(0).astype(np.float32)
                weighted_tag_frame = weighted_tag_frame.mul(weights, axis=0)
                weighted_tag_frame["userId"] = liked_chunk["userId"].to_numpy(dtype=np.int32)
                weighted_tag_frame["_tag_weight"] = weights
                grouped_tags = weighted_tag_frame.groupby("userId").sum()
                for user_id, row in grouped_tags.iterrows():
                    uid = _coerce_int(user_id)
                    tag_vector = row[tag_feature_columns].to_numpy(dtype=np.float32, copy=True)
                    existing_vector = user_tag_sum.get(uid)
                    if existing_vector is None:
                        user_tag_sum[uid] = tag_vector
                    else:
                        existing_vector += tag_vector
                    user_tag_weight[uid] += float(row["_tag_weight"])

    if not rating_count:
        empty = pd.DataFrame(
            columns=[
                "userId",
                "user_activity_level",
                "user_avg_rating",
                "user_rating_std",
                "user_recency_days_log",
                "user_recency_score",
                "user_rating_freq_trend",
            ]
        )
        with open(XGB_USER_STATS_PATH, "wb") as f:
            pickle.dump(
                {
                    "feature_df": empty,
                    "genre_affinity_map": {},
                    "tag_profile_map": {},
                    "genre_tokens": list(genre_tokens),
                    "tag_feature_columns": tag_feature_columns,
                },
                f,
            )
        return empty, {}, {}

    user_ids = np.array(list(rating_count.keys()), dtype=np.int32)
    recency_days = np.array([
        max(0.0, (max_ts - latest_ts[_coerce_int(uid)]) / 86400.0)
        if max_ts > 0 else 0.0
        for uid in user_ids
    ], dtype=np.float32)
    recent_rates = np.array(
        [recent_30_count.get(_coerce_int(uid), 0) for uid in user_ids],
        dtype=np.float32,
    )
    prior_rates = np.array(
        [prior_90_count.get(_coerce_int(uid), 0) for uid in user_ids],
        dtype=np.float32,
    )
    user_df = pd.DataFrame({
        "userId": user_ids,
        "user_activity_level": np.log1p(
            np.array([rating_count[_coerce_int(uid)] for uid in user_ids], dtype=np.float32)
        ),
        "user_avg_rating": np.array([
            rating_sum[_coerce_int(uid)] / max(1, rating_count[_coerce_int(uid)])
            for uid in user_ids
        ], dtype=np.float32),
        "user_rating_std": np.array([
            float(
                np.sqrt(
                    max(
                        (rating_sq_sum[_coerce_int(uid)] / max(1, rating_count[_coerce_int(uid)]))
                        - ((rating_sum[_coerce_int(uid)] / max(1, rating_count[_coerce_int(uid)])) ** 2),
                        0.0,
                    )
                )
            )
            for uid in user_ids
        ], dtype=np.float32),
        "user_recency_days_log": np.log1p(recency_days),
        "user_recency_score": np.exp(-recency_days / 45.0).astype(np.float32),
        "user_rating_freq_trend": np.log1p(
            (recent_rates + 1.0) / ((prior_rates / 3.0) + 1.0)
        ).astype(np.float32),
    })

    global_genre_means = {
        genre_name: float(global_genre_sum[genre_name] / max(global_genre_count.get(genre_name, 1), 1))
        for genre_name in genre_tokens
        if global_genre_count.get(genre_name, 0) > 0
    }
    genre_affinity_map: dict[int, dict[str, float]] = {}
    for user_id, genre_sum_map in user_genre_sum.items():
        affinity_payload: dict[str, float] = {}
        for genre_name, total_rating in genre_sum_map.items():
            count = max(user_genre_count[user_id].get(genre_name, 0), 1)
            global_mean = global_genre_means.get(genre_name)
            if global_mean is None:
                continue
            user_mean = float(total_rating / count)
            affinity_payload[genre_name] = float(
                np.clip((user_mean - global_mean) / 2.5, -1.5, 1.5)
            )
        if affinity_payload:
            genre_affinity_map[_coerce_int(user_id)] = affinity_payload

    tag_profile_map: dict[int, np.ndarray] = {}
    for user_id, tag_sum_vector in user_tag_sum.items():
        total_weight = max(float(user_tag_weight.get(user_id, 0.0)), 1e-6)
        profile = np.asarray(tag_sum_vector / total_weight, dtype=np.float32)
        norm = float(np.linalg.norm(profile))
        tag_profile_map[_coerce_int(user_id)] = (
            profile / norm if norm > 0 else profile
        ).astype(np.float32)

    with open(XGB_USER_STATS_PATH, "wb") as f:
        pickle.dump(
            {
                "feature_df": user_df,
                "genre_affinity_map": genre_affinity_map,
                "tag_profile_map": tag_profile_map,
                "genre_tokens": list(genre_tokens),
                "tag_feature_columns": tag_feature_columns,
            },
            f,
        )
    return user_df, genre_affinity_map, tag_profile_map


def _load_genome_feature_frame(train_path: Path) -> pd.DataFrame:
    required_columns = {
        "movieId",
        "genome_mean_relevance",
        "genome_max_relevance",
        "genome_high_relevance_log",
        "item_popularity_decay",
        "item_recent_rating_velocity",
    }
    if XGB_GENOME_STATS_PATH.exists() and not FORCE_RETRAIN:
        cached = pd.read_pickle(XGB_GENOME_STATS_PATH)
        has_top_tag_columns = any(
            str(column_name).startswith("genome_tag_")
            for column_name in cached.columns
        )
        if required_columns.issubset(set(cached.columns)) and (
            GENOME_TOP_TAG_FEATURES <= 0 or has_top_tag_columns
        ):
            return cached
        log.info("Genome feature cache missing newer columns – rebuilding")

    log.info("Building movie summary features …")
    with get_db() as conn:
        try:
            genome_df = pd.read_sql_query(
                """
                SELECT
                    movieId,
                    AVG(relevance) AS genome_mean_relevance,
                    MAX(relevance) AS genome_max_relevance,
                    SUM(CASE WHEN relevance >= 0.5 THEN 1 ELSE 0 END) AS genome_high_relevance_count
                FROM genome_scores
                GROUP BY movieId
                """,
                conn,
            )
            if GENOME_TOP_TAG_FEATURES > 0 and not genome_df.empty:
                top_tags = pd.read_sql_query(
                    f"""
                    WITH tag_stats AS (
                        SELECT
                            tagId,
                            AVG(relevance) AS avg_relevance,
                            AVG(relevance * relevance) AS avg_sq_relevance
                        FROM genome_scores
                        GROUP BY tagId
                    )
                    SELECT tagId
                    FROM tag_stats
                    ORDER BY
                        (avg_sq_relevance - (avg_relevance * avg_relevance)) DESC,
                        tagId ASC
                    LIMIT {GENOME_TOP_TAG_FEATURES}
                    """,
                    conn,
                )
                top_tag_ids = [_coerce_int(tag_id) for tag_id in top_tags.get("tagId", pd.Series(dtype=np.int32)).tolist()]
                if top_tag_ids:
                    placeholders = ",".join("?" for _ in top_tag_ids)
                    top_tag_rows = pd.read_sql_query(
                        f"""
                        SELECT movieId, tagId, relevance
                        FROM genome_scores
                        WHERE tagId IN ({placeholders})
                        """,
                        conn,
                        params=tuple(top_tag_ids),
                    )
                    if not top_tag_rows.empty:
                        top_tag_pivot = top_tag_rows.pivot_table(
                            index="movieId",
                            columns="tagId",
                            values="relevance",
                            fill_value=0.0,
                        )
                        top_tag_pivot.columns = [
                            f"genome_tag_{_coerce_int(tag_id)}"
                            for tag_id in top_tag_pivot.columns
                        ]
                        genome_df = genome_df.merge(
                            top_tag_pivot.reset_index(),
                            on="movieId",
                            how="left",
                        )
        except Exception as exc:
            log.warning("Genome feature aggregation failed: %s", exc)
            genome_df = pd.DataFrame(columns=[
                "movieId",
                "genome_mean_relevance",
                "genome_max_relevance",
                "genome_high_relevance_log",
            ])

    popularity_decay: dict[int, float] = defaultdict(float)
    recent_count: dict[int, float] = defaultdict(float)
    prior_count: dict[int, float] = defaultdict(float)
    max_ts = _global_rating_reference_timestamp()
    if train_path.exists() and max_ts > 0:
        log.info("Building recency-weighted popularity features …")
        for chunk in pd.read_csv(
            train_path,
            chunksize=CHUNK,
            usecols=["movieId", "timestamp"],
            dtype={"movieId": np.int32, "timestamp": np.int32},
        ):
            age_days = np.maximum(
                0.0,
                (max_ts - chunk["timestamp"].to_numpy(dtype=np.int64)) / 86400.0,
            )
            popularity_chunk = pd.DataFrame(
                {
                    "movieId": chunk["movieId"].to_numpy(dtype=np.int32),
                    "item_popularity_decay": np.exp(-age_days / 180.0).astype(np.float32),
                    "recent_count": (age_days <= 45.0).astype(np.float32),
                    "prior_count": ((age_days > 45.0) & (age_days <= 180.0)).astype(np.float32),
                }
            )
            grouped_popularity = (
                popularity_chunk.groupby("movieId", as_index=False)[
                    ["item_popularity_decay", "recent_count", "prior_count"]
                ].sum()
            )
            for row in grouped_popularity.itertuples(index=False):
                mid = _coerce_int(row.movieId)
                popularity_decay[mid] += float(row.item_popularity_decay)
                recent_count[mid] += float(row.recent_count)
                prior_count[mid] += float(row.prior_count)

    popularity_df = pd.DataFrame(
        {
            "movieId": list(popularity_decay.keys()),
            "item_popularity_decay": [
                float(np.log1p(popularity_decay[mid]))
                for mid in popularity_decay
            ],
            "item_recent_rating_velocity": [
                float(np.log1p((recent_count[mid] + 1.0) / ((prior_count[mid] / 3.0) + 1.0)))
                for mid in popularity_decay
            ],
        }
    )

    if genome_df.empty:
        genome_df = popularity_df.copy()
        if genome_df.empty:
            genome_df.to_pickle(XGB_GENOME_STATS_PATH)
            return genome_df
        for column_name in (
            "genome_mean_relevance",
            "genome_max_relevance",
            "genome_high_relevance_count",
        ):
            genome_df[column_name] = 0.0

    genome_df["genome_mean_relevance"] = genome_df["genome_mean_relevance"].fillna(0).astype(np.float32)
    genome_df["genome_max_relevance"] = genome_df["genome_max_relevance"].fillna(0).astype(np.float32)
    genome_df["genome_high_relevance_log"] = np.log1p(
        genome_df["genome_high_relevance_count"].fillna(0).astype(np.float32)
    )
    if "genome_high_relevance_count" in genome_df.columns:
        genome_df = genome_df.drop(columns=["genome_high_relevance_count"])
    if not popularity_df.empty:
        genome_df = genome_df.merge(popularity_df, on="movieId", how="outer")
    elif "item_popularity_decay" not in genome_df.columns:
        genome_df["item_popularity_decay"] = 0.0
        genome_df["item_recent_rating_velocity"] = 0.0
    for column_name in genome_df.columns:
        if column_name == "movieId":
            continue
        genome_df[column_name] = genome_df[column_name].fillna(0).astype(np.float32)
    genome_df.to_pickle(XGB_GENOME_STATS_PATH)
    return genome_df


def _load_movie_sbert_embeddings() -> dict[int, np.ndarray]:
    if MOVIE_SBERT_EMBEDDINGS_PATH.exists() and not FORCE_RETRAIN:
        with open(MOVIE_SBERT_EMBEDDINGS_PATH, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            return {
                _coerce_int(movie_id): np.asarray(embedding, dtype=np.float32)
                for movie_id, embedding in payload.items()
            }

    sentence_transformers = _try_import("sentence_transformers")
    SentenceTransformer = getattr(sentence_transformers, "SentenceTransformer", None)
    if SentenceTransformer is None:
        log.warning("sentence-transformers not installed – SBERT taste features skipped")
        return {}

    movies_path = DATA_DIR / "movies.csv"
    if not movies_path.exists():
        log.warning("movies.csv not found – SBERT taste features skipped")
        return {}

    log.info("Building SBERT movie title embeddings …")
    movies_df = pd.read_csv(movies_path, usecols=["movieId", "title"])
    titles = (
        movies_df["title"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
        .tolist()
    )
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        titles,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    payload = {
        _coerce_int(movie_id): embeddings[idx]
        for idx, movie_id in enumerate(movies_df["movieId"].tolist())
    }
    with open(MOVIE_SBERT_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(payload, f)
    return payload


def _load_user_taste_vectors(
    train_path: Path,
    movie_embeddings: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, int]]:
    if USER_TASTE_VECTORS_PATH.exists() and USER_TASTE_ID_MAP_PATH.exists() and not FORCE_RETRAIN:
        matrix = np.asarray(np.load(USER_TASTE_VECTORS_PATH), dtype=np.float32)
        with open(USER_TASTE_ID_MAP_PATH, "rb") as f:
            id_map = pickle.load(f)
        if isinstance(id_map, dict):
            return matrix, {
                _coerce_int(user_id): _coerce_int(row_idx)
                for user_id, row_idx in id_map.items()
            }

    if not movie_embeddings:
        return np.zeros((0, 0), dtype=np.float32), {}

    embedding_dim = len(next(iter(movie_embeddings.values())))
    user_vec_sum: dict[int, np.ndarray] = {}
    user_vec_count: dict[int, int] = {}
    total_rows = 0
    log.info("Building SBERT user taste vectors from train.csv …")
    for chunk in pd.read_csv(
        train_path,
        chunksize=CHUNK,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    ):
        liked = chunk[chunk["rating"] >= 4.0]
        for row in liked.itertuples(index=False):
            movie_id = _coerce_int(row.movieId)
            embedding = movie_embeddings.get(movie_id)
            if embedding is None:
                continue
            user_id = _coerce_int(row.userId)
            current = user_vec_sum.get(user_id)
            if current is None:
                current = np.zeros(embedding_dim, dtype=np.float32)
                user_vec_sum[user_id] = current
            current += embedding
            user_vec_count[user_id] = user_vec_count.get(user_id, 0) + 1
        total_rows += len(chunk)
        if total_rows and total_rows % 5_000_000 == 0:
            log.info("  … streamed %d M rows for taste vectors", total_rows // 1_000_000)

    if not user_vec_sum:
        return np.zeros((0, embedding_dim), dtype=np.float32), {}

    user_ids = sorted(user_vec_sum.keys())
    id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    matrix = np.zeros((len(user_ids), embedding_dim), dtype=np.float32)
    for user_id, row_idx in id_map.items():
        vector = user_vec_sum[user_id] / max(user_vec_count.get(user_id, 1), 1)
        norm = float(np.linalg.norm(vector))
        matrix[row_idx] = vector / norm if norm > 0 else vector

    np.save(USER_TASTE_VECTORS_PATH, matrix)
    with open(USER_TASTE_ID_MAP_PATH, "wb") as f:
        pickle.dump(id_map, f)
    return matrix, id_map


def _predict_sbert_scores(
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    movie_embeddings: dict[int, np.ndarray],
    user_taste_vectors: np.ndarray,
    user_taste_id_map: dict[int, int],
) -> np.ndarray:
    if not movie_embeddings or user_taste_vectors.size == 0 or not user_taste_id_map:
        return np.zeros(len(user_ids), dtype=np.float32)

    scores = np.zeros(len(user_ids), dtype=np.float32)
    for idx, (uid, mid) in enumerate(zip(user_ids, movie_ids)):
        taste_idx = user_taste_id_map.get(_coerce_int(uid))
        movie_embedding = movie_embeddings.get(_coerce_int(mid))
        if taste_idx is None or movie_embedding is None:
            continue
        if taste_idx >= len(user_taste_vectors):
            continue
        scores[idx] = float(np.dot(user_taste_vectors[taste_idx], movie_embedding))
    return scores


def _predict_svd_scores(svd_model: Any, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
    if svd_model is None:
        return np.zeros(len(user_ids), dtype=np.float32)
    preds = [
        float(svd_model.predict(_coerce_int(uid), _coerce_int(mid)).est) / 5.0
        for uid, mid in zip(user_ids, movie_ids)
    ]
    return np.clip(np.asarray(preds, dtype=np.float32), 0.0, 1.0)


def _predict_als_scores(
    als_model: Any,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    item_ids: list[int],
    user_ids_lookup: list[int],
) -> np.ndarray:
    if als_model is None or not item_ids or not user_ids_lookup:
        return np.zeros(len(user_ids), dtype=np.float32)

    item_index = {_coerce_int(mid): idx for idx, mid in enumerate(item_ids)}
    user_index = {_coerce_int(uid): idx for idx, uid in enumerate(user_ids_lookup)}
    item_factors = getattr(als_model, "item_factors", None)
    user_factors = getattr(als_model, "user_factors", None)
    if item_factors is None or user_factors is None:
        return np.zeros(len(user_ids), dtype=np.float32)

    raw_scores = np.zeros(len(user_ids), dtype=np.float32)
    for idx, (uid, mid) in enumerate(zip(user_ids, movie_ids)):
        u_idx = user_index.get(_coerce_int(uid))
        i_idx = item_index.get(_coerce_int(mid))
        if u_idx is None or i_idx is None:
            continue
        if u_idx >= len(user_factors) or i_idx >= len(item_factors):
            continue
        raw_scores[idx] = float(np.dot(user_factors[u_idx], item_factors[i_idx]))
    return _sigmoid_array(raw_scores)


def _predict_ncf_scores(
    ncf_model: Any,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    user_enc: dict[int, int],
    item_enc: dict[int, int],
) -> np.ndarray:
    if ncf_model is None or not user_enc or not item_enc:
        return np.zeros(len(user_ids), dtype=np.float32)

    scores = np.zeros(len(user_ids), dtype=np.float32)
    valid_positions = [
        idx
        for idx, (uid, mid) in enumerate(zip(user_ids, movie_ids))
        if _coerce_int(uid) in user_enc and _coerce_int(mid) in item_enc
    ]
    if not valid_positions:
        return scores

    u_arr = np.array([user_enc[_coerce_int(user_ids[idx])] for idx in valid_positions], dtype=np.int32)
    i_arr = np.array([item_enc[_coerce_int(movie_ids[idx])] for idx in valid_positions], dtype=np.int32)
    preds = ncf_model.predict([u_arr, i_arr], batch_size=8192, verbose=0).flatten()
    scores[np.array(valid_positions, dtype=np.int32)] = np.clip(
        preds.astype(np.float32),
        0.0,
        1.0,
    )
    return scores


def _build_xgb_feature_frame(
    ratings_df: pd.DataFrame,
    movie_meta_df: pd.DataFrame,
    user_feature_df: pd.DataFrame,
    genome_feature_df: pd.DataFrame,
    user_genre_affinity_map: dict[int, dict[str, float]],
    user_tag_profile_map: dict[int, np.ndarray],
    movie_sbert_embeddings: dict[int, np.ndarray],
    user_taste_vectors: np.ndarray,
    user_taste_id_map: dict[int, int],
    svd_model: Any,
    als_model: Any,
    als_item_ids: list[int],
    als_user_ids: list[int],
    ncf_model: Any,
    ncf_user_enc: dict[int, int],
    ncf_item_enc: dict[int, int],
    genre_tokens: list[str],
) -> pd.DataFrame:
    if user_feature_df.empty:
        user_feature_df = pd.DataFrame(
            columns=[
                "userId",
                "user_activity_level",
                "user_avg_rating",
                "user_rating_std",
                "user_recency_days_log",
                "user_recency_score",
                "user_rating_freq_trend",
            ]
        )
    if genome_feature_df.empty:
        genome_feature_df = pd.DataFrame(
            columns=[
                "movieId",
                "genome_mean_relevance",
                "genome_max_relevance",
                "genome_high_relevance_log",
                "item_popularity_decay",
                "item_recent_rating_velocity",
            ]
        )

    df = ratings_df.merge(movie_meta_df, on="movieId", how="left")
    df = df.merge(user_feature_df, on="userId", how="left")
    df = df.merge(genome_feature_df, on="movieId", how="left")
    df = df.fillna(0)

    user_ids = df["userId"].to_numpy(dtype=np.int32)
    movie_ids = df["movieId"].to_numpy(dtype=np.int32)

    svd_scores = _predict_svd_scores(svd_model, user_ids, movie_ids)
    if svd_model is None:
        svd_scores = np.clip(
            df["avg_rating"].to_numpy(dtype=np.float32) / 5.0,
            0.0,
            1.0,
        )
    df["svd_score"] = svd_scores

    als_scores = _predict_als_scores(als_model, user_ids, movie_ids, als_item_ids, als_user_ids)
    df["als_score"] = als_scores if als_model is not None else df["svd_score"]

    ncf_scores = _predict_ncf_scores(ncf_model, user_ids, movie_ids, ncf_user_enc, ncf_item_enc)
    df["ncf_score"] = ncf_scores if ncf_model is not None else df["svd_score"]
    df["sbert_sim"] = _predict_sbert_scores(
        user_ids,
        movie_ids,
        movie_sbert_embeddings,
        user_taste_vectors,
        user_taste_id_map,
    )

    score_frame = df[["svd_score", "als_score", "ncf_score"]].astype(np.float32)
    df["ensemble_mean"] = score_frame.mean(axis=1).astype(np.float32)
    df["ensemble_std"] = score_frame.std(axis=1, ddof=0).astype(np.float32)
    df["score_range"] = (
        score_frame.max(axis=1) - score_frame.min(axis=1)
    ).astype(np.float32)

    df["avg_rating"] = df["avg_rating"].astype(np.float32)
    df["rating_stddev"] = df.get(
        "rating_stddev",
        pd.Series(0.0, index=df.index, dtype=np.float32),
    ).astype(np.float32)
    df["log_num_ratings"] = np.log1p(df["num_ratings"].astype(np.float32))
    df["sentiment_signal"] = ((df["sentiment_score"].astype(np.float32) + 1.0) / 2.0).clip(0.0, 1.0)
    df["user_activity_level"] = df["user_activity_level"].astype(np.float32)
    df["user_avg_rating"] = df["user_avg_rating"].astype(np.float32)
    df["user_rating_std"] = df["user_rating_std"].astype(np.float32)
    df["user_recency_days_log"] = df["user_recency_days_log"].astype(np.float32)
    df["user_recency_score"] = df.get(
        "user_recency_score",
        pd.Series(0.0, index=df.index, dtype=np.float32),
    ).astype(np.float32)
    df["user_rating_freq_trend"] = df.get(
        "user_rating_freq_trend",
        pd.Series(0.0, index=df.index, dtype=np.float32),
    ).astype(np.float32)
    df["user_movie_rating_diff"] = (
        df["svd_score"].astype(np.float32) - df["user_avg_rating"].astype(np.float32)
    ).astype(np.float32)
    df["sbert_sim"] = df["sbert_sim"].astype(np.float32)
    df["genome_mean_relevance"] = df["genome_mean_relevance"].astype(np.float32)
    df["genome_max_relevance"] = df["genome_max_relevance"].astype(np.float32)
    df["genome_high_relevance_log"] = df["genome_high_relevance_log"].astype(np.float32)
    df["item_popularity_decay"] = df.get(
        "item_popularity_decay",
        pd.Series(0.0, index=df.index, dtype=np.float32),
    ).astype(np.float32)
    df["item_recent_rating_velocity"] = df.get(
        "item_recent_rating_velocity",
        pd.Series(0.0, index=df.index, dtype=np.float32),
    ).astype(np.float32)

    genre_sets = cast(
        pd.Series,
        df.get("genres", pd.Series("", index=df.index)).fillna("").map(
            _extract_genre_token_set
        ),
    )
    genre_affinity_scores = np.zeros(len(user_ids), dtype=np.float32)
    for idx, (uid, genres) in enumerate(zip(user_ids, genre_sets.tolist())):
        affinity_map = user_genre_affinity_map.get(_coerce_int(uid))
        if not affinity_map:
            continue
        affinity_values = [
            float(affinity_map.get(token, 0.0))
            for token in cast(set[str], genres)
            if token in affinity_map
        ]
        if affinity_values:
            genre_affinity_scores[idx] = float(np.mean(affinity_values))
    df["genre_affinity_score"] = genre_affinity_scores.astype(np.float32)

    tag_feature_columns = _tag_feature_columns(genome_feature_df)
    tag_cooccurrence_scores = np.zeros(len(user_ids), dtype=np.float32)
    if tag_feature_columns and user_tag_profile_map:
        movie_tag_matrix = df[tag_feature_columns].fillna(0).to_numpy(dtype=np.float32)
        movie_tag_norms = np.linalg.norm(movie_tag_matrix, axis=1)
        for idx, uid in enumerate(user_ids):
            tag_profile = user_tag_profile_map.get(_coerce_int(uid))
            if tag_profile is None or movie_tag_norms[idx] <= 0:
                continue
            movie_vector = movie_tag_matrix[idx]
            if len(tag_profile) != len(movie_vector):
                common_len = min(len(tag_profile), len(movie_vector))
                if common_len <= 0:
                    continue
                profile_vector = np.asarray(tag_profile[:common_len], dtype=np.float32)
                movie_vector = movie_vector[:common_len]
                vector_norm = float(np.linalg.norm(movie_vector))
                if vector_norm <= 0:
                    continue
                tag_cooccurrence_scores[idx] = float(np.dot(profile_vector, movie_vector / vector_norm))
                continue
            tag_cooccurrence_scores[idx] = float(
                np.dot(tag_profile, movie_vector / movie_tag_norms[idx])
            )
    df["tag_cooccurrence_strength"] = tag_cooccurrence_scores.astype(np.float32)

    df = _add_genre_indicator_features(df, genre_tokens)

    for column_name in genome_feature_df.columns.tolist():
        if column_name == "movieId":
            continue
        df[column_name] = df[column_name].astype(np.float32)

    return df


def _build_xgb_feature_context(
    user_feature_df: pd.DataFrame,
    genome_feature_df: pd.DataFrame,
    genre_tokens: list[str],
    feature_columns: list[str],
    user_genre_affinity_map: dict[int, dict[str, float]],
    user_tag_profile_map: dict[int, np.ndarray],
) -> dict[str, Any]:
    user_feature_columns = [
        str(column_name)
        for column_name in user_feature_df.columns.tolist()
        if column_name != "userId"
    ]
    user_stats = {
        _coerce_int(row["userId"]): {
            column_name: float(row[column_name])
            for column_name in user_feature_columns
        }
        for _, row in user_feature_df.iterrows()
    }
    movie_feature_columns = [
        str(column_name)
        for column_name in genome_feature_df.columns.tolist()
        if column_name != "movieId"
    ]
    movie_stats = {
        _coerce_int(row["movieId"]): {
            column_name: float(row[column_name])
            for column_name in movie_feature_columns
        }
        for _, row in genome_feature_df.iterrows()
    }
    return {
        "feature_columns": list(feature_columns),
        "genre_tokens": list(genre_tokens),
        "tag_feature_columns": _tag_feature_columns(genome_feature_df),
        "user_feature_columns": user_feature_columns,
        "movie_feature_columns": movie_feature_columns,
        "user_stats": user_stats,
        "movie_stats": movie_stats,
        "user_genre_affinity": {
            _coerce_int(user_id): {
                str(genre_name): float(score)
                for genre_name, score in affinity_map.items()
            }
            for user_id, affinity_map in user_genre_affinity_map.items()
        },
        "user_tag_profiles": {
            _coerce_int(user_id): np.asarray(tag_profile, dtype=np.float32)
            for user_id, tag_profile in user_tag_profile_map.items()
        },
    }


def _ranking_metrics_by_user(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    k: int = 10,
) -> dict[str, float]:
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    hit_rate_scores: list[float] = []
    reciprocal_ranks: list[float] = []

    for _, user_df in df.groupby("userId"):
        ranked = user_df.sort_values(score_col, ascending=False)
        relevance = ranked[label_col].astype(np.float32).to_numpy()
        total_relevant = float(relevance.sum())
        if total_relevant <= 0:
            continue

        topk = relevance[:k]
        precision_scores.append(float(topk.mean()) if len(topk) else 0.0)
        recall_scores.append(float(topk.sum() / total_relevant))
        ndcg_scores.append(_ndcg(relevance, k=min(k, len(relevance))))
        hit_rate_scores.append(1.0 if np.any(topk > 0) else 0.0)

        positive_positions = np.flatnonzero(relevance > 0)
        reciprocal_ranks.append(
            float(1.0 / (positive_positions[0] + 1)) if len(positive_positions) else 0.0
        )

    def _avg(values: list[float]) -> float:
        return round(float(np.mean(values)), 4) if values else 0.0

    return {
        "Precision@10": _avg(precision_scores),
        "Recall@10": _avg(recall_scores),
        "NDCG@10": _avg(ndcg_scores),
        "HR@10": _avg(hit_rate_scores),
        "MRR": _avg(reciprocal_ranks),
    }


def _split_xgb_train_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df["userId"].to_numpy()
    for random_state in (42, 84, 126):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=XGB_VALIDATION_RATIO,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(df, groups=groups))
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        if train_df["label"].nunique() == 2 and val_df["label"].nunique() == 2:
            return train_df, val_df

    train_df, val_df = train_test_split(
        df,
        test_size=XGB_VALIDATION_RATIO,
        random_state=42,
        stratify=df["label"],
    )
    return train_df.copy(), val_df.copy()


def _build_xgb_group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("userId", sort=False).size().astype(np.int32).tolist()


def _compute_scale_pos_weight(y: np.ndarray, sample_weight: np.ndarray) -> float:
    positives = float(sample_weight[y == 1].sum())
    negatives = float(sample_weight[y == 0].sum())
    if positives <= 0 or negatives <= 0:
        return 1.0
    return max(1.0, negatives / positives)


def _fit_xgb_calibrator(
    scores: np.ndarray,
    y_true: np.ndarray,
) -> tuple[Any | None, str, float]:
    if len(np.unique(y_true)) < 2:
        return None, "none", float(log_loss(y_true, np.clip(scores, 1e-6, 1 - 1e-6)))

    candidates: list[tuple[float, str, Any]] = []

    platt = LogisticRegression(max_iter=1000, solver="lbfgs")
    platt.fit(scores.reshape(-1, 1), y_true)
    platt_proba = np.clip(
        platt.predict_proba(scores.reshape(-1, 1))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    candidates.append((float(log_loss(y_true, platt_proba)), "platt", platt))

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(scores, y_true)
    isotonic_proba = np.clip(
        isotonic.predict(scores),
        1e-6,
        1 - 1e-6,
    )
    candidates.append((float(log_loss(y_true, isotonic_proba)), "isotonic", isotonic))

    loss, name, calibrator = min(candidates, key=lambda item: item[0])
    return calibrator, name, loss


def _apply_xgb_calibrator(
    scores: np.ndarray,
    calibrator: Any | None,
    calibration_kind: str,
) -> np.ndarray:
    if calibrator is None or calibration_kind == "none":
        calibrated = scores
    elif calibration_kind == "platt":
        calibrated = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    else:
        calibrated = calibrator.predict(scores)
    return np.clip(np.asarray(calibrated, dtype=np.float32), 1e-6, 1 - 1e-6)


def _score_xgb_validation(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    objective: str,
    calibration: str,
    config: dict[str, Any],
    scale_pos_weight: float,
) -> dict[str, Any]:
    labels = df["label"].to_numpy(dtype=np.int32)
    val_df = df[["userId", "movieId", "label"]].copy()
    val_df["pred_score"] = np.clip(probabilities, 1e-6, 1 - 1e-6)
    metrics = {
        "objective": objective,
        "calibration": calibration,
        "scale_pos_weight": round(float(scale_pos_weight), 4),
        "AUC": round(float(roc_auc_score(labels, val_df["pred_score"])) if labels.sum() > 0 else 0.0, 4),
        "LogLoss": round(float(log_loss(labels, val_df["pred_score"])), 4),
    }
    metrics.update(_ranking_metrics_by_user(val_df, "pred_score", "label", k=10))
    metrics["best_iteration"] = int(config.get("best_iteration", config["n_estimators"]))
    for key in (
        "max_depth",
        "learning_rate",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ):
        metrics[key] = config[key]
    return metrics


# =============================================================================
#  STEP 8 — TRAIN SVD
# =============================================================================
def _log_scikit_surprise_guidance() -> None:
    for message in _scikit_surprise_install_guidance():
        log.warning("  %s", message)


def _save_ncf_artifacts(
    model: Any,
    user_enc: dict[int, int],
    item_enc: dict[int, int],
    enc_path: Path,
) -> None:
    model.save_weights(str(NCF_WEIGHTS_PATH))
    with open(NCF_META_PATH, "wb") as f:
        pickle.dump(
            {
                "n_users": len(user_enc),
                "n_items": len(item_enc),
                "mf_dim": NCF_EMBED_DIM,
                "mlp_layers": list(NCF_MLP_LAYERS),
                "dropout": NCF_DROPOUT,
                "learning_rate": 1e-3,
            },
            f,
        )
    with open(enc_path, "wb") as f:
        pickle.dump({"user": user_enc, "item": item_enc}, f)


def _load_ncf_artifacts(
    build_ncf: Any,
    enc_path: Path,
    tf_module: Any,
    allow_legacy_keras: bool,
) -> tuple[Any, dict[int, int], dict[int, int]] | None:
    if NCF_WEIGHTS_PATH.exists() and NCF_META_PATH.exists() and enc_path.exists():
        with open(NCF_META_PATH, "rb") as f:
            meta = pickle.load(f)
        with open(enc_path, "rb") as f:
            enc = pickle.load(f)
        model = build_ncf(
            int(meta.get("n_users") or len(enc.get("user", {}))),
            int(meta.get("n_items") or len(enc.get("item", {}))),
            mf_dim=int(meta.get("mf_dim") or NCF_EMBED_DIM),
            mlp_layers=tuple(meta.get("mlp_layers") or list(NCF_MLP_LAYERS)),
            dropout=float(meta.get("dropout") or NCF_DROPOUT),
            learning_rate=float(meta.get("learning_rate") or 1e-3),
        )
        model.load_weights(str(NCF_WEIGHTS_PATH))
        return model, enc["user"], enc["item"]

    legacy_path = MODEL_DIR / "ncf_model.keras"
    if allow_legacy_keras and legacy_path.exists() and enc_path.exists():
        model = tf_module.keras.models.load_model(str(legacy_path))
        with open(enc_path, "rb") as f:
            enc = pickle.load(f)
        _save_ncf_artifacts(model, enc["user"], enc["item"], enc_path)
        return model, enc["user"], enc["item"]

    return None


def train_svd(train_path: Path) -> Any:
    svd_path = MODEL_DIR / "svd_model.pkl"
    surprise = _try_import("surprise")
    if svd_path.exists() and not FORCE_RETRAIN:
        if surprise is None:
            log.warning("scikit-surprise not installed – existing SVD model cannot be loaded; skipping SVD")
            _log_scikit_surprise_guidance()
            return None
        log.info("SVD model already exists – loading")
        with open(svd_path, "rb") as f:
            return pickle.load(f)

    if surprise is None:
        log.warning("scikit-surprise not installed – SVD skipped")
        _log_scikit_surprise_guidance()
        return None

    log.info("Training SVD on up to %d ratings …", SVD_SAMPLE_ROWS)
    df = _sample_csv(train_path, SVD_SAMPLE_ROWS,
                     usecols=["userId", "movieId", "rating"])
    log.info("  SVD sample: %d rows", len(df))

    reader  = surprise.Reader(rating_scale=(0.5, 5.0))
    dataset = surprise.Dataset.load_from_df(
        df[["userId", "movieId", "rating"]], reader
    )
    trainset = dataset.build_full_trainset()

    model = surprise.SVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        lr_all=SVD_LR_ALL,
        reg_all=SVD_REG_ALL,
        biased=True,
        random_state=42,
        verbose=True,
    )
    model.fit(trainset)

    with open(svd_path, "wb") as f:
        pickle.dump(model, f)
    log.info("SVD saved -> %s", svd_path)
    return model


# =============================================================================
#  STEP 9 — TRAIN ALS
# =============================================================================
def train_als(train_path: Path) -> Tuple[Any, list, list]:
    als_path = MODEL_DIR / "als_model.pkl"
    if als_path.exists() and not FORCE_RETRAIN:
        log.info("ALS model already exists – loading")
        with open(als_path, "rb") as f:
            d = pickle.load(f)
        return d["model"], d["item_ids"], d["user_ids"]

    implicit = _try_import("implicit")
    sp       = _try_import("scipy.sparse")
    if implicit is None or sp is None:
        log.warning("implicit / scipy not installed – ALS skipped")
        log.warning("  Install with:  pip install implicit scipy")
        return None, [], []

    log.info("Training ALS on up to %d ratings …", ALS_SAMPLE_ROWS)
    df = _sample_csv(train_path, ALS_SAMPLE_ROWS,
                     usecols=["userId", "movieId", "rating"])
    log.info("  ALS sample: %d rows", len(df))

    users_ = pd.Categorical(df["userId"])
    items_ = pd.Categorical(df["movieId"])
    mat    = sp.csr_matrix(
        (df["rating"].astype(np.float32), (items_.codes, users_.codes))
    )
    model  = implicit.als.AlternatingLeastSquares(
        factors=64, iterations=20, regularization=0.1, random_state=42
    )
    model.fit(mat)

    item_ids = [_coerce_int(x) for x in items_.categories]
    user_ids = [_coerce_int(x) for x in users_.categories]

    with open(als_path, "wb") as f:
        pickle.dump({"model": model, "item_ids": item_ids, "user_ids": user_ids}, f)
    log.info("ALS saved -> %s", als_path)
    return model, item_ids, user_ids


# =============================================================================
#  STEP 10 — TRAIN NCF
# =============================================================================
def train_ncf(train_path: Path, all_item_ids: np.ndarray) -> Tuple[Any, dict, dict]:
    ncf_path = MODEL_DIR / "ncf_model.keras"
    enc_path = MODEL_DIR / "ncf_encoders.pkl"

    tf = _try_import("tensorflow")
    if tf is None:
        log.warning("TensorFlow not installed – NCF skipped")
        log.warning("  Install with:  pip install tensorflow")
        return None, {}, {}

    # Import build_ncf from recommender
    try:
        sys.path.insert(0, str(BASE_DIR))
        from recommender import build_ncf
    except ImportError as e:
        log.error("Could not import build_ncf from recommender.py: %s", e)
        return None, {}, {}

    if (
        (NCF_WEIGHTS_PATH.exists() and NCF_META_PATH.exists() and enc_path.exists())
        or (ncf_path.exists() and enc_path.exists())
    ) and not FORCE_RETRAIN:
        log.info("NCF model already exists – loading")
        try:
            loaded = _load_ncf_artifacts(
                build_ncf=build_ncf,
                enc_path=enc_path,
                tf_module=tf,
                allow_legacy_keras=True,
            )
        except Exception as exc:
            loaded = None
            log.warning("Existing NCF model could not be loaded; retraining NCF")
            log.warning("  Reason: %s", exc)
        if loaded is not None:
            return loaded
    tf_uses_gpu = _configure_tensorflow_training(tf)

    log.info("Sampling implicit interactions for NCF (%d rows) …", NCF_SAMPLE_ROWS)
    interaction_df = _sample_csv(
        train_path,
        NCF_SAMPLE_ROWS,
        usecols=["userId", "movieId", "rating", "timestamp"],
    )
    interaction_df["rating"] = interaction_df["rating"].astype(np.float32)
    interaction_df["timestamp"] = interaction_df["timestamp"].astype(np.int64)

    pos_df = interaction_df[interaction_df["rating"] >= NCF_POSITIVE_RATING].copy()
    if pos_df.empty:
        log.warning("No positive NCF samples found at rating >= %.1f", NCF_POSITIVE_RATING)
        return None, {}, {}

    latest_ts = int(interaction_df["timestamp"].max() or 0)
    recency_days = np.maximum(
        0.0,
        (latest_ts - pos_df["timestamp"].to_numpy(dtype=np.int64)) / 86400.0,
    )
    recency_decay = np.exp(-recency_days / 365.0).astype(np.float32)

    pos_df = pos_df[["userId", "movieId", "rating", "timestamp"]].copy()
    pos_df["label"] = 1.0
    pos_df["sample_weight"] = np.clip(
        (pos_df["rating"].to_numpy(dtype=np.float32) / 5.0) * (0.5 + recency_decay),
        0.25,
        2.0,
    )

    log.info("Generating %dx popularity-weighted hard negatives …", NEG_RATIO)
    item_pool = np.asarray(sorted({_coerce_int(x) for x in all_item_ids}), dtype=np.int32)
    item_pool_set = set(_coerce_int(x) for x in item_pool.tolist())
    user_items: Dict[int, set[int]] = {}
    for uid, mid in zip(interaction_df["userId"], interaction_df["movieId"]):
        user_items.setdefault(_coerce_int(uid), set()).add(_coerce_int(mid))

    with get_db() as conn:
        item_weight_rows = conn.execute(
            """
            SELECT movieId, num_ratings, trending_score
            FROM movies
            """
        ).fetchall()
    popularity_lookup = {
        _coerce_int(row[0]): float(max(float(row[1] or 0), 1.0)) * (1.0 + float(row[2] or 0.0))
        for row in item_weight_rows
    }
    popularity_weights = np.array(
        [np.sqrt(popularity_lookup.get(_coerce_int(movie_id), 1.0)) for movie_id in item_pool],
        dtype=np.float64,
    )
    popularity_weights /= popularity_weights.sum()

    neg_rows = []
    rng = np.random.default_rng(42)

    for row in pos_df.itertuples(index=False):
        uid = _coerce_int(row.userId)
        seen = user_items.get(uid, set())
        sampled_negatives: list[int] = []
        sampled_set: set[int] = set()
        draw_rounds = 0

        while len(sampled_negatives) < NEG_RATIO and draw_rounds < 6:
            draws = rng.choice(
                item_pool,
                size=max(NEG_RATIO * 3, 12),
                replace=True,
                p=popularity_weights,
            )
            for mid in draws.tolist():
                if mid in seen or mid in sampled_set:
                    continue
                sampled_set.add(_coerce_int(mid))
                sampled_negatives.append(_coerce_int(mid))
                if len(sampled_negatives) >= NEG_RATIO:
                    break
            draw_rounds += 1

        if len(sampled_negatives) < NEG_RATIO:
            unseen_candidates = list(item_pool_set - seen - sampled_set)
            if unseen_candidates:
                take = min(NEG_RATIO - len(sampled_negatives), len(unseen_candidates))
                sampled_negatives.extend(
                    _coerce_int(mid)
                    for mid in rng.choice(
                        np.asarray(unseen_candidates, dtype=np.int32),
                        size=take,
                        replace=False,
                    ).tolist()
                )

        for mid in sampled_negatives:
            popularity_weight = float(np.sqrt(popularity_lookup.get(_coerce_int(mid), 1.0)))
            neg_rows.append({
                "userId": uid,
                "movieId": _coerce_int(mid),
                "label": 0.0,
                "sample_weight": float(np.clip(0.9 + (0.1 * popularity_weight), 0.9, 1.5)),
            })

    ncf_df = pd.concat(
        [
            pos_df[["userId", "movieId", "label", "sample_weight"]],
            pd.DataFrame(neg_rows),
        ],
        ignore_index=True,
    )
    ncf_df = ncf_df.sample(frac=1, random_state=42).reset_index(drop=True)
    log.info("  NCF dataset: %d rows (%d pos + %d neg)",
             len(ncf_df), len(pos_df), len(neg_rows))

    users_u  = ncf_df["userId"].unique()
    items_u  = ncf_df["movieId"].unique()
    user_enc = {_coerce_int(u): i for i, u in enumerate(users_u)}
    item_enc = {_coerce_int(m): i for i, m in enumerate(items_u)}

    ncf_df["u_enc"] = ncf_df["userId"].map(user_enc)
    ncf_df["i_enc"] = ncf_df["movieId"].map(item_enc)

    X = ncf_df[["u_enc", "i_enc"]].values
    y = ncf_df["label"].values
    sample_weights = ncf_df["sample_weight"].astype(np.float32).values
    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
        X, y, sample_weights, test_size=0.10, random_state=42, stratify=y
    )

    def make_ds(X_, y_, w_, training=True):
        ds = tf.data.Dataset.from_tensor_slices((X_, y_, w_))
        ds = ds.map(lambda f, l, w: ((f[0], f[1]), l, w),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        if training:
            ds = ds.shuffle(150_000)
        return ds.batch(NCF_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    tf_device = "/GPU:0" if tf_uses_gpu else "/CPU:0"
    with tf.device(tf_device):
        model = build_ncf(
            len(users_u),
            len(items_u),
            mf_dim=NCF_EMBED_DIM,
            mlp_layers=NCF_MLP_LAYERS,
            dropout=NCF_DROPOUT,
            learning_rate=1e-3,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=NCF_PATIENCE, restore_best_weights=True, monitor="val_auc", mode="max"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", factor=0.5, patience=3, min_lr=1e-6, mode="max"
            ),
        ]
        log.info("Training NCF (NeuMF) on %s …", "GPU" if tf_uses_gpu else "CPU")
        history = model.fit(
            make_ds(X_tr, y_tr, w_tr, training=True),
            epochs=NCF_EPOCHS,
            validation_data=make_ds(X_va, y_va, w_va, training=False),
            callbacks=callbacks,
            verbose=1,
        )
    best_val_auc = max(history.history.get("val_auc", [0.0]))
    log.info("  Best validation AUC: %.4f", best_val_auc)

    _save_ncf_artifacts(model, user_enc, item_enc, enc_path)
    log.info("NCF saved -> %s", NCF_WEIGHTS_PATH)
    return model, user_enc, item_enc


# =============================================================================
#  STEP 11 — TRAIN XGBOOST
# =============================================================================
def train_xgb(
    train_path: Path,
    svd_model: Any,
    als_model: Any,
    als_item_ids: list[int],
    als_user_ids: list[int],
    ncf_model: Any,
    ncf_user_enc: dict[int, int],
    ncf_item_enc: dict[int, int],
    movie_sbert_embeddings: dict[int, np.ndarray],
    user_taste_vectors: np.ndarray,
    user_taste_id_map: dict[int, int],
) -> tuple[Any, dict[str, Any]]:
    xgb = _try_import("xgboost")
    if xgb is None:
        log.warning("xgboost not installed – XGB skipped")
        log.warning("  Install with:  pip install xgboost")
        return None, {"feature_columns": list(LEGACY_XGB_FEATURE_COLS), "user_stats": {}, "movie_stats": {}}

    if (
        (XGB_MODEL_JSON_PATH.exists() and XGB_MODEL_META_PATH.exists() or XGB_LEGACY_PATH.exists())
        and XGB_FEATURE_CONTEXT_PATH.exists()
        and not FORCE_RETRAIN
        and not FORCE_XGB_RETRAIN
    ):
        log.info("XGBoost model already exists – loading")
        model = _load_xgb_bundle(xgb)
        if model is None:
            log.info("Existing XGBoost bundle could not be loaded – retraining")
        else:
            with open(XGB_FEATURE_CONTEXT_PATH, "rb") as f:
                feature_context = pickle.load(f)
            feature_columns = set(feature_context.get("feature_columns") or [])
            required_columns = {
                "rating_stddev",
                "ensemble_mean",
                "ensemble_std",
                "score_range",
                "user_rating_std",
                "user_recency_score",
                "user_rating_freq_trend",
                "user_movie_rating_diff",
                "genre_affinity_score",
                "item_popularity_decay",
                "item_recent_rating_velocity",
                "tag_cooccurrence_strength",
                "sbert_sim",
            }
            if required_columns.issubset(feature_columns):
                return model, feature_context
            log.info("Existing XGBoost feature context is outdated – retraining")

    log.info("Sampling for XGBoost (%d rows) …", XGB_SAMPLE_ROWS)
    ratings_df = _sample_csv(
        train_path,
        XGB_SAMPLE_ROWS,
        usecols=["userId", "movieId", "rating", "timestamp"],
    )

    with get_db() as conn:
        movie_meta_df = pd.read_sql(
            """
            SELECT movieId, genres, avg_rating, rating_stddev, num_ratings, sentiment_score, trending_score
            FROM movies
            """,
            conn,
        )
    genre_tokens = _select_genre_tokens(movie_meta_df)
    genome_feature_df = _load_genome_feature_frame(train_path)
    user_feature_df, user_genre_affinity_map, user_tag_profile_map = _load_user_feature_frame(
        train_path,
        genre_tokens=genre_tokens,
        genome_feature_df=genome_feature_df,
        movie_meta_df=movie_meta_df,
    )

    df = _build_xgb_feature_frame(
        ratings_df=ratings_df,
        movie_meta_df=movie_meta_df,
        user_feature_df=user_feature_df,
        genome_feature_df=genome_feature_df,
        user_genre_affinity_map=user_genre_affinity_map,
        user_tag_profile_map=user_tag_profile_map,
        movie_sbert_embeddings=movie_sbert_embeddings,
        user_taste_vectors=user_taste_vectors,
        user_taste_id_map=user_taste_id_map,
        svd_model=svd_model,
        als_model=als_model,
        als_item_ids=als_item_ids,
        als_user_ids=als_user_ids,
        ncf_model=ncf_model,
        ncf_user_enc=ncf_user_enc,
        ncf_item_enc=ncf_item_enc,
        genre_tokens=genre_tokens,
    )

    feature_cols = _compose_xgb_feature_columns(genre_tokens, genome_feature_df)
    for column_name in feature_cols:
        if column_name not in df.columns:
            df[column_name] = 0.0
    df["label"] = (df["rating"] >= 4.0).astype(np.int32)
    max_ts = int(df["timestamp"].max() or 0)
    df["sample_weight"] = np.exp(
        -np.maximum(0.0, (max_ts - df["timestamp"].to_numpy(dtype=np.int64)) / 86400.0) / 540.0
    ).astype(np.float32)

    train_df, val_df = _split_xgb_train_validation(df)
    scaler = MinMaxScaler()
    X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)
    w_train = train_df["sample_weight"].to_numpy(dtype=np.float32)
    w_val = val_df["sample_weight"].to_numpy(dtype=np.float32)
    scale_pos_weight = _compute_scale_pos_weight(y_train, w_train)

    log.info(
        "Training XGBoost with %d train rows, %d val rows, scale_pos_weight=%.4f",
        len(train_df),
        len(val_df),
        scale_pos_weight,
    )
    log.info("XGBoost training device: %s", "GPU (CUDA)" if TRAIN_USE_GPU and _gpu_available() else "CPU")

    binary_configs = [
        {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 5,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        {
            "max_depth": 6,
            "learning_rate": 0.03,
            "n_estimators": 800,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "min_child_weight": 6,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    ]
    rank_configs = [
        {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 5,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        {
            "max_depth": 6,
            "learning_rate": 0.03,
            "n_estimators": 700,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "min_child_weight": 6,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    ] if XGB_ENABLE_RANK_OBJECTIVE else []

    def is_better(candidate: dict[str, Any], incumbent: Optional[dict[str, Any]]) -> bool:
        if incumbent is None:
            return True
        candidate_key = (
            float(candidate["NDCG@10"]),
            float(candidate["AUC"]),
            -float(candidate["LogLoss"]),
        )
        incumbent_key = (
            float(incumbent["NDCG@10"]),
            float(incumbent["AUC"]),
            -float(incumbent["LogLoss"]),
        )
        return candidate_key > incumbent_key

    candidate_results: list[dict[str, Any]] = []
    best_bundle: Optional[XGBModelBundle] = None
    best_metrics: Optional[dict[str, Any]] = None

    for config in binary_configs:
        log.info(
            "XGB trial objective=binary:logistic depth=%d lr=%.3f trees=%d subsample=%.2f colsample=%.2f",
            config["max_depth"],
            config["learning_rate"],
            config["n_estimators"],
            config["subsample"],
            config["colsample_bytree"],
        )
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            min_child_weight=config["min_child_weight"],
            gamma=config["gamma"],
            reg_alpha=config["reg_alpha"],
            scale_pos_weight=scale_pos_weight,
            reg_lambda=config["reg_lambda"],
            eval_metric=["auc", "logloss"],
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            random_state=42,
            n_jobs=-1,
            **_xgb_runtime_kwargs(),
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        base_scores = np.asarray(model.predict_proba(X_val)[:, 1], dtype=np.float32)
        calibrator, calibration_name, _ = _fit_xgb_calibrator(base_scores, y_val)
        calibrated_scores = _apply_xgb_calibrator(base_scores, calibrator, calibration_name)
        metrics = _score_xgb_validation(
            val_df,
            calibrated_scores,
            "binary:logistic",
            calibration_name,
            {
                **config,
                "best_iteration": getattr(model, "best_iteration", config["n_estimators"]),
            },
            scale_pos_weight,
        )
        candidate_results.append(metrics)
        log.info(
            "  Validation AUC=%.4f LogLoss=%.4f NDCG@10=%.4f calibration=%s",
            metrics["AUC"],
            metrics["LogLoss"],
            metrics["NDCG@10"],
            calibration_name,
        )
        if is_better(metrics, best_metrics):
            best_metrics = metrics
            best_bundle = XGBModelBundle(
                model=model,
                scaler=scaler,
                calibrator=calibrator,
                model_kind="binary:logistic",
                calibration_kind=calibration_name,
            )

    if rank_configs:
        train_rank_df = train_df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
        val_rank_df = val_df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
        X_train_rank = scaler.transform(train_rank_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        X_val_rank = scaler.transform(val_rank_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        y_train_rank = train_rank_df["label"].to_numpy(dtype=np.int32)
        y_val_rank = val_rank_df["label"].to_numpy(dtype=np.int32)
        train_qid = train_rank_df["userId"].to_numpy(dtype=np.int64)
        val_qid = val_rank_df["userId"].to_numpy(dtype=np.int64)

        for config in rank_configs:
            log.info(
                "XGB trial objective=rank:ndcg depth=%d lr=%.3f trees=%d subsample=%.2f colsample=%.2f",
                config["max_depth"],
                config["learning_rate"],
                config["n_estimators"],
                config["subsample"],
                config["colsample_bytree"],
            )
            model = xgb.XGBRanker(
                objective="rank:ndcg",
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                min_child_weight=config["min_child_weight"],
                gamma=config["gamma"],
                reg_alpha=config["reg_alpha"],
                reg_lambda=config["reg_lambda"],
                eval_metric=["ndcg@10"],
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                random_state=42,
                n_jobs=-1,
                **_xgb_runtime_kwargs(),
            )
            model.fit(
                X_train_rank,
                y_train_rank,
                qid=train_qid,
                eval_set=[(X_val_rank, y_val_rank)],
                eval_qid=[val_qid],
                verbose=False,
            )
            base_scores = np.asarray(model.predict(X_val_rank), dtype=np.float32)
            calibrator, calibration_name, _ = _fit_xgb_calibrator(base_scores, y_val_rank)
            calibrated_scores = _apply_xgb_calibrator(base_scores, calibrator, calibration_name)
            metrics = _score_xgb_validation(
                val_rank_df,
                calibrated_scores,
                "rank:ndcg",
                calibration_name,
                {
                    **config,
                    "best_iteration": getattr(model, "best_iteration", config["n_estimators"]),
                },
                scale_pos_weight,
            )
            candidate_results.append(metrics)
            log.info(
                "  Validation AUC=%.4f LogLoss=%.4f NDCG@10=%.4f calibration=%s",
                metrics["AUC"],
                metrics["LogLoss"],
                metrics["NDCG@10"],
                calibration_name,
            )
            if is_better(metrics, best_metrics):
                best_metrics = metrics
                best_bundle = XGBModelBundle(
                    model=model,
                    scaler=scaler,
                    calibrator=calibrator,
                    model_kind="rank:ndcg",
                    calibration_kind=calibration_name,
                )
    else:
        log.info(
            "Skipping XGB rank:ndcg trials; set XGB_ENABLE_RANK_OBJECTIVE=1 to enable grouped ranking search."
        )

    if best_bundle is None or best_metrics is None:
        raise RuntimeError("XGBoost training failed to produce a model")

    log.info(
        "Selected XGBoost objective=%s calibration=%s with validation AUC=%.4f LogLoss=%.4f NDCG@10=%.4f",
        best_metrics["objective"],
        best_metrics["calibration"],
        best_metrics["AUC"],
        best_metrics["LogLoss"],
        best_metrics["NDCG@10"],
    )

    _save_xgb_bundle(best_bundle)
    feature_context = _build_xgb_feature_context(
        user_feature_df,
        genome_feature_df,
        genre_tokens=genre_tokens,
        feature_columns=feature_cols,
        user_genre_affinity_map=user_genre_affinity_map,
        user_tag_profile_map=user_tag_profile_map,
    )
    feature_context.update(
        {
            "scaler": "minmax",
            "best_objective": best_metrics["objective"],
            "calibration": best_metrics["calibration"],
            "sbert_enabled": bool(movie_sbert_embeddings) and bool(user_taste_id_map),
            "validation_metrics": {
                "AUC": best_metrics["AUC"],
                "LogLoss": best_metrics["LogLoss"],
                "NDCG@10": best_metrics["NDCG@10"],
            },
            "objective_search": candidate_results,
            "scale_pos_weight": round(float(scale_pos_weight), 4),
            "xgb_device": "cuda" if TRAIN_USE_GPU and _gpu_available() else "cpu",
        }
    )
    with open(XGB_FEATURE_CONTEXT_PATH, "wb") as f:
        pickle.dump(feature_context, f)
    log.info("XGBoost saved -> %s", XGB_LEGACY_PATH)
    return best_bundle, feature_context


# =============================================================================
#  STEP 12 — EVALUATE + REPORT
# =============================================================================
def evaluate_and_report(
    svd_model,
    ncf_model,
    ncf_user_enc,
    ncf_item_enc,
    als_model,
    als_item_ids,
    als_user_ids,
    xgb_model,
    xgb_feature_context: dict[str, Any],
    movie_sbert_embeddings: dict[int, np.ndarray],
    user_taste_vectors: np.ndarray,
    user_taste_id_map: dict[int, int],
):
    import json
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, roc_auc_score, f1_score,
        log_loss, precision_score, recall_score,
    )

    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        log.warning("test.csv not found – skipping evaluation")
        return {}

    log.info("Loading random test sample for evaluation …")
    test_df = _sample_csv(
        test_path,
        EVAL_SAMPLE_ROWS,
        usecols=["userId", "movieId", "rating"],
    )
    log.info("  Evaluation sample: %d rows", len(test_df))

    report: Dict[str, Dict] = {}

    # SVD
    if svd_model is not None:
        log.info("Evaluating SVD …")
        try:
            preds   = [svd_model.predict(int(str(r.userId)), int(str(r.movieId))).est
                       for r in test_df.itertuples()]
            actuals = np.asarray(test_df["rating"].values, dtype=np.float32)
            preds_arr = np.asarray(preds, dtype=np.float32)
            svd_scores = np.clip(preds_arr / 5.0, 1e-6, 1 - 1e-6)
            svd_eval_df = test_df[["userId", "movieId"]].copy()
            svd_eval_df["label"] = (test_df["rating"] >= 4.0).astype(np.int32)
            svd_eval_df["pred_score"] = svd_scores
            metrics = {
                "RMSE": round(float(np.sqrt(mean_squared_error(actuals, preds_arr))), 4),
                "MAE":  round(float(mean_absolute_error(actuals, preds_arr)), 4),
                "AUC":  round(
                    float(roc_auc_score(svd_eval_df["label"], svd_scores))
                    if int(svd_eval_df["label"].sum()) > 0
                    else 0.0,
                    4,
                ),
            }
            metrics.update(_ranking_metrics_by_user(svd_eval_df, "pred_score", "label", k=10))
            report["SVD"] = metrics
        except Exception as e:
            log.warning("SVD eval error: %s", e)

    # NCF
    if ncf_model is not None and ncf_user_enc and ncf_item_enc:
        log.info("Evaluating NCF …")
        try:
            df = test_df[
                test_df["userId"].isin(ncf_user_enc) &
                test_df["movieId"].isin(ncf_item_enc)
            ].copy()
            if df.empty:
                raise ValueError("No overlapping user/item IDs for NCF evaluation")
            df["u_enc"] = df["userId"].map(ncf_user_enc)
            df["i_enc"] = df["movieId"].map(ncf_item_enc)
            df["label"] = (df["rating"] >= 4.0).astype(float)
            preds = np.asarray(
                ncf_model.predict(
                [df["u_enc"].values, df["i_enc"].values], verbose=0
                ).flatten(),
                dtype=np.float32,
            )
            y_true = np.asarray(df["label"].to_numpy(dtype=np.float32, copy=False), dtype=np.float32)
            y_bin = np.asarray(preds >= 0.5, dtype=np.int32)
            metrics = {
                "BCE":       round(float(log_loss(y_true, np.clip(preds, 1e-7, 1-1e-7))), 4),
                "AUC":       round(float(roc_auc_score(y_true, preds)) if y_true.sum() > 0 else 0.0, 4),
                "F1":        round(float(f1_score(y_true, y_bin, zero_division=0)), 4),
                "Precision": round(float(precision_score(y_true, y_bin, zero_division=0)), 4),
                "Recall":    round(float(recall_score(y_true, y_bin, zero_division=0)), 4),
            }
            df["pred_score"] = preds
            metrics.update(_ranking_metrics_by_user(df, "pred_score", "label", k=10))
            report["NCF"] = metrics
        except Exception as e:
            log.warning("NCF eval error: %s", e)

    # XGB
    if xgb_model is not None:
        log.info("Evaluating XGBoost …")
        try:
            with get_db() as conn:
                movie_meta_df = pd.read_sql(
                    """
                    SELECT movieId, genres, avg_rating, rating_stddev, num_ratings, sentiment_score, trending_score
                    FROM movies
                    """,
                    conn,
                )

            user_stats_map = dict(xgb_feature_context.get("user_stats") or {})
            movie_stats_map = dict(xgb_feature_context.get("movie_stats") or {})
            user_genre_affinity_map = {
                _coerce_int(user_id): {
                    str(genre_name): float(score)
                    for genre_name, score in dict(affinity_map).items()
                }
                for user_id, affinity_map in dict(xgb_feature_context.get("user_genre_affinity") or {}).items()
            }
            user_tag_profile_map = {
                _coerce_int(user_id): np.asarray(tag_profile, dtype=np.float32)
                for user_id, tag_profile in dict(xgb_feature_context.get("user_tag_profiles") or {}).items()
            }
            user_feature_columns = list(
                xgb_feature_context.get("user_feature_columns")
                or [
                    "user_activity_level",
                    "user_avg_rating",
                    "user_rating_std",
                    "user_recency_days_log",
                    "user_recency_score",
                    "user_rating_freq_trend",
                ]
            )
            movie_feature_columns = list(
                xgb_feature_context.get("movie_feature_columns")
                or [
                    "genome_mean_relevance",
                    "genome_max_relevance",
                    "genome_high_relevance_log",
                    "item_popularity_decay",
                    "item_recent_rating_velocity",
                ]
            )
            user_feature_df = pd.DataFrame(
                [
                    {"userId": _coerce_int(uid), **stats}
                    for uid, stats in user_stats_map.items()
                ]
            )
            movie_feature_df = pd.DataFrame(
                [
                    {"movieId": _coerce_int(mid), **stats}
                    for mid, stats in movie_stats_map.items()
                ]
            )
            if user_feature_df.empty:
                user_feature_df = pd.DataFrame(
                    columns=["userId", *user_feature_columns]
                )
            if movie_feature_df.empty:
                movie_feature_df = pd.DataFrame(
                    columns=["movieId", *movie_feature_columns]
                )
            genre_tokens = list(xgb_feature_context.get("genre_tokens") or [])

            df = _build_xgb_feature_frame(
                ratings_df=test_df.copy(),
                movie_meta_df=movie_meta_df,
                user_feature_df=user_feature_df,
                genome_feature_df=movie_feature_df,
                user_genre_affinity_map=user_genre_affinity_map,
                user_tag_profile_map=user_tag_profile_map,
                movie_sbert_embeddings=movie_sbert_embeddings,
                user_taste_vectors=user_taste_vectors,
                user_taste_id_map=user_taste_id_map,
                svd_model=svd_model,
                als_model=als_model,
                als_item_ids=als_item_ids,
                als_user_ids=als_user_ids,
                ncf_model=ncf_model,
                ncf_user_enc=ncf_user_enc,
                ncf_item_enc=ncf_item_enc,
                genre_tokens=genre_tokens,
            )
            feature_cols = list(xgb_feature_context.get("feature_columns") or XGB_FEATURE_COLS)
            if not feature_cols:
                feature_cols = list(LEGACY_XGB_FEATURE_COLS)
            X = np.asarray(df[feature_cols].to_numpy(dtype=np.float32, copy=False), dtype=np.float32)
            df["label"] = (df["rating"] >= 4.0).astype(int)
            y = np.asarray(df["label"].to_numpy(dtype=np.int32, copy=False), dtype=np.int32)
            proba = np.asarray(xgb_model.predict_proba(X)[:, 1], dtype=np.float32)
            y_bin = np.asarray(proba >= 0.5, dtype=np.int32)
            metrics = {
                "LogLoss":   round(float(log_loss(y, np.clip(proba, 1e-7, 1-1e-7))), 4),
                "AUC":       round(float(roc_auc_score(y, proba)) if y.sum() > 0 else 0.0, 4),
                "F1":        round(float(f1_score(y, y_bin, zero_division=0)), 4),
            }
            df["pred_score"] = proba
            metrics.update(_ranking_metrics_by_user(df, "pred_score", "label", k=10))
            report["XGB"] = metrics
        except Exception as e:
            log.warning("XGB eval error: %s", e)

    # Print report
    log.info("=" * 60)
    log.info("  EVALUATION REPORT")
    log.info("=" * 60)
    for model_name, metrics in report.items():
        log.info("  %s", model_name)
        for k, v in metrics.items():
            log.info("    %-14s %s", k, v)
        log.info("")

    out_path = MODEL_DIR / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": _utc_now().isoformat(),
            "test_ratio":   TEST_RATIO,
            "metrics":      report,
        }, f, indent=2)
    log.info("Evaluation report saved -> %s", out_path)
    return report


# =============================================================================
#  MAIN
# =============================================================================
def main():
    t0 = time.time()

    # Step 0 — checks
    preflight()

    # Step 1 — DB
    init_db()

    # Steps 2-3-4 — load data
    movies_df = load_and_aggregate()
    movies_df = load_tags(movies_df)
    load_genome()

    # Step 5 — write to DB
    preserved_admin_movies = write_movies_to_db(movies_df)

    # Step 6 — split
    train_path, test_path = temporal_split()

    log.info("=" * 60)
    log.info("  SBERT TASTE FEATURES")
    log.info("=" * 60)
    movie_sbert_embeddings = _load_movie_sbert_embeddings()
    user_taste_vectors, user_taste_id_map = _load_user_taste_vectors(
        train_path,
        movie_sbert_embeddings,
    )

    # Collect all unique item IDs for NCF negative sampling
    with get_db() as conn:
        all_item_ids = np.array(
            [r[0] for r in conn.execute("SELECT movieId FROM movies").fetchall()],
            dtype=np.int32,
        )

    restore_admin_movies_to_db(preserved_admin_movies)

    # Steps 7-10 — train models
    log.info("=" * 60)
    log.info("  TRAINING MODELS")
    log.info("=" * 60)

    svd_model                           = train_svd(train_path)
    als_model, als_item_ids, als_user_ids = train_als(train_path)
    ncf_model, ncf_user_enc, ncf_item_enc = train_ncf(train_path, all_item_ids)
    xgb_model, xgb_feature_context      = train_xgb(
        train_path,
        svd_model,
        als_model,
        als_item_ids,
        als_user_ids,
        ncf_model,
        ncf_user_enc,
        ncf_item_enc,
        movie_sbert_embeddings,
        user_taste_vectors,
        user_taste_id_map,
    )

    # Step 11 — evaluate
    evaluate_and_report(
        svd_model, ncf_model, ncf_user_enc, ncf_item_enc,
        als_model, als_item_ids, als_user_ids, xgb_model, xgb_feature_context,
        movie_sbert_embeddings, user_taste_vectors, user_taste_id_map,
    )

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    log.info("=" * 60)
    log.info("  Training complete in %d min %d sec", mins, secs)
    log.info("  Models saved to:  %s", MODEL_DIR)
    log.info("  Now start the server with:")
    log.info("    uvicorn app:app --reload --host 0.0.0.0 --port 8000")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
