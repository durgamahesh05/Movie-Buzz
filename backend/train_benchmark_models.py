"""
Benchmark model trainer for MovieBuzz.

Trains LightGBM, CatBoost, Logistic Regression, and Random Forest on a
manageable sample of the MovieLens train/test CSVs, writes model artifacts,
updates eval_report.json with a normalized metrics schema, and mirrors the
summary to MongoDB when available.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import env_int
from db import format_db_target, get_collection

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover - optional dependency guard
    lgb = None
    LIGHTGBM_IMPORT_ERROR = str(exc)
else:
    LIGHTGBM_IMPORT_ERROR = ""

try:
    from catboost import CatBoostClassifier, Pool
except Exception as exc:  # pragma: no cover - optional dependency guard
    CatBoostClassifier = None
    Pool = None
    CATBOOST_IMPORT_ERROR = str(exc)
else:
    CATBOOST_IMPORT_ERROR = ""

try:
    import surprise  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency guard
    SURPRISE_IMPORT_ERROR = str(exc)
else:
    SURPRISE_IMPORT_ERROR = ""


log = logging.getLogger("benchmark_training")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_CSV_PATH = DATA_DIR / "train.csv"
TEST_CSV_PATH = DATA_DIR / "test.csv"
MOVIES_CSV_PATH = DATA_DIR / "movies.csv"

LIGHTGBM_MODEL_PATH = MODEL_DIR / "lightgbm_model.txt"
CATBOOST_MODEL_PATH = MODEL_DIR / "catboost_model.cbm"
LOGREG_MODEL_PATH = MODEL_DIR / "logreg_model.joblib"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
BENCHMARK_SCHEMA_PATH = MODEL_DIR / "benchmark_feature_schema.json"
EVAL_REPORT_PATH = MODEL_DIR / "eval_report.json"
SVD_MODEL_PATH = MODEL_DIR / "svd_model.pkl"

RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
YEAR_PATTERN = re.compile(r"\((\d{4})\)")
DEFAULT_REFERENCE_YEAR = 2026
TARGETS = {
    "auc": 0.85,
    "f1": 0.75,
    "precision_at_10": 0.85,
    "mrr": 0.95,
    "ndcg_10": 0.95,
}
DEFAULT_TRAIN_ROWS = env_int("MOVIEBUZZ_BENCHMARK_TRAIN_ROWS", default=300_000)
DEFAULT_VAL_ROWS = env_int("MOVIEBUZZ_BENCHMARK_VAL_ROWS", default=75_000)
DEFAULT_TEST_ROWS = env_int("MOVIEBUZZ_BENCHMARK_TEST_ROWS", default=120_000)
DEFAULT_CHUNK_SIZE = env_int("MOVIEBUZZ_BENCHMARK_CHUNK_SIZE", default=200_000)
DEFAULT_TOP_GENRES = env_int("MOVIEBUZZ_BENCHMARK_TOP_GENRES", default=16)


@dataclass
class FeatureBundle:
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    numeric_columns: list[str]
    lightgbm_categorical_columns: list[str]
    catboost_categorical_columns: list[str]
    target_column: str
    user_column: str
    top_genres: list[str]
    reference_year: int
    label_threshold: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _round_metric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, 4)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    try:
        return _round_metric(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _ensure_input_files() -> None:
    missing = [path.name for path in (TRAIN_CSV_PATH, TEST_CSV_PATH, MOVIES_CSV_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing training inputs in backend/data: " + ", ".join(sorted(missing))
        )


def _normalize_genres(raw_value: Any) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    tokens = [
        token.strip()
        for token in text.replace("|", " ").split()
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


def _extract_year(title: Any) -> float:
    match = YEAR_PATTERN.search(str(title or ""))
    if not match:
        return float("nan")
    try:
        return float(match.group(1))
    except Exception:
        return float("nan")


def _sample_csv_rows(
    path: Path,
    sample_size: int,
    seed: int,
    chunksize: int,
) -> pd.DataFrame:
    if sample_size <= 0:
        return pd.DataFrame(columns=RATING_COLUMNS)

    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None
    dtype_map = {
        "userId": "int64",
        "movieId": "int64",
        "rating": "float32",
        "timestamp": "int64",
    }

    for chunk in pd.read_csv(path, usecols=RATING_COLUMNS, dtype=dtype_map, chunksize=chunksize):
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["userId", "movieId", "rating"]).copy()
        if chunk.empty:
            continue
        chunk["_priority"] = rng.random(len(chunk))
        if reservoir is None:
            reservoir = chunk
        else:
            reservoir = pd.concat([reservoir, chunk], ignore_index=True, copy=False)

        trim_size = max(sample_size * 2, sample_size + chunksize)
        if len(reservoir.index) > trim_size:
            reservoir = (
                reservoir.nsmallest(sample_size, columns="_priority")
                .reset_index(drop=True)
            )

    if reservoir is None or reservoir.empty:
        return pd.DataFrame(columns=RATING_COLUMNS)

    return (
        reservoir.nsmallest(sample_size, columns="_priority")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )


def _load_movies_catalog(top_genre_limit: int) -> tuple[pd.DataFrame, list[str]]:
    movies = pd.read_csv(
        MOVIES_CSV_PATH,
        usecols=["movieId", "title", "genres"],
        dtype={"movieId": "int64", "title": "string", "genres": "string"},
    ).copy()
    movies["title"] = movies["title"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    movies["movie_year"] = movies["title"].map(_extract_year)
    movies["genre_tokens"] = movies["genres"].map(_normalize_genres)
    movies["primary_genre"] = movies["genre_tokens"].map(lambda tokens: tokens[0] if tokens else "unknown")
    movies["genre_count"] = movies["genre_tokens"].map(len).astype("int16")

    exploded = movies[["movieId", "genre_tokens"]].explode("genre_tokens").dropna()
    if exploded.empty:
        top_genres: list[str] = []
    else:
        top_genres = (
            exploded["genre_tokens"]
            .value_counts()
            .head(max(0, top_genre_limit))
            .index
            .tolist()
        )

    for genre in top_genres:
        feature_name = _genre_feature_name(genre)
        movies[feature_name] = movies["genre_tokens"].map(
            lambda tokens, target=genre: 1.0 if target in tokens else 0.0
        ).astype("float32")

    return movies, top_genres


def _prepare_interactions(df: pd.DataFrame, movies: pd.DataFrame, label_threshold: float) -> pd.DataFrame:
    frame = df.copy()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce").astype("float32")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0).astype("int64")
    frame["label"] = (frame["rating"] >= float(label_threshold)).astype("int8")
    merged = frame.merge(movies, on="movieId", how="left", copy=False)
    merged["genres"] = merged["genres"].fillna("")
    merged["genre_tokens"] = merged["genre_tokens"].map(
        lambda tokens: tokens if isinstance(tokens, list) else []
    )
    merged["primary_genre"] = merged["primary_genre"].fillna("unknown")
    return merged


def _load_svd_model() -> Any | None:
    if SURPRISE_IMPORT_ERROR or not SVD_MODEL_PATH.exists():
        return None
    try:
        with open(SVD_MODEL_PATH, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        log.warning("Could not load SVD artifact for benchmark features: %s", exc)
        return None


def _predict_svd_scores(model: Any, frame: pd.DataFrame) -> np.ndarray:
    scores = np.empty(len(frame.index), dtype=np.float32)
    if len(frame.index) == 0:
        return scores

    user_ids = frame["userId"].to_numpy(dtype=np.int64, copy=False)
    movie_ids = frame["movieId"].to_numpy(dtype=np.int64, copy=False)
    for index, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
        try:
            scores[index] = float(model.predict(int(user_id), int(movie_id)).est)
        except Exception:
            scores[index] = 0.0
    return scores


def _build_user_stats(train_frame: pd.DataFrame, global_mean: float, reference_ts: int) -> pd.DataFrame:
    user_stats = (
        train_frame.groupby("userId", observed=True)
        .agg(
            user_avg_rating=("rating", "mean"),
            user_rating_std=("rating", "std"),
            user_num_ratings=("rating", "size"),
            user_positive_ratio=("label", "mean"),
            user_first_ts=("timestamp", "min"),
            user_last_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0.0)
    user_stats["user_log_num_ratings"] = np.log1p(user_stats["user_num_ratings"]).astype("float32")
    user_stats["user_bias"] = user_stats["user_avg_rating"] - float(global_mean)
    user_stats["user_recency_days"] = (
        (float(reference_ts) - user_stats["user_last_ts"]) / 86400.0
    ).clip(lower=0.0)
    user_stats["user_activity_span_days"] = (
        (user_stats["user_last_ts"] - user_stats["user_first_ts"]) / 86400.0
    ).clip(lower=0.0)
    return user_stats.drop(columns=["user_first_ts", "user_last_ts"])


def _build_movie_stats(train_frame: pd.DataFrame, global_mean: float, reference_ts: int) -> pd.DataFrame:
    movie_stats = (
        train_frame.groupby("movieId", observed=True)
        .agg(
            movie_avg_rating=("rating", "mean"),
            movie_rating_std=("rating", "std"),
            movie_num_ratings=("rating", "size"),
            movie_positive_ratio=("label", "mean"),
            movie_first_ts=("timestamp", "min"),
            movie_last_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    movie_stats["movie_rating_std"] = movie_stats["movie_rating_std"].fillna(0.0)
    movie_stats["movie_log_num_ratings"] = np.log1p(movie_stats["movie_num_ratings"]).astype("float32")
    movie_stats["movie_bias"] = movie_stats["movie_avg_rating"] - float(global_mean)
    movie_stats["movie_recency_days"] = (
        (float(reference_ts) - movie_stats["movie_last_ts"]) / 86400.0
    ).clip(lower=0.0)
    movie_stats["movie_activity_span_days"] = (
        (movie_stats["movie_last_ts"] - movie_stats["movie_first_ts"]) / 86400.0
    ).clip(lower=0.0)
    return movie_stats.drop(columns=["movie_first_ts", "movie_last_ts"])


def _build_user_genre_stats(train_frame: pd.DataFrame, top_genres: list[str]) -> pd.DataFrame:
    if not top_genres:
        return pd.DataFrame(
            columns=[
                "userId",
                "genre_tokens",
                "user_genre_positive_ratio",
                "user_genre_avg_rating",
                "user_genre_interactions",
            ]
        )

    genre_frame = (
        train_frame[["userId", "label", "rating", "genre_tokens"]]
        .explode("genre_tokens")
        .dropna(subset=["genre_tokens"])
    )
    if genre_frame.empty:
        return pd.DataFrame(
            columns=[
                "userId",
                "genre_tokens",
                "user_genre_positive_ratio",
                "user_genre_avg_rating",
                "user_genre_interactions",
            ]
        )

    genre_frame = genre_frame[genre_frame["genre_tokens"].isin(top_genres)]
    if genre_frame.empty:
        return pd.DataFrame(
            columns=[
                "userId",
                "genre_tokens",
                "user_genre_positive_ratio",
                "user_genre_avg_rating",
                "user_genre_interactions",
            ]
        )

    return (
        genre_frame.groupby(["userId", "genre_tokens"], observed=True)
        .agg(
            user_genre_positive_ratio=("label", "mean"),
            user_genre_avg_rating=("rating", "mean"),
            user_genre_interactions=("label", "size"),
        )
        .reset_index()
    )


def _attach_user_genre_features(frame: pd.DataFrame, user_genre_stats: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    base = frame.reset_index(drop=True).copy()
    base["_row_id"] = np.arange(len(base.index))

    if user_genre_stats.empty:
        for column in (
            "user_genre_affinity_mean",
            "user_genre_affinity_max",
            "user_genre_rating_mean",
            "user_genre_history",
        ):
            base[column] = 0.0
        return base.drop(columns=["_row_id"])

    exploded = (
        base[["_row_id", "userId", "genre_tokens"]]
        .explode("genre_tokens")
        .dropna(subset=["genre_tokens"])
    )
    if exploded.empty:
        for column in (
            "user_genre_affinity_mean",
            "user_genre_affinity_max",
            "user_genre_rating_mean",
            "user_genre_history",
        ):
            base[column] = 0.0
        return base.drop(columns=["_row_id"])

    merged = exploded.merge(user_genre_stats, on=["userId", "genre_tokens"], how="left")
    aggregated = (
        merged.groupby("_row_id", observed=True)
        .agg(
            user_genre_affinity_mean=("user_genre_positive_ratio", "mean"),
            user_genre_affinity_max=("user_genre_positive_ratio", "max"),
            user_genre_rating_mean=("user_genre_avg_rating", "mean"),
            user_genre_history=("user_genre_interactions", "sum"),
        )
        .reset_index()
    )
    base = base.merge(aggregated, on="_row_id", how="left")
    for column in (
        "user_genre_affinity_mean",
        "user_genre_affinity_max",
        "user_genre_rating_mean",
        "user_genre_history",
    ):
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)

    return base.drop(columns=["_row_id"])


def _prepare_feature_bundle(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    top_genres: list[str],
    label_threshold: float,
) -> FeatureBundle:
    svd_model = _load_svd_model()
    global_mean = float(train_frame["rating"].mean())
    global_positive_rate = float(train_frame["label"].mean())
    reference_ts = int(train_frame["timestamp"].max())
    reference_year = DEFAULT_REFERENCE_YEAR
    movie_year_median = float(train_frame["movie_year"].dropna().median()) if train_frame["movie_year"].notna().any() else 2000.0

    user_stats = _build_user_stats(train_frame, global_mean, reference_ts)
    movie_stats = _build_movie_stats(train_frame, global_mean, reference_ts)
    user_genre_stats = _build_user_genre_stats(train_frame, top_genres)
    genre_feature_columns = [_genre_feature_name(genre) for genre in top_genres]

    def enrich(frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.merge(user_stats, on="userId", how="left", copy=False)
        enriched = enriched.merge(movie_stats, on="movieId", how="left", copy=False)
        enriched = _attach_user_genre_features(enriched, user_genre_stats)
        enriched["movie_year"] = pd.to_numeric(enriched["movie_year"], errors="coerce").fillna(movie_year_median)
        enriched["movie_age"] = (float(reference_year) - enriched["movie_year"]).clip(lower=0.0)
        enriched["interaction_recency_days"] = (
            (float(reference_ts) - enriched["timestamp"]) / 86400.0
        ).clip(lower=0.0)
        enriched["bias_gap"] = (
            pd.to_numeric(enriched["user_avg_rating"], errors="coerce")
            - pd.to_numeric(enriched["movie_avg_rating"], errors="coerce")
        )
        enriched["positive_rate_gap"] = (
            pd.to_numeric(enriched["user_positive_ratio"], errors="coerce")
            - pd.to_numeric(enriched["movie_positive_ratio"], errors="coerce")
        )
        genre_count = pd.to_numeric(enriched["genre_count"], errors="coerce").fillna(0.0)
        enriched["genre_count"] = genre_count
        enriched["user_genre_known_share"] = (
            pd.to_numeric(enriched["user_genre_history"], errors="coerce").fillna(0.0)
            / np.maximum(genre_count.to_numpy(dtype=np.float32), 1.0)
        )
        enriched["primary_genre"] = enriched["primary_genre"].fillna("unknown").astype("string")
        enriched["userId_cat"] = enriched["userId"].astype("string")
        enriched["movieId_cat"] = enriched["movieId"].astype("string")
        if svd_model is not None:
            enriched["svd_score"] = _predict_svd_scores(svd_model, enriched)
        else:
            enriched["svd_score"] = global_mean
        enriched["svd_user_gap"] = (
            pd.to_numeric(enriched["svd_score"], errors="coerce")
            - pd.to_numeric(enriched["user_avg_rating"], errors="coerce")
        )
        enriched["svd_movie_gap"] = (
            pd.to_numeric(enriched["svd_score"], errors="coerce")
            - pd.to_numeric(enriched["movie_avg_rating"], errors="coerce")
        )

        fill_defaults: dict[str, float] = {
            "user_avg_rating": global_mean,
            "user_rating_std": 0.0,
            "user_num_ratings": 0.0,
            "user_positive_ratio": global_positive_rate,
            "user_log_num_ratings": 0.0,
            "user_bias": 0.0,
            "user_recency_days": 0.0,
            "user_activity_span_days": 0.0,
            "movie_avg_rating": global_mean,
            "movie_rating_std": 0.0,
            "movie_num_ratings": 0.0,
            "movie_positive_ratio": global_positive_rate,
            "movie_log_num_ratings": 0.0,
            "movie_bias": 0.0,
            "movie_recency_days": 0.0,
            "movie_activity_span_days": 0.0,
            "movie_year": movie_year_median,
            "movie_age": max(float(reference_year) - movie_year_median, 0.0),
            "interaction_recency_days": 0.0,
            "bias_gap": 0.0,
            "positive_rate_gap": 0.0,
            "genre_count": 0.0,
            "user_genre_affinity_mean": global_positive_rate,
            "user_genre_affinity_max": global_positive_rate,
            "user_genre_rating_mean": global_mean,
            "user_genre_history": 0.0,
            "user_genre_known_share": 0.0,
            "svd_score": global_mean,
            "svd_user_gap": 0.0,
            "svd_movie_gap": 0.0,
        }
        for feature_name, default_value in fill_defaults.items():
            enriched[feature_name] = pd.to_numeric(enriched[feature_name], errors="coerce").fillna(default_value)

        for feature_name in genre_feature_columns:
            enriched[feature_name] = pd.to_numeric(enriched.get(feature_name), errors="coerce").fillna(0.0)

        return enriched

    prepared_train = enrich(train_frame)
    prepared_val = enrich(val_frame)
    prepared_test = enrich(test_frame)

    numeric_columns = [
        "user_avg_rating",
        "user_rating_std",
        "user_num_ratings",
        "user_positive_ratio",
        "user_log_num_ratings",
        "user_bias",
        "user_recency_days",
        "user_activity_span_days",
        "movie_avg_rating",
        "movie_rating_std",
        "movie_num_ratings",
        "movie_positive_ratio",
        "movie_log_num_ratings",
        "movie_bias",
        "movie_recency_days",
        "movie_activity_span_days",
        "movie_year",
        "movie_age",
        "interaction_recency_days",
        "bias_gap",
        "positive_rate_gap",
        "genre_count",
        "user_genre_affinity_mean",
        "user_genre_affinity_max",
        "user_genre_rating_mean",
        "user_genre_history",
        "user_genre_known_share",
        "svd_score",
        "svd_user_gap",
        "svd_movie_gap",
        *genre_feature_columns,
    ]

    for feature_frame in (prepared_train, prepared_val, prepared_test):
        for column in numeric_columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(0.0).astype("float32")

    lightgbm_categorical_columns = ["movieId_cat", "primary_genre"]
    catboost_categorical_columns = ["userId_cat", "movieId_cat", "primary_genre"]

    for column in lightgbm_categorical_columns:
        category_values = pd.Index(
            pd.concat(
                [
                    prepared_train[column].astype("string"),
                    prepared_val[column].astype("string"),
                    prepared_test[column].astype("string"),
                ],
                ignore_index=True,
            )
            .fillna("unknown")
            .unique()
        )
        dtype = pd.CategoricalDtype(categories=category_values.tolist())
        prepared_train[column] = prepared_train[column].astype("string").fillna("unknown").astype(dtype)
        prepared_val[column] = prepared_val[column].astype("string").fillna("unknown").astype(dtype)
        prepared_test[column] = prepared_test[column].astype("string").fillna("unknown").astype(dtype)

    return FeatureBundle(
        train_frame=prepared_train.reset_index(drop=True),
        val_frame=prepared_val.reset_index(drop=True),
        test_frame=prepared_test.reset_index(drop=True),
        numeric_columns=numeric_columns,
        lightgbm_categorical_columns=lightgbm_categorical_columns,
        catboost_categorical_columns=catboost_categorical_columns,
        target_column="label",
        user_column="userId",
        top_genres=top_genres,
        reference_year=reference_year,
        label_threshold=float(label_threshold),
    )


def _choose_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.linspace(0.2, 0.8, 31),
                    np.quantile(probabilities, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                ]
            ),
            0.05,
            0.95,
        )
    )
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = -1.0
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int8)
        current_f1 = f1_score(y_true, predictions, zero_division=0)
        current_precision = precision_score(y_true, predictions, zero_division=0)
        if current_f1 > best_f1 or (
            math.isclose(current_f1, best_f1) and current_precision > best_precision
        ):
            best_threshold = float(threshold)
            best_f1 = float(current_f1)
            best_precision = float(current_precision)
    return best_threshold, best_f1


def _ranking_metrics(
    user_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "userId": pd.Series(user_ids, dtype="int64"),
            "label": pd.Series(labels, dtype="int8"),
            "score": pd.Series(scores, dtype="float32"),
        }
    )
    grouped = frame.groupby("userId", sort=False, observed=True)

    precision_values: list[float] = []
    recall_values: list[float] = []
    hr_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []

    for _, group in grouped:
        positives = int(group["label"].sum())
        if positives <= 0:
            continue
        top_group = group.nlargest(k, columns="score")
        relevance = top_group["label"].to_numpy(dtype=np.int8)
        hits = int(relevance.sum())
        precision_values.append(hits / float(k))
        recall_values.append(hits / float(positives))
        hr_values.append(1.0 if hits > 0 else 0.0)

        if hits > 0:
            first_hit_index = int(np.flatnonzero(relevance > 0)[0]) + 1
            mrr_values.append(1.0 / float(first_hit_index))
        else:
            mrr_values.append(0.0)

        dcg = 0.0
        for rank, rel in enumerate(relevance, start=1):
            if rel > 0:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(positives, k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg_values.append((dcg / ideal_dcg) if ideal_dcg > 0 else 0.0)

    def average(values: list[float]) -> float:
        return round(float(np.mean(values)) if values else 0.0, 4)

    return {
        "Precision@10": average(precision_values),
        "Recall@10": average(recall_values),
        "HR@10": average(hr_values),
        "NDCG@10": average(ndcg_values),
        "MRR": average(mrr_values),
    }


def _evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    user_ids: np.ndarray,
    loss_label: str = "LogLoss",
) -> dict[str, Any]:
    clipped_probs = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
    predictions = (clipped_probs >= float(threshold)).astype(np.int8)
    metrics: dict[str, Any] = {
        "Loss": _round_metric(log_loss(labels, clipped_probs, labels=[0, 1])),
        "LossLabel": loss_label,
        "AUC": _safe_auc(labels, clipped_probs),
        "F1": _round_metric(f1_score(labels, predictions, zero_division=0)),
        "Precision": _round_metric(precision_score(labels, predictions, zero_division=0)),
        "Recall": _round_metric(recall_score(labels, predictions, zero_division=0)),
        "Threshold": _round_metric(threshold),
    }
    metrics.update(_ranking_metrics(user_ids=user_ids, labels=labels, scores=clipped_probs, k=10))
    return metrics


def _artifact_status() -> dict[str, list[str]]:
    model_files = {
        "SVD": (MODEL_DIR / "svd_model.pkl").exists(),
        "ALS": (MODEL_DIR / "als_model.pkl").exists(),
        "NCF": ((MODEL_DIR / "ncf_model.keras").exists() or (MODEL_DIR / "ncf_model.weights.h5").exists())
        and (MODEL_DIR / "ncf_encoders.pkl").exists(),
        "SBERT": (MODEL_DIR / "sbert_embeddings.npy").exists() and (MODEL_DIR / "sbert_index.pkl").exists(),
        "XGB": (MODEL_DIR / "xgb_ranker.pkl").exists() or (MODEL_DIR / "xgb_ranker.json").exists(),
        "LightGBM": LIGHTGBM_MODEL_PATH.exists(),
        "CatBoost": CATBOOST_MODEL_PATH.exists(),
        "LogReg": LOGREG_MODEL_PATH.exists(),
        "RandomForest": RANDOM_FOREST_MODEL_PATH.exists(),
    }
    return {
        "available_models": [name for name, present in model_files.items() if present],
        "missing_models": [name for name, present in model_files.items() if not present],
    }


def _build_api_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics_block = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    comparison: list[dict[str, Any]] = []

    for model_name, raw_values in metrics_block.items():
        if not isinstance(raw_values, dict):
            continue
        comparison.append(
            {
                "model": str(model_name),
                "auc": _round_metric(raw_values.get("AUC")),
                "f1": _round_metric(raw_values.get("F1")),
                "precision": _round_metric(raw_values.get("Precision")),
                "recall": _round_metric(raw_values.get("Recall")),
                "precision_at_10": _round_metric(raw_values.get("Precision@10")),
                "recall_at_10": _round_metric(raw_values.get("Recall@10")),
                "ndcg_10": _round_metric(raw_values.get("NDCG@10")),
                "hr_10": _round_metric(raw_values.get("HR@10")),
                "mrr": _round_metric(raw_values.get("MRR")),
                "loss": _round_metric(raw_values.get("Loss")),
                "loss_label": str(raw_values.get("LossLabel") or "Loss"),
            }
        )

    payload: dict[str, Any] = {
        "run_id": str(report.get("run_id") or ""),
        "model": "Benchmark suite",
        "report_generated_at": str(report.get("generated_at") or ""),
        "updated_at": str(report.get("generated_at") or ""),
        "test_ratio": _round_metric(report.get("test_ratio")),
        "report_metrics": metrics_block,
        "comparison": comparison,
        **_artifact_status(),
    }

    best_metric_map = {
        "best_auc": "AUC",
        "best_f1": "F1",
        "best_precision_at_10": "Precision@10",
        "best_mrr": "MRR",
        "best_ndcg_10": "NDCG@10",
    }
    for payload_key, report_key in best_metric_map.items():
        best_model = ""
        best_value: float | None = None
        for model_name, raw_values in metrics_block.items():
            if not isinstance(raw_values, dict):
                continue
            value = _round_metric(raw_values.get(report_key))
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_model = str(model_name)
        if best_value is not None:
            payload[payload_key] = best_value
            payload[f"{payload_key}_model"] = best_model

    return payload


def _load_legacy_baselines() -> dict[str, dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    candidate_paths = [EVAL_REPORT_PATH]
    candidate_paths.extend(sorted(MODEL_DIR.glob("eval_report.json.bak_*"), reverse=True))

    for path in candidate_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for model_name in ("NCF", "XGB", "SVD"):
            if model_name in baselines:
                continue
            raw_values = metrics.get(model_name)
            if not isinstance(raw_values, dict):
                continue
            baseline_values = dict(raw_values)
            baseline_values.setdefault("Source", "legacy_baseline")
            baselines[model_name] = baseline_values
        if len(baselines) == 3:
            break

    return baselines


def _write_eval_report(report: dict[str, Any]) -> None:
    if EVAL_REPORT_PATH.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = MODEL_DIR / f"eval_report.json.bak_{timestamp}"
        shutil.copy2(EVAL_REPORT_PATH, backup_path)
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _write_schema(bundle: FeatureBundle) -> None:
    payload = {
        "generated_at": _utc_now_iso(),
        "numeric_columns": bundle.numeric_columns,
        "lightgbm_categorical_columns": bundle.lightgbm_categorical_columns,
        "catboost_categorical_columns": bundle.catboost_categorical_columns,
        "top_genres": bundle.top_genres,
        "label_threshold": bundle.label_threshold,
        "reference_year": bundle.reference_year,
    }
    BENCHMARK_SCHEMA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary_to_mongo(report: dict[str, Any]) -> bool:
    summary_payload = _build_api_summary(report)
    summary_payload["kind"] = "benchmark_report"
    summary_payload["ts"] = report.get("generated_at") or _utc_now_iso()
    summary_payload["db_target"] = format_db_target()
    try:
        get_collection("model_metrics").update_one(
            {"run_id": summary_payload["run_id"]},
            {"$set": summary_payload},
            upsert=True,
        )
        return True
    except Exception as exc:
        log.warning("MongoDB metrics sync skipped: %s", exc)
        return False


def _train_lightgbm(bundle: FeatureBundle, seed: int) -> tuple[dict[str, Any] | None, str | None]:
    if lgb is None:
        return None, f"LightGBM unavailable: {LIGHTGBM_IMPORT_ERROR or 'not installed'}"

    feature_columns = bundle.numeric_columns + bundle.lightgbm_categorical_columns
    train_X = bundle.train_frame[feature_columns].copy()
    val_X = bundle.val_frame[feature_columns].copy()
    test_X = bundle.test_frame[feature_columns].copy()
    train_y = bundle.train_frame[bundle.target_column].to_numpy(dtype=np.int8)
    val_y = bundle.val_frame[bundle.target_column].to_numpy(dtype=np.int8)
    test_y = bundle.test_frame[bundle.target_column].to_numpy(dtype=np.int8)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=40,
        reg_alpha=0.15,
        reg_lambda=0.25,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        train_X,
        train_y,
        eval_set=[(val_X, val_y)],
        eval_metric="auc",
        categorical_feature=bundle.lightgbm_categorical_columns,
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=50)],
    )

    val_probabilities = model.predict_proba(val_X)[:, 1]
    threshold, best_val_f1 = _choose_best_threshold(val_y, val_probabilities)
    test_probabilities = model.predict_proba(test_X)[:, 1]
    metrics = _evaluate_predictions(
        labels=test_y,
        probabilities=test_probabilities,
        threshold=threshold,
        user_ids=bundle.test_frame[bundle.user_column].to_numpy(dtype=np.int64),
    )
    metrics["BestIteration"] = int(getattr(model, "best_iteration_", 0) or 0)
    metrics["ValidationAUC"] = _safe_auc(val_y, val_probabilities)
    metrics["ValidationF1"] = _round_metric(best_val_f1)
    metrics["TargetReached"] = bool((metrics.get("AUC") or 0.0) >= TARGETS["auc"])

    model.booster_.save_model(str(LIGHTGBM_MODEL_PATH))
    return metrics, None


def _train_catboost(bundle: FeatureBundle, seed: int) -> tuple[dict[str, Any] | None, str | None]:
    if CatBoostClassifier is None or Pool is None:
        return None, f"CatBoost unavailable: {CATBOOST_IMPORT_ERROR or 'not installed'}"

    feature_columns = bundle.numeric_columns + bundle.catboost_categorical_columns
    train_X = bundle.train_frame[feature_columns].copy()
    val_X = bundle.val_frame[feature_columns].copy()
    test_X = bundle.test_frame[feature_columns].copy()
    for column in bundle.catboost_categorical_columns:
        train_X[column] = train_X[column].astype("string").fillna("unknown")
        val_X[column] = val_X[column].astype("string").fillna("unknown")
        test_X[column] = test_X[column].astype("string").fillna("unknown")
    train_y = bundle.train_frame[bundle.target_column].to_numpy(dtype=np.int8)
    val_y = bundle.val_frame[bundle.target_column].to_numpy(dtype=np.int8)
    test_y = bundle.test_frame[bundle.target_column].to_numpy(dtype=np.int8)

    model = CatBoostClassifier(
        iterations=1600,
        learning_rate=0.04,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        auto_class_weights="Balanced",
        verbose=False,
        allow_writing_files=False,
    )
    train_pool = Pool(train_X, label=train_y, cat_features=bundle.catboost_categorical_columns)
    val_pool = Pool(val_X, label=val_y, cat_features=bundle.catboost_categorical_columns)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        early_stopping_rounds=20,
        verbose=50,
    )

    val_probabilities = model.predict_proba(val_X)[:, 1]
    threshold, best_val_f1 = _choose_best_threshold(val_y, val_probabilities)
    test_probabilities = model.predict_proba(test_X)[:, 1]
    metrics = _evaluate_predictions(
        labels=test_y,
        probabilities=test_probabilities,
        threshold=threshold,
        user_ids=bundle.test_frame[bundle.user_column].to_numpy(dtype=np.int64),
    )
    metrics["BestIteration"] = int(model.get_best_iteration())
    metrics["ValidationAUC"] = _safe_auc(val_y, val_probabilities)
    metrics["ValidationF1"] = _round_metric(best_val_f1)
    metrics["TargetReached"] = bool((metrics.get("AUC") or 0.0) >= TARGETS["auc"])

    model.save_model(str(CATBOOST_MODEL_PATH))
    return metrics, None


def _train_logistic_regression(bundle: FeatureBundle, seed: int) -> tuple[dict[str, Any] | None, str | None]:
    train_X = bundle.train_frame[bundle.numeric_columns].copy()
    val_X = bundle.val_frame[bundle.numeric_columns].copy()
    test_X = bundle.test_frame[bundle.numeric_columns].copy()
    train_y = bundle.train_frame[bundle.target_column].to_numpy(dtype=np.int8)
    val_y = bundle.val_frame[bundle.target_column].to_numpy(dtype=np.int8)
    test_y = bundle.test_frame[bundle.target_column].to_numpy(dtype=np.int8)

    candidate_c_values = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_model: Pipeline | None = None
    best_threshold = 0.5
    best_val_f1 = -1.0
    best_val_auc: float | None = None
    best_c = candidate_c_values[0]

    for c_value in candidate_c_values:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=600,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipeline.fit(train_X, train_y)
        val_probabilities = pipeline.predict_proba(val_X)[:, 1]
        threshold, current_val_f1 = _choose_best_threshold(val_y, val_probabilities)
        current_val_auc = _safe_auc(val_y, val_probabilities)

        if current_val_f1 > best_val_f1:
            best_model = pipeline
            best_threshold = threshold
            best_val_f1 = current_val_f1
            best_val_auc = current_val_auc
            best_c = c_value

        log.info("LogReg C=%.2f -> val F1 %.4f", c_value, current_val_f1)
        if current_val_f1 >= TARGETS["f1"]:
            log.info("LogReg hit target F1 %.2f; stopping search", TARGETS["f1"])
            break

    if best_model is None:
        return None, "Logistic regression failed to train"

    test_probabilities = best_model.predict_proba(test_X)[:, 1]
    metrics = _evaluate_predictions(
        labels=test_y,
        probabilities=test_probabilities,
        threshold=best_threshold,
        user_ids=bundle.test_frame[bundle.user_column].to_numpy(dtype=np.int64),
    )
    metrics["ValidationAUC"] = best_val_auc
    metrics["ValidationF1"] = _round_metric(best_val_f1)
    metrics["BestC"] = best_c
    metrics["TargetReached"] = bool((metrics.get("F1") or 0.0) >= TARGETS["f1"])

    dump(best_model, LOGREG_MODEL_PATH)
    return metrics, None


def _train_random_forest(bundle: FeatureBundle, seed: int) -> tuple[dict[str, Any] | None, str | None]:
    train_X = bundle.train_frame[bundle.numeric_columns].copy()
    val_X = bundle.val_frame[bundle.numeric_columns].copy()
    test_X = bundle.test_frame[bundle.numeric_columns].copy()
    train_y = bundle.train_frame[bundle.target_column].to_numpy(dtype=np.int8)
    val_y = bundle.val_frame[bundle.target_column].to_numpy(dtype=np.int8)
    test_y = bundle.test_frame[bundle.target_column].to_numpy(dtype=np.int8)
    val_user_ids = bundle.val_frame[bundle.user_column].to_numpy(dtype=np.int64)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
        warm_start=True,
    )

    best_model: RandomForestClassifier | None = None
    best_threshold = 0.5
    best_precision_at_10 = -1.0
    best_val_f1 = -1.0
    best_estimators = 100
    best_val_auc: float | None = None

    for estimator_count in (100, 200, 300, 400):
        model.set_params(n_estimators=estimator_count)
        model.fit(train_X, train_y)
        val_probabilities = model.predict_proba(val_X)[:, 1]
        threshold, current_val_f1 = _choose_best_threshold(val_y, val_probabilities)
        current_ranking = _ranking_metrics(val_user_ids, val_y, val_probabilities, k=10)
        current_precision_at_10 = float(current_ranking["Precision@10"])
        current_val_auc = _safe_auc(val_y, val_probabilities)
        log.info(
            "RandomForest estimators=%d -> val Precision@10 %.4f",
            estimator_count,
            current_precision_at_10,
        )

        if current_precision_at_10 > best_precision_at_10:
            best_model = copy.deepcopy(model)
            best_threshold = threshold
            best_precision_at_10 = current_precision_at_10
            best_val_f1 = current_val_f1
            best_val_auc = current_val_auc
            best_estimators = estimator_count

        if current_precision_at_10 >= TARGETS["precision_at_10"]:
            log.info(
                "RandomForest hit target Precision@10 %.2f; stopping growth",
                TARGETS["precision_at_10"],
            )
            break

    if best_model is None:
        return None, "Random forest failed to train"

    test_probabilities = best_model.predict_proba(test_X)[:, 1]
    metrics = _evaluate_predictions(
        labels=test_y,
        probabilities=test_probabilities,
        threshold=best_threshold,
        user_ids=bundle.test_frame[bundle.user_column].to_numpy(dtype=np.int64),
    )
    metrics["ValidationAUC"] = best_val_auc
    metrics["ValidationF1"] = _round_metric(best_val_f1)
    metrics["ValidationPrecision@10"] = _round_metric(best_precision_at_10)
    metrics["BestEstimators"] = best_estimators
    metrics["TargetReached"] = bool((metrics.get("Precision@10") or 0.0) >= TARGETS["precision_at_10"])

    dump(best_model, RANDOM_FOREST_MODEL_PATH)
    return metrics, None


def train_benchmark_suite(
    train_rows: int,
    val_rows: int,
    test_rows: int,
    seed: int,
    chunksize: int,
    label_threshold: float,
    top_genres: int,
) -> dict[str, Any]:
    _ensure_input_files()

    total_train_sample = max(2, int(train_rows) + int(val_rows))
    total_test_sample = max(1, int(test_rows))

    log.info(
        "Sampling MovieLens rows -> train=%d val=%d test=%d",
        train_rows,
        val_rows,
        test_rows,
    )
    sampled_train = _sample_csv_rows(TRAIN_CSV_PATH, total_train_sample, seed, chunksize)
    sampled_test = _sample_csv_rows(TEST_CSV_PATH, total_test_sample, seed + 1, chunksize)
    if sampled_train.empty or sampled_test.empty:
        raise RuntimeError("Sampling returned no data")

    movies, top_genre_tokens = _load_movies_catalog(top_genres)
    prepared_train = _prepare_interactions(sampled_train, movies, label_threshold)
    prepared_test = _prepare_interactions(sampled_test, movies, label_threshold)

    train_split, val_split = train_test_split(
        prepared_train,
        test_size=min(max(val_rows / float(len(prepared_train.index)), 0.1), 0.4),
        random_state=seed,
        stratify=prepared_train["label"] if prepared_train["label"].nunique() > 1 else None,
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    bundle = _prepare_feature_bundle(
        train_frame=train_split,
        val_frame=val_split,
        test_frame=prepared_test,
        top_genres=top_genre_tokens,
        label_threshold=label_threshold,
    )
    _write_schema(bundle)

    trained_metrics: dict[str, Any] = {}
    skipped_models: dict[str, str] = {}

    log.info("Training LightGBM benchmark ...")
    lightgbm_metrics, lightgbm_error = _train_lightgbm(bundle, seed)
    if lightgbm_metrics is not None:
        trained_metrics["LightGBM"] = lightgbm_metrics
    elif lightgbm_error:
        skipped_models["LightGBM"] = lightgbm_error

    log.info("Training CatBoost benchmark ...")
    catboost_metrics, catboost_error = _train_catboost(bundle, seed)
    if catboost_metrics is not None:
        trained_metrics["CatBoost"] = catboost_metrics
    elif catboost_error:
        skipped_models["CatBoost"] = catboost_error

    log.info("Training Logistic Regression benchmark ...")
    logreg_metrics, logreg_error = _train_logistic_regression(bundle, seed)
    if logreg_metrics is not None:
        trained_metrics["LogisticRegression"] = logreg_metrics
    elif logreg_error:
        skipped_models["LogisticRegression"] = logreg_error

    log.info("Training Random Forest benchmark ...")
    rf_metrics, rf_error = _train_random_forest(bundle, seed)
    if rf_metrics is not None:
        trained_metrics["RandomForest"] = rf_metrics
    elif rf_error:
        skipped_models["RandomForest"] = rf_error

    if not trained_metrics:
        raise RuntimeError("No benchmark models were trained successfully")

    legacy_baselines = _load_legacy_baselines()
    for model_name, baseline_values in legacy_baselines.items():
        trained_metrics.setdefault(model_name, baseline_values)

    report = {
        "run_id": f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "generated_at": _utc_now_iso(),
        "data_source": {
            "train_csv": str(TRAIN_CSV_PATH),
            "test_csv": str(TEST_CSV_PATH),
            "movies_csv": str(MOVIES_CSV_PATH),
            "db_target": format_db_target(),
        },
        "sampled_rows": {
            "train": int(len(train_split.index)),
            "validation": int(len(val_split.index)),
            "test": int(len(prepared_test.index)),
        },
        "label_threshold": float(label_threshold),
        "test_ratio": 0.2,
        "metrics": trained_metrics,
        "skipped_models": skipped_models,
        "legacy_baseline_models": sorted(legacy_baselines.keys()),
        "targets": TARGETS,
    }
    _write_eval_report(report)
    report["mongo_synced"] = _write_summary_to_mongo(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train benchmark recommender models and write normalized metrics."
    )
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--val-rows", type=int, default=DEFAULT_VAL_ROWS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=_env_float("MOVIEBUZZ_POSITIVE_RATING_THRESHOLD", 4.0),
        help="Ratings greater than or equal to this value are treated as positive.",
    )
    parser.add_argument("--top-genres", type=int, default=DEFAULT_TOP_GENRES)
    args = parser.parse_args()

    report = train_benchmark_suite(
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        test_rows=args.test_rows,
        seed=args.seed,
        chunksize=args.chunk_size,
        label_threshold=args.label_threshold,
        top_genres=args.top_genres,
    )

    summary = _build_api_summary(report)
    best_auc = summary.get("best_auc")
    best_auc_model = summary.get("best_auc_model")
    log.info(
        "Benchmark training complete. Best AUC: %s (%s)",
        best_auc if best_auc is not None else "n/a",
        best_auc_model or "n/a",
    )


if __name__ == "__main__":
    main()
