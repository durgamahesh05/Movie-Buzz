"""
data_pipeline.py  –  MovieBuzz Train / Test / Evaluate
========================================================
Steps:
  1. Load MovieLens 25M in chunks (memory safe)
  2. Temporal train/test split  (last 20% of each user's ratings = test)
  3. Negative sampling          (for NCF / BPR training)
  4. Feature engineering        (genre dummies, log counts, normalised ratings)
  5. Save splits to SQLite + CSV
  6. Evaluate all models:
       SVD  → RMSE, MAE
       NCF  → BCE, BPR, AUC, Precision@K, Recall@K, NDCG@K
       ALS  → Hit Rate@K, MRR
       XGB  → Log-loss, F1
       Full → NDCG@10, MAP@10
  7. Print a formatted report + save to models/eval_report.json
"""

from __future__ import annotations

import importlib
import importlib.util
import json, os, pickle, sqlite3, logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, f1_score, log_loss,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import env_path


def _load_optional_module(module_name: str):
    if not importlib.util.find_spec(module_name):
        return None
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        logging.getLogger("pipeline").warning(
            "Optional dependency %s unavailable: %s",
            module_name,
            exc,
        )
        return None


# ── Optional deps ─────────────────────────────────────────────────────────────
Dataset = Reader = SVD = accuracy = None
surprise_module = _load_optional_module("surprise")
if surprise_module is not None:
    Dataset = surprise_module.Dataset
    Reader = surprise_module.Reader
    SVD = surprise_module.SVD
    accuracy = surprise_module.accuracy
    HAS_SURPRISE = True
else:
    HAS_SURPRISE = False

tf = _load_optional_module("tensorflow")
keras = None
if tf is not None:
    keras = tf.keras
    HAS_TF = True
else:
    HAS_TF = False

xgb = _load_optional_module("xgboost")
if xgb is not None:
    HAS_XGB = True
else:
    HAS_XGB = False

implicit = _load_optional_module("implicit")
sp = _load_optional_module("scipy.sparse")
if implicit is not None and sp is not None:
    HAS_IMPLICIT = True
else:
    HAS_IMPLICIT = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("pipeline")


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

    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if (resolved / "movies.csv").exists():
            return resolved
    return (base_dir / "data").resolve(strict=False)


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = _resolve_data_dir(BASE_DIR)
MODEL_DIR = BASE_DIR / "models"
DB_PATH   = env_path("DATABASE_URL", "DB_PATH", "MOVIEBUZZ_DB_PATH", default=BASE_DIR / "moviebuzz.db")
MODEL_DIR.mkdir(exist_ok=True)

CHUNK        = 500_000
TEST_RATIO   = 0.20     # last 20% of each user's ratings go to test
NEG_RATIO    = 4        # 4 negatives per positive (for NCF / BPR)
TOP_K        = 10       # evaluation @K
ALS_SAMPLE_ROWS = 5_000_000
SVD_SAMPLE_ROWS = 2_000_000
NCF_POSITIVE_SAMPLE_ROWS = 400_000
XGB_SAMPLE_ROWS = 2_000_000
EVAL_SAMPLE_ROWS = 250_000

RATING_DTYPES: dict[str, Any] = {
    "userId": np.int32,
    "movieId": np.int32,
    "rating": np.float32,
    "timestamp": np.int32,
}


class RatingsMeta(TypedDict):
    ratings_path: Path
    all_item_ids: NDArray[np.int32]
    n_ratings: int
    n_users: int
    n_items: int


def build_rating_dtype_map(usecols: List[str]) -> dict[str, Any]:
    return {key: value for key, value in RATING_DTYPES.items() if key in usecols}


def rounded_metric(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC FUNCTIONS  (all pure NumPy – no framework needed)
# ═══════════════════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def ndcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain @k"""
    r = np.asarray(relevance[:k], dtype=float)
    if r.sum() == 0:
        return 0.0
    dcg  = float(np.sum(r / np.log2(np.arange(2, len(r) + 2))))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0

def mean_ap_at_k(relevant_list: List[List[int]],
                 predicted_list: List[List[int]], k: int = 10) -> float:
    """Mean Average Precision @k"""
    aps = []
    for rel, pred in zip(relevant_list, predicted_list):
        rel_set = set(rel)
        hits, s = 0, 0.0
        for i, p in enumerate(pred[:k], 1):
            if p in rel_set:
                hits += 1
                s   += hits / i
        aps.append(s / min(len(rel_set), k) if rel_set else 0.0)
    return float(np.mean(aps))

def hit_rate_at_k(relevant_list: List[List[int]],
                  predicted_list: List[List[int]], k: int = 10) -> float:
    hits = sum(
        1 for rel, pred in zip(relevant_list, predicted_list)
        if set(pred[:k]) & set(rel)
    )
    return hits / len(relevant_list) if relevant_list else 0.0

def mrr(relevant_list: List[List[int]],
        predicted_list: List[List[int]]) -> float:
    """Mean Reciprocal Rank"""
    rrs = []
    for rel, pred in zip(relevant_list, predicted_list):
        rel_set = set(rel)
        rr = 0.0
        for i, p in enumerate(pred, 1):
            if p in rel_set:
                rr = 1.0 / i
                break
        rrs.append(rr)
    return float(np.mean(rrs))

def bpr_loss_np(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Bayesian Personalised Ranking loss (NumPy)"""
    diff = pos_scores - neg_scores[:len(pos_scores)]
    return float(-np.mean(np.log(1 / (1 + np.exp(-diff)) + 1e-8)))

def bce_loss_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def iter_rating_chunks(
    csv_path: Path,
    usecols: List[str] | None = None,
    chunksize: int = CHUNK,
) -> Iterator[pd.DataFrame]:
    effective_usecols = usecols or ["userId", "movieId", "rating", "timestamp"]
    dtype_map = build_rating_dtype_map(effective_usecols)

    yield from pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=effective_usecols,
        dtype=cast(Any, dtype_map),
    )


def export_table_to_csv(
    conn: sqlite3.Connection,
    table_name: str,
    output_path: Path,
    chunksize: int = CHUNK,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    first_chunk = True
    query = f"SELECT userId, movieId, rating, timestamp FROM {table_name}"
    for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
        chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False


def read_rating_sample(
    csv_path: Path,
    nrows: int | None = None,
    usecols: List[str] | None = None,
) -> pd.DataFrame:
    effective_usecols = usecols or ["userId", "movieId", "rating", "timestamp"]
    dtype_map = build_rating_dtype_map(effective_usecols)
    return pd.read_csv(
        csv_path,
        nrows=nrows,
        usecols=effective_usecols,
        dtype=cast(Any, dtype_map),
    )


def sample_rating_rows(
    csv_path: Path,
    sample_rows: int,
    usecols: List[str] | None = None,
) -> pd.DataFrame:
    effective_usecols = usecols or ["userId", "movieId", "rating", "timestamp"]
    total_rows = 0
    for chunk in iter_rating_chunks(csv_path, usecols=effective_usecols):
        total_rows += len(chunk)

    if total_rows <= sample_rows:
        return read_rating_sample(csv_path, usecols=effective_usecols)

    ratio = sample_rows / total_rows
    sampled_chunks = []
    for index, chunk in enumerate(iter_rating_chunks(csv_path, usecols=effective_usecols)):
        take = max(1, int(round(len(chunk) * ratio)))
        take = min(take, len(chunk))
        sampled_chunks.append(chunk.sample(n=take, random_state=42 + index))

    sampled = pd.concat(sampled_chunks, ignore_index=True)
    if len(sampled) > sample_rows:
        sampled = sampled.sample(n=sample_rows, random_state=42)
    return sampled.reset_index(drop=True)


def load_data() -> Tuple[pd.DataFrame, RatingsMeta]:
    """
    Returns movies metadata plus ratings file metadata.
    Ratings are scanned in chunks so ML25M can be processed without building
    one giant in-memory DataFrame during the import stage.
    """
    log.info("Loading movies.csv …")
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    movies_df["genres_clean"] = (
        movies_df["genres"]
        .str.replace("|", " ", regex=False)
        .str.replace("(no genres listed)", "", regex=False)
        .str.strip()
    )

    # genre dummy features
    all_genres = set()
    for g in movies_df["genres_clean"]:
        all_genres.update(g.split())
    all_genres.discard("")

    for genre in all_genres:
        movies_df[f"g_{genre}"] = movies_df["genres_clean"].str.contains(
            genre, case=False, na=False
        ).astype(np.float32)

    ratings_path = DATA_DIR / "ratings.csv"
    log.info("Scanning ratings.csv in chunks …")
    sum_ratings: Dict[int, float] = {}
    count_ratings: Dict[int, int] = {}
    unique_users: set[int] = set()
    unique_items: set[int] = set()
    total_ratings = 0

    for chunk in iter_rating_chunks(ratings_path):
        total_ratings += len(chunk)
        unique_users.update(chunk["userId"].unique().tolist())
        unique_items.update(chunk["movieId"].unique().tolist())

        grouped = chunk.groupby("movieId")["rating"].agg(["sum", "count"])
        for movie_id, row in grouped.iterrows():
            movie_id_int = cast(int, movie_id)
            sum_ratings[movie_id_int] = sum_ratings.get(movie_id_int, 0.0) + float(row["sum"])
            count_ratings[movie_id_int] = count_ratings.get(movie_id_int, 0) + int(row["count"])

    log.info(
        "Scanned %d ratings from %d users on %d movies",
        total_ratings,
        len(unique_users),
        len(unique_items),
    )

    agg = pd.DataFrame(
        {
            "movieId": list(sum_ratings.keys()),
            "avg_rating": [
                sum_ratings[movie_id] / max(count_ratings[movie_id], 1)
                for movie_id in sum_ratings
            ],
            "num_ratings": [count_ratings[movie_id] for movie_id in sum_ratings],
        }
    )
    agg["log_ratings"]  = np.log1p(agg["num_ratings"])
    agg["norm_rating"] = MinMaxScaler().fit_transform(agg[["avg_rating"]]).ravel()
    movies_df = movies_df.merge(agg, on="movieId", how="left")
    movies_df["avg_rating"]  = movies_df["avg_rating"].fillna(0)
    movies_df["num_ratings"] = movies_df["num_ratings"].fillna(0).astype(int)
    movies_df["log_ratings"] = movies_df["log_ratings"].fillna(0)
    movies_df["norm_rating"] = movies_df["norm_rating"].fillna(0)

    return movies_df, {
        "ratings_path": ratings_path,
        "all_item_ids": np.array(sorted(unique_items), dtype=np.int32),
        "n_ratings": total_ratings,
        "n_users": len(unique_users),
        "n_items": len(unique_items),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – TEMPORAL TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_split(
    ratings_path: Path,
    test_ratio: float = TEST_RATIO,
) -> Tuple[Path, Path]:
    """
    Import ratings.csv into SQLite in chunks, then use window functions to
    build temporal train/test splits without sorting the full 25M DataFrame in memory.
    """
    log.info("Performing temporal train/test split (%.0f%% test) …", test_ratio * 100)

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS ratings_raw")
        conn.execute("DROP TABLE IF EXISTS train_ratings")
        conn.execute("DROP TABLE IF EXISTS test_ratings")

        first_chunk = True
        for chunk in iter_rating_chunks(ratings_path):
            chunk.to_sql(
                "ratings_raw",
                conn,
                if_exists="replace" if first_chunk else "append",
                index=False,
                chunksize=10_000,
            )
            first_chunk = False

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ratings_raw_user_ts ON ratings_raw(userId, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ratings_raw_movie ON ratings_raw(movieId)"
        )

        n_test_sql = (
            f"CASE WHEN CAST(cnt * {test_ratio} AS INTEGER) < 1 "
            f"THEN 1 ELSE CAST(cnt * {test_ratio} AS INTEGER) END"
        )

        conn.executescript(
            f"""
            CREATE TABLE train_ratings AS
            WITH ranked AS (
                SELECT
                    userId,
                    movieId,
                    rating,
                    timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY userId
                        ORDER BY timestamp
                    ) AS rn,
                    COUNT(*) OVER (
                        PARTITION BY userId
                    ) AS cnt
                FROM ratings_raw
            )
            SELECT userId, movieId, rating, timestamp
            FROM ranked
            WHERE rn <= cnt - {n_test_sql};

            CREATE TABLE test_ratings AS
            WITH ranked AS (
                SELECT
                    userId,
                    movieId,
                    rating,
                    timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY userId
                        ORDER BY timestamp
                    ) AS rn,
                    COUNT(*) OVER (
                        PARTITION BY userId
                    ) AS cnt
                FROM ratings_raw
            )
            SELECT userId, movieId, rating, timestamp
            FROM ranked
            WHERE rn > cnt - {n_test_sql};
            """
        )

        train_count = conn.execute("SELECT COUNT(*) FROM train_ratings").fetchone()[0]
        test_count = conn.execute("SELECT COUNT(*) FROM test_ratings").fetchone()[0]
        log.info("Train: %d  |  Test: %d", train_count, test_count)

        export_table_to_csv(conn, "train_ratings", train_path)
        export_table_to_csv(conn, "test_ratings", test_path)

    log.info("Saved data/train.csv and data/test.csv")
    log.info("Saved train_ratings / test_ratings tables to DB")

    return train_path, test_path


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – NEGATIVE SAMPLING  (for NCF / BPR)
# ═══════════════════════════════════════════════════════════════════════════════

def negative_sample(
    train_df: pd.DataFrame,
    all_item_ids: NDArray[np.int32],
    neg_ratio: int = NEG_RATIO,
) -> pd.DataFrame:
    """
    For each positive (user, item) pair in train_df, sample neg_ratio
    items that the user has NOT interacted with.
    Returns a DataFrame with columns [userId, movieId, label].
    """
    log.info("Generating %d× negative samples …", neg_ratio)

    user_items: Dict[int, set[int]] = {}
    for user_id, movie_ids in train_df.groupby("userId")["movieId"]:
        user_id_int = cast(int, user_id)
        user_items[user_id_int] = {
            cast(int, movie_id) for movie_id in movie_ids.astype(np.int32).tolist()
        }
    item_pool = {int(item_id) for item_id in all_item_ids.tolist()}

    positives = train_df[["userId", "movieId"]].copy()
    positives["label"] = 1.0

    neg_rows = []
    rng = np.random.default_rng(42)
    for uid in positives["userId"]:
        uid_int = int(uid)
        seen = user_items.get(uid_int, set())
        unseen = list(item_pool - seen)
        if len(unseen) == 0:
            continue
        sampled = rng.choice(
            np.asarray(unseen, dtype=np.int32),
            size=min(neg_ratio, len(unseen)),
            replace=False,
        )
        for neg in sampled:
            neg_rows.append({"userId": uid_int, "movieId": int(neg), "label": 0.0})

    negatives = pd.DataFrame(neg_rows)
    combined  = pd.concat([positives, negatives], ignore_index=True).sample(
        frac=1, random_state=42
    )
    log.info("Total samples: %d  (%d pos, %d neg)",
             len(combined), len(positives), len(negatives))

    combined.to_csv(DATA_DIR / "train_ncf.csv", index=False)
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – FEATURE MATRIX  (for XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    ratings_source: pd.DataFrame | Path,
    movies_df: pd.DataFrame,
    svd_model=None,
    sample_rows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds feature matrix X and label y for XGBoost.
    Features:
      avg_rating, log_ratings, norm_rating,
      svd_predicted_rating (if model available),
      genre dummies
    """
    log.info("Building feature matrix …")

    if isinstance(ratings_source, pd.DataFrame):
        ratings_df = ratings_source.copy()
    else:
        ratings_df = read_rating_sample(
            ratings_source,
            nrows=sample_rows,
            usecols=["userId", "movieId", "rating"],
        )

    df = ratings_df.merge(
        movies_df[["movieId", "avg_rating", "log_ratings", "norm_rating"]],
        on="movieId", how="left"
    ).fillna(0)

    if svd_model is not None:
        log.info("Adding SVD predictions to features …")
        df["svd_pred"] = df.apply(
            lambda r: svd_model.predict(int(r["userId"]), int(r["movieId"])).est,
            axis=1
        )
    else:
        df["svd_pred"] = df["avg_rating"]

    genre_cols = [c for c in movies_df.columns if c.startswith("g_")]
    if genre_cols:
        df = df.merge(movies_df[["movieId"] + genre_cols],
                      on="movieId", how="left").fillna(0)

    base_features = ["avg_rating", "log_ratings", "norm_rating", "svd_pred"]
    feature_cols  = base_features + genre_cols
    X = np.asarray(df[[c for c in feature_cols if c in df.columns]].values, dtype=np.float32)
    y = np.asarray((df["rating"] >= 4.0).astype(int))

    log.info("Feature matrix: %s  label balance: %.1f%%",
             X.shape, 100 * float(np.mean(y)))
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – TRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def train_svd(train_source: pd.DataFrame | Path):
    if not HAS_SURPRISE or Dataset is None or Reader is None or SVD is None:
        log.warning("scikit-surprise not installed – SVD skipped")
        return None

    dataset_cls = Dataset
    reader_cls = Reader
    svd_cls = SVD

    log.info("Training SVD …")
    if isinstance(train_source, pd.DataFrame):
        reader = reader_cls(rating_scale=(0.5, 5.0))
        dataset = dataset_cls.load_from_df(
            train_source[["userId", "movieId", "rating"]], reader
        )
        trainset = dataset.build_full_trainset()
    else:
        sampled_train = sample_rating_rows(
            train_source,
            SVD_SAMPLE_ROWS,
            usecols=["userId", "movieId", "rating"],
        )
        reader = reader_cls(rating_scale=(0.5, 5.0))
        dataset = dataset_cls.load_from_df(
            sampled_train[["userId", "movieId", "rating"]],
            reader,
        )
        trainset = dataset.build_full_trainset()

    model = svd_cls(n_factors=100, n_epochs=25, lr_all=0.005,
                    reg_all=0.02, random_state=42)
    model.fit(trainset)

    path = MODEL_DIR / "svd_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("SVD saved → %s", path)
    return model


def train_als(train_df: pd.DataFrame):
    if not HAS_IMPLICIT or implicit is None or sp is None:
        log.warning("implicit not installed – ALS skipped")
        return None, None, None

    implicit_module = implicit
    sparse_module = sp

    log.info("Training ALS …")
    users_ = pd.Categorical(train_df["userId"])
    items_ = pd.Categorical(train_df["movieId"])
    mat = sparse_module.csr_matrix(
        (train_df["rating"].astype(np.float32),
         (items_.codes, users_.codes))
    )
    model = implicit_module.als.AlternatingLeastSquares(
        factors=64, iterations=20, regularization=0.1, random_state=42
    )
    model.fit(mat)

    path = MODEL_DIR / "als_model.pkl"
    item_ids = [int(item_id) for item_id in items_.categories.tolist()]
    user_ids = [int(user_id) for user_id in users_.categories.tolist()]

    with open(path, "wb") as f:
        pickle.dump({
            "model":    model,
            "item_ids": item_ids,
            "user_ids": user_ids,
        }, f)
    log.info("ALS saved → %s", path)
    return model, item_ids, user_ids


def train_ncf(ncf_df: pd.DataFrame, n_users: int, n_items: int):
    if not HAS_TF or keras is None:
        log.warning("TensorFlow not installed – NCF skipped")
        return None, {}, {}

    assert tf is not None
    tf_module = cast(Any, tf)
    keras_module = keras

    log.info("Training NCF …")
    from recommender import build_ncf

    users_u = ncf_df["userId"].unique()
    items_u = ncf_df["movieId"].unique()
    user_enc = {u: i for i, u in enumerate(users_u)}
    item_enc = {m: i for i, m in enumerate(items_u)}

    ncf_df = ncf_df.copy()
    ncf_df["u_enc"] = ncf_df["userId"].map(user_enc)
    ncf_df["i_enc"] = ncf_df["movieId"].map(item_enc)

    X_tr, X_va, y_tr, y_va = train_test_split(
        ncf_df[["u_enc", "i_enc"]].values,
        ncf_df["label"].values,
        test_size=0.05, random_state=42
    )

    # ── tf.data Optimizations: Mapping, Caching, Sharding, Batching, Prefetching ──
    def make_dataset(X, y, is_training=True, num_shards=1, shard_index=0):
        dataset = tf_module.data.Dataset.from_tensor_slices((X, y))
        
        if num_shards > 1:
            dataset = dataset.shard(num_shards, shard_index)
            
        def preprocess_func(features, label):
            return (features[0], features[1]), label
            
        dataset = dataset.map(preprocess_func, num_parallel_calls=tf_module.data.AUTOTUNE)
        dataset = dataset.cache()
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=100_000)
            
        dataset = dataset.batch(2048)
        dataset = dataset.prefetch(tf_module.data.AUTOTUNE)
        return dataset

    train_dataset = make_dataset(X_tr, y_tr, is_training=True)
    val_dataset = make_dataset(X_va, y_va, is_training=False)

    model = build_ncf(len(users_u), len(items_u))
    callbacks = [
        keras_module.callbacks.EarlyStopping(patience=2, restore_best_weights=True,
                                             monitor="val_auc", mode="max"),
        keras_module.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.5, patience=1),
    ]
    model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=callbacks, verbose=1,
    )
    path = MODEL_DIR / "ncf_model.keras"
    model.save(str(path))

    enc_path = MODEL_DIR / "ncf_encoders.pkl"
    with open(enc_path, "wb") as f:
        pickle.dump({"user": user_enc, "item": item_enc}, f)
    log.info("NCF saved → %s", path)
    return model, user_enc, item_enc


def train_xgb(X_train: np.ndarray, y_train: np.ndarray):
    if not HAS_XGB or xgb is None:
        log.warning("xgboost not installed – XGB skipped")
        return None

    xgb_module = xgb

    log.info("Training XGBoost …")
    model = xgb_module.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=50)

    path = MODEL_DIR / "xgb_ranker.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("XGBoost saved → %s", path)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 – EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_svd(model, test_df: pd.DataFrame) -> dict:
    if model is None:
        return {}
    log.info("Evaluating SVD …")
    preds, actuals = [], []
    for _, row in test_df.iterrows():
        preds.append(model.predict(int(row["userId"]), int(row["movieId"])).est)
        actuals.append(row["rating"])
    preds   = np.array(preds)
    actuals = np.array(actuals)
    return {
        "RMSE": round(rmse(actuals, preds), 4),
        "MAE":  round(mae(actuals, preds),  4),
    }


def evaluate_ncf(model, test_df: pd.DataFrame,
                 user_enc: dict, item_enc: dict) -> dict:
    if model is None:
        return {}
    log.info("Evaluating NCF …")
    df = test_df.copy()
    df = df[df["userId"].isin(user_enc) & df["movieId"].isin(item_enc)]
    df["u_enc"] = df["userId"].map(user_enc)
    df["i_enc"] = df["movieId"].map(item_enc)
    df["label"] = (df["rating"] >= 4.0).astype(float)

    raw = model.predict([df["u_enc"].values, df["i_enc"].values],
                        verbose=0).flatten()

    y_true  = df["label"].values
    y_pred  = raw
    y_binary = (y_pred >= 0.5).astype(int)
    
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    y_binary_np = np.asarray(y_binary, dtype=int)

    # BPR
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    bpr = bpr_loss_np(y_pred[pos_mask], y_pred[neg_mask]) if np.sum(pos_mask) > 0 else 0.0

    return {
        "BCE":       rounded_metric(bce_loss_np(y_true_np, y_pred_np)),
        "BPR":       rounded_metric(bpr),
        "AUC":       rounded_metric(roc_auc_score(y_true_np, y_pred_np) if np.sum(y_true_np) > 0 else 0.0),
        "Precision": rounded_metric(precision_score(y_true_np, y_binary_np, zero_division=0)),
        "Recall":    rounded_metric(recall_score(y_true_np, y_binary_np, zero_division=0)),
        "F1":        rounded_metric(f1_score(y_true_np, y_binary_np, zero_division=0)),
    }


def evaluate_als(model, test_df: pd.DataFrame,
                 item_ids: list, user_ids: list) -> dict:
    if model is None or not item_ids or sp is None:
        return {}

    sparse_module = sp
    log.info("Evaluating ALS (Hit Rate, MRR) …")

    sample_users = test_df["userId"].unique()[:200]   # 200-user sample for speed

    rel_list, pred_list = [], []
    for uid in sample_users:
        uid_int = int(uid)
        if uid_int not in user_ids:
            continue
        u_idx = user_ids.index(uid_int)
        pos_items = test_df[
            (test_df["userId"] == uid_int) & (test_df["rating"] >= 4.0)
        ]["movieId"].astype(int).tolist()
        if not pos_items:
            continue

        try:
            mat_dummy = sparse_module.csr_matrix(
                (np.ones(1), ([0], [u_idx])),
                shape=(1, len(user_ids))
            )
            recs, _ = model.recommend(u_idx, mat_dummy, N=TOP_K,
                                      filter_already_liked_items=False)
            rec_items = [item_ids[int(i)] for i in recs if int(i) < len(item_ids)]
        except Exception:
            continue

        rel_list.append(pos_items)
        pred_list.append(rec_items)

    if not rel_list:
        return {}
    return {
        "HitRate@10": round(hit_rate_at_k(rel_list, pred_list, TOP_K), 4),
        "MRR":        round(mrr(rel_list, pred_list), 4),
        "MAP@10":     round(mean_ap_at_k(rel_list, pred_list, TOP_K), 4),
    }


def evaluate_xgb(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    if model is None:
        return {}
    log.info("Evaluating XGBoost …")
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "LogLoss":  rounded_metric(log_loss(y_test, proba)),
        "F1":       rounded_metric(f1_score(y_test, preds, zero_division=0)),
        "AUC":      rounded_metric(roc_auc_score(y_test, proba) if y_test.sum() > 0 else 0.0),
        "Precision":rounded_metric(precision_score(y_test, preds, zero_division=0)),
        "Recall":   rounded_metric(recall_score(y_test, preds, zero_division=0)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7 – FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(report: dict):
    print("\n" + "═" * 60)
    print("  MovieBuzz – Model Evaluation Report")
    print("═" * 60)
    for model_name, metrics in report.items():
        if not metrics:
            continue
        print(f"\n  {model_name}")
        print("  " + "─" * 40)
        for k, v in metrics.items():
            bar_len = int(float(str(v)) * 30) if float(str(v)) <= 1 else min(30, int(float(str(v)) * 6))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    {k:<14} {v:>7}  {bar}")
    print("\n" + "═" * 60)


def run_pipeline():
    # ── Load ──────────────────────────────────────────────────────────────────
    movies_df, ratings_meta = load_data()
    ratings_path = ratings_meta["ratings_path"]
    all_item_ids = ratings_meta["all_item_ids"]

    # ── Split ─────────────────────────────────────────────────────────────────
    train_path, test_path = temporal_split(ratings_path)

    # ── Chunk-safe training/eval samples ─────────────────────────────────────
    als_train_df = read_rating_sample(train_path, nrows=ALS_SAMPLE_ROWS)
    ncf_base_df = read_rating_sample(train_path, nrows=NCF_POSITIVE_SAMPLE_ROWS)
    test_df = read_rating_sample(test_path, nrows=EVAL_SAMPLE_ROWS)

    # ── Train ─────────────────────────────────────────────────────────────────
    svd_model = train_svd(train_path)
    als_model, item_ids, user_ids = train_als(als_train_df)

    # NCF stays sampled for CPU safety: 400K positives + 4 negatives ≈ 2M rows
    ncf_df = negative_sample(ncf_base_df, all_item_ids)
    ncf_sample = ncf_df.head(2_000_000)
    ncf_model, user_enc, item_enc = train_ncf(
        ncf_sample, ncf_sample["userId"].nunique(), ncf_sample["movieId"].nunique()
    )

    # XGBoost feature matrix on a large streamed sample instead of the full train set
    X_all, y_all = build_feature_matrix(
        train_path,
        movies_df,
        svd_model,
        sample_rows=XGB_SAMPLE_ROWS,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    xgb_model = train_xgb(X_tr, y_tr)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    report = {
        "SVD  (RMSE, MAE)":        evaluate_svd(svd_model, test_df),
        "NCF  (BCE, BPR, AUC, F1)":evaluate_ncf(ncf_model, test_df, user_enc, item_enc),
        "ALS  (HitRate, MRR, MAP)": evaluate_als(als_model, test_df, item_ids or [], user_ids or []),
        "XGB  (LogLoss, F1, AUC)":  evaluate_xgb(xgb_model, X_te, y_te),
    }

    print_report(report)

    out_path = MODEL_DIR / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "test_ratio":   TEST_RATIO,
            "top_k":        TOP_K,
            "split_method": "temporal (last 20% per user)",
            "neg_ratio":    NEG_RATIO,
            "metrics":      report,
        }, f, indent=2)
    log.info("Report saved → %s", out_path)

    return report


if __name__ == "__main__":
    run_pipeline()
