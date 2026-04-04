"""
train_svd.py  –  Standalone SVD trainer for MovieLens 25M
Run once:  python train_svd.py
"""

from typing import Any, cast
import os
import pickle
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).parent


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
        if (resolved / "ratings.csv").exists():
            return resolved
    return (base_dir / "data").resolve(strict=False)


DATA_DIR  = _resolve_data_dir(BASE_DIR)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CHUNK = 500_000
SVD_SAMPLE_ROWS = int(os.getenv("SVD_SAMPLE_ROWS", "2000000"))
RATING_DTYPES = {
    "userId": np.int32,
    "movieId": np.int32,
    "rating": np.float32,
}


def _load_surprise():
    if not importlib.util.find_spec("surprise"):
        raise RuntimeError("scikit-surprise is not installed in backend/.venv")
    try:
        surprise_module = importlib.import_module("surprise")
    except Exception as exc:
        raise RuntimeError(
            "scikit-surprise could not be imported. Reinstall it with a NumPy-compatible build before running train_svd.py."
        ) from exc
    return surprise_module.Dataset, surprise_module.Reader, surprise_module.SVD


def _sample_ratings(csv_path: Path, sample_rows: int) -> pd.DataFrame:
    total_rows = 0
    for chunk in pd.read_csv(
        csv_path,
        chunksize=CHUNK,
        usecols=["userId", "movieId", "rating"],
        dtype=cast(Any, RATING_DTYPES),
    ):
        total_rows += len(chunk)

    if total_rows <= sample_rows:
        return pd.read_csv(
            csv_path,
            usecols=["userId", "movieId", "rating"],
            dtype=cast(Any, RATING_DTYPES),
        )

    ratio = sample_rows / total_rows
    sampled_chunks = []
    for index, chunk in enumerate(
        pd.read_csv(
            csv_path,
            chunksize=CHUNK,
            usecols=["userId", "movieId", "rating"],
            dtype=cast(Any, RATING_DTYPES),
        )
    ):
        take = max(1, int(round(len(chunk) * ratio)))
        take = min(take, len(chunk))
        sampled_chunks.append(chunk.sample(n=take, random_state=42 + index))

    sampled = pd.concat(sampled_chunks, ignore_index=True)
    if len(sampled) > sample_rows:
        sampled = sampled.sample(n=sample_rows, random_state=42)
    return sampled.reset_index(drop=True)


def main():
    Dataset, Reader, SVD = _load_surprise()
    ratings_path = DATA_DIR / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.csv not found at {ratings_path}")

    print(f"Building Surprise trainset from a streamed sample of up to {SVD_SAMPLE_ROWS:,} ratings …")
    sampled_ratings = _sample_ratings(ratings_path, SVD_SAMPLE_ROWS)
    print(f"Sampled ratings loaded: {len(sampled_ratings):,}")

    reader = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(
        sampled_ratings[["userId", "movieId", "rating"]],
        reader,
    )
    trainset = dataset.build_full_trainset()

    svd = SVD(
        n_factors=100,
        n_epochs=25,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        verbose=True,
    )
    svd.fit(trainset)

    out = MODEL_DIR / "svd_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(svd, f)

    print(f"✅ SVD model saved to {out}")


if __name__ == "__main__":
    main()
