"""
explore_data.py  –  Quick stats on your MovieLens 25M data
Run:  python explore_data.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CHUNK    = 500_000

print("\n" + "═" * 55)
print("  MovieBuzz – Dataset Explorer")
print("═" * 55)

# ── movies ────────────────────────────────────────────────────────────────────
movies = pd.read_csv(DATA_DIR / "movies.csv")
print(f"\n  Movies       : {len(movies):>10,}")
print(f"  Genres found : {movies['genres'].str.split('|').explode().nunique():>10,}")

top_genres = (
    movies["genres"].str.split("|").explode()
    .value_counts().head(10)
)
print("\n  Top genres:")
for g, cnt in top_genres.items():
    bar = "█" * int(cnt / top_genres.max() * 20)
    print(f"    {g:<18} {cnt:>6,}  {bar}")

# ── ratings (chunked) ─────────────────────────────────────────────────────────
n_rows = n_users = n_items = 0
rating_counts = {}
min_ts = max_ts = None

for chunk in pd.read_csv(
    DATA_DIR / "ratings.csv", chunksize=CHUNK,
    usecols=["userId", "movieId", "rating", "timestamp"]
):
    n_rows  += len(chunk)
    n_users += chunk["userId"].nunique()
    n_items += chunk["movieId"].nunique()
    for r, cnt in chunk["rating"].value_counts().items():
        rating_counts[r] = rating_counts.get(r, 0) + cnt
    ts_min = chunk["timestamp"].min()
    ts_max = chunk["timestamp"].max()
    min_ts = ts_min if min_ts is None else min(min_ts, ts_min)
    max_ts = ts_max if max_ts is None else max(max_ts, ts_max)

import datetime
date_min = datetime.datetime.fromtimestamp(min_ts).strftime("%Y-%m-%d") if min_ts is not None else "N/A"
date_max = datetime.datetime.fromtimestamp(max_ts).strftime("%Y-%m-%d") if max_ts is not None else "N/A"

print(f"\n  Ratings      : {n_rows:>10,}")
print(f"  Unique users : {n_users:>10,}  (approx – chunked)")
print(f"  Unique items : {n_items:>10,}  (approx – chunked)")
print(f"  Date range   : {date_min}  →  {date_max}")

print("\n  Rating distribution:")
for r in sorted(rating_counts):
    cnt = rating_counts[r]
    bar = "█" * int(cnt / max(rating_counts.values()) * 25)
    print(f"    ★ {r:<4}  {cnt:>9,}  {bar}")

# ── train / test check ────────────────────────────────────────────────────────
train_path = DATA_DIR / "train.csv"
test_path  = DATA_DIR / "test.csv"
if train_path.exists() and test_path.exists():
    tr = pd.read_csv(train_path, usecols=["rating"])
    te = pd.read_csv(test_path,  usecols=["rating"])
    total = len(tr) + len(te)
    print(f"\n  Train split  : {len(tr):>10,}  ({100*len(tr)/total:.1f}%)")
    print(f"  Test split   : {len(te):>10,}  ({100*len(te)/total:.1f}%)")
else:
    print("\n  train.csv / test.csv not found → run data_pipeline.py first")

# ── tags ──────────────────────────────────────────────────────────────────────
tags_path = DATA_DIR / "tags.csv"
if tags_path.exists():
    tags = pd.read_csv(tags_path, usecols=["tag"])
    print(f"\n  Tags         : {len(tags):>10,}")
    print(f"  Unique tags  : {tags['tag'].nunique():>10,}")

print("\n" + "═" * 55 + "\n")
