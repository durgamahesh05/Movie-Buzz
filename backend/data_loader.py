"""
📊 CSV Data Loader Utility
Converts CSV files into in-memory tabular format (pandas DataFrames) for efficient access.
Handles large datasets with chunking and caching.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache
import warnings
from pandas.errors import DtypeWarning

warnings.filterwarnings('ignore', category=DtypeWarning)

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA DIRECTORY & PATHS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"

# CSV file paths
MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"
TAGS_CSV = DATA_DIR / "tags.csv"
GENOME_TAGS_CSV = DATA_DIR / "genome-tags.csv"
GENOME_SCORES_CSV = DATA_DIR / "genome-scores.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_DATAFRAMES_CACHE: Dict[str, pd.DataFrame] = {}
_SAMPLE_CACHE: Dict[str, pd.DataFrame] = {}
_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}

# ═══════════════════════════════════════════════════════════════════════════════
#  LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_movies(use_cache: bool = True) -> pd.DataFrame:
    """
    Load movies.csv as DataFrame
    
    Columns: movieId (int), title (str), genres (str - pipe-separated)
    
    Returns:
        pd.DataFrame: Movies data
    """
    if use_cache and "movies" in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE["movies"].copy()
    
    try:
        df = pd.read_csv(MOVIES_CSV)
        df["movieId"] = df["movieId"].astype(int)
        df["title"] = df["title"].astype(str)
        df["genres"] = df["genres"].fillna("").astype(str)
        
        _DATAFRAMES_CACHE["movies"] = df
        _METADATA_CACHE["movies"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
        log.info(f"✅ Loaded movies.csv → {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"❌ Failed to load movies.csv: {e}")
        return pd.DataFrame()


def load_ratings(use_cache: bool = True, sample: bool = False, sample_size: int = 100000) -> pd.DataFrame:
    """
    Load ratings.csv as DataFrame with optional sampling for large files
    
    Columns: userId (int), movieId (int), rating (float), timestamp (int)
    
    Args:
        use_cache: Use cached data if available
        sample: If True, return sample instead of full data
        sample_size: Number of rows to sample (if sample=True)
    
    Returns:
        pd.DataFrame: Ratings data (full or sample)
    """
    cache_key = f"ratings_sample_{sample_size}" if sample else "ratings"
    
    if use_cache and cache_key in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE[cache_key].copy()
    
    try:
        if sample:
            df = pd.read_csv(RATINGS_CSV, skiprows=lambda i: i > 0 and np.random.random() > (sample_size / 125000000))
            df = df.head(sample_size)
            log.info(f"✅ Loaded ratings sample → {len(df):,} rows")
        else:
            df = pd.read_csv(RATINGS_CSV)
            log.info(f"✅ Loaded ratings.csv → {len(df):,} rows")
        
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        df["rating"] = df["rating"].astype(float)
        df["timestamp"] = df["timestamp"].astype(int)
        
        _DATAFRAMES_CACHE[cache_key] = df
        _METADATA_CACHE[cache_key] = {
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "userId_unique": df["userId"].nunique(),
            "movieId_unique": df["movieId"].nunique(),
        }
        return df
    except Exception as e:
        log.error(f"❌ Failed to load ratings.csv: {e}")
        return pd.DataFrame()


def load_tags(use_cache: bool = True) -> pd.DataFrame:
    """
    Load tags.csv as DataFrame
    
    Columns: userId (int), movieId (int), tag (str), timestamp (int)
    
    Returns:
        pd.DataFrame: User-generated tags
    """
    if use_cache and "tags" in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE["tags"].copy()
    
    try:
        df = pd.read_csv(TAGS_CSV)
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        df["tag"] = df["tag"].astype(str)
        df["timestamp"] = df["timestamp"].astype(int)
        
        _DATAFRAMES_CACHE["tags"] = df
        _METADATA_CACHE["tags"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
        log.info(f"✅ Loaded tags.csv → {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"❌ Failed to load tags.csv: {e}")
        return pd.DataFrame()


def load_genome_tags(use_cache: bool = True) -> pd.DataFrame:
    """
    Load genome-tags.csv as DataFrame
    
    Columns: tagId (int), tag (str)
    
    Returns:
        pd.DataFrame: Genome tag definitions
    """
    if use_cache and "genome_tags" in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE["genome_tags"].copy()
    
    try:
        df = pd.read_csv(GENOME_TAGS_CSV)
        df["tagId"] = df["tagId"].astype(int)
        df["tag"] = df["tag"].astype(str)
        
        _DATAFRAMES_CACHE["genome_tags"] = df
        _METADATA_CACHE["genome_tags"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
        log.info(f"✅ Loaded genome-tags.csv → {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"❌ Failed to load genome-tags.csv: {e}")
        return pd.DataFrame()


def load_genome_scores(use_cache: bool = True, sample: bool = False, sample_size: int = 100000) -> pd.DataFrame:
    """
    Load genome-scores.csv as DataFrame with optional sampling
    
    Columns: movieId (int), tagId (int), relevance (float)
    
    Args:
        use_cache: Use cached data if available
        sample: If True, return sample instead of full data
        sample_size: Number of rows to sample (if sample=True)
    
    Returns:
        pd.DataFrame: Genome scores (tag relevance per movie)
    """
    cache_key = f"genome_scores_sample_{sample_size}" if sample else "genome_scores"
    
    if use_cache and cache_key in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE[cache_key].copy()
    
    try:
        if sample:
            df = pd.read_csv(GENOME_SCORES_CSV, skiprows=lambda i: i > 0 and np.random.random() > (sample_size / 14000000))
            df = df.head(sample_size)
            log.info(f"✅ Loaded genome-scores sample → {len(df):,} rows")
        else:
            df = pd.read_csv(GENOME_SCORES_CSV)
            log.info(f"✅ Loaded genome-scores.csv → {len(df):,} rows")
        
        df["movieId"] = df["movieId"].astype(int)
        df["tagId"] = df["tagId"].astype(int)
        df["relevance"] = df["relevance"].astype(float)
        
        _DATAFRAMES_CACHE[cache_key] = df
        _METADATA_CACHE[cache_key] = {
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
        return df
    except Exception as e:
        log.error(f"❌ Failed to load genome-scores.csv: {e}")
        return pd.DataFrame()


def load_train_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load train.csv (ML training set)
    
    Returns:
        pd.DataFrame: Training data
    """
    if use_cache and "train" in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE["train"].copy()
    
    try:
        df = pd.read_csv(TRAIN_CSV)
        _DATAFRAMES_CACHE["train"] = df
        _METADATA_CACHE["train"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
        log.info(f"✅ Loaded train.csv → {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"❌ Failed to load train.csv: {e}")
        return pd.DataFrame()


def load_test_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load test.csv (ML test set)
    
    Returns:
        pd.DataFrame: Test data
    """
    if use_cache and "test" in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE["test"].copy()
    
    try:
        df = pd.read_csv(TEST_CSV)
        _DATAFRAMES_CACHE["test"] = df
        _METADATA_CACHE["test"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }
        log.info(f"✅ Loaded test.csv → {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"❌ Failed to load test.csv: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data(use_cache: bool = True, skip_large: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files at once
    
    Args:
        use_cache: Use cached data
        skip_large: Skip very large files (ratings, genome-scores)
    
    Returns:
        Dict mapping filename to DataFrame
    """
    dataframes = {
        "movies": load_movies(use_cache=use_cache),
        "tags": load_tags(use_cache=use_cache),
        "genome_tags": load_genome_tags(use_cache=use_cache),
        "train": load_train_data(use_cache=use_cache),
        "test": load_test_data(use_cache=use_cache),
    }
    
    if not skip_large:
        dataframes["ratings"] = load_ratings(use_cache=use_cache, sample=False)
        dataframes["genome_scores"] = load_genome_scores(use_cache=use_cache, sample=False)
    
    log.info(f"✅ Loaded {len(dataframes)} datasets")
    return dataframes


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION & STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def get_movie_stats() -> pd.DataFrame:
    """
    Compute per-movie statistics from ratings
    
    Returns:
        DataFrame with: movieId, avg_rating, num_ratings, min_rating, max_rating, std_rating
    """
    if "movie_stats" in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE["movie_stats"].copy()
    
    ratings = load_ratings(sample=True, sample_size=1000000)
    if ratings.empty:
        return pd.DataFrame()
    
    try:
        stats = ratings.groupby("movieId").agg(
            avg_rating=pd.NamedAgg(column="rating", aggfunc="mean"),
            num_ratings=pd.NamedAgg(column="rating", aggfunc="count"),
            min_rating=pd.NamedAgg(column="rating", aggfunc="min"),
            max_rating=pd.NamedAgg(column="rating", aggfunc="max"),
            std_rating=pd.NamedAgg(column="rating", aggfunc="std"),
        ).reset_index()
        
        stats["avg_rating"] = stats["avg_rating"].round(2)
        stats["std_rating"] = stats["std_rating"].fillna(0).round(2)
        
        _DATAFRAMES_CACHE["movie_stats"] = stats
        log.info(f"✅ Computed movie stats → {len(stats):,} movies")
        return stats
    except Exception as e:
        log.error(f"❌ Failed to compute movie stats: {e}")
        return pd.DataFrame()


def get_user_stats() -> pd.DataFrame:
    """
    Compute per-user statistics from ratings
    
    Returns:
        DataFrame with: userId, num_ratings, avg_rating, review_activity
    """
    if "user_stats" in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE["user_stats"].copy()
    
    ratings = load_ratings(sample=True, sample_size=1000000)
    if ratings.empty:
        return pd.DataFrame()
    
    try:
        stats = ratings.groupby("userId").agg(
            num_ratings=pd.NamedAgg(column="rating", aggfunc="count"),
            avg_rating=pd.NamedAgg(column="rating", aggfunc="mean"),
            min_rating=pd.NamedAgg(column="rating", aggfunc="min"),
            max_rating=pd.NamedAgg(column="rating", aggfunc="max"),
        ).reset_index()
        
        stats["avg_rating"] = stats["avg_rating"].round(2)
        
        _DATAFRAMES_CACHE["user_stats"] = stats
        log.info(f"✅ Computed user stats → {len(stats):,} users")
        return stats
    except Exception as e:
        log.error(f"❌ Failed to compute user stats: {e}")
        return pd.DataFrame()


def get_genre_distribution() -> pd.DataFrame:
    """
    Compute genre distribution from movies
    
    Returns:
        DataFrame with: genre, movie_count, avg_rating, total_ratings
    """
    if "genre_distribution" in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE["genre_distribution"].copy()
    
    movies = load_movies()
    ratings = load_ratings(sample=True, sample_size=1000000)
    
    if movies.empty or ratings.empty:
        return pd.DataFrame()
    
    try:
        # Explode genres
        movie_genres = movies[["movieId", "genres"]].copy()
        movie_genres["genre"] = movie_genres["genres"].str.split("|")
        movie_genres = movie_genres.explode("genre")
        movie_genres["genre"] = movie_genres["genre"].str.strip()
        
        # Merge with ratings
        genre_stats = movie_genres.merge(ratings, on="movieId", how="left")
        
        # Aggregate
        dist = genre_stats.groupby("genre").agg({
            "movieId": "nunique",
            "rating": ["count", "mean"]
        }).round(2)
        
        dist.columns = ["movie_count", "total_ratings", "avg_rating"]
        dist = dist.sort_values("movie_count", ascending=False).reset_index()
        
        _DATAFRAMES_CACHE["genre_distribution"] = dist
        log.info(f"✅ Computed genre distribution → {len(dist)} genres")
        return dist
    except Exception as e:
        log.error(f"❌ Failed to compute genre distribution: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def search_movies_by_title(query: str, movies_df: Optional[pd.DataFrame] = None, limit: int = 20) -> pd.DataFrame:
    """
    Search movies by title (case-insensitive substring match)
    
    Args:
        query: Search query
        movies_df: Movies DataFrame (loads if None)
        limit: Max results
    
    Returns:
        Matching movies
    """
    if movies_df is None:
        movies_df = load_movies()
    
    query_lower = query.lower().strip()
    if not query_lower:
        return pd.DataFrame()
    
    mask = movies_df["title"].str.lower().str.contains(query_lower, regex=False, na=False)
    return movies_df[mask].head(limit)


def search_movies_by_genre(genre: str, movies_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Search movies by genre (case-insensitive)
    
    Args:
        genre: Genre name
        movies_df: Movies DataFrame (loads if None)
    
    Returns:
        Movies with matching genre
    """
    if movies_df is None:
        movies_df = load_movies()
    
    genre_lower = genre.lower().strip()
    if not genre_lower:
        return pd.DataFrame()
    
    mask = movies_df["genres"].str.lower().str.contains(genre_lower, regex=False, na=False)
    return movies_df[mask]


def get_top_rated_movies(limit: int = 50) -> pd.DataFrame:
    """Get top-rated movies by average rating"""
    movies = load_movies()
    stats = get_movie_stats()
    
    if movies.empty or stats.empty:
        return pd.DataFrame()
    
    top = stats.nlargest(limit, "avg_rating")
    return movies.merge(top, on="movieId")[["movieId", "title", "genres", "avg_rating", "num_ratings"]]


def get_most_rated_movies(limit: int = 50) -> pd.DataFrame:
    """Get most-rated movies by number of ratings"""
    movies = load_movies()
    stats = get_movie_stats()
    
    if movies.empty or stats.empty:
        return pd.DataFrame()
    
    top = stats.nlargest(limit, "num_ratings")
    return movies.merge(top, on="movieId")[["movieId", "title", "genres", "avg_rating", "num_ratings"]]


def get_data_cache_info() -> Dict[str, Any]:
    """Get cache statistics"""
    cache_info = {
        "cached_dataframes": list(_DATAFRAMES_CACHE.keys()),
        "total_cached": len(_DATAFRAMES_CACHE),
        "metadata": dict(_METADATA_CACHE),
        "total_memory_mb": sum(
            meta.get("size_mb", 0) for meta in _METADATA_CACHE.values()
        ),
    }
    return cache_info


def clear_cache(keep_movies: bool = True):
    """Clear cache to free memory"""
    global _DATAFRAMES_CACHE, _METADATA_CACHE
    
    if keep_movies:
        movies = _DATAFRAMES_CACHE.pop("movies", None)
        movies_meta = _METADATA_CACHE.pop("movies", None)
        _DATAFRAMES_CACHE.clear()
        _METADATA_CACHE.clear()
        if movies is not None:
            _DATAFRAMES_CACHE["movies"] = movies
        if movies_meta is not None:
            _METADATA_CACHE["movies"] = movies_meta
        log.info("✅ Cache cleared (kept movies)")
    else:
        _DATAFRAMES_CACHE.clear()
        _METADATA_CACHE.clear()
        log.info("✅ Cache fully cleared")


def display_data_summary() -> str:
    """Display summary of all loaded datasets"""
    summary = "\n" + "="*70 + "\n"
    summary += "📊 DATA LOADER SUMMARY\n"
    summary += "="*70 + "\n"
    
    cache_info = get_data_cache_info()
    
    for dataset_name, metadata in cache_info["metadata"].items():
        rows = metadata.get("rows", "?")
        size_mb = metadata.get("size_mb", 0)
        summary += f"\n📁 {dataset_name.upper()}\n"
        summary += f"   Rows: {rows:,}\n"
        summary += f"   Size: {size_mb:.2f} MB\n"
        summary += f"   Columns: {metadata.get('columns', [])}\n"
    
    summary += f"\n💾 Total Cached Memory: {cache_info['total_memory_mb']:.2f} MB\n"
    summary += "="*70 + "\n"
    
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n🔄 Loading all datasets...")
    all_data = load_all_data(skip_large=False)
    print(display_data_summary())
    
    print("\n📊 Computing statistics...")
    print("\n🎬 Movie Stats (sample):")
    print(get_top_rated_movies(10))
    
    print("\n🏷️ Top Genres:")
    print(get_genre_distribution().head(10))
