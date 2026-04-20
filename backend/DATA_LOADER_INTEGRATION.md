"""
🔗 Data & Recommender Integration Examples
Shows how to use the CSV data loader in recommender.py and other modules
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 1: Use in recommender.py
# ═══════════════════════════════════════════════════════════════════════════════

"""
# At top of recommender.py, add:

from data_loader import (
    load_movies,
    load_ratings,
    load_genome_tags,
    load_genome_scores,
    get_movie_stats,
    get_genre_distribution,
    search_movies_by_title,
    search_movies_by_genre,
    get_top_rated_movies,
    get_most_rated_movies,
)

# Use in RecommenderEngine.__init__:
class RecommenderEngine:
    def __init__(self):
        log.info("Loading datasets...")
        self.movies_df = load_movies()
        self.ratings_df = load_ratings(sample=True)  # Sample for memory
        self.movie_stats = get_movie_stats()
        self.genre_dist = get_genre_distribution()
        self.genome_tags = load_genome_tags()
        
        log.info(f"✅ Engine initialized with {len(self.movies_df):,} movies")

# Use in search function:
def search(self, q: str, limit: int = 50) -> list:
    results = search_movies_by_title(q, self.movies_df, limit=limit)
    return results.to_dict('records')

# Use in recommendation function:
def recommend(self, title: str, top_n: int = 50) -> dict:
    # Find similar movies
    query_results = search_movies_by_title(title, self.movies_df, limit=1)
    if query_results.empty:
        return {"results": []}
    
    query_movie = query_results.iloc[0]
    
    # Merge with stats
    similar = self.movie_stats.merge(
        self.movies_df, on="movieId"
    ).sort_values("avg_rating", ascending=False)
    
    return {
        "resolved_title": query_movie["title"],
        "results": similar.head(top_n).to_dict('records'),
    }
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 2: Use in auth_routes.py (User Preferences)
# ═══════════════════════════════════════════════════════════════════════════════

"""
# In auth_routes.py for user preferences:

from data_loader import get_genre_distribution, search_movies_by_genre

@app.get("/api/genres")
async def get_available_genres():
    '''List all available genres for user preferences'''
    genre_dist = get_genre_distribution()
    return {
        "genres": genre_dist["genre"].tolist(),
        "stats": genre_dist.to_dict('records'),
    }

@app.get("/api/movies/by-genre/{genre}")
async def movies_by_genre(genre: str):
    '''Get movies in a specific genre'''
    results = search_movies_by_genre(genre)
    return {
        "genre": genre,
        "count": len(results),
        "movies": results.head(50).to_dict('records'),
    }
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 3: Use in FastAPI endpoints
# ═══════════════════════════════════════════════════════════════════════════════

"""
# In app.py:

from data_loader import (
    get_top_rated_movies,
    get_most_rated_movies,
    get_data_cache_info,
    search_movies_by_title,
)

@app.get("/api/movies/top-rated")
async def top_rated_movies(limit: int = 50):
    '''Get top-rated movies'''
    df = get_top_rated_movies(limit)
    return {
        "query": "top_rated",
        "limit": limit,
        "count": len(df),
        "results": df.to_dict('records'),
    }

@app.get("/api/movies/trending")
async def trending_movies(limit: int = 50):
    '''Get most-rated (trending) movies'''
    df = get_most_rated_movies(limit)
    return {
        "query": "trending",
        "limit": limit,
        "count": len(df),
        "results": df.to_dict('records'),
    }

@app.get("/api/system/cache")
async def cache_info():
    '''Display cache statistics'''
    info = get_data_cache_info()
    return {
        "status": "ok",
        "cached_datasets": info["cached_dataframes"],
        "total_datasets": info["total_cached"],
        "memory_usage_mb": round(info["total_memory_mb"], 2),
    }
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 4: Use in ML training scripts
# ═══════════════════════════════════════════════════════════════════════════════

"""
# In train_svd.py or other ML scripts:

from data_loader import (
    load_ratings,
    load_train_data,
    load_test_data,
    load_movies,
)
import numpy as np
from sklearn.decomposition import TruncatedSVD

def train_svd_model():
    '''Train SVD on ratings data'''
    ratings = load_ratings(sample=True, sample_size=1000000)
    movies = load_movies()
    
    # Create user-item matrix
    user_item_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    
    # Train SVD
    svd = TruncatedSVD(n_components=50)
    user_factors = svd.fit_transform(user_item_matrix)
    
    print(f"Trained SVD: {user_factors.shape}")
    return svd, user_factors

def load_ml_splits():
    '''Load pre-split train/test data'''
    train = load_train_data()
    test = load_test_data()
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")
    return train, test
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 5: Use in admin dashboard
# ═══════════════════════════════════════════════════════════════════════════════

"""
# In admin endpoints:

from data_loader import (
    get_movie_stats,
    get_user_stats,
    get_genre_distribution,
    get_data_cache_info,
)

@app.get("/api/admin/stats/movies")
async def admin_movie_stats(limit: int = 100):
    '''Admin movie statistics'''
    stats = get_movie_stats()
    return {
        "total_movies": len(stats),
        "avg_rating_overall": stats["avg_rating"].mean(),
        "top_movies": stats.nlargest(limit, "avg_rating").to_dict('records'),
    }

@app.get("/api/admin/stats/users")
async def admin_user_stats(limit: int = 100):
    '''Admin user statistics'''
    stats = get_user_stats()
    return {
        "total_users": len(stats),
        "avg_reviews_per_user": stats["num_ratings"].mean(),
        "top_reviewers": stats.nlargest(limit, "num_ratings").to_dict('records'),
    }

@app.get("/api/admin/stats/genres")
async def admin_genre_stats():
    '''Admin genre statistics'''
    dist = get_genre_distribution()
    return {
        "total_genres": len(dist),
        "genres": dist.to_dict('records'),
    }
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 6: Quick reference - Basic usage patterns
# ═══════════════════════════════════════════════════════════════════════════════

"""
# ─ Load specific datasets ─────────────────────────────────────────────────────
from data_loader import load_movies, load_ratings, load_tags

# Load movies
movies_df = load_movies()

# Load ratings (with caching - fast)
ratings_df = load_ratings(use_cache=True)

# Load ratings sample for memory efficiency
ratings_sample = load_ratings(sample=True, sample_size=100000)

# Load user tags
tags_df = load_tags()


# ─ Query data ─────────────────────────────────────────────────────────────────
from data_loader import search_movies_by_title, search_movies_by_genre

# Search by title
results = search_movies_by_title("Toy Story")

# Search by genre
horror_movies = search_movies_by_genre("Horror")


# ─ Get statistics ─────────────────────────────────────────────────────────────
from data_loader import get_movie_stats, get_user_stats, get_genre_distribution

# Movie statistics
movie_stats = get_movie_stats()
top_rated = movie_stats.nlargest(10, "avg_rating")

# User statistics
user_stats = get_user_stats()
top_reviewers = user_stats.nlargest(10, "num_ratings")

# Genre information
genres = get_genre_distribution()


# ─ Get curated lists ──────────────────────────────────────────────────────────
from data_loader import get_top_rated_movies, get_most_rated_movies

# Top 50 by rating
top_50 = get_top_rated_movies(limit=50)

# Most reviewed (trending)
trending_50 = get_most_rated_movies(limit=50)


# ─ DataFrame operations (pandas) ──────────────────────────────────────────────
# All functions return pandas DataFrames, so you can:

movies = load_movies()

# Filter
action_movies = movies[movies["genres"].str.contains("Action", na=False)]

# Group & aggregate
genre_counts = movies["genres"].str.split("|").explode().value_counts()

# Merge with other data
stats = get_movie_stats()
movies_with_stats = movies.merge(stats, on="movieId")

# Convert to dict/JSON
json_data = movies.head(10).to_dict('records')


# ─ Memory management ──────────────────────────────────────────────────────────
from data_loader import get_data_cache_info, clear_cache

# Check what's cached
info = get_data_cache_info()
print(f"Cached: {info['total_cached']} datasets")
print(f"Memory: {info['total_memory_mb']:.2f} MB")

# Clear cache to free memory (keep movies)
clear_cache(keep_movies=True)

# Clear everything
clear_cache(keep_movies=False)
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 7: Integration into recommender_mongodb.py
# ═══════════════════════════════════════════════════════════════════════════════

"""
# Update recommender_mongodb.py to use data_loader:

from data_loader import (
    load_movies,
    load_ratings,
    get_movie_stats,
    search_movies_by_title,
)

class MovieOperations:
    @staticmethod
    def search(query: str, limit: int = 50) -> list[Dict]:
        '''Search using data_loader instead of MongoDB'''
        results = search_movies_by_title(query, limit=limit)
        return results.to_dict('records')
    
    @staticmethod
    def get_trending(limit: int = 50, genre: Optional[str] = None) -> list[Dict]:
        '''Get trending movies by ratings'''
        stats = get_movie_stats()
        top = stats.nlargest(limit, "avg_rating")
        return top.to_dict('records')
    
    @staticmethod
    def load_ml25m_to_db():
        '''Load MovieLens 25M using data_loader'''
        movies = load_movies()
        ratings = load_ratings(sample=True, sample_size=1000000)
        
        # Aggregate by movie
        movie_stats = ratings.groupby("movieId")["rating"].agg([
            ("avg_rating", "mean"),
            ("num_ratings", "count"),
        ]).reset_index()
        
        # Merge and return
        result = movies.merge(movie_stats, on="movieId", how="left")
        return result
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════════

"""
❌ Issue: "MemoryError when loading ratings"
✅ Solution: Use sample=True parameter
   ratings = load_ratings(sample=True, sample_size=100000)

❌ Issue: "CSV file not found"
✅ Solution: Ensure CSV files are in backend/data/ directory
   Check: ls backend/data/

❌ Issue: "Slow when loading large files"
✅ Solution: Results are cached after first load
   Second call will be instant (from cache)

❌ Issue: "Need to reload data"
✅ Solution: Use use_cache=False
   movies = load_movies(use_cache=False)

❌ Issue: "Need to free memory"
✅ Solution: Clear cache
   from data_loader import clear_cache
   clear_cache(keep_movies=True)
"""
