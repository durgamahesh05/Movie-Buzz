# MovieBuzz Recommendation Redesign

This redesign keeps the current MovieBuzz product behavior intact and upgrades the recommendation stack in an additive way.

Current codebase fit:
- API layer: `backend/app.py`
- Main recommender logic: `backend/recommender.py`
- Training script: `backend/run_training.py`
- User and wishlist tables: `backend/user_model.py`
- Database today: SQLite

The main bottleneck is not just model choice. The real issue is weak interaction data. Right now the system mostly learns from wishlist and a small amount of feedback, so the Neural Collaborative Filtering model is trying to solve a hard ranking problem with too little signal.

## 1. Recommended Architecture

Text diagram:

```text
Frontend
  -> sends events: view, click, search, wishlist, dwell/watch time

FastAPI backend
  -> /events/* endpoints write interactions
  -> /recommend endpoint builds recommendation request

Feature store layer
  -> SQLite now
  -> PostgreSQL later
  -> cached aggregates per user and per movie

Candidate generation
  -> NCF implicit model
  -> content-based retrieval
  -> popularity/trending/recent fallback

Candidate merge
  -> union top candidates from all generators

Ranking layer
  -> XGBoost ranker
  -> uses user, item, and context features

Serving layer
  -> returns ranked movies
  -> logs impression and click events for continuous learning
```

Recommended online flow:
1. If the user has enough interactions, use NCF to fetch top 100-300 candidates.
2. Always fetch 50-100 content-based candidates using genres, actors, plot keywords.
3. Always fetch 20-50 fallback candidates from trending, popular, and recently added lists.
4. Merge and deduplicate candidates.
5. Build ranking features.
6. Score with XGBoost.
7. Return top K.

This is beginner-friendly because each piece can be added one phase at a time.

## 2. Why the current NCF is weak

Based on the current training code in `backend/run_training.py`, the model has these limitations:
- It learns mostly from ratings and wishlist-like positives, not rich implicit behavior.
- The older NCF path under-sampled negatives. The current WSL training run moved to `6x` harder negatives using a random-plus-popular mix, and that improvement should be kept.
- Labels are binary and coarse.
- Sparse users are not supported with a strong fallback path.
- Headline evaluation should be ranking-first. `Precision@10`, `Recall@10`, `NDCG@10`, `HR@10`, and `MRR` are more meaningful than standalone AUC for recommendation quality.

That explains why BCE is high and AUC is close to random for a recommendation problem with sparse data.

## 2.1 Current Offline Benchmark

Latest measured run from the current WSL pipeline (`2026-03-25`, Run 2):

- `SVD`: `RMSE 0.9087`, `MAE 0.6937`
- `NCF`: `BCE 0.7574`, `AUC 0.6376`, `F1 0.2129`, `Precision 0.6953`, `Recall 0.1257`, `Precision@10 0.6391`, `Recall@10 0.5606`, `NDCG@10 0.7267`, `HR@10 0.9914`, `MRR 0.8218`
- `XGBoost`: `LogLoss 1.0616`, `AUC 0.6745`, `F1 0.5892`, `Precision@10 0.6304`, `Recall@10 0.6818`, `NDCG@10 0.7443`, `HR@10 0.9909`, `MRR 0.7984`

Interpretation:
- The harder-negative NCF run improved top-K ranking quality materially, even though threshold metrics such as raw `F1` and `Recall` moved in a less favorable direction.
- For design reviews and release sign-off, the headline metrics should now be `NDCG@10`, `HR@10`, and `MRR`, followed by `Precision@10` and `Recall@10`.
- `AUC`, `F1`, `BCE`, and `LogLoss` remain useful diagnostics, but they are secondary for this recommender.
- In the current codebase, `sentence-transformers` is used in `backend/recommender.py` for serving-time content similarity. The present `backend/run_training.py` XGBoost feature frame does not yet consume SBERT-derived features, so SBERT availability should not be treated as the current explanation for the XGBoost training delta.

## 3. Data Collection Improvements

### 3.1 New implicit feedback signals

Add these events:
- `impression`: movie shown in a list
- `view`: movie detail page opened
- `click`: movie card clicked
- `search`: user searched a query
- `wishlist_add`: added to wishlist
- `wishlist_remove`: removed from wishlist
- `trailer_open`: clicked trailer
- `watch_time`: dwell time on movie details or trailer page
- `rating_like`: explicit like
- `rating_dislike`: explicit dislike

### 3.2 Interaction weights

Use weighted implicit feedback instead of a single binary signal:

```text
impression      = 0.05
view            = 0.20
click           = 0.35
search_match    = 0.25
trailer_open    = 0.45
watch_time_30s  = 0.55
wishlist_add    = 1.00
like            = 1.20
dislike         = -0.80
wishlist_remove = -0.50
```

Important rule:
- Store raw events first.
- Compute weights later in a feature pipeline.
- Do not overwrite raw history.

### 3.3 Updated database schema

Additive schema for SQLite now:

```sql
CREATE TABLE IF NOT EXISTS user_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    movieId INTEGER,
    event_type TEXT NOT NULL,
    event_value REAL DEFAULT 1.0,
    weight REAL DEFAULT 0.0,
    session_id TEXT DEFAULT '',
    query_text TEXT DEFAULT '',
    source_page TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '',
    ts TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ui_user_ts
    ON user_interactions(user_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_ui_movie_ts
    ON user_interactions(movieId, ts DESC);

CREATE INDEX IF NOT EXISTS idx_ui_event_ts
    ON user_interactions(event_type, ts DESC);

CREATE TABLE IF NOT EXISTS movie_features (
    movieId INTEGER PRIMARY KEY,
    genres_text TEXT DEFAULT '',
    actors_text TEXT DEFAULT '',
    plot_text TEXT DEFAULT '',
    keywords_text TEXT DEFAULT '',
    language TEXT DEFAULT '',
    country TEXT DEFAULT '',
    year INTEGER DEFAULT 0,
    content_vector_path TEXT DEFAULT '',
    updated_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    genre_profile_json TEXT DEFAULT '',
    actor_profile_json TEXT DEFAULT '',
    keyword_profile_json TEXT DEFAULT '',
    last_active_at TEXT DEFAULT '',
    total_events INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS recommendation_impressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    movieId INTEGER NOT NULL,
    rank_position INTEGER NOT NULL,
    generator TEXT DEFAULT '',
    score REAL DEFAULT 0.0,
    ts TEXT NOT NULL
);
```

Why this helps:
- `user_interactions` becomes the source of truth for training.
- `movie_features` stores content-based metadata in a queryable format.
- `user_profiles` reduces online compute cost.
- `recommendation_impressions` lets you measure CTR and ranking quality later.

## 4. Cold Start Strategy

### 4.1 New user cold start

If a user has fewer than `N=5` strong interactions:
- show trending movies by genre if a search/query exists
- show popular movies overall
- show recently added movies
- add content-based matches from recent clicks or searches

### 4.2 New movie cold start

When an admin adds or uploads a movie:
1. Fetch OMDb metadata.
2. Extract genres, actors, director, plot text, year.
3. Build a content vector.
4. Insert the movie into content retrieval immediately.
5. Allow ranking even if the movie has no collaborative history yet.

### 4.3 Content-based method

Use a content representation built from:
- genres
- actors
- director
- plot keywords
- tags if available

Simple college-friendly option:
- `TfidfVectorizer` on combined text
- cosine similarity

Better option if already using sentence transformers:
- combine SBERT embedding for plot plus sparse TF-IDF features for genres/actors

Combined content text example:

```text
title + genres + actors + director + plot + tags
```

Example:

```text
Inception science fiction thriller Leonardo DiCaprio Christopher Nolan dream heist mind bending subconscious
```

## 5. NCF Redesign

### 5.1 Training objective

Move from weak explicit-only classification to weighted implicit recommendation:
- positive interactions: view, click, wishlist, trailer open, long dwell
- sampled negatives: unseen items, skipped items, bounce events

### 5.2 Better labels

Build an interaction score:

```text
implicit_score = sum(event_weight over a rolling 90-day window)
label = 1 if implicit_score >= 0.6 else 0
```

Also keep a confidence score:

```text
confidence = min(1.0, log1p(total_weight))
```

### 5.3 Better negative sampling

Current state:
- the latest training path already uses `6` negatives per positive
- the mix is currently random unseen items plus popular unseen items
- this harder-negative setup improved ranking quality and should remain the default baseline

Recommended:
- 4 to 10 negatives per positive
- mix three negative types:
  - random unseen items
  - popular unseen items
  - near-miss content-similar items the user ignored

Recommended ratio:
- `50%` random negatives
- `30%` popular negatives
- `20%` hard negatives

### 5.4 Better NCF architecture

Suggested model:
- user embedding: 32 or 64
- item embedding: 32 or 64
- MLP layers: `[128, 64, 32]`
- dropout: `0.1` to `0.2`
- batch normalization optional
- loss: binary cross entropy
- sample weights: use interaction confidence

### 5.5 Proper split

Use temporal split by user:
- train: oldest 80%
- validation: next 10%
- test: most recent 10%

Do not random split interactions across time for recommendation training.

### 5.6 Example NCF training pipeline

Sample code:

```python
def build_implicit_training_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        events_df.groupby(["user_id", "movieId"], as_index=False)
        .agg(
            total_weight=("weight", "sum"),
            last_ts=("ts", "max"),
            num_events=("event_type", "count"),
        )
    )
    grouped["label"] = (grouped["total_weight"] >= 0.6).astype("float32")
    grouped["sample_weight"] = np.clip(
        np.log1p(grouped["total_weight"].clip(lower=0.01)),
        0.05,
        2.0,
    )
    return grouped


def sample_negatives(positives_df: pd.DataFrame, all_movie_ids: np.ndarray, ratio: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    user_seen = positives_df.groupby("user_id")["movieId"].apply(set).to_dict()
    negative_rows = []

    for row in positives_df.itertuples():
        seen = user_seen.get(row.user_id, set())
        unseen = np.array(list(set(all_movie_ids) - seen), dtype=np.int32)
        if len(unseen) == 0:
            continue
        take = min(ratio, len(unseen))
        sampled = rng.choice(unseen, size=take, replace=False)
        for movie_id in sampled:
            negative_rows.append({
                "user_id": row.user_id,
                "movieId": int(movie_id),
                "label": 0.0,
                "sample_weight": 1.0,
            })

    return pd.DataFrame(negative_rows)


def train_ncf_implicit(events_df: pd.DataFrame, all_movie_ids: np.ndarray):
    tf = __import__("tensorflow")
    positives = build_implicit_training_frame(events_df)
    positives = positives[positives["label"] == 1].copy()
    negatives = sample_negatives(positives, all_movie_ids, ratio=6)
    train_df = pd.concat([positives, negatives], ignore_index=True)

    user_codes = {u: i for i, u in enumerate(train_df["user_id"].unique())}
    item_codes = {m: i for i, m in enumerate(train_df["movieId"].unique())}

    train_df["u_idx"] = train_df["user_id"].map(user_codes)
    train_df["i_idx"] = train_df["movieId"].map(item_codes)

    user_in = tf.keras.Input(shape=(1,), name="user_id")
    item_in = tf.keras.Input(shape=(1,), name="movie_id")

    user_emb = tf.keras.layers.Embedding(len(user_codes), 64)(user_in)
    item_emb = tf.keras.layers.Embedding(len(item_codes), 64)(item_in)

    user_vec = tf.keras.layers.Flatten()(user_emb)
    item_vec = tf.keras.layers.Flatten()(item_emb)

    x = tf.keras.layers.Concatenate()([user_vec, item_vec])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[user_in, item_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    model.fit(
        [train_df["u_idx"], train_df["i_idx"]],
        train_df["label"],
        sample_weight=train_df["sample_weight"],
        batch_size=2048,
        epochs=8,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=2,
                restore_best_weights=True,
                mode="max",
            )
        ],
        verbose=1,
    )
    return model, user_codes, item_codes
```

Expected benefit:
- more positives
- better negatives
- stronger user-item signal
- higher AUC than the current setup

## 6. XGBoost Feature Engineering

XGBoost should rank candidates, not learn recommendation from scratch.

### 6.1 Candidate generators feeding XGBoost

Use these candidate sources:
- `ncf`
- `content`
- `popular`
- `trending`
- `recent`

### 6.2 Better ranking features

Add the following features:

User features:
- total interactions in 7d, 30d, 90d
- number of wishlist adds
- number of views
- favorite genres distribution
- favorite actors distribution
- recency since last active

Movie features:
- global popularity
- 7d and 30d popularity
- CTR from impressions to clicks
- wishlist rate
- average dwell time
- recency since added
- OMDb rating
- num ratings

User x movie features:
- genre overlap score
- actor overlap score
- cosine similarity between movie content vector and user profile vector
- candidate source flags
- NCF score
- content similarity score
- popularity score
- search match score
- whether the movie was previously impressed but ignored

Context features:
- hour of day
- weekday
- source page
- query length
- search present yes/no

### 6.3 Better labels for XGBoost

Instead of:
- label = rating >= 4

Use session-aware labels:
- positive if clicked after impression
- stronger positive if wishlist add or long dwell follows
- negative if impressed and skipped many times

Sample label policy:

```text
wishlist_add or like        -> label 3
click + long dwell          -> label 2
click only                  -> label 1
impression without action   -> label 0
explicit dislike            -> label -1
```

For college-level simplicity:
- convert to binary classification first
- positive: click, wishlist_add, like
- negative: impression with no click after 24h

### 6.4 Example ranking feature frame

```python
feature_cols = [
    "ncf_score",
    "content_score",
    "popularity_score",
    "trending_score",
    "recent_score",
    "avg_rating",
    "imdb_rating",
    "num_ratings_log",
    "user_events_30d",
    "user_wishlist_30d",
    "genre_similarity",
    "actor_similarity",
    "keyword_similarity",
    "hours_since_last_user_event",
    "days_since_movie_added",
    "is_from_search",
    "is_from_content",
    "is_from_ncf",
]
```

## 7. Hybrid Recommendation Design

Recommended serving logic:

```text
If user interactions >= 5:
  candidates = NCF + content + trending
Else:
  candidates = content + popular + recent

If query exists:
  boost search-matching and content-similar movies

If candidate count is low:
  backfill from trending and recent

Rank final candidates with XGBoost
```

### 7.1 Score blending before ranking

Before XGBoost is ready, use a simple weighted blend:

```text
final_score =
    0.45 * ncf_score
  + 0.30 * content_score
  + 0.15 * trending_score
  + 0.10 * popularity_score
```

This gives a usable transition path while training better rankers.

### 7.2 Final hybrid stack

Phase 1:
- content + popularity + trending rules

Phase 2:
- NCF candidate generator

Phase 3:
- XGBoost reranker

This staged plan is safer than trying to deploy everything at once.

## 8. Evaluation Metrics

### 8.1 Model metrics to track

- Headline ranking metrics:
- Precision@5
- Precision@10
- Recall@5
- Recall@10
- NDCG@5
- NDCG@10
- HR@10
- MRR
- Diagnostic metrics:
- AUC
- BCE
- LogLoss
- F1
- MAP@10 if possible

### 8.2 Practical targets

Realistic targets for a college-to-production upgrade:
- keep `HR@10` around the current `~0.99` range or better
- improve `NDCG@10` beyond the current `0.7267` for NCF and `0.7443` for XGBoost
- improve `MRR` beyond the current `0.8218` for NCF and `0.7984` for XGBoost
- improve `Precision@10` and `Recall@10` week over week
- use `AUC`, `BCE`, and `LogLoss` as secondary diagnostics, not release headline metrics

### 8.3 Baselines to compare against

Always compare against:
- popularity-only
- trending-only
- content-only
- current NCF
- hybrid without XGBoost

If the fancy model does not beat popularity-only, do not ship it.

### 8.4 Example ranking metrics code

```python
def precision_at_k(y_true_ranked, k=10):
    topk = y_true_ranked[:k]
    return sum(topk) / max(1, k)


def recall_at_k(y_true_ranked, total_relevant, k=10):
    topk = y_true_ranked[:k]
    return sum(topk) / max(1, total_relevant)


def ndcg_at_k(y_true_ranked, k=10):
    dcg = 0.0
    for idx, rel in enumerate(y_true_ranked[:k], start=1):
        dcg += rel / np.log2(idx + 1)
    ideal = sorted(y_true_ranked, reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal[:k], start=1):
        idcg += rel / np.log2(idx + 1)
    return dcg / idcg if idcg > 0 else 0.0
```

## 9. Scalability and Production Plan

### 9.1 Why SQLite is fine for now

SQLite is okay for:
- course demos
- small user counts
- early prototype training
- single-machine development

### 9.2 When to move to PostgreSQL

Migrate to PostgreSQL when:
- concurrent writes increase
- event logging becomes heavy
- analytics queries slow down
- recommendation traffic grows

Recommended migration order:
1. keep current schema names
2. move event tables first
3. move user and wishlist tables
4. move recommendation logging
5. keep model files in the filesystem or object storage

### 9.3 Redis and caching

Optional but very useful:
- cache home page recommendations by `(user_id, context)`
- cache trending lists
- cache content neighbors for popular movies
- cache user profiles

Suggested TTLs:
- trending cache: 10 minutes
- home recommendations: 2 to 5 minutes
- movie detail cache: 1 hour

### 9.4 API latency optimization

Fast wins:
- precompute top content neighbors offline
- precompute popularity and trending tables
- precompute user profiles every few minutes
- avoid live feature joins over very large tables
- return top candidates first, then enrich details

Target latency:
- candidate generation: under 80 ms
- ranking: under 40 ms
- full API response: under 200 ms locally

## 10. Example API Endpoints

Keep your existing endpoints and add these:

```text
POST /events/impression
POST /events/view
POST /events/click
POST /events/search
POST /events/watch-time
GET  /recommendations/home?user_id=...&limit=20
GET  /recommendations/similar?movie_id=...&user_id=...&limit=20
GET  /recommendations/for-you?user_id=...&limit=20
GET  /recommendations/trending?genre=...&limit=20
GET  /recommendations/recent?limit=20
```

Example request body:

```json
{
  "user_id": "23012531027@gnu.ac.in",
  "movie_id": 296,
  "event_type": "view",
  "event_value": 1.0,
  "session_id": "sess_123",
  "source_page": "movie_detail",
  "query_text": "",
  "timestamp": "2026-03-25T18:30:00Z"
}
```

Example recommendation response:

```json
{
  "user_id": "23012531027@gnu.ac.in",
  "strategy": "hybrid_ranked",
  "request_id": "rec_20260325_183500_001",
  "items": [
    {
      "movie_id": 296,
      "title": "Pulp Fiction",
      "score": 0.923,
      "reasons": [
        "Because you often watch crime drama titles",
        "Popular among users with similar taste",
        "Matches your recent searches"
      ],
      "generator": "ncf+xgb"
    }
  ]
}
```

## 11. Recommended file-level changes in MovieBuzz

Minimal-impact implementation map:

- `backend/user_model.py`
  - add `user_interactions`, `user_profiles`, and `recommendation_impressions`

- `backend/app.py`
  - add event logging endpoints
  - keep current endpoints untouched
  - add new recommendation endpoints in parallel

- `backend/recommender.py`
  - add content-based candidate generator
  - add popularity and recent fallback functions
  - add candidate merge and reranking utilities

- `backend/run_training.py`
  - build interaction dataset from `user_interactions`
  - retrain NCF with implicit weighted labels
  - build XGBoost ranking dataset
  - report Precision@K, Recall@K, NDCG

- `frontend`
  - fire event logging requests on search, click, detail open, wishlist add, and dwell time
  - no UI redesign required

## 12. Step-by-step implementation plan

### Phase 1: Tracking and data foundation

1. Add `user_interactions` and `recommendation_impressions` tables.
2. Create event logging endpoints in FastAPI.
3. Send events from frontend:
   - search
   - card click
   - detail view
   - wishlist add
   - trailer open
   - dwell time
4. Build a daily aggregation job that converts raw events into weighted user-item interactions.

Success check:
- interaction table grows every day
- you can inspect user histories

### Phase 2: Cold-start and content-based retrieval

1. Build `movie_features` from OMDb metadata.
2. Generate content text per movie.
3. Train TF-IDF or SBERT content embeddings.
4. Add `get_similar_movies_content(movie_id)` function.
5. Add popular, trending, and recent fallback functions.

Success check:
- new users still receive sensible recommendations
- new admin-added movies can be recommended immediately

### Phase 3: Better NCF

1. Build implicit interaction dataset from event aggregates.
2. Introduce better negative sampling.
3. Add validation split and early stopping.
4. Tune embedding size and hidden layers.
5. Save NCF candidate scores for top-N generation.

Success check:
- AUC improves past `0.75`
- BCE drops significantly

### Phase 4: Better XGBoost reranking

1. Build candidate sets from NCF, content, and fallback generators.
2. Join user, movie, and pairwise features.
3. Train XGBoost classifier or ranker.
4. Evaluate with LogLoss, AUC, Precision@K, Recall@K, and NDCG.

Success check:
- reranked results beat popularity-only baseline

### Phase 5: Production hardening

1. Precompute trending lists and user profiles.
2. Add caching.
3. Log impressions for online evaluation.
4. Add dashboard metrics for:
   - CTR
   - wishlist rate
   - Precision@10
   - NDCG@10
5. Migrate heavy-write event tables to PostgreSQL when needed.

## 13. Practical recommendation for your current project

If the goal is a strong college project with industry relevance, do this in order:

1. Add event logging.
2. Add content-based fallback.
3. Rebuild NCF using weighted implicit interactions.
4. Upgrade XGBoost into a real reranker.
5. Add ranking metrics and an admin analytics panel.

If you do only one thing, do not start with model tuning. Start with better data collection. That will improve the whole system more than changing layer sizes alone.

## 14. Final recommendation

The best production-style version of MovieBuzz is:
- event-driven implicit feedback collection
- content-based retrieval for cold start
- NCF for collaborative candidate generation
- popularity/trending/recent as safe fallback
- XGBoost as the final ranking layer
- ranking metrics and impression logging for evaluation

This design is practical, additive, and realistic for your current MovieBuzz codebase.
