# MovieBuzz – Complete Setup & Run Guide

## Folder Structure

```
moviebuzz-backend/
│
├── app.py                  ← FastAPI server (main entry point)
├── recommender.py          ← Full ML engine (TF-IDF, SBERT, SVD, NCF, ALS, XGB)
├── data_pipeline.py        ← Train/test split + model training + evaluation
├── explore_data.py         ← Quick dataset stats
├── train_svd.py            ← Standalone SVD pre-trainer
├── requirements.txt
├── moviebuzz.db            ← Auto-created SQLite (users + movies + cache)
│
├── auth/
│   ├── __init__.py         ← Empty file (create manually)
│   ├── auth_routes.py      ← Signup / Login / OTP (SQLite)
│   └── user_model.py       ← SQLite user CRUD helpers
│
├── data/                   ← Download from MovieLens 25M
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   ├── genome-scores.csv
│   ├── genome-tags.csv
│   ├── train.csv           ← Auto-created by data_pipeline.py
│   └── test.csv            ← Auto-created by data_pipeline.py
│
└── models/                 ← Auto-created, models saved here
    ├── svd_model.pkl
    ├── als_model.pkl
    ├── ncf_model.keras
    ├── xgb_ranker.pkl
    ├── sbert_embeddings.npy
    └── eval_report.json
```

---

## Step 1 — Download MovieLens 25M

Go to: https://grouplens.org/datasets/movielens/25m/

Download and extract. Copy these files into your `data/` folder:
- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `genome-scores.csv`
- `genome-tags.csv`

---

## Step 2 — Create auth/__init__.py

```bash
mkdir auth
echo "# auth package" > auth/__init__.py
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

---

## Step 4 — Explore your data (optional but recommended)

```bash
python explore_data.py
```

Output shows total movies, ratings, date range, rating distribution,
and train/test split sizes.

---

## Step 5 — Run the data pipeline (train all models)

```bash
python data_pipeline.py
```

This does everything in one command:
1. Loads 25M ratings in 500K chunks (RAM safe)
2. Temporal train/test split — last 20% of each user's ratings = test
3. Negative sampling for NCF (4 negatives per positive)
4. Trains SVD → ALS → NCF → XGBoost
5. Evaluates all models (RMSE, BCE, BPR, AUC, NDCG@10, etc.)
6. Saves `models/eval_report.json`

Takes 20–40 minutes on first run. After that all models are cached on disk.

---

## Step 6 — Start the server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

On first server start:
- SQLite `moviebuzz.db` is created automatically
- Users table initialised
- Movies loaded from CSVs into DB
- If models already trained in Step 5, loads instantly

---

## API Endpoints

### Auth

| Method | URL | Body | Description |
|--------|-----|------|-------------|
| POST | `/auth/signup` | `{name, email, password}` | Register — sends OTP email |
| POST | `/auth/verify-otp` | `{email, otp}` | Verify OTP |
| POST | `/auth/resend-otp` | `{email}` | Resend OTP |
| POST | `/auth/login` | `{email, password}` | Login — returns name, email, role |

### Movies

| Method | URL | Params | Description |
|--------|-----|--------|-------------|
| GET | `/search` | `?q=inception` | Search movies |
| GET | `/recommend` | `?title=inception&user_id=1&mood=excited` | Hybrid recommendations |
| GET | `/mood/{mood}` | — | Browse by mood |
| GET | `/moods` | — | List all valid moods |

### User Features

| Method | URL | Body | Description |
|--------|-----|------|-------------|
| POST | `/feedback` | `{user_id, movie_id, feedback}` | Like / dislike a movie |

feedback values: `"like"` `"dislike"` `"neutral"`

### Admin — Movies

| Method | URL | Body/File | Description |
|--------|-----|-----------|-------------|
| POST | `/admin/movies/manual` | `[{title, genres, rating, year}]` | Add movies manually |
| POST | `/admin/movies/csv` | CSV file upload | Upload CSV from admin dashboard |

### Admin — Users

| Method | URL | Body | Description |
|--------|-----|------|-------------|
| GET | `/admin/users` | — | List all users |
| DELETE | `/admin/users/{email}` | — | Delete a user |
| PATCH | `/admin/users/{email}/role` | `{role}` | Change role: user / mod / admin |

---

## Mood Options

| Mood | Genres matched |
|------|----------------|
| happy | Comedy, Animation, Family, Musical |
| sad | Drama, Romance |
| excited | Action, Adventure, Thriller |
| scared | Horror, Thriller, Mystery |
| romantic | Romance, Drama |
| thoughtful | Documentary, Drama, Sci-Fi |
| adventurous | Adventure, Action, Fantasy |
| relaxed | Comedy, Family, Animation |

---

## How the 6 files connect

```
data_pipeline.py
   └─ loads data/ratings.csv + movies.csv
   └─ creates data/train.csv + data/test.csv
   └─ trains → saves SVD, ALS, NCF, XGB to models/

app.py
   └─ imports recommender.py  (all ML logic)
   └─ imports auth/auth_routes.py  (login/signup)

auth/auth_routes.py
   └─ imports auth/user_model.py  (SQLite CRUD)

recommender.py
   └─ reads models/ (loads pre-trained models)
   └─ reads/writes moviebuzz.db  (movies, tags, genome, omdb cache, feedback)

user_model.py
   └─ writes to moviebuzz.db  (users table)
```

All 6 files share ONE database: `moviebuzz.db`
No MongoDB. No separate database server. SQLite handles everything.

---

## Quick test after setup

```bash
# Search
curl http://localhost:8000/search?q=inception

# Recommend with mood
curl "http://localhost:8000/recommend?title=inception&mood=excited"

# Signup
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@email.com","password":"pass123"}'

# Admin: list users
curl http://localhost:8000/admin/users

# Admin: add movie manually
curl -X POST http://localhost:8000/admin/movies/manual \
  -H "Content-Type: application/json" \
  -d '[{"title":"My Movie","genres":"Action Drama","rating":8.5,"year":"2024"}]'
```
