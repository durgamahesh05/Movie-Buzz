# MovieBuzz

MovieBuzz is a full-stack movie discovery and recommendation platform built with a React + Vite frontend and a FastAPI backend. It combines search, personalized recommendations, mood-based browsing, wishlist management, OTP-based authentication, trailer playback, and admin tools for managing users, movies, and model metrics.

## Features

- Hybrid recommendation engine using TF-IDF, SBERT, SVD, ALS, NCF, sentiment, and ranking signals
- Movie search, home feed, movie details, mood browsing, ratings, and feedback
- OTP signup, verification, password reset, and account deletion flows
- User preference survey for age, genres, and moods
- Wishlist management with saved movie metadata
- Trailer lookup and caching using TMDb, OMDb, and YouTube embeds
- Admin dashboard for users, movies, CSV/manual uploads, and model metrics

## Tech Stack

- Frontend: React 18, TypeScript, Vite, React Router, Zustand, React Query, Radix UI
- Backend: FastAPI, Uvicorn, MongoDB, PyMongo
- ML/Data: scikit-learn, sentence-transformers, TensorFlow/Keras, implicit, XGBoost, LightGBM, CatBoost
- External services: MongoDB Atlas, OMDb, TMDb, SMTP email

## Project Structure

```text
MovieBuzz/
├── backend/                 # FastAPI API, auth, recommender, training/bootstrap scripts
├── frontend/MOVIEBUZZ/      # React + Vite frontend
├── Dockerfile               # Docker deployment
├── render.yaml              # Render backend config
├── railway.toml             # Railway config
├── package.json             # Root frontend shortcut scripts
└── .env.example             # Example backend environment variables
```

## Prerequisites

- Python 3.10.13
- Node.js and npm
- MongoDB Atlas cluster or local MongoDB
- OMDb API key
- TMDb API key
- SMTP credentials for OTP/email flows

## Local Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd MovieBuzz
```

### 2. Configure environment variables

Create a root `.env` file from `.env.example` and update the important values:

- `MONGODB_URI`
- `DATABASE_NAME`
- `SECRET_KEY`
- `FRONTEND_URL`
- `OMDB_API_KEY`
- `TMDB_API_KEY`
- `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`
- `MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON`

Example admin seed:

```env
MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON=[{"name":"Admin","email":"admin@example.com","password":"change-me"}]
```

### 3. Install backend dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
python -m textblob.download_corpora
```

### 4. Start the backend

```bash
cd backend
uvicorn app:app --reload
```

Backend runs on `http://localhost:8000`.

### 5. Configure and start the frontend

Create `frontend/MOVIEBUZZ/.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Then run:

```bash
cd frontend/MOVIEBUZZ
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.

## Optional Data Bootstrap

To load MovieLens data into MongoDB and prepare recommender assets:

```bash
cd backend
python bootstrap_backend.py --prefetch-posters
```

If you only want to import data without training/warming the engine:

```bash
python bootstrap_backend.py --skip-train
```

To run the fuller training pipeline and benchmark models:

```bash
python run_training.py --train-benchmarks
```

## Useful Root Scripts

From the repo root:

```bash
npm run dev
npm run build
```

These proxy to `frontend/MOVIEBUZZ`.

## API Highlights

- `GET /health` - backend health check
- `GET /search` - catalog search
- `GET /movies/home` - home movie feed
- `GET /movies/{movie_id}/details` - movie details
- `GET /recommend` - personalized recommendations
- `GET /mood/{mood}` and `GET /moods` - mood browsing
- `POST /feedback` and `POST /feedback/rating` - feedback and ratings
- `GET /api/trailer/{movie_id}` - trailer lookup
- `POST /auth/signup`, `POST /auth/login`, `POST /auth/verify-otp`
- `GET /auth/wishlist/{email}` - wishlist management
- `GET /admin/overview`, `GET /admin/movies`, `GET /admin/model-metrics`

## Local Development Notes

- On localhost, OTP flows can fall back to returning the OTP in the API response if email is not configured.
- `MOVIEBUZZ_DEV=1` allows flexible local CORS handling.
- For local development, install from `backend/requirements.txt`.
- The root `requirements.txt` is intended for deployment-friendly environments.

## Deployment

This repo already includes deployment support for:

- Vercel for the frontend
- Render for the backend
- Railway via `railway.toml`
- Docker via the included `Dockerfile`

Set your production environment variables before deploying, especially:

- `FRONTEND_URL`
- `MONGODB_URI`
- `SECRET_KEY`
- `OMDB_API_KEY`
- `TMDB_API_KEY`
- SMTP settings
- `MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON`

## Summary

MovieBuzz is designed as a movie recommendation and management platform with a modern frontend, a FastAPI backend, MongoDB storage, and a hybrid ML recommendation workflow that supports both user-facing discovery and admin-side catalog control.
