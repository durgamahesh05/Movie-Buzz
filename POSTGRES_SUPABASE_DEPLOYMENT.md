# MovieBuzz PostgreSQL + Supabase + Render Deployment Guide

## Overview

This guide walks you through migrating MovieBuzz from SQLite to PostgreSQL (via Supabase) and deploying the backend on Render.

---

## Prerequisites

- Supabase account (free tier: https://supabase.com)
- Render account (free tier: https://render.com)
- GitHub repository with MovieBuzz code pushed
- local MovieBuzz database with trained models (if migrating existing data)

---

## Step 1: Set Up Supabase PostgreSQL Database

### 1.1 Create Supabase Project

1. Go to **https://supabase.com** and sign in
2. Click **New Project**
3. Name: `moviebuzz`
4. Password: Create a strong database password (copy this!)
5. Region: Choose closest to your users
6. Click **Create new project** (wait 2-3 minutes)

### 1.2 Get Database Credentials

1. Once ready, click the project name to open dashboard
2. Go to **Settings > Database**
3. Copy the following:
   - **Project Reference**: Look in the connection string URL or under "General"
   - **Host**: `db.[PROJECT_REF].supabase.co`
   - **Port**: `5432`
   - **Database**: `postgres`
   - **User**: `postgres`
   - **Password**: Your created password

Example connection string format:
```
postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres?sslmode=require
```

### 1.3 Initialize Database Schema

#### Option A: Using Migration Tool (Recommended)

If you have an existing SQLite database with schema:

```bash
cd backend
python migrate_sqlite_to_postgres.py \
  --source-db moviebuzz.db \
  --dest-url "postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres?sslmode=require"
```

#### Option B: Manual Init

Run the schema initialization in `db.py`:

```bash
python -c "from db import init_db; init_db()"
```

This creates all tables:
- `movies` - Movie catalog
- `tags` - Movie tags
- `genome_scores` - Tag relevance scores
- `users` - User accounts
- `wishlist` - Saved movies
- `user_interactions` - Viewing history
- `user_profiles` - Preference profiles
- `user_feedback` - Ratings/likes
- `omdb_cache` - Cached OMDB metadata
- `trailer_cache` - YouTube trailer links
- `rating_timestamps` - Temporal data
- `model_metrics` - Evaluation reports
- `recommendation_impressions` - A/B test data

---

## Step 2: Configure Local Environment

### 2.1 Create .env file

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 2.2 Fill in Supabase Credentials

Edit `.env`:
```bash
# Option 1: Use individual Supabase variables
SUPABASE_PROJECT_REF=your_project_ref
SUPABASE_DB_HOST=db.your_project_ref.supabase.co
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=your_strong_password
SUPABASE_DB_NAME=postgres
SUPABASE_DB_PORT=5432
SUPABASE_DB_SSLMODE=require

# Or Option 2: Full connection string (comment out above)
# DATABASE_URL=postgresql://postgres:password@db.project_ref.supabase.co:5432/postgres?sslmode=require
```

### 2.3 Add API Keys

```bash
OMDB_API_KEY=your_key_from_omdbapi.com
TMDB_API_KEY=your_key_from_themoviedb.org
SMTP_USER=your_email@domain.com
SMTP_PASSWORD=your_app_password
```

---

## Step 3: Test Local Connection

```bash
# Install dependencies
pip install -r requirements.txt

# Test database connection
python -c "from db import is_postgres; print(f'Using PostgreSQL: {is_postgres()}')"

# Optional: Load MovieLens data
python -c "from recommender import load_ml25m_to_db; load_ml25m_to_db()"

# Start backend
uvicorn app:app --reload
```

Test endpoints:
- Health: `http://localhost:8000/health`
- Search: `http://localhost:8000/search?q=inception`
- Recommend: `http://localhost:8000/recommend?title=Inception`

---

## Step 4: Deploy Backend on Render

### 4.1 Connect GitHub Repository

1. Push your code to GitHub (with `.env.example` in repo)
2. Go to **https://render.com**
3. Click **Dashboard > New > Web Service**
4. Click **Connect Repository** and authorize GitHub
5. Select `MovieBuzz` repository

### 4.2 Configure Deployment

1. **General Settings**:
   - Name: `moviebuzz-backend`
   - Environment: `Python`
   - Region: Choose closest to users
   - Plan: **Free** (or Starter for production)

2. **Build & Deploy**:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Auto-deploy: Enable

3. **Environment Variables**:
   Click **Add Environment Variable** for each:

   ```
   DATABASE_URL = (leave blank if using SUPABASE_* vars below)
   SUPABASE_PROJECT_REF = your_project_ref
   SUPABASE_DB_HOST = db.your_project_ref.supabase.co
   SUPABASE_DB_PASSWORD = your_password
   SUPABASE_DB_USER = postgres
   SUPABASE_DB_NAME = postgres
   SUPABASE_DB_PORT = 5432
   SUPABASE_DB_SSLMODE = require
   OMDB_API_KEY = your_key
   TMDB_API_KEY = your_key
   SMTP_HOST = smtp.zoho.com
   SMTP_PORT = 587
   SMTP_USER = your_email
   SMTP_PASSWORD = your_app_password
   SMTP_FROM = noreply@moviebuzz.app
   SUPPORT_EMAIL = support@moviebuzz.app
   MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON = ["admin@domain.com"]
   MOVIEBUZZ_SKIP_STARTUP = 0
   MOVIEBUZZ_WARM_ENGINE = 0
   ```

   **Note**: Mark sensitive vars as "Secret" (in newer Render UI)

### 4.3 Deploy

1. Click **Create Web Service**
2. Wait 3-5 minutes for Render to build and deploy
3. Check **Logs** tab for any errors
4. Get your public URL: `https://moviebuzz-backend.onrender.com`

### 4.4 Verify Deployment

```bash
# Test from terminal
curl https://moviebuzz-backend.onrender.com/health

# Test search
curl "https://moviebuzz-backend.onrender.com/search?q=inception"

# View logs on Render dashboard
```

---

## Step 5: Deploy Frontend and Connect to Backend

### 5.1 Update Frontend API Endpoint

In `frontend/MOVIEBUZZ/src/api.ts` (or similar):

```typescript
const API_BASE = process.env.REACT_APP_API_URL || 'https://moviebuzz-backend.onrender.com';

export const search = (q: string) => fetch(`${API_BASE}/search?q=${q}`).then(r => r.json());
export const recommend = (title: string) => fetch(`${API_BASE}/recommend?title=${title}`).then(r => r.json());
```

### 5.2 Configure CORS

In `backend/app.py`, update CORS middleware:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local dev
        "http://localhost:3000",
        "https://moviebuzz-frontend.onrender.com",  # Render frontend
        "https://yourdomain.com",  # Your custom domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 5.3 Deploy Frontend on Render

1. Go to Render Dashboard
2. **New > Static Site** (or **Web Service** for Node apps)
3. Connect GitHub repo
4. Build Command: `cd frontend/MOVIEBUZZ && npm run build`
5. Publish Directory: `frontend/MOVIEBUZZ/build` or `dist`
6. Deploy

Your frontend will be accessible at: `https://moviebuzz-frontend.onrender.com`

---

## Step 6: Connect Frontend to Backend

### 6.1 Add Backend URL to Frontend

Update environment variables in frontend `.env`:

```bash
VITE_API_URL=https://moviebuzz-backend.onrender.com
```

Or directly in code:

```typescript
const API_BASE = 'https://moviebuzz-backend.onrender.com';
```

### 6.2 Test End-to-End

1. Visit `https://moviebuzz-frontend.onrender.com`
2. Search for movies
3. Check browser DevTools Console for any CORS errors
4. Verify requests are reaching backend

---

## Troubleshooting

### Connection Issues

**Issue**: `psycopg2.OperationalError: connection failed`

**Solution**:
```bash
# Test Supabase connection locally
python -c "
import os
os.environ['SUPABASE_PROJECT_REF'] = 'your_ref'
os.environ['SUPABASE_DB_PASSWORD'] = 'your_password'
from db import get_db
with get_db() as conn:
    print('Connected!', conn.execute('SELECT version()').fetchone())
"
```

### Render Deployment Failures

**Issue**: Build fails during deployment

**Check**:
- `requirements.txt` has all dependencies
- `render.yaml` matches your settings
- Environment variables are set correctly
- Check Render logs for details

```bash
# View logs
render-cli logs moviebuzz-backend
```

### CORS Errors

**Issue**: Frontend can't reach backend (`OPTIONS 403`)

**Solution**: Add frontend URL to `CORS_ORIGINS` in `app.py`

### Database Empty after Deploy

**Issue**: Data not visible after deployment

**Solution**: Initialize data via Render Shell or migrations:

```bash
# On Render via shell
python migrate_sqlite_to_postgres.py --skip-truncate
# or
python -c "from recommender import load_ml25m_to_db; load_ml25m_to_db()"
```

---

## Advanced Configuration

### Custom Domain

1. Purchase domain (GoDaddy, Namecheap, etc.)
2. On Render, go to **Settings > Custom Domain**
3. Add domain name
4. Update DNS records (Render will show instructions)
5. Update CORS_ORIGINS for new domain

### Environment-Specific Configuration

```bash
# Production (Render)
MOVIEBUZZ_WARM_ENGINE=1
TRAIN_USE_GPU=1
LOG_LEVEL=INFO

# Development (Local)
DATABASE_URL=sqlite:///moviebuzz.db
LOG_LEVEL=DEBUG
MOVIEBUZZ_SKIP_STARTUP=1
```

### Monitoring & Logs

- **Render Logs**: Dashboard > Logs tab
- **PostgreSQL Monitoring**: Supabase > Database > Logs
- **Error Tracking**: Consider Sentry integration

### Scaling

#### Free Tier (Render)
- Auto-pauses after 15 min of inactivity
- 0.5 CPU, 512 MB RAM

#### Paid Tier
- Always running
- More CPU/RAM
- SSL certificates included
- Recommend: **Starter** ($7/mo) for small projects

---

## Cleanup & Maintenance

### Backup Database

```bash
# Export Supabase data
pg_dump postgresql://postgres:password@db.project_ref.supabase.co:5432/postgres > backup.sql

# Or use Supabase automated backups (Settings > Backups)
```

### Monitor Database Performance

On Supabase:
1. Go to **Database > Query Performance**
2. Identify slow queries
3. Add indexes if needed:
```sql
CREATE INDEX idx_movies_title ON movies(title);
CREATE INDEX idx_users_email ON users(email);
```

### Update Dependencies

```bash
# Check for updates
pip list --outdated

# Update in requirements.txt and test locally
pip install -U fastapi psycopg

# Redeploy
git push origin main  # Render auto-deploys
```

---

## Summary

| Component | Service | Status |
|-----------|---------|--------|
| Database | Supabase PostgreSQL | ✅ Deployed |
| Backend | Render Web Service | ✅ Deployed |
| Frontend | Render Static/Web | ✅ Deployed |
| Storage | Supabase (PostgreSQL) | ✅ Connected |
| API | FastAPI + CORS | ✅ Connected |

Your MovieBuzz application is now fully deployed in production! 🚀

For updates, just push to GitHub and Render will auto-deploy.
