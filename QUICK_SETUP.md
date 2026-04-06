# MovieBuzz - PostgreSQL + Supabase + Render Deployment - Quick Setup Guide

## ✅ What's Been Done

### 1. **Fixed Pylance Type Errors** ✓
   - **8 reportOptionalSubscript errors** resolved in `app.py` and `recommender.py`
   - Added null checks for all `fetchone()` calls
   - All database queries now handle `None` results safely

### 2. **Database Configuration** ✓
   - Backend already supports both SQLite and PostgreSQL
   - Database abstraction layer in place (`db.py`)
   - Supabase connection strings fully configured

### 3. **Frontend-Backend Integration** ✓
   - Frontend API client configured to use `VITE_API_URL`
   - CORS middleware updated with environment-aware origins
   - Support for localhost, Render, and custom domains

### 4. **Environment Configuration** ✓
   - `.env.example` with all Supabase parameters
   - Frontend `.env.local` and `.env.example` created
   - `render.yaml` already configured for Render deployment

---

## 🚀 Quick Start (5 Steps)

### Step 1: Create Supabase Account & Database
```bash
# 1. Go to https://supabase.com
# 2. Sign up and create new project
# 3. Note down: Project Ref, Host, Password

# Example values:
SUPABASE_PROJECT_REF=helgktwlzbrjjnrpqyoz
SUPABASE_DB_HOST=db.helgktwlzbrjjnrpqyoz.supabase.co
SUPABASE_DB_PASSWORD=your_secure_password
```

### Step 2: Configure Backend Local Environment
```bash
cd backend

# Copy and edit environment file
cp .env.example .env

# Edit .env with your Supabase credentials:
# SUPABASE_PROJECT_REF=your_ref
# SUPABASE_DB_HOST=db.your_ref.supabase.co
# SUPABASE_DB_PASSWORD=your_password
# OMDB_API_KEY=your_key
# TMDB_API_KEY=your_key
```

### Step 3: Initialize Database (First Time Only)
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Initialize PostgreSQL database
python -c "from db import init_db; init_db()"

# (Optional) Load MovieLens data
python -c "from recommender import load_ml25m_to_db; load_ml25m_to_db()"
```

### Step 4: Configure & Run Frontend Locally
```bash
cd frontend/MOVIEBUZZ

# Install dependencies
npm install

# Create environment
cp .env.example .env.local

# Make sure VITE_API_URL=http://localhost:8000
# Start dev server
npm run dev
```

### Step 5: Deploy to Render
```bash
# 1. Push code to GitHub with .env.example (NOT .env)
git add .
git commit -m "PostgreSQL + Supabase + Render setup"
git push origin main

# 2. Go to https://render.com
# 3. Create Backend Web Service:
#    - Connect GitHub repo
#    - Root Directory: backend
#    - Build: pip install -r requirements.txt
#    - Start: uvicorn app:app --host 0.0.0.0 --port $PORT
#    - Add environment variables from Section 🔑 below

# 4. Create Frontend Static Site:
#    - Connect GitHub repo
#    - Root Directory: .
#    - Build: cd frontend/MOVIEBUZZ && npm install && npm run build
#    - Publish: frontend/MOVIEBUZZ/dist
#    - Add: VITE_API_URL=https://moviebuzz-backend.onrender.com
```

---

## 🔑 Environment Variables Checklist

### Backend (.env on Render)
```
# Supabase PostgreSQL
SUPABASE_PROJECT_REF=your_project_ref
SUPABASE_DB_HOST=db.your_project_ref.supabase.co
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=your_strong_password
SUPABASE_DB_NAME=postgres
SUPABASE_DB_PORT=5432
SUPABASE_DB_SSLMODE=require

# APIs
OMDB_API_KEY=your_omdb_key
TMDB_API_KEY=your_tmdb_key

# Email
SMTP_HOST=smtp.zoho.com
SMTP_PORT=587
SMTP_USER=your_email@zoho.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=noreply@moviebuzz.app

# Admin
MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON=["admin@example.com"]

# Deployment
MOVIEBUZZ_SKIP_STARTUP=0
MOVIEBUZZ_WARM_ENGINE=1
```

### Frontend (.env.local on Render)
```
VITE_API_URL=https://moviebuzz-backend.onrender.com
VITE_ENABLE_MOOD_BROWSE=true
VITE_ENABLE_ADMIN_PANEL=true
VITE_ENABLE_PREFERENCES=true
VITE_ENABLE_WISHLIST=true
```

---

## 📋 File References

### Key Files Updated:
1. **`backend/app.py`**
   - ✅ Fixed 4 Pylance errors 
   - ✅ Added environment-aware CORS

2. **`backend/recommender.py`**
   - ✅ Fixed 4 Pylance errors
   - ✅ Better null handling for database results

3. **`backend/.env.example`**
   - ✅ Comprehensive Supabase configuration

4. **`frontend/MOVIEBUZZ/.env.local`** (NEW)
   - Created for local development

5. **`render.yaml`**
   - ✅ Already configured for backend deployment

### Documentation Created:
1. **`POSTGRES_SUPABASE_DEPLOYMENT.md`** (50+ sections)
   - Complete step-by-step guide
   - Supabase setup instructions
   - Render deployment walkthrough
   - Troubleshooting section

2. **`FRONTEND_SETUP_DEPLOYMENT.md`** (40+ sections)
   - Frontend development setup
   - Render deployment options
   - API endpoint reference
   - Performance optimization

3. **This file: Quick Setup Summary**

---

## 🔗 Connection Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Browser                         │
│         (Frontend on https://yourdomain.com)            │
└──────────────┬──────────────────────────────────────────┘
               │ (REST API + CORS)
               ↓
┌──────────────────────────────────────────────────────────┐
│     Backend (FastAPI on Render)                          │
│   https://moviebuzz-backend.onrender.com/                │
└──────────────┬──────────────────────────────────────────┘
               │ (psycopg PostgreSQL driver)
               ↓
┌──────────────────────────────────────────────────────────┐
│        PostgreSQL Database (Supabase)                     │
│   db.project_ref.supabase.co:5432/postgres              │
│   - movies, users, ratings, recommendations              │
│   - Full transaction support, backups, monitoring         │
└──────────────────────────────────────────────────────────┘
```

---

## ⚡ Testing Checklist

### Local Testing
```bash
# Test backend PostgreSQL connection
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# Test frontend-backend connection
curl -H "Origin: http://localhost:5173" \
     "http://localhost:8000/search?q=inception"

# Test search with data
curl "http://localhost:8000/search?q=movie"
```

### Production Testing
```bash
# Test deployed backend
curl https://moviebuzz-backend.onrender.com/health

# Test CORS from frontend
# Open browser console and run:
fetch('https://moviebuzz-backend.onrender.com/health')
  .then(r => r.json())
  .then(console.log)

# Test full flow
# Visit https://moviebuzz-frontend.onrender.com
# Search for a movie - should work!
```

---

## 📊 Database Statistics

After setup, you'll have:
- ✓ 13 tables for users, movies, ratings, interactions
- ✓ Automatic backups on Supabase
- ✓ SSL connections (required)
- ✓ Connection pooling available
- ✓ Monitoring and analytics dashboard

### Example Schema:
```sql
movies (movieId, title, genres, avg_rating, trending_score, poster, ...)
users (id, name, email, password, verified, role, ...)
user_interactions (user_id, movieId, event_type, weight, timestamp, ...)
recommendation_impressions (request_id, user_id, movieId, rank, ...)
user_feedback (user_id, movieId, feedback, timestamp, ...)
```

---

## 🎓 Learning Resources

### Database & ORM:
- Supabase Docs: https://supabase.com/docs
- psycopg (PostgreSQL driver): https://www.psycopg.org/
- Query examples in `backend/db.py` and `backend/recommender.py`

### Backend Deployment:
- Render Docs: https://render.com/docs
- FastAPI: https://fastapi.tiangolo.com/
- CORS: https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS

### Frontend Deployment:
- Vite Docs: https://vitejs.dev/
- React: https://react.dev/
- Render Static Sites: https://render.com/static-sites

---

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `Connection refused` | Check Supabase project is running, credentials are correct |
| `CORS error in console` | Verify frontend origin in `app.py` CORS_ORIGINS list |
| `API returns 404` | Check backend is deployed, VITE_API_URL is correct |
| `psycopg2 not found` | Run `pip install psycopg[binary]` |
| `Module not found errors` | Run `pip install -r requirements.txt` |
| `Build fails on Render` | Check build command matches directory structure |
| `Blank page on frontend` | Check browser console for errors, network tab for failed requests |

---

## 📦 Deployment Checklist

### Before Deploying Backend:
- [ ] `.env.example` has all Supabase variables
- [ ] `.env` file has real credentials
- [ ] Database initialized: `python -c "from db import init_db; init_db()"`
- [ ] Code committed to GitHub (NOT `.env`)
- [ ] `render.yaml` configured correctly
- [ ] OMDB/TMDB API keys obtained

### Before Deploying Frontend:
- [ ] `.env.example` has VITE_API_URL
- [ ] `npm run build` succeeds locally
- [ ] `dist/` folder created with optimized files
- [ ] Backend API URL set correctly
- [ ] Code committed to GitHub

### After Deployment:
- [ ] Test `/health` endpoint
- [ ] Test search works
- [ ] Test recommendations work
- [ ] Check authentication flow
- [ ] Monitor logs for errors
- [ ] Set up monitoring (optional)

---

## 🎯 Next Steps

1. **Follow `POSTGRES_SUPABASE_DEPLOYMENT.md`** for detailed setup
2. **Follow `FRONTEND_SETUP_DEPLOYMENT.md`** for frontend deployment
3. **Test locally first** before deploying to Render
4. **Set up monitoring** (Sentry, Datadog, etc.)
5. **Configure custom domain** (optional but recommended)
6. **Plan scaling** based on usage

---

## ✨ You're All Set!

Your MovieBuzz application is now configured for:
- ✅ **PostgreSQL** database (via Supabase)
- ✅ **High availability** (managed by Supabase + Render)
- ✅ **SSL/TLS encryption** (secure connections)
- ✅ **Automatic backups** (Supabase)
- ✅ **Auto-deployment** (GitHub push → live)
- ✅ **CORS properly configured** (frontend-backend communication)
- ✅ **Type safe code** (all Pylance errors fixed)

```
Architecture:
Frontend (React) → Backend (FastAPI) → PostgreSQL (Supabase)
  on Render         on Render          on Supabase Cloud
```

Ready to go live! 🚀

For detailed instructions, see:
- **Backend**: `POSTGRES_SUPABASE_DEPLOYMENT.md`
- **Frontend**: `FRONTEND_SETUP_DEPLOYMENT.md`
