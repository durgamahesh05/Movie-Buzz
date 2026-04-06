# MovieBuzz Frontend - Setup & Deployment Guide

## Local Development Setup

### Prerequisites

- Node.js 18+ and npm
- Git
- Vite (included in project)
- Backend running locally (http://localhost:8000)

### Step 1: Install Dependencies

```bash
cd frontend/MOVIEBUZZ
npm install
```

### Step 2: Configure API Endpoint

Create or edit `.env.local`:

```bash
# Copy from example
cp .env.example .env.local

# Edit with your backend URL (default is localhost:8000)
VITE_API_URL=http://localhost:8000
```

### Step 3: Run Development Server

```bash
npm run dev
```

Frontend will be available at: **http://localhost:5173**

If port 5173 is in use, Vite will automatically use another port.

### Step 4: Build for Production

```bash
npm run build
```

Creates optimized build in `dist/` directory

---

## Deployment on Render

### Option 1: Static Site (Fastest, Recommended for SPA)

Best for React SPA that connects to separate backend API.

#### 1.1 Create Render Static Site

1. Go to **https://render.com > Dashboard**
2. Click **New > Static Site**
3. Connect your GitHub repository
4. Fill in settings:
   - **Name**: `moviebuzz-frontend`
   - **Build Command**: 
     ```bash
     cd frontend/MOVIEBUZZ && npm install && npm run build
     ```
   - **Publish Directory**: `frontend/MOVIEBUZZ/dist`
   - **Root Directory**: `.` (or blank)

#### 1.2 Add Environment Variables

Click **Environment > Add Environment Variable**:

```
VITE_API_URL=https://moviebuzz-backend.onrender.com
VITE_ENABLE_MOOD_BROWSE=true
VITE_ENABLE_ADMIN_PANEL=true
VITE_ENABLE_PREFERENCES=true
```

#### 1.3 Deploy

Click **Create Static Site** and wait 2-3 minutes.

Your frontend URL: `https://moviebuzz-frontend.onrender.com`

---

### Option 2: Web Service (More Control, Node.js Server)

Better for server-side rendering or custom server logic.

#### 2.1 Create Render Web Service

1. Go to **https://render.com > Dashboard**
2. Click **New > Web Service**
3. Connect repository
4. Settings:
   - **Name**: `moviebuzz-frontend`
   - **Environment**: `Node`
   - **Build Command**:
     ```bash
     cd frontend/MOVIEBUZZ && npm install && npm run build
     ```
   - **Start Command**:
     ```bash
     cd frontend/MOVIEBUZZ && npx serve -s dist -l 3000
     ```

#### 2.2 Add Environment Variables

```
VITE_API_URL=https://moviebuzz-backend.onrender.com
NODE_ENV=production
PORT=3000
```

#### 2.3 Deploy

Click **Create Web Service**

---

## Connecting Frontend to Backend

### 1. Update API Endpoint in Frontend

In `.env` or `.env.local`:

```bash
# Local development
VITE_API_URL=http://localhost:8000

# Production (Render)
VITE_API_URL=https://moviebuzz-backend.onrender.com
```

### 2. Verify Backend CORS

Backend must allow frontend origin. Check `backend/app.py`:

```python
CORS_ORIGINS = [
    "http://localhost:5173",                        # Local
    "http://localhost:3000",                        # Local
    "https://moviebuzz-frontend.onrender.com",     # Production
    "https://your-domain.com",                      # Custom domain
]
```

### 3. Test Connection

```bash
# From browser console on frontend:
fetch('https://moviebuzz-backend.onrender.com/health')
  .then(r => r.json())
  .then(console.log)

# Should return: { "status": "ok" }
```

---

## API Endpoints Reference

### Available Routes

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/search?q=query` | Search movies |
| GET | `/recommend?title=movie&top_n=50` | Get recommendations |
| GET | `/home?limit=50` | Get home feed |
| GET | `/browse/mood?mood=happy` | Browse by mood |
| POST | `/feedback` | Record user feedback |
| POST | `/auth/signup` | Register user |
| POST | `/auth/login` | Login user |
| GET | `/auth/preferences` | Get user preferences |
| POST | `/auth/preferences` | Save user preferences |
| GET | `/admin/overview` | Admin dashboard |
| GET | `/admin/movies` | List admin movies |
| POST | `/admin/movies` | Add manual movie |
| DELETE | `/admin/movies/{id}` | Delete movie |

### Example Requests

**Search:**
```bash
curl "http://localhost:8000/search?q=inception"
```

**Recommend:**
```bash
curl "http://localhost:8000/recommend?title=Inception&top_n=10"
```

**Browse by Mood:**
```bash
curl "http://localhost:8000/browse/mood?mood=happy"
```

**Login:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"pass"}'
```

---

## Troubleshooting

### Build Fails

**Issue**: `npm ERR! code ERESOLVE`

**Solution**: 
```bash
npm install --legacy-peer-deps
# or
npm install --force
```

### CORS Errors in Console

**Issue**: `Access to XMLHttpRequest blocked by CORS policy`

**Solution**: 
1. Check backend has frontend URL in `CORS_ORIGINS`
2. Restart backend
3. Clear browser cache: `Ctrl+Shift+Delete`

### API Endpoint Returns 404

**Issue**: `Fetch failed, status: 404`

**Solutions**:
```bash
# 1. Check backend is running
curl http://localhost:8000/health

# 2. Verify VITE_API_URL in .env.local
cat .env.local

# 3. Check browser DevTools Network tab for actual URL being called
```

### Frontend loads but no data appears

**Issue**: Page renders but lists are empty

**Solutions**:
1. Open browser DevTools > Console for errors
2. Check Network tab for failed API calls
3. Verify backend has data:
   ```bash
   curl "http://localhost:8000/search?q=movie"
   ```

### Render Build Fails

**Issue**: `Command failed: npm run build`

**Check**:
- `package.json` exists in `frontend/MOVIEBUZZ/`
- **Build Command** points to correct directory
- **Root Directory** is set properly
- All dependencies install without errors

```bash
# Test locally
cd frontend/MOVIEBUZZ
npm ci  # Clean install
npm run build
```

---

## Environment Variables Reference

### Browser-Accessible (Public)

Variables starting with `VITE_` are built into frontend bundle:

```
VITE_API_URL              - Backend API base URL
VITE_ENABLE_MOOD_BROWSE   - Show mood browsing feature
VITE_ENABLE_ADMIN_PANEL   - Show admin panel
VITE_ENABLE_PREFERENCES   - Show user preferences
VITE_ENABLE_WISHLIST      - Show wishlist feature
VITE_ANALYTICS_KEY        - Analytics/Sentry tracking
```

### Build-Time Only

Variables without `VITE_` prefix are not accessible in browser:

```
NODE_ENV                  - 'development' or 'production'
PORT                      - Server port (for Web Service)
```

---

## Performance Optimization

### 1. Enable Code Splitting

Already configured in Vite, but verify in `vite.config.ts`:

```typescript
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
        },
      },
    },
  },
});
```

### 2. Image Optimization

Movie posters are already cached and lazy-loaded. Check `src/lib/movie-cache.ts`

### 3. Bundle Analysis

```bash
npm install -D rollup-plugin-visualizer

# Then add to vite.config.ts and run build
npm run build
# Opens dist/stats.html
```

### 4. Minimize JavaScript

Vite automatically minifies in production build. Output should be < 500KB gzipped.

---

## Monitoring & Debugging

### 1. Check Render Logs

On Render dashboard:
- Click service name
- Go to **Logs** tab
- Monitor build and runtime errors

### 2. Browser DevTools

**F12 > Console**:
- Look for CORS errors
- API fetch failures
- React errors

**F12 > Network**:
- Check API requests status (200, 404, 500, etc.)
- Verify response payloads
- Monitor request timing

### 3. Backend Logs

```bash
# View backend logs
curl https://api.render.com/services/moviebuzz-backend/logs
# Or use Render CLI
render-cli logs moviebuzz-backend
```

---

## Custom Domain Setup

### 1. Point Domain to Render

1. Get your Render URL: `https://moviebuzz-frontend.onrender.com`
2. On domain registrar (GoDaddy, Namecheap, etc.):
   - Add **CNAME** record:
     ```
     www => moviebuzz-frontend.onrender.com
     @ => moviebuzz-frontend.onrender.com
     ```

### 2. Configure on Render

1. Go to **Settings > Custom Domain**
2. Enter your domain: `moviebuzz.app`
3. Render generates SSL certificate automatically

### 3. Update Frontend Config

```bash
# Update .env with custom domain
VITE_API_URL=https://api.moviebuzz.app  # or backend domain
```

---

## Summary Checklist

- [ ] Node.js 18+ installed
- [ ] Dependencies installed: `npm install`
- [ ] `.env.local` created with `VITE_API_URL`
- [ ] Backend running and accessible
- [ ] Local dev runs: `npm run dev`
- [ ] Build succeeds: `npm run build`
- [ ] Render account created
- [ ] GitHub repo connected
- [ ] Environment variables set on Render
- [ ] Frontend deployed
- [ ] Backend deployed
- [ ] Frontend → Backend connection tested
- [ ] Search and recommendations working
- [ ] Authentication working
- [ ] [Optional] Custom domain configured

---

## Next Steps

1. **Set up monitoring**: Integrate Sentry for error tracking
2. **Configure analytics**: Track user behavior
3. **Optimize performance**: Profile and reduce bundle size
4. **Plan scaling**: Monitor usage and upgrade Render tier if needed
5. **Implement CI/CD**: Auto-run tests before deployment

Your MovieBuzz frontend is now deployed and connected! 🎬
