# MovieBuzz GitHub Publish Guide

This file documents exactly what belongs in GitHub for this repository and what must stay local.

## Push / Skip Matrix

Priority meanings:

- `must` = include or exclude this on the first public push
- `optional` = safe either way, but the recommendation below is best for a clean repo

### Project structure items

| Path | Decision | Reason | Priority |
| --- | --- | --- | --- |
| `backend/` | PUSH | Main backend source folder | must |
| `backend/__pycache__/` | SKIP | Python bytecode cache | must |
| `backend/.venv/` | SKIP | Local Python environment | must |
| `backend/.venv-wsl/` | SKIP | WSL-only Python environment | must |
| `backend/data/README.md` | PUSH | Explains how to download and place data locally | must |
| `backend/data/train.csv` | SKIP | Generated split, large local artifact | must |
| `backend/data/test.csv` | SKIP | Generated split, large local artifact | must |
| `backend/models/README.md` | PUSH | Explains how to regenerate models locally | must |
| `backend/models/svd_model.pkl` | SKIP | Generated ML artifact | must |
| `backend/models/als_model.pkl` | SKIP | Generated ML artifact | must |
| `backend/models/ncf_model.keras` | SKIP | Generated ML artifact | must |
| `backend/models/xgb_ranker.pkl` | SKIP | Generated ML artifact | must |
| `backend/models/eval_report.json` | SKIP | Generated training report | must |
| `backend/app.py` | PUSH | FastAPI app entrypoint | must |
| `backend/auth_routes.py` | PUSH | Auth routes and admin bootstrap logic | must |
| `backend/bootstrap_backend.py` | PUSH | Startup helper | must |
| `backend/config.py` | PUSH | Local `.env` loader and safe config helpers | must |
| `backend/data_pipeline.py` | PUSH | Data processing and split logic | must |
| `backend/email_service.py` | PUSH | SMTP integration with env-based secrets | must |
| `backend/explore_data.py` | PUSH | Data exploration utility | optional |
| `backend/recommender.py` | PUSH | Recommendation engine | must |
| `backend/requirements.txt` | PUSH | Backend dependency lock point | must |
| `backend/run_training.py` | PUSH | Main training entrypoint | must |
| `backend/run_training_wsl.sh` | PUSH | WSL training launcher | must |
| `backend/train_svd.py` | PUSH | SVD helper script | optional |
| `backend/trailer_router.py` | PUSH | Trailer and metadata routes | must |
| `backend/user_model.py` | PUSH | SQLite user store code | must |
| `backend/view_movies.py` | PUSH | Admin/local inspection helper | optional |
| `backend/view_users.py` | PUSH | Admin/local inspection helper | optional |
| `backend/moviebuzz.db` | SKIP | SQLite database with local/user data | must |
| `backend/moviebuzz.corrupt_*` | SKIP | Corrupt database backups | must |
| `backend/training.log*` | SKIP | Local logs and backups | must |
| `backend/backend_8000.err.log` | SKIP | Local server log | must |
| `backend/backend_8000.out.log` | SKIP | Local server log | must |
| `backend/.env` | SKIP | Secret config file | must |
| `backend/.env.example` | PUSH | Public-safe config template | must |
| `frontend/` | PUSH | Frontend source tree | must |
| `frontend/MOVIEBUZZ/node_modules/` | SKIP | Installed frontend dependencies | must |
| `frontend/MOVIEBUZZ/build/` | SKIP | Generated frontend build output | must |
| `frontend/MOVIEBUZZ/src/` | PUSH | React + TypeScript source | must |
| `frontend/MOVIEBUZZ/public/` | PUSH | Static assets | must |
| `frontend/MOVIEBUZZ/package.json` | PUSH | Frontend dependencies and scripts | must |
| `frontend/MOVIEBUZZ/package-lock.json` | PUSH | Deterministic frontend installs | must |
| `frontend/MOVIEBUZZ/tsconfig.json` | PUSH | TypeScript config | must |
| `frontend/MOVIEBUZZ/pyrightconfig.json` | PUSH if present | Helpful type-check config | optional |
| `docs/` | PUSH | Project documentation | must |
| `.vscode/` | SKIP | Personal IDE settings | optional |
| `package.json` | PUSH | Root workspace scripts | must |
| `package-lock.json` | PUSH | Root lock file | must |
| `pyrightconfig.json` | PUSH | Root type-check config | optional |
| `training.log` | SKIP | Local training log | must |

### Additional local artifacts found in this checkout

| Path | Decision | Reason | Priority |
| --- | --- | --- | --- |
| `.venv/` | SKIP | Tracked virtualenv content is polluting git history | must |
| `frontend/MOVIEBUZZ/frontend_preview.*.log` | SKIP | Local preview logs | must |
| `moviebuzz.db` | SKIP | Root-level local SQLite file | must |
| `node_modules/` | SKIP | Root-level local Node install | must |
| `.pytest_cache/` | SKIP | Test cache | optional |

## Secret handling rules

- Never commit SMTP credentials.
- Never commit API keys.
- Never commit `.env`.
- Never commit any SQLite database file containing local or user data.
- If you need seeded local admin accounts, supply them through `MOVIEBUZZ_SYSTEM_ADMIN_ACCOUNTS_JSON` in `backend/.env`, not in source code.

## Git commands

These commands are written for PowerShell and for the current repository you want to push to:

`https://github.com/durgamahesh05/Movie-Buzz.git`

### 1. Initialize / fix the remote

```powershell
git init
git branch -M main
git remote remove origin 2>$null
git remote add origin https://github.com/durgamahesh05/Movie-Buzz.git
```

### 2. Stop tracking local-only artifacts without deleting local files

```powershell
git rm -r --cached --ignore-unmatch .venv backend/.venv backend/.venv-wsl frontend/MOVIEBUZZ/node_modules node_modules frontend/MOVIEBUZZ/build .pytest_cache .vscode
git rm -r --cached --ignore-unmatch backend/data backend/models
git rm --cached --ignore-unmatch moviebuzz.db backend/moviebuzz.db training.log backend/training.log
```

### 3. Stage the publish-safe project

After `.gitignore` is in place and the cached junk is removed, this is the safe staging command:

```powershell
git add .gitignore backend/.env.example backend/config.py backend/data/README.md backend/models/README.md docs/github-publish-guide.md
git add backend frontend docs package.json package-lock.json pyrightconfig.json
```

### 4. Verify nothing sensitive is staged

```powershell
git status --short
git diff --cached --name-only
git diff --cached --stat
```

You should **not** see:

- `.env`
- any `*.db`
- `backend/data/*.csv`
- `backend/models/*` binaries
- `node_modules/`
- logs or backups

### 5. Commit

```powershell
git commit -m "prepare MovieBuzz for GitHub"
```

### 6. Push

```powershell
git push -u origin main
```

### 7. If the remote already has commits and push is rejected

```powershell
git pull --rebase origin main
git push -u origin main
```
