# firehawk-project
 
Programming Lab / Applied Project final work for the Bachelor in Informatics
Engineering, ISLA Gaia — Polytechnic University of Management and Technology,
Porto, Portugal.
 
FireHawk is a decision-support demonstrator for wildfire response in Portugal. It
serves the dissertation's tier classifier and two-stage aerial model: for each live
incident it predicts an expected response tier (minimal / standard / reinforced) and
an aerial-mobilisation probability, instead of exact resource counts.
 
## Architecture
 
Single containerised hybrid monolith deployed on Railway:
 
- `AI_FirehawkLab/` — Python (Flask) inference server. Builds the live incident
  feed, runs `pipeline_active.run_pipeline()`, serves the built SPA and the API.
- `firehawk-app/` — React (Vite) frontend; map rendered with Leaflet over
  OpenStreetMap.
- Auth backend — separate Node.js/Express service with a managed MySQL database
  (role-based access: operator vs viewer).
The active model is the self-contained bundle `AI_FirehawkLab/model_tier_pipeline.pkl`
(KMeans tiers + Random Forest tier classifier + two-stage aerial model).
 
## Model and results

The active model is the dissertation tier formulation, validated on a strict temporal
split (train 2019-2022, validation 2023, test 2024-2025).

- Tier classifier (Random Forest), test 2024-2025: accuracy 0.488, balanced accuracy
  0.406, weighted F1 0.480; per-tier recall T0/T1/T2 = 0.261 / 0.677 / 0.280; vehicle
  range accuracy 72.2%.
- Two-stage aerial model: operational threshold 0.40, selected by maximum F1 on the
  validation set (2023) and frozen before test reporting (not tuned on the test set);
  test ROC-AUC 0.655, PR-AUC 0.012.
- `n_concurrent_fires` is leakage-free: it counts only fires in the same district and
  calendar day that ignited at or before the incident (no look-ahead; DHFIM/duration
  are not used). It is a minor predictor (MDI importance rank ~19 of 21), not a top feature.


## Environment variables
 
| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_DIR` | `/data` | Directory for the persisted `fires.json` snapshot. On Railway this is a mounted Volume; otherwise the server falls back to `firehawk-app/dist/data` (prod build) or `public/data` (local dev). |
| `REFRESH_CRON` | `0 6,12,18,0 * * *` | Cron (UTC) for the background refresh — 06:00, 12:00, 18:00, 00:00. |
| `STARTUP_MAX_AGE_HOURS` | `6` | On boot, refresh once if the snapshot is missing or older than this. |
| `PORT` | `5001` | HTTP port. |
 
## Data refresh and persistence
 
- An in-app APScheduler `BackgroundScheduler` runs the pipeline per `REFRESH_CRON`
  (`max_instances=1`, `coalesce=True`); the frontend reads the cached snapshot.
- `POST /api/refresh-data` is a manual trigger. It starts the pipeline in a
  background thread and returns immediately (HTTP 202); the frontend polls
  `fires.json` for the result, so the request never blocks or times out. A run
  already in progress returns HTTP 409 (shared anti-overlap guard).
- On first boot against an empty volume, the snapshot is seeded from the built
  `dist/data/fires.json`, then replaced by the first live run.
## Run locally
 
Prerequisites: Python 3.10+, Node.js 18+, Git LFS.
> Windows (PowerShell): set environment variables on their own line — e.g.
> `$env:DATA_DIR="./_vol"` then `python server.py`. If `python`/`pip` are not
> recognised, use the `py` launcher: `py server.py`, `py -m pip install -r requirements.txt`.
 
1. Fetch the model bundle (LFS):
```bash
   git lfs install && git lfs pull
```
2. Inference server:
```bash
   cd AI_FirehawkLab
   pip install -r requirements.txt
   DATA_DIR=./_vol python server.py
   # GET  http://localhost:5001/data/fires.json
   # POST http://localhost:5001/api/refresh-data
```
3. Frontend (separate terminal):
```bash
   cd firehawk-app
   npm install
   npm run dev      # Vite dev server on :5173 (calls Python :5001 and auth :5000 via config.js)
   # or: npm run build   # static build served by the Flask server
```
4. Auth backend (separate terminal; needed for login / RBAC):
```bash
   cd firehawk-backend
   npm install
   # create .env with MySQL settings: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME (optional PORT)
   node setup.js     # provisions and seeds the MySQL schema (default admin user)
   npm start         # http://localhost:5000  (config.js -> AUTH_API)
```
## Deploy to Railway
 
The `Dockerfile` builds the whole stack (Python + Node + React) into one image.
 
1. Connect the GitHub repo to the Railway service; deploys trigger on push to
   `master`.
2. Attach a Volume mounted at `/data` and set `DATA_DIR=/data` (optionally
   `REFRESH_CRON`, `STARTUP_MAX_AGE_HOURS`).
3. Push to `master` -> Railway builds and deploys automatically.
### Model artifact at build time
 
`model_tier_pipeline.pkl` (~135 MB) exceeds GitHub's 100 MB blob limit and is stored
via Git LFS. Railway's build context does **not** include the `.git` directory, so
`git lfs pull` cannot run inside the build; loading the LFS pointer instead of the
real binary fails with `KeyError: 118` (the pointer text starts with `v` = byte 118).
 
The `Dockerfile` therefore downloads the real bundle by URL from GitHub's LFS media
endpoint (public repo, no auth):
 
```
https://media.githubusercontent.com/media/<owner>/<repo>/master/AI_FirehawkLab/model_tier_pipeline.pkl
```
 
A size gate fails the build if the download is too small, and a `joblib.load` check
verifies the bundle loads, so a broken image is never shipped. For a private repo,
pass a token to the `curl` call or publish the bundle as a GitHub Release asset and
download that instead.