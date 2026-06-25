# firehawk-project
Programming Lab/Applied Project final work for Bachelor in Informatics Engineering, ISLA Gaia - Polytechnic University of Management and Technology, Porto - Portugal

## Data refresh & persistence

The Flask service (`AI_FirehawkLab/server.py`) serves the built SPA and runs the
incident pipeline (`pipeline_active.run_pipeline()`) **on a schedule**, decoupled
from the user's click. The result snapshot (`fires.json`) is written to a
persistent location so it survives restarts/redeploys.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_DIR` | `/data` | Directory for the persisted `fires.json` snapshot. On Railway this is a mounted **Volume**. When the directory exists the pipeline writes there and the server serves `/data/fires.json` from it; otherwise it falls back to `firehawk-app/dist/data` (prod build) or `public/data` (local dev). |
| `REFRESH_CRON` | `0 6,12,18,0 * * *` | Cron expression (**UTC**) for the background refresh — by default 06:00, 12:00, 18:00 and 00:00. |
| `STARTUP_MAX_AGE_HOURS` | `6` | On boot, a refresh runs once if the snapshot is missing or older than this many hours. |
| `PORT` | `5001` | HTTP port. |

### Behaviour

- An in-app **APScheduler** `BackgroundScheduler` runs the pipeline per `REFRESH_CRON`
  (`max_instances=1`, `coalesce=True`).
- `POST /api/refresh-data` remains a manual trigger and shares the same anti-overlap
  guard, so the pipeline never runs twice at once (returns HTTP 409 while busy).
- On first boot against an empty volume, the snapshot is seeded from the built
  `dist/data/fires.json` so the dashboard renders immediately, then replaced by the
  first live run.

### Local run

```bash
cd AI_FirehawkLab
DATA_DIR=./_vol python server.py
# GET  http://localhost:5001/data/fires.json   -> served from ./_vol
# POST http://localhost:5001/api/refresh-data  -> manual refresh
```

### Railway (project: mindful-perfection)

Attach a Volume mounted at **`/data`** and set `DATA_DIR=/data` (plus optionally
`REFRESH_CRON`). No Dockerfile/CMD or frontend-build changes are required.
