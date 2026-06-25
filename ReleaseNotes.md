Firehawk Project - Release Notes

Version: v2.1 (Scheduled Refresh & Persistent Snapshot) Date: June 25, 2026 Deployment Target: Railway PaaS / Local Hybrid Environment

🦅 Overview
Release v2.1 decouples data collection from request serving. The incident pipeline now runs on an in-app schedule and persists its snapshot to a Railway Volume, so the dashboard no longer depends on a synchronous run at click time and survives restarts/redeploys.

✨ Changes
- In-App Scheduler: server.py runs run_pipeline() via an APScheduler BackgroundScheduler on REFRESH_CRON (default "0 6,12,18,0 * * *", UTC), with max_instances=1 and coalesce=True. A refresh also runs once on boot when the snapshot is missing or older than STARTUP_MAX_AGE_HOURS (default 6h).
- Persistent Snapshot (DATA_DIR): pipeline_active.py writes fires.json to DATA_DIR (default "/data") when that directory exists, and server.py serves /data/fires.json from it. On an empty volume the snapshot is seeded from the built dist/data/fires.json so the UI renders immediately.
- Shared Anti-Overlap Guard: POST /api/refresh-data stays as a manual trigger and shares the scheduler's lock; concurrent triggers return HTTP 409 instead of running the pipeline twice.
- No changes to the frontend build or the Dockerfile CMD.

Configuration: see README ("Data refresh & persistence") for DATA_DIR, REFRESH_CRON, STARTUP_MAX_AGE_HOURS. Railway: attach a Volume mounted at /data and set DATA_DIR=/data.

---

Version: v2.0 (Production Release) Date: January 4, 2026 Deployment Target: Railway PaaS / Local Hybrid Environment

🦅 Overview
Release v2.0 marks the transition of the Firehawk Project from a development prototype to a production-capable Hybrid Monolith. This version introduces a unified container architecture that orchestrates a Python AI pipeline, a Node.js authentication backend, and a React frontend simultaneously. It addresses critical data persistence issues on cloud infrastructure and refines the predictive algorithms for the 2026 fire season.

✨ Key Functionalities
1. Hybrid Infrastructure & Deployment

Unified Containerization: Implements a single Dockerfile utilizing a hybrid image (python3.10-nodejs22) to execute both the Flask data server and the React build process.



Automated SPA Routing: The Python backend (server.py) now includes a "catch-all" routing mechanism to serve the React index.html for client-side paths (e.g., /dashboard), preventing 404 errors on page refresh.


Production-Aware File Paths: The system dynamically detects the environment to serve static assets correctly from the dist folder in production or public in development.

2. Intelligent Data Pipeline (ETL & AI)

Adaptive Backfill Algorithm: The pipeline_active.py script now features an expanding search window (up to 365 days) to locate historical fire data even during low-activity seasons (Winter/Autumn).

Advanced Weather Integration: Integration with Open-Meteo APIs to fetch real-time and historical data, including new meteorological variables:

Vapor Pressure Deficit (VPD) in kPa.

Atmospheric Pressure (MSL).

Wind Direction.


FWI System Implementation: Full algorithmic implementation of the Canadian Forest Fire Weather Index (FWI) system, calculating FFMC, DMC, DC, ISI, and BUI based on raw weather inputs.


Resource Prediction: Deployment of the model_resources_lite.pkl (Random Forest) to predict Human, Terrestrial, and Aerial resource requirements based on environmental conditions.


3. Authentication & Security

Role-Based Access Control (RBAC): Distinct user roles ("Operator" vs. "Viewer") managed via a dedicated Node.js backend.


Automated Database Seeding: The setup.js utility ensures the MySQL schema (Unidade table) is automatically provisioned and seeded with default administrative credentials upon first deployment.

🐛 Main Corrections & Fixes
Railway Persistence Fix: Resolved an issue where the frontend displayed outdated JSON data. The pipeline now detects the production build folder (dist/data) and writes the fresh fires.json directly to the active web directory.

Date Sorting Logic: Fixed the frontend parsing logic to correctly handle date formats (converting DD-MM-YYYY to valid JavaScript Date objects), ensuring incidents from 2026 and late 2025 appear at the top of the dashboard.


VPD & Pressure Visibility: Corrected the data transformation layer to ensure vpd_kpa and pressao values from the Python backend are correctly mapped and displayed in the Incident Details UI.


404 Routing Error: Patched server.py to handle non-API requests by serving the static entry point, fixing the "Not Found" error when refreshing deep links.

⚠️ Known Limitations
Ephemeral Storage (Railway): [RESOLVED in v2.1]

While the pipeline now writes to the correct folder, the filesystem on Railway is ephemeral. If the deployment restarts, the fires.json file may revert to the build-time version until "Update Incidents" is triggered manually via the dashboard. (v2.1 resolves this by persisting fires.json to a Railway Volume at DATA_DIR and refreshing on a schedule.)

Slope Approximation:

Terrain slope is currently calculated using a simplified sampling method (N/S/E/W elevation points) rather than high-precision Digital Elevation Models (DEM). This provides a relative estimation rather than survey-grade accuracy.

External API Dependency:

The system relies on fogos.pt and open-meteo.com. Rate limits or downtime on these third-party services may temporarily affect the ability to refresh incident data.

Model Scope:

The currently deployed model (lite) is optimized for performance over deep complexity. It does not currently factor in live vegetation indices (NDVI) from satellite imagery.

Authorized by: Firehawk Development Team Status: Stable / Production Ready
