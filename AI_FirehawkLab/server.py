from flask import Flask, jsonify, send_from_directory, abort
from flask_cors import CORS
import atexit
import os
import shutil
import sys
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Import the pipeline
import pipeline_active

# ==========================================
# PATH / CONFIG
# ==========================================
# FRONTEND_FOLDER: built SPA (Vite 'dist'); '..' goes up to repo root.
FRONTEND_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'firehawk-app', 'dist')

# Persistent snapshot dir. On Railway a Volume is mounted here (default "/data"),
# so fires.json survives restarts/redeploys (the container disk is ephemeral).
DATA_DIR = os.environ.get('DATA_DIR', '/data')

# Cron schedule for the background refresh (UTC). Default: 06:00, 12:00, 18:00, 00:00.
REFRESH_CRON = os.environ.get('REFRESH_CRON', '0 6,12,18,0 * * *')

# Run a refresh on boot if the snapshot is missing or older than this many hours.
STARTUP_MAX_AGE_HOURS = float(os.environ.get('STARTUP_MAX_AGE_HOURS', '6'))

SNAPSHOT_NAME = 'fires.json'

# Single in-process guard shared by the scheduler AND the manual trigger so the
# pipeline never runs twice at once (a run can take ~1 minute).
_pipeline_lock = threading.Lock()
_scheduler = None

# Configure Flask to serve static files from the built SPA folder
app = Flask(__name__, static_folder=FRONTEND_FOLDER)
CORS(app)


# ==========================================
# SNAPSHOT HELPERS
# ==========================================
def snapshot_path():
    return os.path.join(DATA_DIR, SNAPSHOT_NAME)


def snapshot_age_hours():
    """Age of the persisted snapshot in hours, or None when it does not exist."""
    p = snapshot_path()
    if not os.path.exists(p):
        return None
    return (time.time() - os.path.getmtime(p)) / 3600.0


def seed_snapshot():
    """Make sure DATA_DIR exists and holds an initial fires.json.

    Seeds from the built SPA (dist/data/fires.json) so GET works immediately on a
    fresh volume, before the first pipeline run produces a live snapshot.
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except OSError as e:
        print(f"[seed] WARNING: cannot create DATA_DIR '{DATA_DIR}': {e}")
        return
    dst = snapshot_path()
    if os.path.exists(dst):
        return
    src = os.path.join(FRONTEND_FOLDER, 'data', SNAPSHOT_NAME)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"[seed] Seeded {dst} from {src}")
    else:
        print(f"[seed] No seed source at {src}; volume will fill on first refresh.")


def run_pipeline_guarded(source):
    """Run the pipeline under the shared lock. Returns 'ok' or 'busy'.

    Exceptions from the pipeline propagate to the caller (the manual endpoint maps
    them to HTTP 500; the scheduler wrapper logs them).
    """
    if not _pipeline_lock.acquire(blocking=False):
        print(f"[{source}] Refresh already in progress — skipping this trigger.")
        return 'busy'
    try:
        print(f"[{source}] --- 🦅 FireHawk: running pipeline ---")
        t0 = time.time()
        pipeline_active.run_pipeline()
        print(f"[{source}] Pipeline finished in {time.time() - t0:.1f}s.")
        return 'ok'
    finally:
        _pipeline_lock.release()


def scheduled_refresh():
    """Scheduler entry point: same guard as the manual trigger, errors logged."""
    try:
        run_pipeline_guarded('scheduler')
    except Exception as e:  # never let a bad run kill the scheduler thread
        print(f"[scheduler] Pipeline error: {type(e).__name__}: {e}")


def start_background():
    """Seed the volume, start the cron scheduler, and refresh once if stale."""
    global _scheduler
    # Decide staleness from the LIVE snapshot before seeding, since seeding writes a
    # copy with a fresh mtime that would otherwise look up-to-date.
    age = snapshot_age_hours()
    seed_snapshot()

    _scheduler = BackgroundScheduler(timezone='UTC')
    _scheduler.add_job(
        scheduled_refresh,
        trigger=CronTrigger.from_crontab(REFRESH_CRON, timezone='UTC'),
        id='refresh',
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False))
    print(f"[scheduler] Started. REFRESH_CRON='{REFRESH_CRON}' (UTC), DATA_DIR='{DATA_DIR}'.")

    if age is None or age > STARTUP_MAX_AGE_HOURS:
        reason = 'missing' if age is None else f'{age:.1f}h old (> {STARTUP_MAX_AGE_HOURS}h)'
        print(f"[scheduler] Snapshot {reason} — running an initial refresh in the background.")
        # Background thread so app startup / port binding is not blocked (~1 min run).
        threading.Thread(
            target=lambda: scheduled_refresh(), name='startup-refresh', daemon=True
        ).start()
    else:
        print(f"[scheduler] Snapshot {age:.1f}h old — no startup refresh needed.")


# ==========================================
# ROUTES
# ==========================================
# --- 1. API ROUTES (priority) ---
@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    """Manual trigger. Shares the anti-overlap guard with the scheduler."""
    try:
        status = run_pipeline_guarded('manual')
        if status == 'busy':
            return jsonify({"success": False, "error": "A refresh is already running."}), 409
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --- 2. SNAPSHOT DATA (served from the persistent volume) ---
@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve /data/* from DATA_DIR (the volume), falling back to the built seed."""
    for base in (DATA_DIR, os.path.join(FRONTEND_FOLDER, 'data')):
        if os.path.isfile(os.path.join(base, filename)):
            resp = send_from_directory(base, filename)
            resp.headers['Cache-Control'] = 'no-cache'
            return resp
    abort(404)


# --- 3. "CATCH-ALL" ROUTE FOR REACT ---
# Any route that is not /api/... or /data/... falls here.
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # Serve a real static file if it exists (logo.png, main.js, css, ...)
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # Otherwise hand off to the SPA so React Router can resolve the URL.
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    # Start the scheduler only in the main process. With the Werkzeug reloader the
    # parent process has WERKZEUG_RUN_MAIN unset; use_reloader=False below means we
    # never fork, so this guard simply starts it once.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "false":
        start_background()
    app.run(host='0.0.0.0', port=port, use_reloader=False)
