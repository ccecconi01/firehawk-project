from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

# Import the pipeline
import pipeline_active 

# PATH CONFIGURATION
# 1. os.path.dirname(__file__) -> Get where I am (AI_FirehawkLab)
# 2. '..' -> Go up to root (/app)
# 3. 'firehawk-app' -> Enter the frontend folder
# 4. 'dist' -> Enter the build folder (If Vite it's 'dist', if CRA it's 'build')
FRONTEND_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'firehawk-app', 'dist')

# Configure Flask to serve static files from this folder
app = Flask(__name__, static_folder=FRONTEND_FOLDER)
CORS(app)

# --- 1. API ROUTES (Have priority) ---
@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    try:
        print("--- 🦅 FireHawk: Pipeline Triggered ---")
        pipeline_active.run_pipeline()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# --- 2. "CATCH-ALL" ROUTE FOR REACT (The Secret) ---
# Any route that is not /api/... falls here.
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # Check if the file physically exists (e.g., logo.png, main.js, css)
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    
    # If the file doesn't exist (e.g., /dashboard, /alert/5), serve index.html
    # React Router will read the URL and show the correct page.
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Port 5001 as configured
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)