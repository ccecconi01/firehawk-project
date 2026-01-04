from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

# Import the pipeline
import pipeline_active 

# PATH CONFIGURATION
# Try to find the 'build' or 'dist' folder from React
FRONTEND_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'firehawk-app', 'dist')
# If you use VITE, change 'build' to 'dist' in the line above!

app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app)

# 1. Route to run the pipeline
@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    try:
        print("--- FireHawk: Pipeline Triggered ---")
        pipeline_active.run_pipeline()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# 2. Route to serve React (Home Page)
@app.route('/')
def serve_react():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

# 3. Route to serve static files (CSS, JS, Images, JSON)
@app.route('/<path:path>')
def serve_static(path):
    # If the file exists in the React folder, serve it
    if os.path.exists(os.path.join(FRONTEND_FOLDER, path)):
        return send_from_directory(FRONTEND_FOLDER, path)
    # If not, serve index.html (for React Router to work)
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)