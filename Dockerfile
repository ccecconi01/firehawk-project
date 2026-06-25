# 1. Hybrid Image (Python + Node)
FROM nikolaik/python-nodejs:python3.10-nodejs22

# 2. Configure working folder
WORKDIR /app

# 2b. Ensure curl is available to fetch the model bundle (used in step 3b).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy the application source. Railway's build context does NOT include .git, so
#    `git lfs pull` is impossible here and the model_*.pkl files arrive as Git LFS
#    POINTERS (134-byte text files); joblib.load() on a pointer crashes with
#    "KeyError: 118" (the pointer starts with 'v' = 0x76 = 118). Replaced in 3b.
COPY . .

# 3b. Download the real model bundle from GitHub's LFS media endpoint. The repo is
#     public, so no authentication is required. This replaces `git lfs pull`.
RUN curl -fL -o AI_FirehawkLab/model_tier_pipeline.pkl \
      "https://media.githubusercontent.com/media/ccecconi01/firehawk-project/master/AI_FirehawkLab/model_tier_pipeline.pkl" \
    && echo "model_tier_pipeline.pkl size: $(stat -c%s AI_FirehawkLab/model_tier_pipeline.pkl) bytes"

# 3c. Fail fast if the download did not return the real binary (wrong path/branch,
#     repo turned private, or LFS object missing) instead of shipping a broken image.
RUN PKL=AI_FirehawkLab/model_tier_pipeline.pkl; \
    SIZE=$(stat -c%s "$PKL"); \
    if [ "$SIZE" -lt 1000000 ]; then \
      echo "ERROR: ${PKL} is only ${SIZE} bytes — model download failed."; \
      echo "Verify the repo is public and the path/branch in the media URL are correct."; \
      exit 1; \
    fi

# 4. Install Python dependencies (Pipeline + Server)
WORKDIR /app/AI_FirehawkLab
RUN pip install -r requirements.txt

# 4b. Verify the model is a real, loadable bundle (catches pointers AND lib mismatches).
RUN python -c "import joblib; b = joblib.load('model_tier_pipeline.pkl'); print('bundle OK:', type(b).__name__, '| version', b.get('version'), '| features', len(b['features']))"

# 5. Build the Frontend (React)
WORKDIR /app/firehawk-app
RUN npm install
# If using Vite it's 'npm run build', if Create-React-App also.

RUN npm run build

# 6. Return to Python and expose the port
WORKDIR /app/AI_FirehawkLab
EXPOSE 5001

# 7. Run the Python server (which will also serve the website)
CMD ["python", "server.py"]
