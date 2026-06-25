# 1. Hybrid Image (Python + Node)
FROM nikolaik/python-nodejs:python3.10-nodejs22

# 2. Configure working folder
WORKDIR /app

# 2b. Install git-lfs. The base image has git but NOT git-lfs. It is required to
#     materialise the model_*.pkl objects: Railway's checkout leaves Git LFS
#     POINTERS (134-byte text files) in the build context, and joblib.load() on a
#     pointer crashes with "KeyError: 118" (the pointer starts with 'v' = 0x76 = 118).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy everything inside (includes .git so LFS can resolve/fetch its objects)
COPY . .

# 3b. Replace the LFS pointers with the real binaries.
#     Requires .git in the build context (no .dockerignore, so it is present) and,
#     for a PRIVATE repo, credentials on the cloned remote. If LFS cannot fetch,
#     the size gate in 3c fails the build (the running container is kept by Railway).
RUN git config --global --add safe.directory /app \
    && git lfs install --local \
    && git lfs pull \
    && echo "== LFS files after pull ==" && git lfs ls-files

# 3c. Fail fast if the model is still a pointer (LFS did not materialise) instead of
#     shipping a broken image. See README for the runtime-download fallback.
RUN PKL=AI_FirehawkLab/model_tier_pipeline.pkl; \
    SIZE=$(stat -c%s "$PKL"); \
    echo "model_tier_pipeline.pkl size: ${SIZE} bytes"; \
    if [ "$SIZE" -lt 1000000 ]; then \
      echo "ERROR: ${PKL} is still an LFS pointer (${SIZE} bytes) — Railway did not pull LFS."; \
      echo "Ensure .git is in the build context and the LFS remote is reachable (auth for private repos),"; \
      echo "or switch to the runtime model-download fallback documented in the README."; \
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
