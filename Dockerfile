# 1. Hybrid Image (Python + Node)
FROM nikolaik/python-nodejs:python3.10-nodejs22

# 2. Configure working folder
WORKDIR /app

# 3. Copy everything inside
COPY . .

# 4. Install Python dependencies (Pipeline + Server)
WORKDIR /app/AI_FirehawkLab
RUN pip install -r requirements.txt

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