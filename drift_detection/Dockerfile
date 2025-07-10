# drift-monitor/Dockerfile

FROM python:3.11-slim

# System dependencies (optional, depending on your base)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY requirements_drift.txt ./
COPY drift_monitor.py ./
COPY preprocessing.py ./

# Copy folder structure
COPY containers/ ./containers/
COPY models/ ./models/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the script on container start
CMD ["python", "drift_monitor.py"]
