# ML Container for Rakuten Product Classification
# Contains all dependencies needed for preprocessing and training scripts

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements_ml.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_ml.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copy ML scripts
COPY training.py ./scripts/training.py
COPY preprocessing.py ./scripts/preprocessing.py
COPY preprocessing_test_data.py ./scripts/preprocessing_test_data.py
COPY drift_detection.py ./scripts/drift_detection.py

# Create directories for data and models
RUN mkdir -p processed_data models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "--version"]