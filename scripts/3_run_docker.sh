#!/bin/bash

# Enable strict error handling for deployment safety
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

echo "Updating .env file..."
echo "AIRFLOW_UID=1000" > "$ENV_FILE"
echo "PROJECT_ROOT=$PROJECT_ROOT" >> "$ENV_FILE"

echo "Building Docker image for Rakuten ML..."
docker build -t rakuten-ml "$PROJECT_ROOT/containers/rakuten-ml"

cd "$PROJECT_ROOT"
echo "Starting Docker Compose services..."
docker compose up -d