#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Boot up dockers ..."
cd "$PROJECT_ROOT"
docker compose up -d

echo "Waiting for PostgreSQL on port 5432 ..."
until nc -z localhost 5432; do
  sleep 1
done
echo "PostgreSQL is ready."

echo "Waiting for MinIO on port 9000 ..."
until nc -z localhost 9000; do
  sleep 1
done
echo "MinIO is ready."

echo "Loading data ..."
source "$PROJECT_ROOT/venv/bin/activate"
python "$PROJECT_ROOT/src/load_postgres.py"
python "$PROJECT_ROOT/src/load_minio.py"