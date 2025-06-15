#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing unzip ..."
sudo apt update && sudo apt install -y unzip

echo "Extracting images ..."
cd "$PROJECT_ROOT/raw_data"
unzip -o images.zip

echo "Installing docker plugin ..."
sudo apt-get update
sudo apt-get install docker-compose-plugin

echo "Creating enviroment ..."
cd "$PROJECT_ROOT"
python3 -m venv venv
source "$PROJECT_ROOT/venv/bin/activate"
pip install -r requirements.txt