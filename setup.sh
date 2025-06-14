#!/bin/bash
set -e

echo "Installing unzip ..."
sudo apt update && sudo apt install -y unzip

echo "Extracting images ..."
cd raw_data
unzip -o images.zip

echo "Setting up PostgreSQL & MinIO ..."
docker compose up -d