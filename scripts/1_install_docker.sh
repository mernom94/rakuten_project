#!/bin/bash

# Enable strict error handling for deployment safety
set -e

echo "Updating package list..."
sudo apt-get update

# Check if Docker is already installed
if command -v docker &> /dev/null
then
    echo "Docker is already installed. Skipping Docker installation."
else
    echo "Installing prerequisites..."
    sudo apt-get install -y \
      ca-certificates \
      curl \
      gnupg \
      lsb-release

    echo "Adding Docker's official GPG key..."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      sudo tee /etc/apt/trusted.gpg.d/docker.asc > /dev/null

    echo "Setting up Docker repository..."
    echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/docker.asc] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    echo "Updating package list again..."
    sudo apt-get update

    echo "Installing Docker Compose plugin..."
    sudo apt-get install -y docker-compose-plugin

    echo "Verifying Docker Compose installation..."
    docker compose version

    echo "Docker Compose installation completed successfully."
fi

# Check if unzip is already installed
if command -v unzip &> /dev/null
then
    echo "Unzip is already installed. Skipping unzip installation."
else
    echo "Installing unzip..."
    sudo apt-get install -y unzip
    echo "Unzip installation completed successfully."
fi