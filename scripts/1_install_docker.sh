#!/bin/bash

# Enable strict error handling for deployment safety
set -e

echo "Updating package list..."
sudo apt-get update

# Check if Docker is already installed
if command -v docker &> /dev/null
then
    echo "Docker is already installed. Checking for Docker Compose plugin..."
    
    # Check if the new docker compose command works (the missing piece we discovered)
    if docker compose version &> /dev/null
    then
        echo "Docker Compose plugin is already installed. Skipping Docker installation."
    else
        echo "Docker Compose plugin is missing. Installing it..."
        
        # Add Docker repository if not already present (needed for compose plugin)
        if [ ! -f /etc/apt/sources.list.d/docker.list ]; then
            echo "Adding Docker repository for plugin installation..."
            
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
        fi
        
        # Install the missing compose plugin and update CLI if needed
        echo "Installing Docker Compose plugin and updating CLI..."
        sudo apt-get install -y docker-compose-plugin docker-ce-cli

        echo "Verifying Docker Compose installation..."
        docker compose version

        echo "Docker Compose plugin installation completed successfully."
    fi
else
    echo "Installing Docker from scratch..."
    
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

    # Install complete Docker suite (fixed the original's incomplete installation)
    echo "Installing Docker Engine, CLI, and Compose plugin..."
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

    echo "Verifying Docker Compose installation..."
    docker compose version

    echo "Docker installation completed successfully."
fi

# Check if unzip is already installed (unchanged from original)
if command -v unzip &> /dev/null
then
    echo "Unzip is already installed. Skipping unzip installation."
else
    echo "Installing unzip..."
    sudo apt-get install -y unzip
    echo "Unzip installation completed successfully."
fi