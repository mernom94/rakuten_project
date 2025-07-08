#!/bin/bash

# Enable strict error handling for deployment safety
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Checking for existing data..."

# Download header
COOKIE="csrftoken=iz1ktRciOB8KNKuKuPfZ7mMjhw02xxPC; sessionid=co8bqylktof1ys347jx8w5tbyg3safeb"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"

mkdir -p "$PROJECT_ROOT/raw_data"
cd "$PROJECT_ROOT/raw_data"


ask_user() {
    local prompt="$1"
    local choice
    
    while true; do
        read -p "$prompt (y/n): " choice
        case "${choice,,}" in  # ${choice,,} converts to lowercase
            y|yes)
                return 0  # Shell convention: 0 means success/true
                ;;
            n|no)
                return 1  # Non-zero means failure/false
                ;;
            *)
                echo "Please answer y (yes) or n (no)"
                ;;
        esac
    done
}


check_and_handle_csv_files() {
    if [[ -f "x_train.csv" || -f "y_train.csv" ]]; then
        echo "Found existing training data files:"
        [[ -f "x_train.csv" ]] && echo "  - x_train.csv"
        [[ -f "y_train.csv" ]] && echo "  - y_train.csv"
        
        if ask_user "Do you want to re-download the training data"; then
            echo "Removing existing files..."
            rm -f x_train.csv y_train.csv
            return 0  # Proceed with download
        else
            echo "Keeping existing training data files."
            return 1  # Skip download
        fi
    else
        echo "No existing training data found."
        return 0  # Proceed with download
    fi
}


check_and_handle_images() {
    local has_zip=false
    local has_extracted=false
    
    [[ -f "images.zip" ]] && has_zip=true
    # Check if images directory exists and contains files
    [[ -d "images" && -n "$(ls -A images/ 2>/dev/null)" ]] && has_extracted=true
    
    if [[ "$has_zip" == true || "$has_extracted" == true ]]; then
        echo "Found existing image data:"
        [[ "$has_zip" == true ]] && echo "  - images.zip file"
        [[ "$has_extracted" == true ]] && echo "  - extracted images directory"
        
        if ask_user "Do you want to re-download and extract the images"; then
            echo "Cleaning up existing image data..."
            rm -f images.zip
            rm -rf images/
            return 0  # Proceed with download
        else
            echo "Keeping existing image data."
            return 1  # Skip download
        fi
    else
        echo "No existing image data found."
        return 0  # Proceed with download
    fi
}


# Main execution flow
download_csv=false
download_images=false

if check_and_handle_csv_files; then
    download_csv=true
fi

if check_and_handle_images; then
    download_images=true
fi

# Perform downloads based on user choices
if [[ "$download_csv" == true ]]; then
    echo "Downloading training data..."
    wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/x-train" -O x_train.csv
    wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/y-train" -O y_train.csv
    echo "Training data download completed."
fi

if [[ "$download_images" == true ]]; then
    echo "Downloading and extracting images..."
    wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/supplementary-files" -O images.zip
    unzip -o images.zip
    rm images.zip
    echo "Image download and extraction completed."
fi

echo "Script completed successfully!"