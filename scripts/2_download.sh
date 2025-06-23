#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Downloading raw data ..."

#Download Header
COOKIE="csrftoken=PgtE5ZWyVyULTYn7K54IhQaQ73tcCXqT; sessionid=qm863u47vqs8pqor4oariah186ovgpu0"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"

mkdir -p "$PROJECT_ROOT/raw_data"
cd "$PROJECT_ROOT/raw_data"
wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/x-train" -O x_train.csv
wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/y-train" -O y_train.csv
wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/x-test" -O x_test.csv
wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/supplementary-files" -O images.zip


echo "Extracting images ..."
cd "$PROJECT_ROOT/raw_data"
unzip -o images.zip

echo "Deleting zip file ..."
rm images.zip