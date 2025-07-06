#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Downloading raw data ..."

#Download Header
COOKIE="csrftoken=iz1ktRciOB8KNKuKuPfZ7mMjhw02xxPC; sessionid=co8bqylktof1ys347jx8w5tbyg3safeb"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"

mkdir -p "$PROJECT_ROOT/raw_data"
cd "$PROJECT_ROOT/raw_data"
rm -f x_train.csv y_train.csv
wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/x-train" -O x_train.csv
wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/y-train" -O y_train.csv
# wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/x-test" -O x_test.csv
# wget --header="Cookie: $COOKIE" --header="User-Agent: $UA" "https://challengedata.ens.fr/participants/challenges/35/download/supplementary-files" -O images.zip


# echo "Extracting images ..."
# cd "$PROJECT_ROOT/raw_data"
# unzip -o images.zip

# echo "Deleting zip file ..."
# rm images.zip