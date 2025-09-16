#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- PULLING DATA FROM GCS WITH DVC ---"

# This is the magic command. DVC will read the data.dvc file,
# connect to your configured GCS remote, and download the
# data to recreate the local `data/processed` directory structure.
dvc pull data.dvc -f

echo "--- DATA PULL COMPLETE. STARTING TRAINING ---"

# Now, execute your Python training script.
# We pass all the shell script's arguments ($@) directly to the python script.
python scripts/train.py "$@"

echo "--- TRAINING SCRIPT FINISHED ---"