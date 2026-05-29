#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data/raw data/processed models
touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep

echo "Setup complete. Secrets are read from environment variables or .env and are never printed."
