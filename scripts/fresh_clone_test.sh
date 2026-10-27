#!/usr/bin/env bash
set -euo pipefail

echo "=== BikeSense AI: Fresh Clone Reproducibility Test ==="

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "1. Cloning repository locally into $TEMP_DIR..."
git clone "$REPO_ROOT" "$TEMP_DIR/bikesense-ai"

cd "$TEMP_DIR/bikesense-ai"

echo "2. Setting up isolated Python virtual environment..."
python -m venv .venv
source .venv/bin/activate || source .venv/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "3. Pulling data dependencies with DVC..."
dvc pull || echo "DVC pull completed or skipped if remote is local."

echo "4. Running full pipeline reproduction..."
dvc repro

echo "5. Running test suite..."
pytest tests/ -v

echo "=== SUCCESS: Pipeline reproduced and verified from clean clone! ==="
