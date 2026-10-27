#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "      BikeSense AI: End-to-End MLOps Demonstration         "
echo "=========================================================="

# 1. Environment check
echo -e "\n[Step 1] Ensuring Python virtual environment is active..."
source .venv/bin/activate || source .venv/Scripts/activate

# 2. Train initial models on Dataset v1
echo -e "\n[Step 2] Training models (Baseline, Ridge, LightGBM) on Dataset v1..."
python -m src.models.train --version v1

# 3. Register best candidate as Champion
echo -e "\n[Step 3] Registering best LightGBM model as initial @champion..."
python -m src.registry.register --as-champion

# 4. Verify 100% Lineage
echo -e "\n[Step 4] Running automated lineage audit across MLflow Registry..."
python -m scripts.registry_audit

# 5. Evaluate Champion on Replay Window
echo -e "\n[Step 5] Evaluating @champion on simulated production replay window..."
python -m src.models.evaluate --version v1 --alias champion

# 6. Simulate Next Window Retraining on v2
echo -e "\n[Step 6] Simulating incoming data drift: retraining candidate on Dataset v2..."
python -m src.models.train --version v2

# 7. Register new candidate as Challenger
echo -e "\n[Step 7] Registering new model as @challenger..."
python -m src.registry.register

# 8. Run Promotion Gate with +3% MAE improvement requirement
echo -e "\n[Step 8] Running promotion gate (checking +3% margin & shortfall safeguard)..."
python -m src.registry.promote --version v1 --auto-approve

# 9. Demonstrate One-Step Alias Rollback (<5 min SLA)
echo -e "\n[Step 9] Demonstrating zero-downtime alias rollback..."
python -m src.registry.rollback

echo -e "\n=========================================================="
echo "          Demonstration Completed Successfully!           "
echo "=========================================================="
