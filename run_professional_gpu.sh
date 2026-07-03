#!/bin/bash
# Professional RL Training Pipeline for Remote GPU servers

set -e

echo "=== Starting Professional RL Pipeline ==="

# 1. Environment Abstraction via Conda
# Assumes Miniconda is installed. If not, it will fail gracefully.
if ! command -v conda &> /dev/null
then
    echo "ERROR: conda could not be found. Please install Miniconda."
    exit 1
fi

echo "Setting up Conda Environment 'orca_rl'..."
conda create -n orca_rl python=3.10 -y
# Activate conda in bash script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate orca_rl

# 2. Strict Dependency Pinning
echo "Installing dependencies securely from pinned requirements..."
pip install --upgrade pip
pip install -r requirements.txt
cd ../orca_sim && pip install -e .
cd ../fv-orca-hand-rl

# 3. Execution & Automated Alerting
echo "Authenticating WandB..."
wandb login

echo "Launching Training with aggressive 100k checkpoints..."
# Trap errors to send alert
python train.py --timesteps 5000000 --run-name phase1-5M-pro-gpu --device cuda --upload-model
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "CRITICAL FAILURE: train.py exited with code $EXIT_CODE"
    # Placeholder for webhook alert (Slack/Discord)
    # curl -X POST -H 'Content-type: application/json' --data '{"text":"CRITICAL: GPU RL Training Failed!"}' https://hooks.slack.com/services/...
    exit $EXIT_CODE
else
    echo "SUCCESS: Training completed successfully."
fi
