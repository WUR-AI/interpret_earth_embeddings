#!/bin/bash
#SBATCH --job-name=interpret_embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --cpus-per-task=2          # CPU cores
#SBATCH --mem=16G                  # Memory
#SBATCH --time=02:00:00            # Runtime

module load 2023

# Activate your virtual environment
source ../aether/.venv/bin/activate

# Run your Python script
python src/run_cluster.py