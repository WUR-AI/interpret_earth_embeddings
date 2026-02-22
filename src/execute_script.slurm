#!/bin/bash
#SBATCH --job-name=interpret_embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --cpus-per-task=2          # CPU cores
#SBATCH --mem=16G                  # Memory
#SBATCH --time=00:10:00            # Runtime

# Load Python module (adjust version to match your env setup)
module load 2023

# Activate your virtual environment
source ~/venvs/bayesenv/bin/activate

# Run your Python script
python src/run_cluster.py