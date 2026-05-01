#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu
#SBATCH --gpus=0
#SBATCH --job-name=neureo
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=logs/neureo_%j.out
#SBATCH --error=logs/neureo_%j.err

# Activate environment
source ../aether/.venv/bin/activate

# Runs
#srun python src/train.py experiment=alignment trainer=$TRAINER_PROFILE logger=$LOGGER
srun python -u src/get_dim_tessera.py 