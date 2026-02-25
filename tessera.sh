#!/usr/bin/env bash
#SBATCH --job-name=tessera_download
#SBATCH --output=logs/tessera_%A_%a.out
#SBATCH --error=logs/tessera_%A_%a.err
#SBATCH --array=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Activate environment
source ../aether/.venv/bin/activate

N=190000             
P=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

CHUNK=$((N / P))

START=$((TASK_ID * CHUNK))
END=$(( (TASK_ID + 1) * CHUNK ))

# last task takes remainder
if [ "$TASK_ID" -eq $((P - 1)) ]; then
    END=$N
fi

python -u src/download_tessera.py \
        --start $START \
        --stop $END \
        --root_dir /lustre/backup/SHARED/AIN/embed_interpret/ \
        --cache_root /lustre/scratch/WUR/AIN/tijun001/ \
        --year 2024 \
        --size 128 
