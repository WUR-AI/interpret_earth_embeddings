#!/usr/bin/env bash
#SBATCH --job-name=tessera_download
#SBATCH --output=logs/tessera_%A_%a.out
#SBATCH --error=logs/tessera_%A_%a.err
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Activate environment
source ../aether/.venv/bin/activate

N=20000              # total points
P=10                 # total array jobs
TASK_ID=$SLURM_ARRAY_TASK_ID

CHUNK=$((N / P))

START=$((TASK_ID * CHUNK))
END=$(( (TASK_ID + 1) * CHUNK ))

# last task takes remainder
if [ "$TASK_ID" -eq $((P - 1)) ]; then
    END=$N
fi

echo "Task $TASK_ID processing $START -> $END"

python -u src/download_tessera.py \
        --start $START \
        --stop $END \
        --root_dir /lustre/backup/SHARED/AIN/embed_interpret/ \
        --year 2024 \
        --size 128