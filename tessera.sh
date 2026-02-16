#!/usr/bin/env bash
#SBATCH --job-name=gee_download
#SBATCH --output=logs/gee_%j.out
#SBATCH --error=logs/gee_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=compute

# Activate environment
source ../aether/.venv/bin/activate

N=20000
P=$SLURM_CPUS_PER_TASK
CHUNK=$((N / P))

echo "Using $P processes"

for ((i=0; i<P; i++)); do
    START=$((i * CHUNK))
    END=$(( (i+1) * CHUNK ))

    if [ $i -eq $((P-1)) ]; then
        END=$N
    fi

    echo "Launching $START -> $END"

    srun --exclusive -N1 -n1 \
        python -u download_gee_data.py \
            --start $START \
            --stop $END \
            --root_dir /lustre/backup/SHARED/AIN/embed_interpret/ \
            --year 2024 \
            --size 128 &
done

wait
echo "All workers finished."