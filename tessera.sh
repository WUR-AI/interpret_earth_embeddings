#!/usr/bin/env bash

# Activate environment
source ../aether/.venv/bin/activate

N=20000        # max total points
P=10           # number of processes
CHUNK=$((N / P))

for ((i=0; i<P; i++)); do
    START=$((i * CHUNK))
    END=$(( (i+1) * CHUNK ))

    # last chunk takes the remainder
    if [ $i -eq $((P-1)) ]; then
        END=$N
    fi

    echo "Launching $START -> $END"
    python -u download_gee_data.py \
            --start $START \
            --stop $END  \
            --root_dir /lustre/backup/SHARED/AIN/embed_interpret/ \
            --year 2024 \
            --size 128 &
done

wait
echo "All workers finished."
