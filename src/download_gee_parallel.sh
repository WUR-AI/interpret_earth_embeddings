#!/usr/bin/env bash

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
    python download_gee_data.py --start $START --stop $END &
done

wait
echo "All workers finished."
