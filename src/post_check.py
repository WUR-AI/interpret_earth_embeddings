import os
import glob
import re

import numpy as np
import rasterio

def sanity_chek_rasters(path: str, threshold=0.05, nan_val=0.0) -> str:
    """Return True if raster seem logical (not filled with 0s)"""
    with rasterio.open(path) as src:
        data = src.read()
        size = data.shape[0] * data.shape[1] * data.shape[2]

        if data.min() == nan_val and data.max() == nan_val:
            return 'completely_empty'
        if np.sum(data == nan_val) / size > threshold:
            return f'partially_empty: {np.sum(data == nan_val) / size}'
        return ''


def main(root_dir):
    pattern = os.path.join(root_dir, "*.tif")
    tif_files = sorted(glob.glob(pattern))

    if not tif_files:
        print(f"No .tif files found in {root_dir}")
        # return

    print(f"Found {len(tif_files)} .tif files in {root_dir}\n")

    empty_files = []
    partial_files = {}

    for path in tif_files:
        try:
            flag = sanity_chek_rasters(path, threshold=0.02)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        if flag == 'completely_empty':
            empty_files.append(path)
            print(f"[EMPTY]     {os.path.basename(path)}")
        elif 'partially_empty' in flag:
            partial_files[path] = flag.split(':')[-1].strip()
            print(f"[PARTIAL]    {os.path.basename(path)}", partial_files[path])

    print("\nSummary:")
    print(f"  Empty     : {len(empty_files)}")
    print(f"  Partial     : {len(partial_files)}")

    if empty_files:
        with open(f'{root_dir}/empty_tessera.txt', 'a') as f:
            f.write('\n'.join(empty_files))

    if partial_files:
        with open(f'{root_dir}/partial_tessera.txt', 'a') as f:
            for k, v  in partial_files.items():
                f.write(f'{k} {v}\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory path.")
    args = parser.parse_args()
    main(args['root_dir'])
