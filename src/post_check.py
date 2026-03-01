import os
import pandas as pd

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

    csv_path = os.path.join(root_dir, 'data', 'dw_locations_2026-02-13-1659_year-2024_50m_spherical_100k_random_stratified.csv')
    df = pd.read_csv(csv_path)

    # Subset for selected samples
    df = df[(df['random_sample'] == 1) | (df['lc_stratified_sample'] == 1)]
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={'id': 'row_id'}, inplace=True)

    save_dir = os.path.join(root_dir, 'data', f'tessera_2024_v1')

    empty_files = []
    partial_files = {}
    missing_files = []

    for row in df.itertuples():
        path = os.path.join(save_dir, f'{row.row_id}_tessera_y-2024.tif')
        if not os.path.exists(path):
            missing_files.append(path)
            print(f"[MISSING]     {os.path.basename(path)}")
        else:
            flag = sanity_chek_rasters(path, threshold=0.02)
            if flag == 'completely_empty':
                empty_files.append(path)
                print(f"[EMPTY]     {os.path.basename(path)}")
            elif 'partially_empty' in flag:
                partial_files[path] = flag.split(':')[-1].strip()
                print(f"[PARTIAL]    {os.path.basename(path)}", partial_files[path])

    print("\nSummary:")
    print(f"  Missing     : {len(missing_files)}")
    print(f"  Empty     : {len(empty_files)}")
    print(f"  Partial     : {len(partial_files)}")

    if len(missing_files) > 0:
        with open(os.path.join(save_dir, 'tessera_skipped.txt'), 'w') as f:
            f.write('\n'.join(missing_files))

    if len(empty_files) > 0:
        with open(f'{save_dir}/empty_tessera.txt', 'a') as f:
            f.write('\n'.join(empty_files))

    if len(partial_files) > 0:
        with open(f'{save_dir}/partial_tessera.txt', 'a') as f:
            for k, v  in partial_files.items():
                f.write(f'{k} {v}\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory path.")
    args = parser.parse_args()
    main(args['root_dir'])
