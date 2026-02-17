import os
import argparse

import pandas as pd
import torch
from geoclip import LocationEncoder


def main(root_dir):
    # Data points
    csv_path = os.path.join(root_dir, 'data', 'dw_locations_2026-02-13-1659_year-2024_50m_spherical_100k_random_stratified.csv')
    df = pd.read_csv(csv_path)
    df.reset_index(drop=True, inplace=True)

    # Encoder
    encoder = LocationEncoder()
    encoder.eval()

    # Subset df per sampling str
    modes = {'random_sample': None, 'lc_stratified_sample': None}
    for m in modes:
        df_sub = df[df[m] == 1].reset_index(drop=True)

        # Coords for the encoder
        coords = torch.tensor(
            df_sub[["lat", "lon"]].values,
            dtype=torch.float32,
        )

        # Encode
        with torch.no_grad():
            feats = encoder(coords)

        # Save csv
        embed_df = pd.DataFrame(
            feats.detach().cpu().numpy(),
            columns=[f"emb_{i}" for i in range(feats.shape[1])]
        )
        embed_df["id"] = df_sub["id"]

        save_path = os.path.join(root_dir, 'data', 'geoclip', f'{m}_geoclip.csv')
        embed_df.to_csv(save_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run main script with configurable parameters.")

    parser.add_argument("--root_dir", type=str, required=True, help="Root directory path.")

    args = parser.parse_args()
    main(root_dir=args.root_dir)
