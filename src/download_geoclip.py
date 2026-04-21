import os
import argparse

import pandas as pd
import torch
from geoclip import LocationEncoder

DATASET_NAME = 'cropharvest'
DATA_DIR = 'data'


def main(dataset=DATASET_NAME, data_dir=DATA_DIR):
    print(f"Fetching {dataset} GeoClip point embeddings to data_dir={data_dir}")

    # Target save_dir and save_csv_name
    target_dict = {
        "biomass":  ("downstream_tasks", "biomass_cleaned_centre.csv", "biomass_geoclip_centre.csv"),
        'cropharvest': ("downstream_tasks", "cropharvest_cleaned_global_threshold-200-sample.csv","cropharvest_200_geoclip_centre.csv"),
       'global': ("geoclip_centre", "dw_locations_2026-02-13-1659_year-2024_50m_spherical_100k_random_stratified.csv", "geoclip_centre.csv")
    }

    assert dataset in target_dict.keys(), KeyError

    # CSV with input coords
    csv_path = os.path.join(data_dir, target_dict[dataset][0], target_dict[dataset][1])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.reset_index(drop=True, inplace=True)

    # Encoder
    encoder = LocationEncoder()
    encoder.eval()

    # Subset df per sampling str
    if dataset == 'global':
        modes = {'random_sample', 'lc_stratified_sample'}
    else:
        modes = {''}
    for m in modes:
        if len(m) > 0:
            df_sub = df[df[m] == 1].reset_index(drop=True)
            save_path = os.path.join(data_dir, target_dict[dataset][0], target_dict[dataset][-1][:-4] + m + '.csv')
        else:
            df_sub = df
            save_path = os.path.join(data_dir, target_dict[dataset][0], target_dict[dataset][-1])

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
        if dataset == 'biomass':
            col_name = "index"
        elif dataset == "cropharvest":
            df.reset_index(inplace=True)
            col_name = "level_0"
        else:
            col_name = "id"
        embed_df["id"] = df_sub[col_name]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        embed_df.to_csv(save_path, index=False)
        print(f"Saved {save_path}")


if __name__ == '__main__':
    os.chdir('..')
    main()
