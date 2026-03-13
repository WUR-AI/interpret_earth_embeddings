import os, sys 
import argparse
import gee_utils as gu
import pandas as pd 
import numpy as np

def main(start=0, stop=2000, content='gee_aux_vals'):
    assert content in ['gee_ims', 'gee_aux_vals'], f'{content} not recognised.'
    folder_save = '/Users/tplas/data/dw_sampled_locations/data_spherical/gee_aux_values'
    path_csv_locations = '/Users/tplas/data/dw_sampled_locations/dw_locations_2026-02-13-1659_year-2024_50m_spherical_100k_random_stratified.csv'
    
    assert os.path.exists(folder_save), f"Save folder does not exist: {folder_save}"
    assert os.path.exists(path_csv_locations), f"CSV file with locations does not exist: {path_csv_locations}"

    df_locations = pd.read_csv(path_csv_locations)
    df_locations = df_locations[np.logical_or(df_locations.random_sample == 1, df_locations.lc_stratified_sample == 1)]
    assert len(df_locations) > 0, "No locations to process after filtering for random_sample and lc_stratified_sample."
    assert len(df_locations) <= 20000, f"Number of locations to process ({len(df_locations)}) exceeds expected maximum of 20,000."

    assert start >= 0 and stop > start, f"Invalid start ({start}) and stop ({stop}) values. Ensure that 0 <= start < stop."
    if stop > len(df_locations):
        print(f"Warning: stop index ({stop}) exceeds number of available locations ({len(df_locations)}). Adjusting stop to {len(df_locations)}.")
        stop = len(df_locations)
    if stop <= start:
        return None
    ## coords should be (lon, lat) for gu.get_gee_image_from_point() 

    coords_list = [(row.lon, row.lat) for _, row in df_locations.iterrows()]
    name_list = df_locations.index.values

    if content == 'gee_ims':
        inds_none = gu.download_list_coord(coord_list=coords_list, name_list=name_list, path_save=folder_save, bool_buffer_in_deg=False, buffer_deg=None, buffer_m=800,
                           name_group='2026-02-13-1659', start_index=start, stop_index=stop, resize_image=True, threshold_size=128,
                           list_collections=['alphaearth', 'dynamicworld', 'dsm'], save_coords_json=False)
    elif content == 'gee_aux_vals':
        results, save_path, inds_none = gu.get_aux_data_from_coords_list(coords_list=coords_list, id_list=name_list, 
                                                              name_group='2026-02-13-1659', start_index=start, stop_index=stop,
                                                              save_folder=folder_save, save_file=True)
    return inds_none

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    print(f"Starting download of GEE data for locations from index {args.start} to {args.stop}...")
    main(start=args.start, stop=args.stop)