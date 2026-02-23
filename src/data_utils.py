import os, sys, json 
import numpy as np 
import pandas as pd 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgb
import rasterio
import xarray as xr
import rioxarray as rxr
import datetime
from tqdm import tqdm, trange
# from skimage import exposure
import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))

from constants import DW_CLASSES

def load_tiff(tiff_file_path, datatype='np', verbose=0):
    '''Load tiff file as np or da'''
    with rasterio.open(tiff_file_path) as f:
        if verbose > 0:
            print(f.profile)
        if datatype == 'np':  # handle different file types 
            im = f.read()
            assert type(im) == np.ndarray
        elif datatype == 'da':
            im = rxr.open_rasterio(f)
            assert type(im) == xr.DataArray
        else:
            assert False, 'datatype should be np or da'
    return im

def create_cmap_dynamic_world(colorblind_friendly=True):
    if colorblind_friendly:
        dict_classes = {
            'water': '#332288',
            'trees': '#117733',
            'grass': '#44aa99',
            'flooded_vegetation': '#882255',
            'crops': '#cc6677',
            'shrub_and_scrub': '#999933',
            'built': '#aa4499',
            'bare': '#ddcc77',
            'snow_and_ice': '#88ccee'
        }

        #CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499'
    else:
        dict_classes = {
            'water': '#419bdf',
            'trees': '#397d49',
            'grass': '#88b053',
            'flooded_vegetation': '#7a87c6',
            'crops': '#e49635',
            'shrub_and_scrub': '#dfc35a',
            'built': '#c4281b',
            'bare': '#a59b8f',
            'snow_and_ice': '#b39fe1'
        }
    return dict_classes

def white_to_color_cmap(color, name="white_to_color"):
    return LinearSegmentedColormap.from_list(
        name,
        [(1, 1, 1), to_rgb(color)],
        N=256
    )

def create_mpl_cmap_dynamic_world():
    dict_classes = create_cmap_dynamic_world()
    cmap_per_class = {}
    for i, (cls, color) in enumerate(dict_classes.items()):
        cmap_per_class[cls] = white_to_color_cmap(color, name=f'{cls}_cmap')  # white to class color
    cmap_all = [dict_classes[dw] for dw in DW_CLASSES]
    cmap_all = ListedColormap(cmap_all)
    
    return {'individual': cmap_per_class, 'all': cmap_all}

def get_hyp_names(include_dsm=True):
    color_dict_dw = create_cmap_dynamic_world()
    hyp_names = list(color_dict_dw.keys())
    if include_dsm:
        hyp_names.append('dsm')
    return hyp_names

def create_timestamp(include_seconds=False):
    dt = datetime.datetime.now()
    timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
    if include_seconds:
        timestamp += ':' + str(dt.second).zfill(2)
    return timestamp


def get_images_from_name(path_folder=path_dict['data_folder'], name='sample-0'):
    assert os.path.exists(path_folder), path_folder
    contents = [f for f in os.listdir(path_folder) if name == f.split('_')[0]]
    assert len(contents) > 0, f'No files found starting with {name}_ in {path_folder}'
    assert len(contents) <= 5, f'More than 5 files found in {path_folder}, please specify name more precisely: {contents}'

    file_sent, file_alpha, file_dynamic, file_worldclimbio, file_dsm = None, None, None, None, None
    for file in contents:
        if file.split('_')[1].startswith('sent2'):
            file_sent = file
        elif file.split('_')[1].startswith('alphaearth'):
            file_alpha = file
        elif file.split('_')[1].startswith('dynamicworld'):
            file_dynamic = file
        elif file.split('_')[1].startswith('worldclimbio'):
            file_worldclimbio = file
        elif file.split('_')[1].startswith('dsm'):
            file_dsm = file

    if file_sent is None and file_alpha is None and file_dynamic is None and file_worldclimbio is None and file_dsm is None:
        raise ValueError(f'No recognised files found in {path_folder} with name {name}')
    return (file_sent, file_alpha, file_dynamic, file_worldclimbio, file_dsm)

def load_all_modalities_from_name(path_folder=path_dict['data_folder'], name='sample-0', verbose=0):
    ## Check if file exists, otherwise return None:
    tmp_files = [f for f in os.listdir(path_folder) if name == f.split('_')[0]]
    if len(tmp_files) == 0:
        if verbose:
            print(f'No files found in {path_folder} with name {name}')
        return None, None, None, None, None
    
    (file_sent, file_alpha, file_dynamic, file_worldclimbio, file_dsm) = get_images_from_name(path_folder=path_folder, name=name)
    path_sent = os.path.join(path_folder, file_sent) if file_sent is not None else None
    path_alpha = os.path.join(path_folder, file_alpha) if file_alpha is not None else None
    path_dynamic = os.path.join(path_folder, file_dynamic) if file_dynamic is not None else None
    path_worldclimbio = os.path.join(path_folder, file_worldclimbio) if file_worldclimbio is not None else None
    path_dsm = os.path.join(path_folder, file_dsm) if file_dsm is not None else None

    im_loaded_alpha, im_loaded_s2, im_loaded_dynamic, im_loaded_worldclimbio, im_loaded_dsm = None, None, None, None, None

    if path_sent is not None:
        im_loaded_s2 = load_tiff(path_sent, datatype='da')
        if verbose:
            print('Sentinel-2:', im_loaded_s2.shape, type(im_loaded_s2))
    else:
        if verbose:
            print('No sentinel-2 image found')

    if path_alpha is not None:
        im_loaded_alpha = load_tiff(path_alpha, datatype='da')
        ## vertical flip:
        im_loaded_alpha = im_loaded_alpha[:, ::-1, :]
        if verbose:
            print('AlphaEarth:', im_loaded_alpha.shape, type(im_loaded_alpha))
    else:
        if verbose:
            print('No alphaearth image found')

    if path_dynamic is not None:
        im_loaded_dynamic = load_tiff(path_dynamic, datatype='da')
        if verbose:
            print('Dynamic World:', im_loaded_dynamic.shape, type(im_loaded_dynamic))
    else:
        if verbose:
            print('No dynamic world image found')
    if path_worldclimbio is not None:
        with open(path_worldclimbio, 'r') as f:
            im_loaded_worldclimbio = json.load(f)
        if verbose:
            print('WorldClimBio:', type(im_loaded_worldclimbio), im_loaded_worldclimbio.keys())
    else:
        if verbose:
            print('No worldclimbio data found')

    if path_dsm is not None:
        im_loaded_dsm = load_tiff(path_dsm, datatype='da')
        if verbose:
            print('DSM:', im_loaded_dsm.shape, type(im_loaded_dsm))
    else:
        if verbose:
            print('No DSM image found')

    return im_loaded_s2, im_loaded_alpha, im_loaded_dynamic, im_loaded_worldclimbio, im_loaded_dsm

def load_all_data(path_folder='/Users/tplas/data/2025-10 neureo/pecl-100-subsample-30km_v2', 
                  prefix_name='pecl176-', rotate_90deg=False, zscore_features=False, 
                  zscore_hypotheses=False, equalize_sentinel=False, nonnan_only=False,
                  complete_only=False, n_max_patches=None, nancheck=True):
    assert os.path.exists(path_folder), path_folder
    
    n_patches = len(os.listdir(path_folder))  ## overestimate, doesnt matter.
    hypotheses = []
    features = []
    sentinel = []
    
    for p in trange(n_patches):
        (data_sent, data_alpha, data_dyn, data_worldclim, data_dsm) = load_all_modalities_from_name(name=f'{prefix_name}{p}', 
                                                                                path_folder=path_folder, verbose=0)

        if data_alpha is None:
            continue
        if complete_only:
            if data_sent is None or data_dyn is None or data_dsm is None or data_alpha is None:
                continue
        if nonnan_only:
            if data_sent is None or data_dyn is None or data_dsm is None or data_alpha is None:
                continue
            if np.sum(np.isnan(data_alpha.data)) > 0 or np.sum(np.isinf(data_alpha.data)) > 0:
                continue
            if np.sum(np.isnan(data_dyn.data)) > 0:
                continue
            if np.sum(np.isnan(data_dsm.data)) > 0:
                continue
            if np.sum(np.isnan(data_sent.data)) > 0:
                continue
        # Land coverage and DSM serve as hypotheses
        assert len(data_dyn.data.shape) == 3 and len(data_dsm.data.shape) == 3 and data_dyn.data.shape[1:] == data_dsm.data.shape[1:]
        hyp_tmp = np.concatenate([data_dyn.data, data_dsm.data], axis=0)
        f_dat = data_alpha.data
        f_dat[~np.isfinite(f_dat)] = np.nan        
        if rotate_90deg:
            hyp_tmp = np.rot90(hyp_tmp, k=1, axes=(1, 2))  # rotate counter-clockwise 90 degrees
            f_dat = np.rot90(f_dat, k=1, axes=(1, 2))  # rotate counter-clockwise 90 degrees
        hypotheses.append(hyp_tmp)
        features.append(f_dat)    
        sentinel.append(data_sent.data)

        if n_max_patches is not None and len(features) >= n_max_patches:
            break
    
    if len(features) == 0:
        print('ERROR: No patches were loaded, please check the path and prefix_name.')
        return None, None, None, None

    if nancheck:
        assert np.sum([np.sum(np.isnan(f)) for f in features]) == 0, "NaNs found in features, please check the data."
        assert np.sum([np.sum(np.isnan(h)) for h in hypotheses]) == 0, "NaNs found in hypotheses, please check the data."
        assert np.sum([np.sum(np.isinf(f)) for f in features]) == 0, "Infs found in features, please check the data."
        assert np.sum([np.sum(np.isinf(h)) for h in hypotheses]) == 0, "Infs found in hypotheses, please check the data."
        print(f'Loaded {len(features)} patches from {path_folder}, no NaNs or Infs found.')

    if zscore_features:
        # Z-score across patches for each feature and each hypothesis
        feat_m = np.stack([np.nanmean(f) for f in np.stack(features, axis=-1)])
        feat_std = np.stack([np.nanstd(f) for f in np.stack(features, axis=-1)])
        features = [(f - feat_m[:,None,None])/feat_std[:,None,None] for f in features]
    if zscore_hypotheses:
        hyp_m = np.stack([np.nanmean(h) for h in np.stack(hypotheses, axis=-1)])
        hyp_std = np.stack([np.nanstd(h) for h in np.stack(hypotheses, axis=-1)])
        hypotheses = [(h - hyp_m[:,None,None])/hyp_std[:,None,None] for h in hypotheses]
    if equalize_sentinel:
        ## equalize histogram across patches for better visualization
        sentinel_eq = np.stack(sentinel, -1)
        # sentinel_eq.shape
        sentinel_eq = sentinel_eq[:3, ...]
        # sentinel_eq = np.swapaxes(np.swapaxes(sentinel_eq, 1, 3), 1, 2)
        sentinel_eq = sentinel_eq.reshape([3, sentinel_eq.shape[1], -1])
        sentinel_eq = np.clip(sentinel_eq, 0, 2000) / 2000
        # sentinel_eq = exposure.equalize_hist(sentinel_eq)
        sentinel_eq = sentinel_eq.reshape([3, sentinel_eq.shape[1], sentinel_eq.shape[1], -1])
        sentinel_eq = [sentinel_eq[:, :, :, i] for i in range(sentinel_eq.shape[-1])]
    else:
        sentinel_eq = None

    return sentinel, sentinel_eq, features, hypotheses

def get_modality_folders(parent_folder):
    '''Finds all recognised modality folders and load the points csv if it exists.'''
    assert os.path.exists(parent_folder), parent_folder
    possible_modalities = ['sentinel2', 'alphaearth', 'dynamicworld', 'worldclimbio', 'dsm', 
                           'tessera', 'tessera_2024', 'geoclip', 'satclip']
    contents = {}
    df_points = None
    for f in os.listdir(parent_folder):
        if f in possible_modalities:
            if f == 'tessera_2024' and 'tessera' not in contents:
                name = 'tessera'
            else: 
                name = f
            contents[name] = os.path.join(parent_folder, f)
        elif f == '.DS_Store':
            continue
        elif f.startswith('dw_locations_') and f.endswith('.csv'):
            if df_points is not None:
                print(f'Warning: Multiple files starting with dw_locations_ found in {parent_folder}, skipping {f}.')
                continue
            df_points = pd.read_csv(os.path.join(parent_folder, f))
        else:
            print(f'Warning: {f} in {parent_folder} is not a recognised modality folder, skipping.')

    return contents, df_points

def get_list_complete_ids(parent_folder):
    '''Finds the complete list of ids that have all modalities available, and returns the modality folders and (filtered) points dataframe if it exists.'''
    modality_folders, df_points = get_modality_folders(parent_folder)
    list_ids_per_modality = {}
    for modality, folder in modality_folders.items():
        ids = set()
        if modality in ['satclip', 'geoclip']:
            csv_files = [x for x in os.listdir(folder) if x.endswith('.csv')]
            for f in csv_files:
                tmp = pd.read_csv(os.path.join(folder, f))
                ids = ids.union(set(tmp.id.values))
        else:
            for f in os.listdir(folder):
                if f.endswith('.tif') or f.endswith('.json'):
                    id = f.split('_')[0]
                    ids.add(int(id))
        list_ids_per_modality[modality] = ids
    complete_ids = set.intersection(*list_ids_per_modality.values())
    complete_ids = np.sort(list(complete_ids))
    if df_points is not None:
        n_original = len(df_points)
        df_points = df_points[df_points.id.isin(complete_ids)]
        n_new = len(df_points)
        if n_new < n_original:  
            print(f'Warning: {n_original - n_new} rows were dropped from df_points because their id was not in the complete_ids set.')
        for col in df_points.columns:
            if col.endswith('_sample'):
                print(f'Sample {col} has {np.sum(df_points[col])} data points out of {len(df_points)}.')

    return complete_ids, modality_folders, df_points

def create_csv_with_points_from_patches(parent_folder, modalities=['tessera', 'alphaearth'], verbose=1):
    list_ids, modality_folders, gdf_points = get_list_complete_ids(parent_folder)
    if gdf_points is None or len(gdf_points) == 0:
        if verbose:
            print(f'No points dataframe found in {parent_folder}, creating a new one with all ids that have complete modalities: {modalities}')
        return None
    for m in modalities:
        if m not in modality_folders:
            print(f'Warning: Modality {m} not found in {parent_folder}, skipping.')
            continue
        folder = modality_folders[m]
    
        if verbose:
            print(f'Processing modality {m} in folder {folder} with {len(list_ids)} complete ids.')
        if m == 'tessera' or m == 'tessera_2024':
            patch_size = 128
            n_dim = 128
        elif m == 'alphaearth':
            patch_size = 128
            n_dim = 64
        emb_cols = [f'emb_{d}' for d in range(n_dim)]
        cols = ['id','pix_x', 'pix_y', 'random_sample', 'lc_stratified_sample'] + emb_cols
        results = {x: [] for x in cols}
        x, y = patch_size // 2, patch_size // 2
        for f in tqdm(os.listdir(folder)):
            if f.endswith('.tif') or f.endswith('.json'):
                id = int(f.split('_')[0])
            else:
                continue 
            if id not in gdf_points.id.values:
                continue
            try:
                im = load_tiff(os.path.join(folder, f), datatype='np')
            except Exception as e:
                print(f'Error loading {f}: {e}')
                continue
            bool_random = gdf_points[gdf_points.id == id]['random_sample'].values[0]
            bool_strat = gdf_points[gdf_points.id == id]['lc_stratified_sample'].values[0]
            if im.shape[0] != n_dim or im.shape[1] != patch_size or im.shape[2] != patch_size:
                print(f'Warning: {f} has shape {im.shape}, expected ({n_dim}, {patch_size}, {patch_size}), skipping.')
                continue
            val = im[:, y, x]
            results['id'].append(id)
            results['pix_x'].append(x)
            results['pix_y'].append(y)
            results['random_sample'].append(bool_random)
            results['lc_stratified_sample'].append(bool_strat)
            for d in range(n_dim):
                results[f'emb_{d}'].append(val[d])

        df_result = pd.DataFrame(results)
        for sample in ['random_sample', 'lc_stratified_sample']:
            df_tmp = df_result[df_result[sample] == True]
            df_tmp = df_tmp[emb_cols + ['id']]
            df_tmp.reset_index(drop=True, inplace=True)
            save_folder = os.path.join(parent_folder, f'{m}_centre')
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f'{sample}_{m}_centre.csv')
            if os.path.exists(save_path):
                print(f'Warning: {save_path} already exists, skipping saving for {m} {sample}.')
                continue
            df_tmp.to_csv(save_path, index=False)

def load_csv_with_points(parent_folder, modality='alphaearth', sample_type='random_sample'):
    if modality == 'tessera_2024' or modality == 'tessera':
        modality = 'tessera_centre'
    elif modality == 'alphaearth':
        modality = 'alphaearth_centre'
    
    assert os.path.exists(parent_folder), f'Parent folder {parent_folder} does not exist.'
    assert modality in os.listdir(parent_folder), f'Modality {modality} not found in {parent_folder}.'
    assert sample_type in ['random_sample', 'lc_stratified_sample'], f'sample_type should be random_sample or lc_stratified_sample, got {sample_type}.'

    folder = os.path.join(parent_folder, modality)
    assert os.path.exists(folder), f'Modality folder {folder} does not exist.'
    file_name = f'{sample_type}_{modality}.csv'
    file_path = os.path.join(folder, file_name)
    assert os.path.exists(file_path), f'File {file_name} not found in {folder}.'
    df = pd.read_csv(file_path)
    return df
