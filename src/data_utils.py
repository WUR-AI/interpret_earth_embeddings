import os, sys, json 
import numpy as np 
import pandas as pd 
# from dwca.read import DwCAReader
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgb
# import seaborn as sns 
import shapely 
import rasterio, rasterio.plot
import xarray as xr
import rioxarray as rxr
# from shapely.geometry import Point, Polygon
import datetime
import utm 

# import warnings
# from shapely.errors import ShapelyDeprecationWarning
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
# import geopandas as gpd
from tqdm import tqdm, trange
from skimage import exposure
# import scipy 
# import scipy.spatial, scipy.cluster
import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))

from sample_locations import DW_CLASSES

ONLINE_ACCESS_TO_GEE = True 
if ONLINE_ACCESS_TO_GEE:
    import api_keys
    import ee, geemap 
    ee.Authenticate()
    ee.Initialize(project=api_keys.GEE_API)
    geemap.ee_initialize()
else:
    print('WARNING: ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE')

# def map_str_tuple_to_coords(tup):
#     """Convert a string tuple to a tuple of floats."""
#     return tuple(float(x) for x in ast.literal_eval(tup))

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

def create_cmap_dynamic_world():
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

def get_epsg_from_latlon(lat, lon):
    """Get the UTM EPSG code from latitude and longitude.
    https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
    """
    utm_result = utm.from_latlon(lat, lon)
    zone_number = utm_result[2]
    hemisphere = '326' if lat >= 0 else '327'
    epsg_code = int(hemisphere + str(zone_number).zfill(2))
    return epsg_code

def get_gee_image_from_point(coords, bool_buffer_in_deg=False, buffer_deg=0.01, buffer_m=800,
                             verbose=0, year=None, threshold_size=128,
                             month_start_str='06', month_end_str='09',
                             image_collection='sentinel2'):
    '''Coords: (lon, lat)'''
    assert ONLINE_ACCESS_TO_GEE, 'Need to set ONLINE_ACCESS_TO_GEE to True to use this function'
    assert image_collection in ['sentinel2', 'alphaearth', 'worldclimbio', 'dynamicworld', 'dsm'], f'image_collection {image_collection} not recognised.'
    if year is None:
        year = 2024
    lon, lat = coords
    epsg_code = get_epsg_from_latlon(lat=lat, lon=lon)

    if bool_buffer_in_deg:  # not ideal https://gis.stackexchange.com/questions/304914/python-shapely-intersection-with-buffer-in-meter
        print('WARNING: using buffer in degrees, which is not ideal for large latitudes.')
        point = shapely.geometry.Point(coords)
        polygon = point.buffer(buffer_deg, cap_style=3)  ## buffer in degrees
        xy_coords = np.array(polygon.exterior.coords.xy).T 
        aoi = ee.Geometry.Polygon(xy_coords.tolist())
    else:
        point = ee.Geometry.Point(coords)
        aoi = point.buffer(buffer_m).bounds()
    
    if image_collection == 'sentinel2':
        ex_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        if ex_collection is None:
            print(f'ERROR: could not load sentinel-2 collection from {coords}')
            return None
        ## also consider creating a mosaic instead: https://gis.stackexchange.com/questions/363163/filter-out-the-least-cloudy-images-in-sentinel-google-earth-engine
        ex_im_gee = ee.Image(ex_collection 
                            .filterBounds(aoi) 
                            .filterDate(ee.Date(f'{year}-{month_start_str}-01'), ee.Date(f'{year}-{month_end_str}-01')) 
                            .select(['B4', 'B3', 'B2', 'B8'])  # 10m bands, RGB and NIR
                            .sort('CLOUDY_PIXEL_PERCENTAGE')
                            .first()  # get the least cloudy image
                            .reproject(f'EPSG:{epsg_code}', scale=10)
                            .clip(aoi))
    elif image_collection == 'alphaearth':
        ex_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        if ex_collection is None:
            print(f'ERROR: could not load alphaearth collection from {coords}')
            return None
        ex_im_gee = ee.Image(ex_collection 
                            .filterBounds(aoi) 
                            .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31')) 
                            .first() 
                            .reproject(f'EPSG:{epsg_code}')  
                            .clip(aoi))

    elif image_collection == 'dynamicworld':
        prob_bands = [
            "water", "trees", "grass", "flooded_vegetation",
            "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"
        ]
        ex_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        if ex_collection is None:
            print(f'ERROR: could not load dynamicworld collection from {coords}')
            return None
        ex_im_gee = ee.Image(ex_collection 
                            #   .project(crs='EPSG:27700', scale=1)
                            .filterBounds(aoi) 
                            .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31'))
                            .select(prob_bands)  # get all probability bands
                            .mean()  # mean over the year
                            .reproject(f'EPSG:{epsg_code}', scale=10)  # reproject to 10m
                            .clip(aoi)
                            )  # mean over the year
    elif image_collection == 'worldclimbio':
        ex_im_gee = ee.Image("WORLDCLIM/V1/BIO").clip(aoi) 
        point = ee.Geometry.Point(coords)  # redefine point for sampling
        values = ex_im_gee.sample(region=point.buffer(1000), scale=1000).first()
        if values is None:
            values = ex_im_gee.sample(region=point.buffer(10000), scale=1000).first()
        if values is None:
            print(f'ERROR: could not sample worldclimbio collection from {coords}')
            return None
        values = values.toDictionary().getInfo()
        return values
    elif image_collection == 'dsm':
        ex_collection = ee.ImageCollection("COPERNICUS/DEM/GLO30")
        if ex_collection is None:
            print(f'ERROR: could not load dsm collection from {coords}')
            return None
        ex_im_gee = ee.Image(ex_collection
                            .filterBounds(aoi)
                            .select(['DEM'])  # select the DEM band
                            # .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31'))
                            .first()
                            .reproject(f'EPSG:{epsg_code}', scale=10)
                            .clip(aoi))
        threshold_size = max(32, threshold_size // 4)  # DSM is 30m resolution, so allow smaller images
    else:
        raise NotImplementedError(image_collection)

    im_dims = ex_im_gee.getInfo()["bands"][0]["dimensions"]
    
    if im_dims[0] < threshold_size or im_dims[1] < threshold_size:
        print('WARNING: image too small, returning None')
        return None
    
    if verbose:
        print(ex_im_gee.projection().getInfo())
        print(f'Area AOI in km2: {aoi.area().getInfo() / 1e6}')
        print(f'Pixel dimensions: {im_dims}')
        print(ex_im_gee.getInfo()['bands'][3])
    
    return ex_im_gee

def create_filename(base_name, image_collection='sentinel2', year=2024,
                    month_start_str='06', month_end_str='09'):
    if image_collection == 'sentinel2':
        filename = f'{base_name}_sent2-4band_y-{year}_m-{month_start_str}-{month_end_str}.tif'
    elif image_collection == 'alphaearth':
        filename = f'{base_name}_alphaearth_y-{year}.tif'
    elif image_collection == 'worldclimbio':
        filename = f'{base_name}_worldclimbio_v1.json'
    elif image_collection == 'dynamicworld':
        filename = f'{base_name}_dynamicworld_y-{year}.tif'
    elif image_collection == 'dsm':
        filename = f'{base_name}_dsm_y-{year}.tif'
    return filename

def download_gee_image(coords, name: str, bool_buffer_in_deg=False, buffer_deg=0.01, buffer_m=800, 
                    verbose=0, year=None, threshold_size=128,
                    month_start_str='06', month_end_str='09',
                    image_collection='sentinel2',
                    path_save=None, resize_image=True):
    assert image_collection in ['sentinel2', 'alphaearth', 'worldclimbio', 'dynamicworld' ,'dsm'], f'image collection {image_collection} not recognised.'
    if year is None:
        year = 2024

    im_gee = get_gee_image_from_point(coords=coords, bool_buffer_in_deg=bool_buffer_in_deg,
                                      buffer_deg=buffer_deg, buffer_m=buffer_m,
                                        verbose=verbose, year=year, 
                                        month_start_str=month_start_str, month_end_str=month_end_str,
                                        image_collection=image_collection,
                                        threshold_size=threshold_size)
    if im_gee is None:  ## if image was too small it was discarded
        return None, None

    if path_save is None:
        path_save = path_dict['data_folder'] 
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        print(f'Created folder {path_save}')

    filename = create_filename(base_name=name, image_collection=image_collection, year=year,
                               month_start_str=month_start_str, month_end_str=month_end_str)
    filepath = os.path.join(path_save, filename)

    if image_collection == 'worldclimbio':  # just return values
        dict_save = {**im_gee, **{'coords': coords, 'name': name}}
        with open(filepath, 'w') as f:
            json.dump(dict_save, f)
        return dict_save, filepath
    
    geemap.ee_export_image(
        im_gee, filename=filepath, 
        scale=10,  # 10m bands
        file_per_band=False,# crs='EPSG:32630'
        verbose=False
    )

    if resize_image:
        ## load & save to size correctly (because of buffer): 
        im = load_tiff(filepath, datatype='da')
        remove_if_too_small = True
        desired_pixel_size = 128
        
        if verbose:
            print('Original size: ', im.shape)
        if im.shape[1] < desired_pixel_size or im.shape[2] < desired_pixel_size:
            print('WARNING: image too small, returning None')
            if remove_if_too_small:
                os.remove(filepath)
            return None, None

        ## crop:
        padding_1 = (im.shape[1] - desired_pixel_size) // 2
        padding_2 = (im.shape[2] - desired_pixel_size) // 2
        im_crop = im[:, padding_1:desired_pixel_size + padding_1, padding_2:desired_pixel_size + padding_2]
        assert im_crop.shape[0] == im.shape[0] and im_crop.shape[1] == desired_pixel_size and im_crop.shape[2] == desired_pixel_size, im_crop.shape
        if verbose:
            print('New size: ', im_crop.shape)
        im_crop.rio.to_raster(filepath)
        im_gee = im_crop 

    return im_gee, filepath

def download_list_coord(coord_list, path_save=None, bool_buffer_in_deg=False, buffer_deg=0.01, buffer_m=800,
                        name_group='sample', start_index=0, stop_index=None, resize_image=True, threshold_size=128,
                        list_collections=['sentinel2', 'alphaearth', 'dynamicworld', 'worldclimbio', 'dsm']):
    assert type(coord_list) == list
    if path_save is None:
        path_save = path_dict['data_folder'] 
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        print(f'Created folder {path_save}')
    else:
        print(f'WARNING: folder {path_save} already exists. OVERWRITING files!')

    ## save list coords:
    filename_coords = os.path.join(path_save, f'{name_group}_coords.json')
    with open(filename_coords, 'w') as f:
        json.dump(coord_list, f)

    inds_none = []
    for i, coords in enumerate(tqdm(coord_list)):
        if i < start_index:
            continue
        if stop_index is not None and i >= stop_index:
            break
        name = f'{name_group}-{i}'
        for im_collection in list_collections:
            try:
                im, path_im = download_gee_image(coords=coords, name=name, 
                                                bool_buffer_in_deg=bool_buffer_in_deg,
                                                buffer_deg=buffer_deg, buffer_m=buffer_m,
                                                path_save=path_save, verbose=0,
                                                resize_image=resize_image,
                                                threshold_size=threshold_size,
                                                image_collection=im_collection)
            except Exception as e:
                print(f'Image {name}, {im_collection} could not be downloaded, error: {e}')
                im = None
            if im is None:
                inds_none.append(f'{i}_{im_collection}')
        
    if len(inds_none) > 0:
        print(f'Images that could not be downloaded: {inds_none}')
    return inds_none

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
        # # sentinel_eq = np.clip(sentinel_eq, 0, 3000) / 3000
        sentinel_eq = exposure.equalize_hist(sentinel_eq)
        sentinel_eq = sentinel_eq.reshape([3, sentinel_eq.shape[1], sentinel_eq.shape[1], -1])
        sentinel_eq = [sentinel_eq[:, :, :, i] for i in range(sentinel_eq.shape[-1])]
    else:
        sentinel_eq = None

    return sentinel, sentinel_eq, features, hypotheses

if __name__ == "__main__":
    print('This is a utility script for creating and processing the dataset.')