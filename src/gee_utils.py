import os, sys, json 
import numpy as np 
import shapely 
import rasterio
import xarray as xr
import rioxarray as rxr
import datetime
import utm 
from tqdm import tqdm, trange
from skimage import exposure
import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))
from data_utils import load_tiff
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
        assert False, 'WARNING: using buffer in degrees, which is not ideal for large latitudes.'
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
                            .mosaic() 
                            .reproject(f'EPSG:{epsg_code}', scale=10)  
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
    
    if threshold_size is not None and (im_dims[0] < threshold_size or im_dims[1] < threshold_size):
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
        desired_pixel_size = threshold_size if threshold_size is not None else 128
        
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
        im_crop = im_crop.astype(np.float32)
        im_crop.rio.to_raster(filepath)
        im_gee = im_crop 

    return im_gee, filepath

def download_list_coord(coord_list, name_list=None, path_save=None, bool_buffer_in_deg=False, buffer_deg=None, buffer_m=800,
                        name_group='sample', start_index=0, stop_index=None, resize_image=True, threshold_size=128,
                        list_collections=['sentinel2', 'alphaearth', 'dynamicworld', 'worldclimbio', 'dsm'],
                        save_coords_json=True):
    assert type(coord_list) == list
    if path_save is None:
        path_save = path_dict['data_folder'] 
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        print(f'Created folder {path_save}')
    else:
        print(f'WARNING: folder {path_save} already exists. OVERWRITING files!')

    if save_coords_json:
        filename_coords = os.path.join(path_save, f'{name_group}_coords.json')
        with open(filename_coords, 'w') as f:
            json.dump(coord_list, f)

    inds_none = []
    if name_list is not None and len(name_list) != len(coord_list):
        print('WARNING: name_list is not the same length as coord_list, ignoring name_list')
        name_list = None
    for i, coords in enumerate(tqdm(coord_list)):
        if i < start_index:
            continue
        if stop_index is not None and i >= stop_index:
            break
        if name_list is not None and len(name_list) == len(coord_list):
            name = name_list[i]
        else:
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

def get_lc_from_coord(lat, lon, year=2024, buffer_m=20):
    n_dw = len(DW_CLASSES)
    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(buffer_m).bounds()
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .mean()
            )
    samples = dw.sample(aoi, scale=10)
    feat = samples.getInfo()['features']
    n_feat = len(feat)
    if n_feat == 0:
        return None
    else:
        probs = np.zeros((n_feat, n_dw))
        for i, f in enumerate(feat):
            # label = f['properties']['label']
            # assert type(label) == int and 0 <= label < n_dw, f'Unexpected label value: {label}'
            probs[i, :] = np.array(list(f['properties'][cls] for cls in DW_CLASSES))
        return feat, probs

if __name__ == "__main__":
    print('This is a utility script for creating and processing the dataset using GEE.')