import os, sys, json 
import numpy as np 
import shapely 
import rasterio
import xarray as xr
import rioxarray as rxr
import datetime
import utm 
import pandas as pd
from tqdm import tqdm, trange
from skimage import exposure
import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))
import data_utils as du
from constants import DW_CLASSES

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

def create_aoi_from_coord_buffer(coords, buffer_m=1000):
    """Create an Earth Engine AOI (Geometry) from a coordinate and buffer in meters."""
    point = shapely.geometry.Point(coords)
    point = ee.Geometry.Point(coords)
    aoi = point.buffer(buffer_m).bounds()
    assert aoi is not None
    return aoi

def get_gee_image_from_point(coords, buffer_m=800,
                             verbose=0, year=None, threshold_size=128,
                             month_start_str='06', month_end_str='09',
                             image_collection='sentinel2'):
    '''Coords: (lon, lat)'''
    assert ONLINE_ACCESS_TO_GEE, 'Need to set ONLINE_ACCESS_TO_GEE to True to use this function'
    assert image_collection in ['sentinel2', 'alphaearth', 'dynamicworld', 'dsm', 'popdensity'], f'image_collection {image_collection} not recognised.'
    if year is None:
        year = 2024
    lon, lat = coords
    epsg_code = get_epsg_from_latlon(lat=lat, lon=lon)
    aoi = create_aoi_from_coord_buffer(coords=coords, buffer_m=buffer_m)
    
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
            return None, None
        ex_im_gee = ee.Image(ex_collection 
                            #   .project(crs='EPSG:27700', scale=1)
                            .filterBounds(aoi) 
                            .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31'))
                            .select(prob_bands)  # get all probability bands
                            .mean()  # mean over the year
                            .reproject(f'EPSG:{epsg_code}', scale=10)  # reproject to 10m
                            .clip(aoi)
                            )  # mean over the year
    elif image_collection == 'dsm':
        ex_collection = ee.ImageCollection("COPERNICUS/DEM/GLO30")
        if ex_collection is None:
            print(f'ERROR: could not load dsm collection from {coords}')
            return None, None
        ex_im_gee = ee.Image(ex_collection
                            .filterBounds(aoi)
                            .select(['DEM'])  # select the DEM band
                            # .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31'))
                            .first()
                            .reproject(f'EPSG:{epsg_code}', scale=10)
                            .clip(aoi))
        threshold_size = max(32, threshold_size // 4)  # DSM is 30m resolution, so allow smaller images
    elif image_collection == 'popdensity':
        year = 2019
        ex_collection = ee.ImageCollection("WorldPop/GP/100m/pop")
        ex_im_gee = ee.Image(
            ex_collection.filterDate(ee.Date(f"{year}-01-01"), ee.Date(f"{year}-12-31"))
            .mosaic()
            .unmask(0)  # set unmasked values to 0 (water bodies)
        )
        return ex_im_gee, aoi  ## return here because no AOI filter so don't want to do threshold size check.  
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
    
    return ex_im_gee, aoi

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
    assert image_collection in ['sentinel2', 'alphaearth', 'dynamicworld' ,'dsm', 'popdensity'], f'image collection {image_collection} not recognised.'
    if year is None:
        year = 2024

    im_gee, _ = get_gee_image_from_point(coords=coords, buffer_m=buffer_m,
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
        im = du.load_tiff(filepath, datatype='da')
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


def get_distance_to_road_within_aoi(aoi, cell_size=30, radius_max=5000):
    """Calculates for each pixel in AOI the distance to the nearest road within radius_max
    and returns the max/mean distance inside the AOI.
    
    Works globally by selecting the relevant GRIP4 regional dataset(s) based on AOI location.
    GRIP4 regions: Africa, Central-South-America, Europe, North-America,
                   Oceania, South-East-Asia, Middle-East-Central-Asia
    """

    # Bounding boxes [west, south, east, north] for each GRIP4 region
    GRIP4_REGIONS = {
        "Africa":                   ee.Geometry.BBox(-26,  -35,  52,   38),
        "Central-South-America":    ee.Geometry.BBox(-82,  -56, -34,   15),
        "Europe":                   ee.Geometry.BBox(-32,   28,  65,   72),
        "North-America":            ee.Geometry.BBox(-180,  15, -52,   84),
        "Oceania":                  ee.Geometry.BBox( 94,  -55, 180,    2),
        "South-East-Asia":          ee.Geometry.BBox( 60,  -12, 150,   55),
        "Middle-East-Central-Asia": ee.Geometry.BBox( 25,   10,  90,   55),
    }

    # Collect all regional datasets whose bounding box intersects the AOI
    matching = []
    for region_name, bbox in GRIP4_REGIONS.items():
        # if bbox.intersects(aoi, maxError=1000).getInfo():
        fc = ee.FeatureCollection(
            f"projects/sat-io/open-datasets/GRIP4/{region_name}"
        ).filterBounds(aoi.buffer(radius_max))
        matching.append(fc)

    if not matching:
        raise ValueError(
            "AOI does not fall within any known GRIP4 region. "
            "Check that your AOI geometry is valid and uses EPSG:4326."
        )

    # Merge all matching regional collections
    roads = matching[0]
    for fc in matching[1:]:
        roads = roads.merge(fc)

    distance = roads.distance(searchRadius=radius_max, maxError=50)
    distance_masked = distance.clip(aoi).rename("distance")

    max_distance = distance_masked.reduceRegion(
        reducer=ee.Reducer.max(), geometry=aoi, scale=cell_size, maxPixels=1e9
    )
    mean_distance = distance_masked.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=cell_size, maxPixels=1e9
    )

    return {
        "maxdist_road":  int(max_distance.get("distance").getInfo() or radius_max),
        "meandist_road": int(mean_distance.get("distance").getInfo() or radius_max),
    }


def convert_popdensity_im_to_sum(popdensity_im, aoi):
    """Convert a population density image to a total population count in the area."""
    assert ONLINE_ACCESS_TO_GEE, "ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE"
    sum_dict = popdensity_im.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=100,  # match the original worldpop image resolution
        maxPixels=1e9,
    )
    total_pop = sum_dict.getInfo().get(
        "population", 0
    )  # get total population, default to 0 if not found
    pop_density = total_pop / (aoi.area(maxError=1).getInfo() / 1e6)  # people per km^2
    if pop_density is None:
        print('pop density none')
    return {"total_population": int(total_pop), "pop_density": int(pop_density)}


def bioclim_schema(data_dir_dataset=None):
    df = pd.read_csv(os.path.join(path_dict['repo'], 'content/bioclim_classes.csv'))  # in content/ folder
    df.sort_values(by=["name"], inplace=True)
    bioclim_variables = df.to_dict("records")
    for v in bioclim_variables:
        v["name"] = v["name"].replace("aux_bioclim_", "bio")
    df["name"] = df["name"].apply(lambda x: x.replace("aux_bioclim_", "bio"))
    return bioclim_variables, df


def get_bioclim_from_coord(coords, buffer_m=1000):
    assert ONLINE_ACCESS_TO_GEE, "ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE"
    aoi = create_aoi_from_coord_buffer(coords, buffer_m=buffer_m)
    im_gee = ee.Image("WORLDCLIM/V1/BIO").clip(aoi)
    point = ee.Geometry.Point(coords)  # redefine point for sampling
    sampled = im_gee.sample(region=point.buffer(buffer_m), scale=1000)
    first = sampled.first()

    if first.getInfo() is None:  ## try increasing buffer if no data found, to account for points just off coastlines.
        print(f"No WORLDCLIM data found for coords {coords} with buffer {buffer_m}m, trying larger buffer...")
        buffer_m *= 10
        aoi = create_aoi_from_coord_buffer(coords, buffer_m=buffer_m)
        im_gee = ee.Image("WORLDCLIM/V1/BIO").clip(aoi)
        sampled = im_gee.sample(region=point.buffer(buffer_m), scale=1000)
        first = sampled.first()
        
        if first.getInfo() is None:
            raise ValueError(f"No WORLDCLIM data found for coords {coords} — point may be over ocean or outside dataset coverage.")
        else:
            print(f"Successfully retrieved WORLDCLIM data with larger buffer of {buffer_m}m.")

    values = first.toDictionary().getInfo()
    return values


def convert_bioclim_to_units(bioclim_dict):
    assert len(bioclim_dict) == 19, "bioclim_dict should have 19 variables"
    for k in range(1, 20):
        assert f"bio{str(k).zfill(2)}" in bioclim_dict, f"bio{str(k).zfill(2)} not in bioclim_dict"
    _, df_bioclim = bioclim_schema()
    for k, v in bioclim_dict.items():
        scale = df_bioclim.loc[df_bioclim["name"] == k, "scale"].values[0]
        bioclim_dict[k] = v * scale

    bioclim_dict = {f'bioclim_{k.lstrip("bio")}': float(v) for k, v in bioclim_dict.items()}
    return bioclim_dict


def get_aux_data_from_coords(
    coords, aux_modalities=[ "bioclim", "pop_density", "dist_road"], patch_size=1280):
    """Get both bioclimatic and land cover data from coordinates."""
    for m in aux_modalities:
        assert m in [
            "bioclim",
            "pop_density",
            "dist_road",
        ], f"Unknown auxiliary modality: {m}"
    aux_data = {}
    aoi = None
    if "bioclim" in aux_modalities:
        bioclim_data = get_bioclim_from_coord(coords, buffer_m=patch_size // 2)
        bioclim_data = convert_bioclim_to_units(bioclim_data)
        aux_data.update(bioclim_data)
    if "pop_density" in aux_modalities:
        popdensity_im, aoi = get_gee_image_from_point(
            coords,
            image_collection="popdensity",
            buffer_m=patch_size // 2,
            threshold_size=None,
        )
        popdensity_data = convert_popdensity_im_to_sum(popdensity_im, aoi)
        aux_data.update(popdensity_data)
    if "dist_road" in aux_modalities:
        if aoi is None:
            _, aoi = get_gee_image_from_point(
                coords,
                image_collection="popdensity",
                buffer_m=patch_size // 2,
                threshold_size=None,
            )
        dist_road = get_distance_to_road_within_aoi(aoi, cell_size=30, radius_max=50000)
        aux_data.update(dist_road)
    return aux_data


def get_aux_data_from_coords_list(
    coords_list,
    id_list=None,
    save_file=True,
    save_folder='./',
    name_group='',
    save_filename="aux_gee_data",
    patch_size=1280,
    start_index=0,
    stop_index=None
):
    """Get all auxiliary data from a list of coordinates."""
    if id_list is not None:
        assert len(id_list) == len(
            coords_list
        ), "id_list and coords_list must have the same length"
    else:
        assert False, 'assuming id_list provided'
    if save_file:
        fname = name_group + save_filename + f'{start_index}-{stop_index}' + '.csv'
        save_path = os.path.join(save_folder, fname)
        assert os.path.exists(save_folder), f"Save folder does not exist: {save_folder}"
        save_every_n = 100  # save every n samples to avoid data loss
        print(f"Will save auxiliary data to {save_path} every {save_every_n} samples")
    else:
        print("WARNING: Not saving auxiliary data to file.")

    results = {}
    inds_none = []
    with tqdm(
        total=len(coords_list) if stop_index is None else stop_index - start_index,
        desc=f"Collecting auxiliary data for {len(coords_list) if stop_index is None else stop_index - start_index} samples",
        position=0,
        leave=True,
        initial=0, 
    ) as pbar:
        for i_coords, coords in enumerate(coords_list):
            if i_coords < start_index:
                continue
            if stop_index is not None and i_coords >= stop_index:
                break
            try:
                result = get_aux_data_from_coords(coords, patch_size=patch_size)
                # result_keys = list(result.keys())
            except Exception as e:
                print(f"Error occurred while processing coordinates {i_coords}, {coords}: {e}")
                inds_none.append(i_coords)
                continue
                # result = {k: np.nan for k in result_keys}  ## issue here if not all keys are present.
            if i_coords == 0 or len(results) == 0:
                for k in result.keys():
                    results[k] = []
                results["coords"] = []
                if id_list is not None:
                    results["id"] = []
            if id_list is not None:
                results["id"].append(id_list[i_coords])
            results["coords"].append(coords)
            for k, v in result.items():  # what if not all keys are present?
                results[k].append(v)
            pbar.update(1)
            pbar.set_postfix({"current": i_coords})  # shows current index
            if save_file and (i_coords + 1) % save_every_n == 0:
                temp_results = pd.DataFrame(results)
                temp_results.to_csv(save_path, index=False)

    results = pd.DataFrame(results)

    if save_file:
        results.to_csv(save_path, index=False)
        print(f"Saved auxiliary data to {save_path}")

    if len(inds_none) > 0:
        print(f"Coordinates that could not be processed: {inds_none}")
    return results, save_path, inds_none


if __name__ == "__main__":
    print('This is a utility script for creating and processing the dataset using GEE.')