import os, sys
import numpy as np 
import pandas as pd 
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import random 
from shapely.geometry import Point
sys.path.append('../content/')
import ee, geemap
import api_keys
import data_utils as du
ee.Authenticate()
ee.Initialize(project=api_keys.GEE_API)
geemap.ee_initialize()

DW_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice",
  ]

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


def random_points_in_polygons(gdf, n):
    minx, miny, maxx, maxy = gdf.total_bounds
    points = []
    while len(points) < n:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if gdf.contains(p).any():
            points.append(p)
    return points


def sample(n=10000, save_every=100, year=2024, save_folder='/Users/tplas/data/', buffer_m=50):
    countries = gpd.read_file('/Users/tplas/data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
    points = random_points_in_polygons(countries, n)
    gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    coords = [(point.y, point.x) for point in gdf_points.geometry]
    majority_distribution = np.zeros(len(DW_CLASSES))
    results = {x: [] for x in ['lat', 'lon', 'label'] + DW_CLASSES}
    timestamp = du.create_timestamp()
    name = f'dw_locations_{timestamp}_year-{year}'
    pbar = tqdm(total=len(coords), desc="Processing coordinates")
    for it, (lat, lon) in enumerate(coords):
        res = get_lc_from_coord(lat, lon, year=year, buffer_m=buffer_m)
        if res is not None:
            probs = res[1]
            probs_mean = probs.mean(axis=0)
            av_argmax_label = int(np.argmax(probs_mean))
            majority_distribution[av_argmax_label] += 1
            results['lat'].append(lat)
            results['lon'].append(lon)
            results['label'].append(av_argmax_label)
            for i, cls in enumerate(DW_CLASSES):
                results[cls].append(float(probs_mean[i]))
        
        # Update progress bar with current majority distribution
        pbar.set_postfix({DW_CLASSES[i]: int(majority_distribution[i]) for i in range(len(DW_CLASSES))})
        pbar.update(1)

        # Save results every `save_every` iterations
        if (it + 1) % save_every == 0 or (it + 1) == len(coords):
            df = pd.DataFrame(results)
            n_samples = len(df)
            save_path = os.path.join(save_folder, f'{name}_{n_samples}-samples_{buffer_m}m.csv')
            df.to_csv(save_path, index=False)

    pbar.close()

if __name__ == "__main__":
    sample(year=2024)