import os, sys
import numpy as np 
import pandas as pd 
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random 
from shapely.geometry import Point
sys.path.append('../content/')
import ee, geemap
import api_keys
import data_utils as du
import gee_utils as gu

DW_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice",
  ]

def random_points_in_polygons(gdf, n):
    '''Sample random points within the given polygons.'''
    assert False, 'Warning that this biases towards poles.'
    minx, miny, maxx, maxy = gdf.total_bounds  # x = longitude, y = latitude
    points = []
    while len(points) < n:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if gdf.contains(p).any():
            points.append(p)
    return points

def random_points_on_sphere_in_polygons(gdf, n):
    '''Sample random points on the sphere within the given polygons.'''
    points = []
    while len(points) < n:
        lat = np.degrees(np.arcsin(random.uniform(-1, 1)))  # Latitude between -90 and 90
        lon = random.uniform(-180, 180)  # Longitude between -180 and 180
        p = Point(lon, lat)
        if gdf.contains(p).any():
            points.append(p)
    return points


def sample_dw_lc_uniformly(n=10000, save_every=100, year=2024, buffer_m=50,
           save_folder='/Users/tplas/data/dw_sampled_locations/spherical/'):
    countries = gpd.read_file('/Users/tplas/data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
    points = random_points_on_sphere_in_polygons(countries, n)
    gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    coords = [(point.y, point.x) for point in gdf_points.geometry]
    majority_distribution = np.zeros(len(DW_CLASSES))
    results = {x: [] for x in ['lat', 'lon', 'label'] + DW_CLASSES}
    timestamp = du.create_timestamp()
    name = f'dw_locations_{timestamp}_year-{year}'
    pbar = tqdm(total=len(coords), desc="Processing coordinates")
    assert os.path.exists(save_folder), f"Save folder does not exist: {save_folder}"
    save_path = None 

    for it, (lat, lon) in enumerate(coords):
        res = gu.get_lc_from_coord(lat, lon, year=year, buffer_m=buffer_m)
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
            if save_path is None:
                save_path = os.path.join(save_folder, f'{name}_{buffer_m}m_spherical_v0.csv')
                ii, max_iters = 1, 100
                while os.path.exists(save_path):
                    save_path = os.path.join(save_folder, f'{name}_{buffer_m}m_spherical_v{ii}.csv')
                    ii += 1
                    if ii > max_iters:
                        raise Exception(f"Too many files with the same name. Please check the save folder: {save_folder}")

            df.to_csv(save_path, index=False)

    pbar.close()

def sample_evenly_from_biased_distr(
    gdf_points,
    size_sample = 10000,
    ratio_start = 0.5,
    ratio_prune = 0.5,
    step_size = 20,
):

    entropy_points = - np.sum(gdf_points[DW_CLASSES].values * np.log(gdf_points[DW_CLASSES].values + 1e-10), axis=1)
    sample_inds_best = set(np.argsort(entropy_points)[-int(size_sample * ratio_start):])

    gdf_prob = gdf_points[DW_CLASSES]
    inds_dropped_ever = set()
    while len(sample_inds_best) < size_sample:
        current_distr = gdf_prob.iloc[np.array(list(sample_inds_best))].sum(0)
        # min_class = np.argmin(current_distr)
        # min_class = np.random.choice(np.argsort(current_distr)[:2])
        mean_val = np.mean(current_distr)
        min_class = np.random.choice(np.where(current_distr < mean_val)[0])
        max_class = np.argmax(current_distr)
        remaining_inds = np.array(list(set(np.arange(len(gdf_prob))).difference(sample_inds_best).difference(inds_dropped_ever)))
        ## get new samples from min class
        tmp = gdf_prob.iloc[remaining_inds].sort_values(DW_CLASSES[min_class], ascending=False)
        
        new_inds = set(tmp[:step_size].index)
        
        assert len(new_inds.intersection(sample_inds_best)) == 0
        if ratio_prune > 0:
            tmp = gdf_prob.iloc[np.array(list(sample_inds_best))].sort_values(DW_CLASSES[max_class], ascending=False)
            drop_inds = set(tmp[:int(step_size * ratio_prune)].index)
            inds_dropped_ever = inds_dropped_ever.union(drop_inds)
            sample_inds_best = sample_inds_best.difference(drop_inds)
        sample_inds_best = sample_inds_best.union(new_inds)
        
    sample_inds_best = np.array(list(sample_inds_best))[:size_sample]
    sample_inds_best = np.sort(sample_inds_best)
    sum_dw_probs_sample = gdf_prob.iloc[sample_inds_best].sum() / len(sample_inds_best)
    c_eff_probs_sample = np.min(sum_dw_probs_sample) / np.mean(sum_dw_probs_sample)
    means_sample = np.mean(gdf_prob.iloc[sample_inds_best], axis=0)
    entropy_means = - np.sum(means_sample * np.log(means_sample + 1e-10))
    return sample_inds_best, c_eff_probs_sample, entropy_means

def plot_map_and_distr(gdf, countries=None, ax_map=None, ax_distr=None, 
                       col_plot='label', name=''):
    if ax_map is None and ax_distr is None:
        fig, (ax_map, ax_distr) = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 0.4]})
    cmap_dict = du.create_mpl_cmap_dynamic_world()
    if ax_map is not None:
        if countries is not None:
            countries.plot(ax=ax_map, color="#EFEFEF", edgecolor='black')
        gdf.plot(ax=ax_map, column=col_plot, markersize=1, 
                cmap=cmap_dict['all'] if col_plot == 'label' else cmap_dict['individual'][col_plot], 
                legend=False if col_plot == 'label' else True)
        ax_map.set_title(f'{name} {len(gdf)} points, showing {col_plot}.')
        
    if ax_distr is not None:
        sum_dw = gdf[DW_CLASSES].sum() / len(gdf)
        c_eff = np.min(sum_dw) / np.mean(sum_dw)
        means = gdf[DW_CLASSES].mean(0)
        entropy = - np.sum(means * np.log(means + 1e-10))
        ax_distr.bar(DW_CLASSES, sum_dw, color=[du.create_cmap_dynamic_world()[dw] for dw in DW_CLASSES])
        ax_distr.set_xticklabels(DW_CLASSES, rotation=45, ha='right')
        # ax_distr.set_title(f'Sum of probabilities per class\nSampling efficiency: {c_eff:.3f}')
        ax_distr.set_title(f'C_eff: {c_eff:.2f}, H = {entropy:.2f}, PP = {np.exp(entropy):.2f}')
        ax_distr.set_ylabel('Density')
        ax_distr.plot([-0.5, len(DW_CLASSES) - 0.5], [1 / len(DW_CLASSES)] * 2, 'k--', alpha=0.8)
        for sp in ['top', 'right']:
            ax_distr.spines[sp].set_visible(False)  

if __name__ == "__main__":
    sample_dw_lc_uniformly(year=2024, n=6000)