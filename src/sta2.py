#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 23 Feb 14:29:10 2026 

@author: jbakermans
"""

import data_utils as du
import vis_utils as vu
import numpy as np
import os
import tqdm
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Rectangle
from scipy.stats import zscore
import scipy.optimize as opt
import geopy.distance
from sklearn.metrics.pairwise import haversine_distances

coords_rad = np.radians(coords)  # shape (n, 2), lat/lon
dist_matrix = haversine_distances(coords_rad) * 6371  # km


# Quick utility function for fitting 2d gaussians
# See https://stackoverflow.com/a/77432576/8919448
def gauss_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

# Core function that calculates receptive fields
# Takes as input a locations x dimensions embedding matrix,
# and a locations x cover x pixels x pixels land cover matrix
def get_receptive_fields(emb, lc, window=32, N_perm=100, do_zscore_lc=True, do_zscore_px=True, do_perm=False, do_regress=False):
    # Extract window around center
    pixels = lc.shape[-1]
    min_window = int(np.floor(pixels/2)-np.floor(window/2))
    max_window = int(np.floor(pixels/2)+np.ceil(window/2))
    lc = lc[:,:,min_window:max_window, min_window:max_window]

    # Extract the matrix for centre pixel embeddings,
    # which is a samples x embedding dimension matrix
    pix_emb = emb.to_numpy()[:,~emb.columns.isin(['id'])]

    # z-score the embeddings and land cover patches, if required
    if do_zscore_lc:
        # z-score within each land cover (which is the second dimension)
        lc = np.stack([zscore(lc[:,i,:,:]) for i in range(lc.shape[1])],axis=1)
    if do_zscore_px:
        # z-score within each embedding dimension (which is second dimension)
        pix_emb = zscore(pix_emb, axis=0)

    # Calculate the weighted sum as receptive field
    # This is a land cover x embedding x pixels x pixels matrix
    # My natural way of implementing this is the below, 
    # but broadcasting creates huge intermediates, and I run out of RAM
    # rec_field = np.mean(lc[:,:,None,:,:] * pix_emb[:,None,:,None,None], axis=0)
    # Let's first make this easier on RAM by using float32
    lc = lc.astype(np.float32)
    pix_emb = pix_emb.astype(np.float32)
    # Then do the actual weighted average with einsum instead of broadcasting
    rec_field = np.einsum('nmxy,nk->mkxy', lc, pix_emb) / lc.shape[0]

    # Find receptive field by permutating feature values between patches
    if do_perm:
        # Now I want to calculate receptive fields as a permutation test
        # I want to repeat it for permuted pixel embeddings,
        # then z-score the unpermuted fields against the permutations
        # To z-score I need mean and variance, but I can't store them for all permutations
        # So calculate both in a "streaming" online way (Welford's algorithm)
        perm_mean = np.zeros_like(rec_field)
        perm_M2 = np.zeros_like(perm_mean)
        for i in tqdm.tqdm(range(N_perm)):
            perm_field = np.einsum('nmxy,nk->mkxy', lc, pix_emb[np.random.permutation(len(pix_emb))]) / lc.shape[0]
            delta = perm_field - perm_mean
            perm_mean += delta / (i + 1)
            delta2 = perm_field - perm_mean
            perm_M2 += delta * delta2
        perm_variance = perm_M2 / (N_perm - 1)
        perm_std = np.sqrt(perm_variance)
        # Finally calculate the z-score
        rec_field = (rec_field - perm_mean) / perm_std

    # Regress out the average land cover value across patches, if required
    if do_regress:
        cleaned_rec_field = np.zeros_like(rec_field)
        # Find the average land cover across all patches
        avg_lc = np.mean(lc, axis=0)        
        for lc_i, (curr_lc, curr_rec_field) in enumerate(zip(avg_lc, rec_field)):
            # Regress out the average land cover from the receptive field
            X = curr_lc.reshape(-1)
            X = np.stack([np.ones(X.size), X], axis=-1)
            Y = curr_rec_field.reshape([curr_rec_field.shape[0],-1]).T
            b = np.linalg.pinv(X) @ Y
            e = Y - X@b 
            cleaned_rec_field[lc_i] = e.T.reshape(curr_rec_field.shape)
        rec_field = cleaned_rec_field

    # Store land cover x embedding x pixels x pixels receptive fields
    # in big list of modalities and sample types
    return rec_field    

### LOAD DATA ###

# Initialise paths
data_folder = '/Users/jbakermans/Documents/Data/Thijs'
list_ids, modality_folders, gdf_points = du.get_list_complete_ids(data_folder)
print(f'Number of samples: {len(list_ids)}')

# Set data sources for which I have pixel values
modalities = ['alphaearth', 'tessera', 'satclip', 'geoclip'] 
samples = ['random_sample', 'lc_stratified_sample']
sample_ids = [gdf_points['id'][gdf_points[s]==1].to_numpy() for s in samples]
embeddings = [[du.load_csv_with_points(parent_folder=data_folder, modality=m, sample_type=s) 
               for s in samples] for m in modalities]

# Then grab the land cover files
land_cover = [{id: du.load_tiff(os.path.join(data_folder, 'dynamicworld', f'{id}_dynamicworld_y-2024.tif'), datatype='np')
               for id in tqdm.tqdm(s_id)} for s_id in sample_ids]
# Get names of hypotheses: different coarse land coverage classes
land_cover_names = [k for k in du.create_cmap_dynamic_world().keys()]

### ANALYSE ALPHA EARTH EMBEDDINGS ###

# Use the alpha earth embeddings for stratified sample
m_i, s_i = 0, 1
emb = embeddings[m_i][s_i][[s_id in land_cover[s_i] for s_id in embeddings[m_i][s_i]['id']]]
emb_np = emb.to_numpy()
lc = np.stack([land_cover[s_i][s_id] for s_id in emb['id']])
rf = get_receptive_fields(emb, lc)

# Find baselines: mean response across window
baseline = np.mean(rf, axis=(-1,-2))
basestd = np.std(rf, axis=(-1,-2))
# Find peakyness: how much is the peak beyond the baseline
peakyness = (np.max(np.abs(rf), axis=(-1,-2)) - np.abs(baseline)) / basestd
# Find baseline z-stat: how much is baseline away from 0
baseline_z = np.abs(baseline) / basestd
# Find examples of *low* baseline but *high* peakyness
# But introduce an additional offset for baseline, 
# otherwise this will just be dominated by low-baseline (and not high-peakyness)
# This offset will determine how much the selection is 
# driven by low baseline (low offset) vs high peakyness (high offset)
# If you put it too low, you just get flat curves
# If you put it too high, the baseline doesn't matter, and you get high overall
prime_examples = peakyness / np.maximum(baseline_z, 2)

# Plot these variables
plt.figure(); 
plt.subplot(4,1,1)
plt.imshow(baseline, vmin=-np.max(np.abs(baseline)), vmax=np.max(np.abs(baseline)), cmap='RdBu_r')
plt.colorbar()
plt.title('Baseline')
plt.subplot(4,1,2)
plt.imshow(peakyness, cmap='Greys')
plt.colorbar()
plt.title('Peakyness')
plt.subplot(4,1,3)
plt.imshow(baseline_z, cmap='Greys')
plt.colorbar()
plt.title('Baseline strength')
plt.subplot(4,1,4)
plt.imshow(prime_examples, cmap='Greys')
plt.colorbar()
plt.title('Peakyness/baseline strength: interesting profile?')

# Get top 10 examples
sorted_examples = np.argsort(prime_examples.reshape(-1))[::-1]
sorted_indices = np.array(np.unravel_index(sorted_examples, prime_examples.shape))
# Collect their receptive fiels
sorted_rfs = rf.reshape([-1, rf.shape[-2], rf.shape[-1]])[sorted_examples]
# Plot these
select_best = 10
plot_lc = 10
rf_lim = np.max(np.abs(sorted_rfs[:select_best]))
plt.figure();
for i in range(select_best):
    plt.subplot(plot_lc+1,select_best,i+1)
    plt.imshow(sorted_rfs[i], vmin=-rf_lim, vmax=rf_lim, cmap='RdBu_r')
    plt.title(f'{sorted_indices[1,i]}: {land_cover_names[sorted_indices[0,i]]}')
    plt.xticks([])
    plt.yticks([])
    for j in range(plot_lc):
        plt.subplot(plot_lc+1,select_best,(j+1)*select_best + i+1)
        plt.imshow(lc[j,sorted_indices[0,i],:,:], vmin=0, vmax=1, cmap='Greys')
        plt.title(f'{emb_np[j,sorted_indices[1,i]]:.2f}')
        plt.xticks([])
        plt.yticks([])        

### EXPLORE SPATIAL VARIATION IN AE EMBEDDING ###

# Get longitude and latitude for all patches
loc = np.stack([[lat, lon] for lon, lat, id in zip(gdf_points['lon'], gdf_points['lat'], gdf_points['id']) 
                if id in emb['id'].to_numpy()])
# Create a approximate distance matrix between all points
# This is inaccurate but fast; geodesic would be better, but slow
coords_rad = np.radians(loc)
dist_matrix = haversine_distances(coords_rad)

# Grab a bunch of points that are relatively far away from each other
N_points = 10
points = [np.random.randint(len(loc))]
for i in range(N_points-1):
    prev_dist = np.min(dist_matrix[points], axis=0)
    new_point = np.argmax(prev_dist)
    points.append(new_point)
points = np.array(points)

# Logic from here:
# 1. Get 1000 closest patches
# 2. Calculate receptive fields
# 3. Find dimensions that change a lot

### CALCULATE ALL RECEPTIVE FIELDS ###

# Collect receptive fields
receptive_fields = [[[] for _ in samples] for _ in modalities]

# Select one embedding matrix
for m_i in range(len(modalities)):
    for s_i in range(len(samples)):
        # Print progress
        print(f'Calculating receptive fields for {modalities[m_i]}, {samples[s_i]}')
        # Temporary fix: some embeddings don't have land cover; exclude those
        emb = embeddings[m_i][s_i][[s_id in land_cover[s_i] for s_id in embeddings[m_i][s_i]['id']]]

        # Stack land cover for these embeddings
        # This is a samples x land covers x pixels x pixels matrix
        lc = np.stack([land_cover[s_i][s_id] for s_id in emb['id']])

        # Calculate receptive fields and store in big matrix
        receptive_fields[m_i][s_i] = get_receptive_fields(emb, lc)

# Plot a selection of receptive fields
for m_i in range(len(modalities)):
    for s_i in range(len(samples)):
        curr_fields = receptive_fields[m_i][s_i]
        h_to_plot = curr_fields.shape[0]
        f_to_plot = 20
        lim = np.nanmax(np.abs(curr_fields[:h_to_plot, :f_to_plot]))
        # Plot one tuning curve per hypothesis per feature
        for plot_lim in [lim, None]:
            plt.figure(figsize=(f_to_plot, h_to_plot))
            plt.suptitle(f'{modalities[m_i]}, {samples[s_i]}')
            for row, hyp_rec in enumerate(curr_fields[:h_to_plot]):
                for col, rec in enumerate(hyp_rec[:f_to_plot]):
                    ax = plt.subplot(h_to_plot, f_to_plot, row * f_to_plot + col + 1)
                    if plot_lim is None:
                        plot_lim = np.nanmax(np.abs(curr_fields[:, col]))
                        ax.imshow(rec,cmap='RdBu_r', vmin=-plot_lim, vmax=plot_lim)
                    else:
                        ax.imshow(rec,cmap='RdBu_r', vmin=-plot_lim, vmax=plot_lim)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0:
                        ax.set_ylabel(land_cover_names[row].replace('_','\n'), rotation=0, labelpad=20)
                    if row == 0:
                        ax.set_title(f'F{col}')
            plt.tight_layout();
            plt.savefig(os.path.join(data_folder, 'output', f'rf_{modalities[m_i]}_{samples[s_i]}_{plot_lim is None}.png'))
