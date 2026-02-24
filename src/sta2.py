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
import scipy.optimize as opt

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

### CALCULATE RECEPTIVE FIELDS ###

# Hyperparams: window size, permutations
window = 32
N_perm = 100

# Collect receptive fields
receptive_fields = [[[] for _ in samples] for _ in modalities]

# Select one embedding matrix
for m_i in range(len(modalities)):
    for s_i in range(len(samples)):
        # Print progress
        print(f'Calculating receptive fields for {modalities[m_i]}, {samples[s_i]}')
        emb = embeddings[m_i][s_i]

        # Stack land cover for these embeddings
        # This is a samples x land covers x pixels x pixels matrix
        lc = np.stack([land_cover[s_i][s_id] for s_id in emb['id']])
        # Extract window around center
        pixels = lc.shape[-1]
        min_window = int(np.floor(pixels/2)-np.floor(window/2))
        max_window = int(np.floor(pixels/2)+np.ceil(window/2))
        lc = lc[:,:,min_window:max_window, min_window:max_window]

        # Extract the matrix for centre pixel embeddings,
        # which is a samples x embedding dimension matrix
        pix_emb = emb.to_numpy()[:,~emb.columns.isin(['id'])]

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
        rec_z = (rec_field - perm_mean) / perm_std
        receptive_fields[m_i][s_i] = rec_z

### PLOT RESULTS ###

# Get names of hypotheses: different coarse land coverage classes
names = [k for k in du.create_cmap_dynamic_world().keys()]

# Plot a selection of stas
for m_i in range(len(modalities)):
    for s_i in range(len(samples)):
        rec_z = receptive_fields[m_i][s_i]
        h_to_plot = rec_z.shape[0]
        f_to_plot = 20
        lim = np.nanmax(np.abs(rec_z[:h_to_plot, :f_to_plot]))
        # Plot one tuning curve per hypothesis per feature
        for plot_lim in [lim, None]:
            plt.figure(figsize=(f_to_plot, h_to_plot))
            plt.suptitle(f'{modalities[m_i]}, {samples[s_i]}')
            for row, hyp_rec in enumerate(rec_z[:h_to_plot]):
                for col, rec in enumerate(hyp_rec[:f_to_plot]):
                    ax = plt.subplot(h_to_plot, f_to_plot, row * f_to_plot + col + 1)
                    if plot_lim is None:
                        ax.imshow(rec,cmap='RdBu_r')
                    else:
                        ax.imshow(rec,cmap='RdBu_r', vmin=-plot_lim, vmax=plot_lim)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0:
                        ax.set_ylabel(names[row].replace('_','\n'), rotation=0, labelpad=20)
                    if row == 0:
                        ax.set_title(f'F{col}')
            plt.tight_layout();
            plt.savefig(os.path.join(data_folder, 'output', f'rf_{modalities[m_i]}_{samples[s_i]}_{plot_lim is None}.png'))
