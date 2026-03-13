#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:20:07 2025

@author: jbakermans
"""

import data_utils as du
import vis_utils as vu
import numpy as np
import os
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

# Then collect from all patches the alpha and dyn data
patches = 200
hypotheses = []
features = []
pictures = []
for p in range(patches):
    print(f'Loading patch {p} / {patches}')
    # Load all data, both features and hypotheses, for the current patch
    (data_sent, data_alpha, data_dyn, data_worldclim, data_dsm) = du.load_all_modalities_from_name(name=f'pecl176-{p}', 
                                                                        path_folder='../content/pecl-100-subsample-30km', verbose=0)    
    if data_sent is None:
        continue
    # Land coverage and DSM serve as hypotheses
    assert len(data_dyn.data.shape) == 3 and len(data_dsm.data.shape) == 3 and data_dyn.data.shape[1:] == data_dsm.data.shape[1:]
    hypotheses.append(np.concatenate([data_dyn.data, data_dsm.data], axis=0))
    # This can definitely be cleaner but I use nan for undefined values
    f_dat = data_alpha.data
    f_dat[~np.isfinite(f_dat)] = np.nan
    features.append(f_dat)    
    pictures.append(data_sent)
# Some patches were undefined so set number to correct value
patches = len(features)

# Z-score across patches for each feature and each hypothesis
feat_m = np.stack([np.nanmean(f) for f in np.stack(features, axis=-1)])
feat_std = np.stack([np.nanstd(f) for f in np.stack(features, axis=-1)])
features = [(f - feat_m[:,None,None])/feat_std[:,None,None] for f in features]
hyp_m = np.stack([np.nanmean(h) for h in np.stack(hypotheses, axis=-1)])
hyp_std = np.stack([np.nanstd(h) for h in np.stack(hypotheses, axis=-1)])
hypotheses = [(h - hyp_m[:,None,None])/hyp_std[:,None,None] for h in hypotheses]

# Get names of hypotheses: different coarse land coverage classes
names = [k for k in du.create_cmap_dynamic_world().keys()] + ['dsm']

# Extract relevant dimensions
radius = 10 # pixels, excluding center pixel (so diameter = 2 * radius + 1)
N_features = features[0].shape[0]
N_pixels = features[0].shape[1]
N_hypotheses = hypotheses[0].shape[0]

### LOAD OR RUN ANALYSIS ###

# Load stas and fits, or recalculate them
load = True
base_dir = '../outputs/radius10'
if load:
    all_stas = np.load(os.path.join(base_dir, 'sta_dat.npy'))
    all_fits = np.load(os.path.join(base_dir, 'sta_fit.npy'))
    all_params = np.load(os.path.join(base_dir, 'sta_par.npy'))
else:
    # Run through all patches, collecting spike triggered averages for each feature for each hypothesis
    patch_stas = []
    for p, (hypothesis, feature) in enumerate(zip(hypotheses, features)):
        print(f'Analysing patch {p+1} / {len(hypotheses)}')
        
        # Create empty region of interest maps: area around each pixel for each band
        rois = np.full([N_hypotheses, N_pixels - 2 * radius, N_pixels - 2 * radius,
                        radius * 2 + 1, radius * 2 + 1], np.nan)
        
        # Collect searchlight data for each pixel from all bands of current modality
        for h, hyp in enumerate(hypothesis):
            print(f'Copying hyp {h} / {len(hypotheses)}')
            for row in range(radius, N_pixels - radius):
                for col in range(radius, N_pixels - radius):
                    # Grab the relevant pixels from the band
                    rois[h, row - radius, col - radius, :, :] = \
                        hyp[(row - radius):(row + radius + 1), (col - radius):(col + radius + 1)]
                        
        # Collect spike time averages for each band
        stas = []
        for h, hyp_rois in enumerate(rois):
            print(f'Calculating spike triggered averages for hyp {h} / {len(rois)}')
            # Then create the spike time average: multiply each roi by the pixel value of the feature pixel
            hyp_stas = [hyp_rois * d[radius:(N_pixels-radius),radius:(N_pixels-radius),None,None] for d in feature]
            # Then average across all pixels and stack to get big output array
            hyp_stas = np.stack([np.nansum(sta.reshape([-1, radius*2+1, radius*2+1]), axis=0) for sta in hyp_stas])
            # Also average the hypothesis itself across all pixels
            hyp_norm = np.nansum(np.reshape(hyp_rois, [-1, radius*2+1, radius*2+1]),axis=0)
            # Regress out the contribution of simple hypothesis geometry from responses
            X = np.stack([np.ones(hyp_norm.size), hyp_norm.reshape(-1)], axis=-1)
            Y = hyp_stas.reshape([N_features,-1]).T
            b = np.linalg.pinv(X) @ Y
            e = Y - X@b
            corr_stas = e.T.reshape([N_features, radius * 2 + 1, radius * 2 + 1])
            # Plot for debugging purposes
            if False:
                plt.figure();
                for r in range(8):
                    for c in range(8):
                        plt.subplot(8,8,r*8+c+1)
                        plt.imshow(feature[r*8+c])
                plt.figure();
                for r in range(8):
                    for c in range(8):
                        plt.subplot(8,8,r*8+c+1)
                        plt.imshow(hyp_stas[r*8+c])
                plt.figure();
                for r in range(8):
                    for c in range(8):
                        plt.subplot(8,8,r*8+c+1)
                        plt.imshow(corr_stas[r*8+c])
                plt.figure(); plt.imshow(hypothesis[h])
                plt.figure(); plt.imshow(hyp_norm)       
                import pdb; pdb.set_trace()             
            # And append to output stas
            stas.append(corr_stas)
        patch_stas.append(np.stack(stas))
        
    # Average across patches to get final stas
    all_stas = np.nanmean(np.stack(patch_stas), axis=0)
    # Save as npy file
    np.save(os.path.join(base_dir, 'sta_dat.npy'), all_stas)
    
    # Fit a gaussian to each sta
    x, y = np.meshgrid(np.arange(2*radius+1), np.arange(2*radius+1))
    # Collect fitted parameters and resulting images
    all_params = np.full(list(all_stas.shape[:-2]) + [7], np.nan)
    all_fits = np.full(all_stas.shape, np.nan)
    for row, hyp_stas in enumerate(all_stas):
        print(f'Fitting Gaussians to hyp {row} / {N_hypotheses}')
        for col, sta in enumerate(hyp_stas):
            curr_fits = []
            curr_pars = []
            for fit_type in ['pos', 'neg']:
                # Set fit type dependent initial guesses: amplitude, offset, center
                if fit_type == 'pos':       
                    a_0 = np.max(sta) - np.min(sta)
                    (y_0, x_0) = np.unravel_index(sta.argmax(), sta.shape)
                    o_0 = np.min(sta)
                else:
                    a_0 = np.min(sta) - np.max(sta)
                    (y_0, x_0) = np.unravel_index(sta.argmin(), sta.shape)
                    o_0 = np.max(sta)
                # Estimate the std as the pixel distance between the peak and the inflexion point:
                # The location where the second derivative is nearest to 0
                curve = np.mean(sta, axis=0)
                curve = np.diff(np.diff(np.mean(sta, axis=0)))
                sx_0 = max(1, np.abs(x_0 - np.argmin(np.abs(np.diff(np.diff(np.mean(sta, axis=0))))) - 1)) # -1 because diff loses dim
                sy_0 = max(1, np.abs(y_0 - np.argmin(np.abs(np.diff(np.diff(np.mean(sta, axis=1))))) - 1))
                theta_0 = 0.0
                # Do the actual fit
                try:
                    # find the optimal Gaussian parameters
                    popt, pcov = opt.curve_fit(gauss_2d, (x, y), sta.ravel(), 
                                               p0=(a_0, x_0, y_0, sx_0, sy_0, theta_0, o_0),
                                               maxfev=int(1e5))   
                    # Store the resulting parameters and fit
                    curr_fits.append(gauss_2d((x, y), *popt).reshape(2*radius+1,2*radius+1))
                    curr_pars.append(popt)
                except RuntimeError as e:
                    # This generally happens when we run out of iterations
                    # That's usually caused by strongly non-gaussian stas. Let's just ignore those
                    print(f'Gaussian fit failed for type {fit_type}, hyp {row}, feature {col}. \n Error message: {e}')
                    # Store the resulting parameters and fit
                    curr_fits.append(np.zeros_like(sta))
                    curr_pars.append(np.zeros(7))
            # Only keep the best fit, between the positive and negative peak
            best_fit = int(np.sum(np.square(sta - curr_fits[0])) > np.sum(np.square(sta - curr_fits[1])))
            # Keep the best fit between min and max
            all_params[row, col, :] = curr_pars[best_fit]
            all_fits[row, col, :, :] = curr_fits[best_fit]
    
    # Save as npy file
    np.save(os.path.join(base_dir, 'sta_fit.npy'), all_fits)
    np.save(os.path.join(base_dir, 'sta_par.npy'), all_params)

### PLOT RESULTS ###

# Plot a selection of stas
h_to_plot = len(all_stas)
f_to_plot = N_features
plt.figure(figsize=(f_to_plot, h_to_plot))
# Color plot borders by average value
lim = np.nanmax(np.abs(all_stas[:h_to_plot, :f_to_plot]))
# Plot one tuning curve per hypothesis per feature
for row, hyp_stas in enumerate(all_stas[:h_to_plot]):
    for col, sta in enumerate(hyp_stas[:f_to_plot]):
        ax = plt.subplot(h_to_plot, f_to_plot, row * f_to_plot + col + 1)
        ax.imshow(sta,cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(names[row].replace('_','\n'), rotation=0, labelpad=20)
        if row == 0:
            ax.set_title(f'F{col}')
plt.tight_layout();
if not load:
    plt.savefig(os.path.join(base_dir, 'sta_dat.png'))
            
# Plot the map for a particular feature that you might like across patches
cols=10
rows=int(np.ceil(patches/cols))
curr_f = 35
lim = np.nanmax(np.abs(np.stack(features, axis=-1)[curr_f]))
plt.figure(figsize=(2*cols, 2*rows))
for p, feature in enumerate(features):
    plt.subplot(rows, cols, p + 1)
    plt.imshow(feature[curr_f], vmin=-lim, vmax=lim, cmap="RdBu_r")
    plt.axis('off')
    
# Plot the map for a particular hypothesis that you might like across patches
cols=10
rows=int(np.ceil(patches/cols))
curr_h = 7
lim = np.nanmax(np.abs(np.stack(hypotheses, axis=-1)[curr_h]))
plt.figure(figsize=(2*cols, 2*rows))
for p, hypothesis in enumerate(hypotheses):
    plt.subplot(rows, cols, p + 1)
    plt.imshow(hypothesis[curr_h], vmin=-lim, vmax=lim, cmap="RdBu_r")
    plt.axis('off')    

# Plot fits for the same stas
plt.figure(figsize=(f_to_plot, h_to_plot))
# Color plot borders by average value
lim = np.nanmax(np.abs(all_fits[:h_to_plot, :f_to_plot]))
# Plot one tuning curve per hypothesis per feature
for row, hyp_fits in enumerate(all_fits[:h_to_plot]):
    for col, fit in enumerate(hyp_fits[:f_to_plot]):
        ax = plt.subplot(h_to_plot, f_to_plot, row * f_to_plot + col + 1)
        ax.imshow(fit, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(names[row].replace('_','\n'), rotation=0, labelpad=20)
        if row == 0:
            ax.set_title(f'F{col}')
plt.tight_layout();            
if not load:
    plt.savefig(os.path.join(base_dir, 'sta_fit.png'))         

# Make overviews of all parameter histograms
plot_params = np.copy(all_params)
plot_params[:,:,1] -= radius
plot_params[:,:,2] -= radius
plot_params[:,:,3] = np.abs(plot_params[:,:,3])
plot_params[:,:,4] = np.abs(plot_params[:,:,4])
plot_params[:,:,5] = np.mod(plot_params[:,:,5], 2*np.pi)
param_names = ['Amplitude', 'Center x', 'Center y', 'Sigma x', 'Sigma y', 'Angle', 'Base'];
param_N_bins = 20;
param_bins = [np.linspace(np.percentile(plot_params[:,:,p], 10),
                          np.percentile(plot_params[:,:,p], 90), param_N_bins) 
              for p in range(plot_params.shape[-1])]
plt.figure(figsize=(2*plot_params.shape[-1], plot_params.shape[0]))
for h, hyp_par in enumerate(plot_params):
    for p, (p_name, p_bins) in enumerate(zip(param_names, param_bins)):
        ax = plt.subplot(plot_params.shape[0], plot_params.shape[-1], h*plot_params.shape[-1] + p + 1)
        plt.hist(hyp_par[:,p], p_bins)
        if p == 0:
            ax.set_ylabel(names[h].replace('_','\n'), rotation=0, labelpad=20)
        if h == 0:
            ax.set_title(p_name)
        if h < plot_params.shape[0] - 1:
            ax.set_xticks([])
plt.tight_layout()
if not load:
    plt.savefig(os.path.join(base_dir, 'sta_par.png'))      
    
# Repeat for features with reasonable peak locations and sigmas
# This avoids fits where the sta is effectively a horizontal/vertical gradient
# so you need a huge gaussian (large radius, large amplitude, far away) to fit it
sta_to_plot = (np.abs(plot_params[:,:,1]) < radius*5) & \
    (np.abs(plot_params[:,:,2]) < radius*5) & \
    (plot_params[:,:,3] < radius * 5) & (plot_params[:,:,4] < radius * 5)
error = np.mean(np.abs(all_stas - all_fits).reshape([N_hypotheses,N_features,-1]),axis=-1)
filtered_params = np.copy(plot_params)
filtered_params[~np.tile(sta_to_plot[:,:,None], [1,1,7])] = np.nan

param_bins = [np.linspace(np.nanmin(filtered_params[:,:,p]), np.nanmax(filtered_params[:,:,p]), param_N_bins)
              for p in range(filtered_params.shape[-1])]
plt.figure(figsize=(2*plot_params.shape[-1], plot_params.shape[0]))
for h, hyp_par in enumerate(filtered_params):
    for p, (p_name, p_bins) in enumerate(zip(param_names, param_bins)):
        ax = plt.subplot(plot_params.shape[0], plot_params.shape[-1], h*plot_params.shape[-1] + p + 1)
        plt.hist(hyp_par[:,p], p_bins)
        if p == 0:
            ax.set_ylabel(names[h].replace('_','\n'), rotation=0, labelpad=20)
        if h == 0:
            ax.set_title(p_name)
        if h < plot_params.shape[0] - 1:
            ax.set_xticks([])
plt.tight_layout()  

# Calculate tuning strength to each hypothesis for each feature
# It's a pretty rough estimate of tuning: 
# How many stds is one hyp's amplitude from the mean of other hyp amplitudes
tuning_strength = np.zeros((N_hypotheses, N_features))
amp_feat = np.max(all_stas.reshape([N_hypotheses, N_features, -1]), axis=-1) - np.min(all_stas.reshape([N_hypotheses, N_features, -1]), axis=-1)
for h in range(N_hypotheses):
    hyp_self = np.eye(N_hypotheses)[h].astype(np.bool)
    self_amp = amp_feat[h]
    other_mean = np.nanmean(amp_feat[~hyp_self], axis=0)
    other_std = np.nanstd(amp_feat[~hyp_self], axis=0)
    tuning_strength[h,:] = np.abs(self_amp - other_mean) / other_std
# Plot average response and significant tuning
plt.figure()
plt.subplot(2,1,1);
plt.imshow(amp_feat)
plt.colorbar()
plt.subplot(2,1,2);
plt.imshow(tuning_strength, vmin=0, vmax=2)
plt.colorbar()

# Plot distribution of strongest tuning across features
sta_lim = np.nanmax(np.abs(all_stas))
feat_lim = np.nanmax(np.abs(all_stas))
plt.figure(figsize=(4,12));
plt.subplot(4,1,1)
plt.hist(np.max(tuning_strength, axis=0));
plt.xlabel('Strongest hypothesis tuning')
plt.ylabel('Nr of features')
# Plot a strongly tuned example
ax = plt.subplot(4,2,3);
vu.plot_image_simple(pictures[15], ax=ax)
plt.title('Sentinel')
plt.subplot(4,2,4);
plt.imshow(hypotheses[15][7])
plt.title('Bare hypothesis')
# Plot the feature that shows strong bare tuning
plt.subplot(4,2,5)
plt.imshow(all_stas[7,9])
plt.title('F9 bare sta')
plt.subplot(4,2,6);
plt.imshow(features[15][9])
plt.title('F9 map')
# Plot the feature that shows weak bare tuning
plt.subplot(4,2,7)
plt.imshow(all_stas[7,10])
plt.title('F10 bare sta')
plt.subplot(4,2,8);
plt.imshow(features[15][10])
plt.title('F10 map')

# Calculate scale: average between x and y sigma, only from reasonable fits
scale = np.mean(filtered_params[:,:,4:6], axis=-1)
plt.figure()
plt.imshow(scale)

# Plot scale across stas
plt.figure(figsize=(4,12));
plt.subplot(4,1,1)
plt.hist(scale[sta_to_plot]);
plt.xlabel('Sta scale in pixels')
plt.ylabel('Nr of stas')
# Plot a short and long scale example
ax = plt.subplot(4,2,3);
vu.plot_image_simple(pictures[4], ax=ax)
plt.title('Sentinel')
plt.subplot(4,2,4);
plt.imshow(hypotheses[4][1])
plt.title('Tree hypothesis')
# Plot the feature with short scale
plt.subplot(4,2,5)
plt.imshow(all_stas[1,30])
plt.title('F30 trees sta')
plt.subplot(4,2,6);
plt.imshow(features[4][30])
plt.title('F30 map')
# Plot the feature with long scale
plt.subplot(4,2,7)
plt.imshow(all_stas[1,36])
plt.title('F36 trees sta')
plt.subplot(4,2,8);
plt.imshow(features[4][36])
plt.title('F36 map')

# Calculate displacement: peak to center distance
displacement = np.sqrt(filtered_params[:,:,1]**2 + filtered_params[:,:,2]**2)
plt.figure()
plt.imshow(displacement)

# Plot displacement across stas
plt.figure(figsize=(4,12));
plt.subplot(4,1,1)
plt.hist(displacement[sta_to_plot]);
plt.xlabel('Sta displacement in pixels')
plt.ylabel('Nr of stas')
# Plot a short and long scale example
ax = plt.subplot(4,2,3);
vu.plot_image_simple(pictures[7], ax=ax)
plt.title('Sentinel')
plt.subplot(4,2,4);
plt.imshow(hypotheses[7][2])
plt.title('Grass hypothesis')
# Plot the feature with small displacement
plt.subplot(4,2,5)
plt.imshow(all_stas[1,13])
plt.title('F13 grass sta')
plt.subplot(4,2,6);
plt.imshow(features[7][13])
plt.title('F13 map')
# Plot the feature with large displacement
plt.subplot(4,2,7)
plt.imshow(all_stas[2,23])
plt.title('F23 trees sta')
plt.subplot(4,2,8);
plt.imshow(features[7][23])
plt.title('F23 map')

# Calculate symmetry: eccentricity
a = np.max(filtered_params[:,:,4:6], axis=-1)
b = np.min(filtered_params[:,:,4:6], axis=-1)
eccentricity = np.sqrt(1-(b/a)**2)
plt.figure()
plt.imshow(eccentricity)

# Plot eccentricity across stas
plt.figure(figsize=(4,12));
plt.subplot(4,1,1)
plt.hist(eccentricity[sta_to_plot]);
plt.xlabel('Sta eccentricity')
plt.ylabel('Nr of stas')
# Plot a short and long scale example
ax = plt.subplot(4,2,3);
vu.plot_image_simple(pictures[55], ax=ax)
plt.title('Sentinel')
plt.subplot(4,2,4);
plt.imshow(hypotheses[55][0])
plt.title('Water hypothesis')
# Plot the feature with small eccentricity
plt.subplot(4,2,5)
plt.imshow(all_stas[0,24])
plt.title('F24 water sta')
plt.subplot(4,2,6);
plt.imshow(features[55][24])
plt.title('F24 map')
# Plot the feature with large eccentiricty
plt.subplot(4,2,7)
plt.imshow(all_stas[0,49])
plt.title('F49 water sta')
plt.subplot(4,2,8);
plt.imshow(features[55][49])
plt.title('F49 map')