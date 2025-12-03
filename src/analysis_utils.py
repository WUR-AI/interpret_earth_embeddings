import os, sys, json 
import numpy as np 
import pandas as pd 
import shapely 
import rasterio
import sklearn.decomposition
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression   
import xarray as xr
import rioxarray as rxr
# from shapely.geometry import Point, Polygon
import datetime
# import warnings
# from shapely.errors import ShapelyDeprecationWarning
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
# import geopandas as gpd
from tqdm import tqdm
# import scipy 
# import scipy.spatial, scipy.cluster
from collections import Counter
import scipy.optimize as opt

import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))
import data_utils as du

def correlation_two_matrices(mat1, mat2):
    print(mat1.shape, mat2.shape)
    assert mat1.shape[1] == mat2.shape[1]
    n1 = mat1.shape[0]
    n2 = mat2.shape[0]
    corr_mat = np.zeros((n1, n2), dtype=np.float32)
    for i1 in range(n1):
        for i2 in range(n2):
            corr_mat[i1, i2] = np.corrcoef(mat1[i1, :], mat2[i2, :])[0, 1]
    return corr_mat

def ravel_features(features):
    assert len(features.shape) == 4
    n_patches, n_features, nx, ny = features.shape
    return np.swapaxes(features, 0, 1).reshape((n_features, -1))

def unravel_features(raveled, n_patches, nx, ny):
    n_features = raveled.shape[0]
    return np.swapaxes(raveled.reshape((n_features, n_patches, nx, ny)), 0, 1)

def gauss_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    # Quick utility function for fitting 2d gaussians
    # See https://stackoverflow.com/a/77432576/8919448
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def load_tuning_surfaces(base_dir='../outputs/radius10', affix='_2025-11-18-1711_mean'):
    all_stas = np.load(os.path.join(base_dir, f'sta_dat{affix}.npy'))
    all_fits = np.load(os.path.join(base_dir, f'sta_fit{affix}.npy'))
    all_params = np.load(os.path.join(base_dir, f'sta_par{affix}.npy'))
    radius = int(base_dir.split('radius')[-1])
    return all_stas, all_fits, all_params, radius

def calculate_tuning_surfaces(hypotheses, features, radius=10, 
                              save_results=True, base_dir='../outputs/radius10',
                              fit_gaussians=False):
    N_features = features[0].shape[0]
    N_pixels = features[0].shape[1]
    N_hypotheses = hypotheses[0].shape[0]
    
    # Run through all patches, collecting spike triggered averages for each feature for each hypothesis
    patch_stas = []
    for p, (hypothesis, feature) in tqdm(enumerate(zip(hypotheses, features))):
        # print(f'Analysing patch {p+1} / {len(hypotheses)}')
        
        # Create empty region of interest maps: area around each pixel for each band
        rois = np.full([N_hypotheses, N_pixels - 2 * radius, N_pixels - 2 * radius,
                        radius * 2 + 1, radius * 2 + 1], np.nan)
        
        # Collect searchlight data for each pixel from all bands of current modality
        for h, hyp in enumerate(hypothesis):
            # print(f'Copying hyp {h} / {len(hypotheses)}')
            for row in range(radius, N_pixels - radius):
                for col in range(radius, N_pixels - radius):
                    # Grab the relevant pixels from the band
                    rois[h, row - radius, col - radius, :, :] = \
                        hyp[(row - radius):(row + radius + 1), (col - radius):(col + radius + 1)]
                        
        # Collect spike time averages for each band
        stas = []
        for h, hyp_rois in enumerate(rois):
            # print(f'Calculating spike triggered averages for hyp {h} / {len(rois)}')
            # Then create the spike time average: multiply each roi by the pixel value of the feature pixel
            hyp_stas = [hyp_rois * d[radius:(N_pixels-radius),radius:(N_pixels-radius),None,None] for d in feature]
            # Then average across all pixels and stack to get big output array
            hyp_stas = np.stack([np.nanmean(sta.reshape([-1, radius*2+1, radius*2+1]), axis=0) for sta in hyp_stas])
            # Also average the hypothesis itself across all pixels
            hyp_norm = np.nanmean(np.reshape(hyp_rois, [-1, radius*2+1, radius*2+1]),axis=0)
            # Regress out the contribution of simple hypothesis geometry from responses
            X = np.stack([np.ones(hyp_norm.size), hyp_norm.reshape(-1)], axis=-1)
            Y = hyp_stas.reshape([N_features,-1]).T
            b = np.linalg.pinv(X) @ Y
            e = Y - X@b
            corr_stas = e.T.reshape([N_features, radius * 2 + 1, radius * 2 + 1])
            # And append to output stas
            stas.append(corr_stas)
        patch_stas.append(np.stack(stas))
        
    # Average across patches to get final stas
    all_stas = np.nanmean(np.stack(patch_stas), axis=0)
    
    # Save as npy file
    if save_results:
        ts = du.create_timestamp()
        np.save(os.path.join(base_dir, f'sta_dat_{ts}.npy'), all_stas)
    
    # Fit a gaussian to each sta
    if fit_gaussians:
        x, y = np.meshgrid(np.arange(2 * radius + 1), np.arange(2 * radius + 1))
        # Collect fitted parameters and resulting images
        n_params_gaussian = 7  # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
        all_params = np.full(list(all_stas.shape[:-2]) + [n_params_gaussian], np.nan)
        all_fits = np.full(all_stas.shape, np.nan)
        for row, hyp_stas in tqdm(enumerate(all_stas)):
            # print(f'Fitting Gaussians to hyp {row} / {N_hypotheses}')
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
        if save_results:
            np.save(os.path.join(base_dir, f'sta_fit_{ts}.npy'), all_fits)
            np.save(os.path.join(base_dir, f'sta_par_{ts}.npy'), all_params)
    else:
        all_fits = None
        all_params = None

    return all_stas, all_fits, all_params, radius

def adjust_fit_parameters_for_plotting(all_params, radius):
    if all_params is None:
        return None, None, None, None, None
    # Prepare parameters for plotting
    plot_params = np.copy(all_params)
    plot_params[:,:,1] -= radius
    plot_params[:,:,2] -= radius
    plot_params[:,:,3] = np.abs(plot_params[:,:,3])
    plot_params[:,:,4] = np.abs(plot_params[:,:,4])
    plot_params[:,:,5] = np.mod(plot_params[:,:,5], 2*np.pi)
    param_names = ['Amplitude', 'Center x', 'Center y', 'Sigma x', 'Sigma y', 'Angle', 'Base'];
    param_N_bins = 20
    param_bins = [np.linspace(np.percentile(plot_params[:,:,p], 10),
                            np.percentile(plot_params[:,:,p], 90), param_N_bins) 
                for p in range(plot_params.shape[-1])]

    # Repeat for features with reasonable peak locations and sigmas
    # This avoids fits where the sta is effectively a horizontal/vertical gradient
    # so you need a huge gaussian (large radius, large amplitude, far away) to fit it
    sta_to_plot = (np.abs(plot_params[:,:,1]) < radius*5) & \
        (np.abs(plot_params[:,:,2]) < radius*5) & \
        (plot_params[:,:,3] < radius * 5) & (plot_params[:,:,4] < radius * 5)
    # error = np.mean(np.abs(all_stas - all_fits).reshape([N_hypotheses,N_features,-1]),axis=-1)
    filtered_params = np.copy(plot_params)
    filtered_params[~np.tile(sta_to_plot[:,:,None], [1,1,7])] = np.nan
    return plot_params, param_names, param_bins, filtered_params, sta_to_plot

def calculate_pca_dim(features_pca, list_n = [1, 5, 10, 50, 100], n_samples=20):
    assert len(features_pca.shape) == 4
    n_feat = features_pca.shape[1]
    dict_expl_var, dict_dim = {n: np.zeros((n_samples, n_feat)) for n in list_n}, {}

    for n in sorted(list_n):
        dim_list = []
        if n == features_pca.shape[0]:
            n_samples = 1
            dict_expl_var[n] = np.zeros((n_samples, n_feat))
        for i in tqdm(range(n_samples)):
            # Randomly sample n features
            inds = np.random.choice(a=features_pca.shape[0], size=n, replace=False)
            features_sampled = features_pca[inds, ...]
            features_sampled = ravel_features(features_sampled)
            features_sampled[np.isnan(features_sampled)] = 0

            # PCA
            pca = sklearn.decomposition.PCA(n_components=64)
            pca.fit(features_sampled.T)

            cum_expl_var = np.cumsum(pca.explained_variance_ratio_)
            dict_expl_var[n][i, :] = cum_expl_var

            sum_squares = np.sum(np.power(pca.explained_variance_, 2))
            square_sum = np.sum(pca.explained_variance_) ** 2
            dim = float(square_sum / sum_squares)
            # print(dim)
            dim_list.append(dim)
        dict_dim[n] = dim_list

    return dict_dim, dict_expl_var

def cca_and_normal_ols_from_hypotheses(features, hypotheses):
    features_all = np.stack(features, axis=0)
    features_all = ravel_features(features_all)
    hypotheses_all = np.stack(hypotheses, axis=0)
    hypotheses_all = ravel_features(hypotheses_all)

    assert np.sum(np.isnan(features_all)) == 0, "NaNs found in features_all, please check the data."
    assert np.sum(np.isnan(hypotheses_all)) == 0, "NaNs found in hypotheses_all, please check the data."

    n_patches = len(features)

    ## CCA:
    feat_ravel = features_all.T # ravel_features(np.stack(features))
    hyp_ravel = hypotheses_all.T #ravel_features(np.stack(hypotheses))
    assert feat_ravel.shape[0] == hyp_ravel.shape[0], (feat_ravel.shape, hyp_ravel.shape)
    print(f'Feature ravel shape: {feat_ravel.shape}, Hypothesis ravel shape: {hyp_ravel.shape}')

    cca = CCA(n_components=10)
    feat_c, hyp_c = cca.fit_transform(feat_ravel, hyp_ravel)
    canonical_corrs = [np.corrcoef(feat_c[:, i], hyp_c[:, i])[0, 1] for i in range(cca.n_components)]
    print("Canonical correlations:", np.round(canonical_corrs, 3))
    print(f'CCA components shape: F {feat_c.shape}, H {hyp_c.shape}')
    
    assert feat_c.shape[0] > feat_c.shape[1], (feat_c.shape)
    assert feat_ravel.shape[0] > feat_ravel.shape[1], (feat_ravel.shape)
    lr = LinearRegression(copy_X=True, fit_intercept=True)
    lr.fit(feat_c, feat_ravel)
    feat_hat_ols_cc = lr.predict(feat_c)
    r2_per_feature_cc = 1 - np.sum((feat_ravel - feat_hat_ols_cc)**2, axis=0) / np.sum((feat_ravel - np.mean(feat_ravel, axis=0))**2, axis=0)
    
    feat_res_ols_cc = feat_ravel - feat_hat_ols_cc
    feat_res_ols_cc_img = unravel_features(feat_res_ols_cc.T, n_patches=n_patches, nx=128, ny=128)
    feat_hat_ols_cc_img = unravel_features(feat_hat_ols_cc.T, n_patches=n_patches, nx=128, ny=128)

    # ## With OLS reprojection from H:
    lr = LinearRegression(copy_X=True, fit_intercept=True)
    lr.fit(hyp_ravel, feat_ravel)
    feat_hat_ols_h = lr.predict(hyp_ravel)
    r2_per_feature_h = 1 - np.sum((feat_ravel - feat_hat_ols_h)**2, axis=0) / np.sum((feat_ravel - np.mean(feat_ravel, axis=0))**2, axis=0)

    feat_res_ols_h = feat_ravel - feat_hat_ols_h
    feat_res_ols_h_img = unravel_features(feat_res_ols_h.T, n_patches=n_patches, nx=128, ny=128)
    feat_hat_ols_h_img = unravel_features(feat_hat_ols_h.T, n_patches=n_patches, nx=128, ny=128)

    return (feat_hat_ols_cc_img, feat_res_ols_cc_img), (feat_hat_ols_h_img, feat_res_ols_h_img), (np.stack(features, axis=0), np.stack(hypotheses, axis=0)), (canonical_corrs, r2_per_feature_cc, r2_per_feature_h)



## CCA with nans:
    # features_all = np.stack(features, axis=0)
    # features_all = ravel_features(features_all)
    # hypotheses_all = np.stack(hypotheses, axis=0)
    # hypotheses_all = ravel_features(hypotheses_all)

    # assert np.sum(np.isnan(features_all)) == 0, "NaNs found in features_all, please check the data."
    # assert np.sum(np.isnan(hypotheses_all)) == 0, "NaNs found in hypotheses_all, please check the data."

    # inds_pixels_nonnan = np.isnan(features_all).sum(0) == 0
    # features_all_nonnan = features_all[:, inds_pixels_nonnan]
    # hypotheses_all_nonnan = hypotheses_all[:, inds_pixels_nonnan]

    # feat_ravel = features_all_nonnan # au.ravel_features(np.stack(features))
    # hyp_ravel = hypotheses_all_nonnan #au.ravel_features(np.stack(hypotheses))
    # assert feat_ravel.shape[1] == hyp_ravel.shape[1], (feat_ravel.shape, hyp_ravel.shape)
    # assert len(inds_pixels_nonnan) == features_all.shape[1], (len(inds_pixels_nonnan), features_all.shape[1])
    # cca = CCA(n_components=10)
    # feat_c, hyp_c = cca.fit_transform(feat_ravel.T, hyp_ravel.T)
    # canonical_corrs = [np.corrcoef(feat_c[:, i], hyp_c[:, i])[0, 1] for i in range(cca.n_components)]
    # print("Canonical correlations:", np.round(canonical_corrs, 3))

    # ## Reprojection:
    # ## With original CCA weights:
    # weight_mat_cc = cca.x_weights_.T
    # # feat_hat_cc = feat_c @ cca.x_weights_.T
    # # feat_res_cc = feat_ravel - feat_hat_cc.T
    # # feat_res_cc_img = au.unravel_features(feat_res_cc, n_patches=n_patches, nx=128, ny=128)
    # # feat_hat_cc_img = au.unravel_features(feat_hat_cc.T, n_patches=n_patches, nx=128, ny=128)

    # ## With OLS reprojection from CC:
    # ## Original:
    # ## feat_ravel = feat_c @ weight_mat_cc.T + residuals
    # ## OLS alternative:
    # ## feat_ravel = feat_c @ pinv(feat_c) @ feat_ravel + residuals
    # weight_mat_ols_cc = np.linalg.pinv(feat_c).dot(feat_ravel.T)
    # assert weight_mat_cc.shape == weight_mat_ols_cc.shape, (weight_mat_cc.shape, weight_mat_ols_cc.shape)
    # feat_hat_ols_cc = feat_c.dot(weight_mat_ols_cc).T
    # feat_res_ols_cc = feat_ravel - feat_hat_ols_cc

    # feat_res_ols_cc_img = np.zeros_like(features_all) + np.nan
    # feat_res_ols_cc_img[:, inds_pixels_nonnan] = feat_res_ols_cc
    # feat_res_ols_cc_img = au.unravel_features(feat_res_ols_cc_img, n_patches=n_patches, nx=128, ny=128)

    # feat_hat_ols_cc_img = np.zeros_like(features_all) + np.nan
    # feat_hat_ols_cc_img[:, inds_pixels_nonnan] = feat_hat_ols_cc
    # feat_hat_ols_cc_img = au.unravel_features(feat_hat_ols_cc_img, n_patches=n_patches, nx=128, ny=128)

    # ## With OLS reprojection from H:
    # weight_mat_ols_h = np.linalg.pinv(hyp_ravel.T).dot(feat_ravel.T)
    # assert weight_mat_cc.shape == weight_mat_ols_h.shape, (weight_mat_cc.shape, weight_mat_ols_h.shape)
    # feat_hat_ols_h = hyp_ravel.T.dot(weight_mat_ols_h).T
    # feat_res_ols_h = feat_ravel - feat_hat_ols_h

    # feat_res_ols_h_img = np.zeros_like(features_all) + np.nan
    # feat_res_ols_h_img[:, inds_pixels_nonnan] = feat_res_ols_h
    # feat_res_ols_h_img = au.unravel_features(feat_res_ols_h_img, n_patches=n_patches, nx=128, ny=128)

    # feat_hat_ols_h_img = np.zeros_like(features_all) + np.nan
    # feat_hat_ols_h_img[:, inds_pixels_nonnan] = feat_hat_ols_h
    # feat_hat_ols_h_img = au.unravel_features(feat_hat_ols_h_img, n_patches=n_patches, nx=128, ny=128)

    