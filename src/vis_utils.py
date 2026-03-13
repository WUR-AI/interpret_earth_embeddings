import os, sys, json 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio, rasterio.plot
from sklearn.decomposition import PCA
import xarray as xr
import rioxarray as rxr
from collections import Counter
from skimage import exposure


import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))
import data_utils as du

def naked(ax):
    '''Remove all spines, ticks and labels'''
    for ax_name in ['top', 'bottom', 'right', 'left']:
        ax.spines[ax_name].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

def despine(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

def plot_image_simple(im, ax=None, name_file=None, use_im_extent=False, verbose=0):
    '''Plot image (as np array or xr DataArray)'''
    if ax is None:
        ax = plt.subplot(111)
    if type(im) == xr.DataArray:
        plot_im = im.to_numpy()
    else:
        plot_im = im
    if verbose > 0:
        print(plot_im.shape, type(plot_im))
    if use_im_extent:
        extent = [im.x.min(), im.x.max(), im.y.min(), im.y.max()]
    else:
        extent = None
    rasterio.plot.show(plot_im, ax=ax, cmap='viridis', 
                       extent=extent)
    naked(ax)
    ax.set_aspect('equal')
    if name_file is None:
        pass 
    else:
        name_tile = name_file.split('/')[-1].rstrip('.tif')
        ax.set_title(name_tile)

def add_scalebar(ax, location: str, fraction: float, label: str, offset=0.02, fontsize=10, text_offset=0.00):
#    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#     scalebar = AnchoredSizeBar(
#         ax.transData,
#         50, '500 m', 'lower left',  # length in data units
#         pad=-2, color='black', frameon=False
#     )
#     ax.add_artist(scalebar)
    '''offset corrects for the bar having rounded edges and not starting exactly at 0.'''
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if location == 'bottom left vertical':
        xy_start = (-0.03, -offset)
        xy_end = (-0.03, (1 + offset) * fraction)
        xy_text = (-0.04 * xlim[1] + text_offset, ylim[0])
        rotation=90
        ha='right'
    elif location == 'bottom right vertical':
        xy_start = (1 + 0.03, -offset)
        xy_end = (1 + 0.03, (1 + offset) * fraction)
        xy_text = ((1 + 0.04) * xlim[1] + text_offset, ylim[0])
        rotation=270
        ha='left'
    else:
        raise ValueError(f"Unknown location {location}")

    ax.annotate(
        '', 
        xy=xy_end, xycoords='axes fraction',
        xytext=xy_start,
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-', lw=1.5, color='k', connectionstyle="arc3,rad=0"),
        annotation_clip=False
    )
    ax.text(s=label, x=xy_text[0], y=xy_text[1], ha=ha, va='bottom', 
                        clip_on=False, rotation=rotation, fontsize=fontsize)



def plot_overview_images(path_folder=path_dict['data_folder'], name='sample-0', 
                         plot_alphaearth=True, plot_dynamicworld_full=True,
                         verbose=0):
    (file_sent, file_alpha, file_dynamic, file_worldclimbio, file_dsm) = du.get_images_from_name(path_folder=path_folder, name=name)
    path_sent = os.path.join(path_folder, file_sent) if file_sent is not None else None
    path_alpha = os.path.join(path_folder, file_alpha) if file_alpha is not None else None
    path_dynamic = os.path.join(path_folder, file_dynamic) if file_dynamic is not None else None
    path_worldclimbio = os.path.join(path_folder, file_worldclimbio) if file_worldclimbio is not None else None
    path_dsm = os.path.join(path_folder, file_dsm) if file_dsm is not None else None

    if path_alpha is not None:
        im_loaded_alpha = du.load_tiff(path_alpha, datatype='da')
        im_plot_alpha = im_loaded_alpha
        ## normalise:
        im_plot_alpha.values[im_plot_alpha.values == -np.inf] = np.nan
        im_plot_alpha.values[im_plot_alpha.values == np.inf] = np.nan
        im_plot_alpha = (im_plot_alpha - np.nanmin(im_plot_alpha)) / (np.nanmax(im_plot_alpha) - np.nanmin(im_plot_alpha))
        # im_plot_alpha = np.swapaxes(im_plot_alpha, 2, 1)  # change to (bands, height, width) to (height, width, bands)`
    else:
        im_plot_alpha = None

    if path_sent is not None:
        im_loaded_s2 = du.load_tiff(path_sent, datatype='da')
        im_loaded_s2 = np.clip(im_loaded_s2, 0, 3000)
        im_loaded_s2 = im_loaded_s2 / (3000)
        im_plot_s2 = im_loaded_s2[:3, ...]
        # im_nir_s2 = im_loaded_s2[1:, ...]
        im_nir_s2 = im_loaded_s2[np.array([0, 1, 3]), ...]  # B4, B3, B8
        ## put B8 band first:
        im_nir_s2 = im_nir_s2[[2, 0, 1], ...]
        size_s2 = im_plot_s2.shape[1]
        assert size_s2 == im_plot_s2.shape[2]
        half_size_s2 = size_s2 // 2
        quarter_size_s2 = size_s2 // 4
        # im_plot_s2 = im_plot_s2[:, quarter_size_s2:half_size_s2 + quarter_size_s2, quarter_size_s2:half_size_s2 + quarter_size_s2]
    else: 
        im_plot_s2 = None
        im_nir_s2 = None

    if path_dynamic is not None:
        im_loaded_dynamic = du.load_tiff(path_dynamic, datatype='da')
        im_argmax_dynamic = np.argmax(im_loaded_dynamic.values, axis=0)
        print('LC pixel count:', Counter(im_argmax_dynamic.flatten()))
        ## normalise:
        # im_plot_dynamic.values[im_plot_dynamic.values == -np.inf] = np.nan
        # im_plot_dynamic.values[im_plot_dynamic.values == np.inf] = np.nan
        # im_plot_dynamic = (im_plot_dynamic - np.nanmin(im_plot_dynamic)) / (np.nanmax(im_plot_dynamic) - np.nanmin(im_plot_dynamic))
    else:
        im_loaded_dynamic = None
        im_argmax_dynamic = None

    if path_dsm is not None:
        im_loaded_dsm = du.load_tiff(path_dsm, datatype='da')
    else:
        im_loaded_dsm = None

    if verbose:
        print(im_loaded_alpha.shape, type(im_loaded_alpha))
        print(im_loaded_s2.shape, type(im_loaded_s2))

    n_rows = 1 + 2 * int(plot_dynamicworld_full) + 4 * int(plot_alphaearth)

    fig, ax = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
    ax = ax.flatten()

    ## Top row:
    if im_plot_s2 is not None:
        plot_image_simple(im_plot_s2, ax=ax[0])
        ax[0].set_title('Sentinel-2 RGB')

    if im_nir_s2 is not None:
        plot_image_simple(im_nir_s2, ax=ax[1])
        ax[1].set_title('Sentinel-2 near infrared')

    if im_argmax_dynamic is not None:
        dict_classes = du.create_cmap_dynamic_world()
        cmap_dw = ListedColormap([v for v in dict_classes.values()])
        im = ax[2].imshow(im_argmax_dynamic, cmap=cmap_dw, interpolation='none', origin='upper', vmax=8.5, vmin=-0.5)
        # Place colorbar outside of ax[2] to avoid shrinking the imshow
        cbar = fig.colorbar(im, ax=ax[2], ticks=np.arange(0, 9), location='right', fraction=0.046, pad=0.04)
        cbar.ax.set_yticks(np.arange(0, 9))
        cbar.ax.set_yticklabels([k for k in dict_classes.keys()])
        # ax[2].set_title('Dynamic World land cover')

    if im_loaded_dsm is not None:
        plot_image_simple(im_loaded_dsm, ax=ax[4], name_file=path_dsm)
        ## cbar:
        mappable = ax[4].images[0]
        cbar = fig.colorbar(mappable, ax=ax[4], location='right', fraction=0.046, pad=0.04)
        ax[4].set_title('DSM (m)')

    for ii in range(2, 5):
        naked(ax[ii])

    ## dynamic world full:
    if im_loaded_dynamic is not None:
        for ii in range(9):
            ax_ind = 5 + ii
            im = ax[ax_ind].imshow(im_loaded_dynamic[ii, ...], cmap='viridis', interpolation='none', 
                                   origin='upper', vmin=0, vmax=1)
            naked(ax[ax_ind])
            ax[ax_ind].set_title(f'DW {ii}: {list(dict_classes.keys())[ii]}')
        ## add cbar to last plot:
        cbar = fig.colorbar(im, ax=ax[ax_ind], ticks=np.linspace(0, 1, 6),
                            location='right', fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Probability', rotation=270, labelpad=15)
        ax_ind += 1
        naked(ax[ax_ind])
    else:
        ax_ind = 14

    ## alpha earth:
    if plot_alphaearth and im_plot_alpha is not None:
        for ii in range(20):
            ax_ind += 1
            bands_alpha_plot = np.arange(ii * 3, (ii + 1) * 3)
            if bands_alpha_plot.max() >= im_plot_alpha.shape[0]:
                if ax_ind < len(ax):
                    naked(ax[ax_ind])
                continue
            plot_image_simple(im_plot_alpha[bands_alpha_plot, ...], ax=ax[ax_ind])
            ax[ax_ind].set_ylim(ax[ax_ind].get_ylim()[::-1])
            ax[ax_ind].set_title(f'AlphaEarth bands {bands_alpha_plot}')

def plot_simple_overview_embeddings(hyp, ax=None, method='first_3_bands', bands=None):
    if ax is None:
        ax = plt.subplot(111)
    assert method in ['first_3_bands', 'select_3_bands', 'pca'], "Unknown method for plotting overview of embeddings"
    assert hyp.ndim == 3 and hyp.shape[1] == hyp.shape[2], "Hyp should be (bands, height, width)"
    if method == 'first_3_bands':
        hyp_plot = hyp[:3, ...]
    elif method == 'select_3_bands':
        assert bands is not None and len(bands) == 3, "Bands must be a list of 3 integers"
        hyp_plot = hyp[bands, ...]
    elif method == 'pca':
        hyp_reshaped = hyp.reshape(hyp.shape[0], -1).T  # (pixels, bands)
        pca = PCA(n_components=3)
        hyp_pca = pca.fit_transform(hyp_reshaped)  # (pixels, 3)
        hyp_plot = hyp_pca.T.reshape(3, hyp.shape[1], hyp.shape[2])  # (3, height, width)
        # Normalize to [0, 1] for visualization
        hyp_plot = (hyp_plot - hyp_plot.min()) / (hyp_plot.max() - hyp_plot.min())

    ax.imshow(hyp_plot.transpose(1, 2, 0), interpolation='none')

def plot_distr_embeddings(path_folder=path_dict['data_folder'], name='sample-0', verbose=0):
    (file_sent, file_alpha, file_dynamic, file_worldclimbio, file_dsm) = du.get_images_from_name(path_folder=path_folder, name=name)
    path_sent = os.path.join(path_folder, file_sent) if file_sent is not None else None
    path_alpha = os.path.join(path_folder, file_alpha) if file_alpha is not None else None

    if path_alpha is not None:
        im_loaded_alpha = du.load_tiff(path_alpha, datatype='da')
        im_plot_alpha = im_loaded_alpha
        ## normalise:
        im_plot_alpha.values[im_plot_alpha.values == -np.inf] = np.nan
        im_plot_alpha.values[im_plot_alpha.values == np.inf] = np.nan
        im_plot_alpha = (im_plot_alpha - np.nanmin(im_plot_alpha)) / (np.nanmax(im_plot_alpha) - np.nanmin(im_plot_alpha))
        # im_plot_alpha = np.swapaxes(im_plot_alpha, 2, 1)  # change to (bands, height, width) to (height, width, bands)`

    if path_sent is not None:
        im_loaded_s2 = du.load_tiff(path_sent, datatype='da')
        im_loaded_s2 = np.clip(im_loaded_s2, 0, 3000)
        im_loaded_s2 = im_loaded_s2 / (3000)
        im_plot_s2 = im_loaded_s2[:3, ...]
        im_nir_s2 = im_loaded_s2[1:, ...]
        size_s2 = im_plot_s2.shape[1]
        assert size_s2 == im_plot_s2.shape[2]
        half_size_s2 = size_s2 // 2
        quarter_size_s2 = size_s2 // 4
        # im_plot_s2 = im_plot_s2[:, quarter_size_s2:half_size_s2 + quarter_size_s2, quarter_size_s2:half_size_s2 + quarter_size_s2]
        
    if verbose:
        print(im_loaded_alpha.shape, type(im_loaded_alpha))
        print(im_loaded_s2.shape, type(im_loaded_s2))

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    ax = ax.flatten()

    plot_image_simple(im_plot_s2, ax=ax[0])
    ax[0].set_title('Sentinel-2 RGB')

    for ii in range(8):
        bands = np.arange(ii * 8, (ii + 1) * 8)
        if bands.max() >= im_plot_alpha.shape[0]:
            if ii + 1 < len(ax):
                naked(ax[ii + 1])
            continue

        ax_ind = ii + 1
        curr_ax = ax[ax_ind]
        for jj, band in enumerate(bands):
            if band >= im_plot_alpha.shape[0]:
                continue
            curr_ax.hist(im_plot_alpha[band, ...].to_numpy().flatten(), bins=np.linspace(0, 1, 41), alpha=0.5, label=f'Band {band}',
                         histtype='stepfilled', density=True, color=plt.cm.viridis(jj / len(bands)))
        curr_ax.set_xlim(-0.1, 1.1)

def plot_sent_feat(sentinel_patch, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    ax.imshow(np.clip(np.swapaxes(np.swapaxes(sentinel_patch[:3], 0, 2), 0, 1), 0, 3000) / 3000)

def plot_feature(feat, ax=None, plot_cbar=False, cax=None, lim_zscore=True):
    if lim_zscore:
        lim = 2.5
    else:
        lim = 0.4
    if ax is None:
        im = plt.imshow(feat, cmap='BrBG', vmin=-lim, vmax=lim, interpolation='none')
        plt.axis('off')
    else:
        im = ax.imshow(feat, cmap='BrBG', vmin=-lim, vmax=lim, interpolation='none')
        naked(ax)
        # ax.axis('off')
    if plot_cbar:
        cbar = ax.figure.colorbar(im, cax=cax, ax=ax, location='left', fraction=0.046, pad=0.04,
                                  ticks=[-lim, 0, lim])
        cbar.set_label('Embed.\n(z-scored)' if lim_zscore else 'Embed.')



def plot_sta(sta, ax=None, plot_cbar=False, cax=None, lim=0.1, add_centre=True):
    if ax is None:
        ax = plt.subplot(111)
    im = ax.imshow(sta, cmap='BrBG', vmin=-lim, vmax=lim, interpolation='none')
    if add_centre:
        ax.annotate(text='+', xy=(0.5, 0.5), ha='center', va='center', xycoords='axes fraction')
    ax.set_xticks([])
    if plot_cbar:
        cbar = ax.figure.colorbar(im, cax=cax, ax=ax, location='left', fraction=0.046, pad=0.04)
        cbar.set_label('TS value')
    ax.set_yticks([])

def plot_sentinel(img, ax=None, eq_hist=False, clip_im=False):
    assert not (eq_hist and clip_im), "Cannot both equalize histogram and clip image"
    if ax is None:
        ax = plt.subplot(111)
    if type(img) == xr.core.dataarray.DataArray:
        img = img.values
    img_plot = np.swapaxes(np.swapaxes(img[:3], 0, 2), 0, 1)
    if eq_hist:
        img_plot = exposure.equalize_hist(img_plot)
    if clip_im:
        img_plot = np.clip(img_plot, 0, 2000) / 2000
    ax.imshow(img_plot, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_dw_landcover_from_hyp(hyp, fig=None, ax=None, cax=None):
    lc = hyp[:9, ...]
    im = np.argmax(lc, axis=0) 
    if ax is None or fig is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)
    dict_classes = du.create_cmap_dynamic_world()
    cmap_dw = ListedColormap([v for v in dict_classes.values()])
    im = ax.imshow(im, cmap=cmap_dw, interpolation='none', origin='upper', vmax=8.5, vmin=-0.5)
    # Place colorbar outside of ax to avoid shrinking the imshow
    cbar = fig.colorbar(im, ax=ax, cax=cax, ticks=np.arange(0, 9), location='right', fraction=0.046, pad=0.04)
    cbar.ax.set_yticks(np.arange(0, 9))
    cbar.ax.set_yticklabels([k for k in dict_classes.keys()])
    ax.axis('off')

def random_gaussian_blob(size=100):
    x, y = np.mgrid[-3:3:size*1j, -3:3:size*1j]  # grid

    # random mean
    mean = np.random.uniform(-2, 0, 2)

    # random covariance matrix -> elongated shapes
    A = np.random.randn(2, 2) / 2
    cov = np.dot(A, A.T)  # positive semi-definite
    cov += np.diag([2, 0.3])  # encourage elongation

    inv_cov = np.linalg.inv(cov)

    pos = np.dstack((x - mean[0], y - mean[1]))
    blob = np.exp(-0.5 * np.einsum('...i,ij,...j->...', pos, inv_cov, pos))
    return blob / blob.max()  # normalize

def plot_sta_example(ax_top=None, ax_bottom=None):
    if ax_top is None or ax_bottom is None:
        fig, ax = plt.subplots(2, 1, figsize=(3, 6))
        ax_top = ax[0]
        ax_bottom = ax[1]
    
    size = 0.12
    centres_inset = [(0.2, 0.2), (0.7, 0.8), (0.8, 0.3), (0.3, 0.7), (0.5, 0.5), (0.55, 0.2)]
    weights = [0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
    blobs = [random_gaussian_blob() for _ in range(len(centres_inset))]
    ## create cmap from BrBG for range -1 to 1:
    cmap_brbg = plt.get_cmap('BrBG_r')
    for i in range(len(centres_inset)):
        ax_inset = ax_top.inset_axes([centres_inset[i][0] - size, centres_inset[i][1] - size, 2 * size, 2 * size])
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.imshow(blobs[i], cmap='Grays', interpolation='none')
        # ax_inset.annotate(text='+', xy=(0.5, 0.5), ha='center', va='center', xycoords='axes fraction')
        ax_inset.plot(50, 50, 'ro', markersize=4.5, markeredgecolor='k', color=cmap_brbg(weights[i] * 0.5 + 0.5))
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title('Individual\ntuning surfaces', fontsize=10)

    ax_bottom.imshow(np.stack([b * weights[j] for j, b in enumerate(blobs)]).mean(0), 
                     cmap='BrBG_r', interpolation='none', alpha=0.9, vmin=-0.4, vmax=0.4)
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])
    ax_bottom.annotate(text='+', xy=(0.5, 0.5), ha='center', va='center', xycoords='axes fraction',
                       fontsize=10)
    # ax_bottom.plot(50, 50, 'ro', markersize=6, markeredgecolor='k')
    ax_bottom.set_title('Weighted tuning\nsurface (TS)', fontsize=10)

def plot_pca_dim(dict_expl_var, dict_dim, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    for i_n, n in enumerate(dict_expl_var.keys()):
        dim_mean = float(np.mean(dict_dim[n]))
        dim_sem = float(np.std(dict_dim[n]) / np.sqrt(len(dict_dim[n])))
        ax.plot(np.concatenate([[0], dict_expl_var[n].mean(0)]) * 100, '.-', c='k', alpha=(i_n + 1) * 0.15,
                label=f'N_patches = {n}, D={np.round(np.mean(dict_dim[n]), 1)} ' + r"$\pm$ " + f"{np.round(dim_sem, 1)}")


    ax.legend(frameon=False, bbox_to_anchor=(0.3, 0.8), fontsize=8)    
    ax.set_xlabel('# PCs')
    ax.set_ylabel('Expl var (%)')
    return 

def plot_overview_cca_reconstruction(features, X_hat_fit, X_res_fit, sentinel,
                                     list_patch_inds=[4, 15, 16, 20, 40, 80, 96], i_f=42):
    fig, all_ax = plt.subplots(len(list_patch_inds), 6, figsize=(10, 1.5 * len(list_patch_inds)), 
                            gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.05]})

    for i_row, i_patch_example in enumerate(list_patch_inds):
        ax = all_ax[i_row, :]
        for ax_ in ax:
            naked(ax_)

        ax[0].imshow(np.clip(np.swapaxes(np.swapaxes(sentinel[i_patch_example][:3], 0, 2), 0, 1), 0, 2000) / 2000)
        ax[0].set_title(f'F{i_f}, P{i_patch_example}', fontsize=10)

        for ii, (name, im_plot) in enumerate(zip(['original', 'reconstructed', 'residual', 'recon + resid'], [features, X_hat_fit, X_res_fit, X_hat_fit + X_res_fit])):
            plot_feature(im_plot[i_patch_example][i_f], ax=ax[ii + 1], plot_cbar=False, lim_zscore=False)
            corr_with_orig_feat = np.corrcoef(im_plot[i_patch_example][i_f].flatten(), features[i_patch_example][i_f].flatten())[0, 1]
            ax[ii + 1].set_xlabel(f'corr: {corr_with_orig_feat:.3f}', fontsize=8)
            if i_row == 0:
                ax[ii + 1].set_title(name, fontsize=10)

        cbar = plt.colorbar(mappable=ax[1].images[0], ax=ax[1], #fraction=0.046, pad=0.04, 
                            cax=ax[5], ticks=[-0.4, 0, 0.4])
        ax[5].set_ylabel('Embedding', rotation=270)
