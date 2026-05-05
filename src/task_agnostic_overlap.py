import data_utils as du
import numpy as np
import os
import tqdm
import scipy
import re
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import haversine_distances
from statsmodels.stats.multitest import multipletests


### DEFINE UTILITIES ###

# Calculate task-agnostic overlap between embeddings
# This basically compares feature similarity structure across examples
# It's based on https://arxiv.org/pdf/1905.00414 and the accompanying demo
# https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
# For now I'll just copy over their utility functions

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)

def cca(features_x, features_y):
  """Compute the mean squared CCA correlation (R^2_{CCA}).

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.

  Returns:
    The mean squared CCA correlations between X and Y.
  """
  qx, _ = np.linalg.qr(features_x)  # Or use SVD with full_matrices=False.
  qy, _ = np.linalg.qr(features_y)
  return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(
      features_x.shape[1], features_y.shape[1])

def rsa(features_x, features_y):
   """Only function that's not taken from Kornblith et al.
  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.

  Returns:
    The similarity-of-similarities, i.e. the correlation between
    the upper triangles of the example x example correlation matrices
    for the both sets of features

   """
   cx = np.corrcoef(features_x)
   cy = np.corrcoef(features_y)
   upper_triangles = np.stack([m[np.triu_indices(features_x.shape[0],1)] for m in [cx, cy]])
   return np.corrcoef(upper_triangles[:, np.sum(np.isnan(upper_triangles),axis=0)==0])[0,1]

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

# Only included samples that are present across all modalities
common_samples = [[e['id'].to_numpy() for e in emb] for emb in embeddings]
common_samples = [reduce(np.intersect1d, ([s[i] for s in common_samples])) for i in range(len(samples))]
common_embeddings = [[e.set_index("id").loc[inc].reset_index(drop=True).to_numpy() for inc, e in zip(common_samples, emb)] for emb in embeddings]

# Load land cover data
land_cover = [{id: du.load_tiff(os.path.join(data_folder, 'dynamicworld', f'{id}_dynamicworld_y-2024.tif'), datatype='np')
               for id in tqdm.tqdm(s_id)} for s_id in common_samples]
land_cover_names = [k for k in du.create_cmap_dynamic_world().keys()]

### CALCULATE OVERLAP ###
# Choose which sample to plot for (s=0: random; s=1: stratified)
s = 1
sim_cka = np.zeros((len(modalities), len(modalities)))
sim_cca = np.zeros((len(modalities), len(modalities)))
sim_rsa = np.zeros((len(modalities), len(modalities)))

# Bit dumb to calculate all entries of symmetric matrices but whatever
for i, e_from in enumerate(common_embeddings):
    for j, e_to in enumerate(common_embeddings):
        sim_cka[i, j] = feature_space_linear_cka(e_from[s], e_to[s])
        sim_cca[i, j] = cca(e_from[s], e_to[s])
        sim_rsa[i, j] = rsa(e_from[s], e_to[s])
        print(f'Finished {i}, {j}')

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(6, 4), constrained_layout=True)
ims = []
for col, (ax, data, method) in enumerate(zip(
    axs,
    [sim_cka, sim_cca, sim_rsa],
    ['cka', 'cca', 'rsa']
)):
    im = ax.imshow(data, vmin=0, vmax=1)
    ims.append(im)
    ax.set_title(method)
    ax.set_xticks(range(4))
    ax.set_xticklabels(modalities, rotation=90)
    if col == 0:
        ax.set_yticks(range(4))
        ax.set_yticklabels(modalities)
    else:
       ax.set_yticks([])
fig.colorbar(ims[0], ax=axs, location='right', shrink=0.5)
plt.savefig(f'figs/jacob/overlap_measures.pdf')    
plt.savefig(f'figs/jacob/overlap_measures.png')    

### CREATE OVERLAP MAPS ###

# Show representational overlap localised around patches:
# compute across a region of nearest neighbours for each patch
# Get longitude and latitude for all patches
locs = [gdf_points.set_index("id").loc[samp].reset_index(drop=True)[['lat', 'lon']].to_numpy() for samp in common_samples]
loc = locs[s]

# Create a approximate distance matrix between all points
# This is inaccurate but fast; geodesic would be better, but slow
coords_rad = np.radians(loc)
dist_matrix = haversine_distances(coords_rad)

# Select regions around points
region_size = 100
regions = np.zeros((len(loc), region_size), dtype=int)
for p, dist in enumerate(dist_matrix):
    regions[p] = np.argsort(dist)[:region_size]

# Just for illustration, plot a bunch of random regions in random colours
plt.figure(figsize=(6,4));
for r in regions[::100]:
   plt.scatter(loc[r,1], loc[r,0], color=np.random.rand(3))
plt.xlim([-180,180])
plt.ylim([-90,90])
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect('equal')
plt.title('A bunch of random regions')
plt.savefig(f'figs/jacob/overlap_regions.pdf')    
plt.savefig(f'figs/jacob/overlap_regions.png')    

# Calculate cka and rsa for each region
# This time don't do feature-space cka, because there are more features than examples
pairs = [[i, j] for i in range(0,len(modalities)) for j in range(i+1,len(modalities))]
region_cka = np.zeros((len(loc), len(pairs)))
region_rsa = np.zeros((len(loc), len(pairs)))
for r, region in enumerate(regions):
  for p, pair in enumerate(pairs):
    region_cka[r, p] = cka(gram_linear(common_embeddings[pair[0]][s][region]), 
                           gram_linear(common_embeddings[pair[1]][s][region]))
    region_rsa[r, p] = rsa(common_embeddings[pair[0]][s][region], 
                           common_embeddings[pair[1]][s][region])
  if r % 100 == 0:
    print(f'Finished region {r} / {len(regions)}')

# Plot results
for data, sim_type in zip([region_cka, region_rsa], ['cka', 'rsa']):
  fig = plt.figure(figsize=(12,6));
  for p, pair in enumerate(pairs):
    ax = plt.subplot(len(modalities)-1, len(modalities)-1, pair[0] * (len(modalities)-1) + (pair[1]-1) + 1)
    ax.set_aspect('equal')
    plt.xlim([-180,180])
    plt.ylim([-90,90])
    plt.scatter(loc[:,1], loc[:,0], 2, np.concatenate([np.clip(data[:,p][:,None], 0, 1), np.zeros((len(data),2))], -1))
    plt.title(f'{modalities[pair[0]]}, {modalities[pair[1]]}')
    plt.xticks([])
    plt.yticks([])
  plt.tight_layout()
  plt.savefig(f'figs/jacob/overlap_{sim_type}_maps.pdf')    
  plt.savefig(f'figs/jacob/overlap_{sim_type}_maps.png')    

### CALCULATE CORRELATION DISTANCES ###

# Get correlation matrix across all pairs of samples
s = 1
all_corr_mats = np.stack([np.corrcoef(e[s]) for e in common_embeddings])
# And get a similarity between land cover too
lc_pix = np.stack([land_cover[s][i][:,64,64] for i in common_samples[s]]).transpose()
lc_pix_z = (lc_pix - np.mean(lc_pix, axis=-1, keepdims=True)) / np.std(lc_pix, axis=-1, keepdims=True)
lc_sim_mats = np.stack([1-np.square(lc[:,None] - lc[None,:]) for lc in lc_pix])
lc_weight_mats = np.stack([np.maximum(lc[:,None], lc[None,:]) for lc in lc_pix])

# I want just the upper triangle, both for distance and correlation; add radius for dist in km
dist_utri = dist_matrix[np.triu_indices(all_corr_mats.shape[-1],1)] * 6371.0
all_corr_utri = np.stack([m[np.triu_indices(all_corr_mats.shape[-1],1)] for m in all_corr_mats])
lc_sim_utri = np.stack([m[np.triu_indices(lc_sim_mats.shape[-1],1)] for m in lc_sim_mats])
lc_weight_utri = np.stack([m[np.triu_indices(lc_weight_mats.shape[-1],1)] for m in lc_weight_mats])

# Store the fitted characteristic distances
entropy_dist = []

# I could just plot all points, i.e. dist vs corr, but there are order 10k^2 so it's too many
# Instead, plot as subsample of points and make a heatmap of all of them
for sim_name, curr_sim, sim_lim, sim_names, sim_type, pair_weights in zip(
  ['emb', 'lc'], 
  [all_corr_utri, lc_sim_utri], 
  [[-1,1],[0,1]], 
  [modalities, land_cover_names], 
  ['Correlation','1 - Diff Sqrd'],
  [np.ones_like(all_corr_utri), lc_weight_utri]):
  # Plot at various distance cutoffs, which show the relevant scales
  for dist_cutoff in [1000]:#, 5000, np.max(dist_utri).astype(int)]:
    plt.figure(figsize=(len(sim_names)*1.5,4))
    for e, (points, weights, name) in enumerate(zip(curr_sim, pair_weights, sim_names)):
      include = dist_utri < dist_cutoff
      # Create a 2d histogram with similarity on y-ax and distance on x-ax
      hist = np.histogram2d(dist_utri[include], points[include], weights=weights[include], 
                            bins=100, range=[[0, np.max(dist_utri[include])], sim_lim], 
                            density=True)
      # First subplot: scatter plot of subsampled pairs
      plt.subplot(2, len(sim_names), e+1)
      steps = int(np.sum(include)/1e4)
      plt.plot(dist_utri[include][::steps], points[include][::steps], 'k.', markersize=1)
      plt.xlim([hist[1][0], hist[1][-1]])
      plt.ylim([hist[2][0], hist[2][-1]])
      if e == 0:
        plt.ylabel(sim_type)
        plt.yticks(np.linspace(sim_lim[0], sim_lim[1], 3))
      else:
        plt.yticks([])
      plt.xticks([])
      plt.title(name)
      # Second subplot: heatmap of log density
      ax1 = plt.subplot(2,len(sim_names), len(sim_names) + e+1)
      ax1.imshow(np.log(hist[0].T),
                interpolation='none',
                origin='lower',
                extent=[hist[1][0], hist[1][-1], hist[2][0], hist[2][-1]])   
      ax1.set_aspect('auto')
      # Plot the entropy on top
      ax2 = ax1.twinx()
      p = hist[0].T / np.sum(hist[0].T, axis=0, keepdims=True)
      entropy = -np.sum(p*np.log(np.clip(p, 1e-12, 1)), axis=0)
      E = np.sum(p*hist[2][:-1][:,None], axis=0)
      ax2.plot(hist[1][:-1], entropy/np.max(entropy), 'r-')
      ax2.plot(hist[1][:-1], E, 'k:')
      ax2.set_ylim([0,1])
      ax2.tick_params(axis="y", colors="red")
      ax2.spines["right"].set_color("red")            
      # Only for short distance cutoff: fit the entropy increase
      if dist_cutoff < 2000:
        pars = scipy.optimize.curve_fit(lambda t,d: (entropy[0]/np.max(entropy)-1)*np.exp(-t/d)+1,  
                                        hist[1][:-1] / hist[1][-2],  
                                        entropy/np.max(entropy),
                                        p0=[100/hist[1][-2]],
                                        maxfev=int(1e5)
                                        )[0]
        ax2.plot(hist[1][:-1], (entropy[0]/np.max(entropy)-1)* np.exp(-hist[1][:-1] / (pars[0] * hist[1][-2])) + 1, 'b:')  
        entropy_dist.append(pars[0]*hist[1][-1])      
        plt.title(label=f'd = {pars[0]*hist[1][-1]:.0f} km')
      # Annotate both axes
      ax1.set_yticks(np.linspace(sim_lim[0], sim_lim[1], 3), [])
      ax2.set_yticks(np.linspace(sim_lim[0], sim_lim[1], 3), [])
      if e == 0:
        ax1.set_ylabel(sim_type)
        ax1.set_yticks(np.linspace(sim_lim[0], sim_lim[1], 3), np.linspace(sim_lim[0], sim_lim[1], 3))
      if e == len(sim_names)-1:
        ax2.set_ylabel('Entropy / Max Entropy', color='red')
        ax2.set_yticks(np.linspace(sim_lim[0], sim_lim[1], 3), np.linspace(sim_lim[0], sim_lim[1], 3))  
      plt.xticks(np.linspace(0, np.max(dist_utri[include]), 3), [f'{d/1000:0.1f}k' for d in np.linspace(0, np.max(dist_utri[include]), 3)])
      plt.xlabel('Distance (km)')    
      plt.tight_layout()
      plt.savefig(f'figs/jacob/dist_{sim_name}_{dist_cutoff}.pdf')    
      plt.savefig(f'figs/jacob/dist_{sim_name}_{dist_cutoff}.png')    

### FIND CORRELATION BETWEEN LC SCALE AND PERFORMANCE ###

# Parse latex tables into numpy arrays (warning: LLM-generated)
# Table 5: land-cover specific R2
r2_scores = r"""
\toprule
Embeddings & \textit{Water} & \textit{Trees} & \textit{Grass} & \textit{Flood.} & \textit{Crops} & \textit{Shrub} & \textit{Built} & \textit{Bare} & \textit{Snow} \\
\midrule
alphaearth & \textbf{89.9 ± 0.4} & \textbf{70.8 ± 0.6} & 63.6 ± 0.9 & 35.1 ± 0.7 & 67.6 ± 0.6 & 56.0 ± 0.8 & \textbf{81.3 ± 0.6} & 92.0 ± 0.4 & 86.2 ± 0.5 \\
tessera & 85.1 ± 0.8 & 65.9 ± 0.7 & \textbf{68.3 ± 0.9} & \textbf{37.1 ± 0.8} & \textbf{67.9 ± 0.7} & \textbf{56.2 ± 0.8} & 72.1 ± 0.6 & \textbf{92.0 ± 0.4} & \textbf{88.1 ± 0.5} \\
geoclip & 4.3 ± 1.7 & 16.1 ± 0.9 & 25.5 ± 1.1 & 7.3 ± 0.9 & 25.8 ± 1.0 & 18.7 ± 1.2 & 19.1 ± 0.9 & 77.1 ± 1.1 & 83.6 ± 0.7 \\
satclip & 8.2 ± 1.5 & 20.3 ± 1.1 & 31.5 ± 1.2 & 13.8 ± 0.7 & 32.0 ± 1.0 & 24.0 ± 1.3 & 19.2 ± 1.0 & 75.8 ± 1.0 & 80.5 ± 0.7 \\ \midrule
\bottomrule
"""
# Table 6: land cover specific complementarity
complementarity_scores = r"""
\toprule
Embeddings & \textit{Water} & \textit{Trees} & \textit{Grass} & \textit{Flood.} & \textit{Crops} & \textit{Shrub} & \textit{Built} & \textit{Bare} & \textit{Snow} \\
\midrule
alphaearth + tessera & \textbf{0.19**} & \textbf{0.15**} & \textbf{0.18**} & \textbf{0.11**} & \textbf{0.24**} & \textbf{0.18**} & \textbf{0.16**} & \textbf{0.22**} & \textbf{0.09**} \\
alphaearth + geoclip & 0.00 & -0.01 & 0.02 & -0.01 & \textbf{0.04**} & 0.01 & 0.00 & 0.02 & \textbf{0.10**} \\
alphaearth + satclip & \textbf{0.04**} & \textbf{0.04**} & \textbf{0.07**} & \textbf{0.03**} & \textbf{0.09**} & \textbf{0.07**} & \textbf{0.03**} & \textbf{0.08**} & \textbf{0.13**} \\
tessera + geoclip & -0.03 & -0.02 & 0.02 & -0.02 & 0.02 & 0.01 & -0.03 & 0.01 & \textbf{0.07**} \\
tessera + satclip & 0.00 & \textbf{0.03**} & \textbf{0.07**} & \textbf{0.02*} & \textbf{0.07**} & \textbf{0.06**} & 0.00 & \textbf{0.05**} & \textbf{0.09**} \\
geoclip + satclip & -0.02 & -0.01 & 0.01 & -0.04 & -0.01 & 0.00 & 0.01 & \textbf{0.27**} & \textbf{0.11**} \\ \midrule
All GFMs & \textbf{0.15**} & \textbf{0.12**} & \textbf{0.18**} & \textbf{0.07**} & \textbf{0.25**} & \textbf{0.17**} & \textbf{0.12**} & \textbf{0.23**} & \textbf{0.17**} \\
\bottomrule"""

def clean_cell(cell):
    # remove LaTeX formatting
    cell = re.sub(r'\\textbf\{([^}]*)\}', r'\1', cell)
    cell = re.sub(r'\\textit\{([^}]*)\}', r'\1', cell)
    # remove ± parts if present
    cell = re.sub(r'±.*', '', cell)
    # remove * or ** markers
    cell = re.sub(r'\*+', '', cell)
    return cell.strip()

# Collect data and find correlation with land cover spatial scale
r_dicts, p_vals = [], []
for latex in [r2_scores, complementarity_scores]:
  rows = []
  for line in latex.splitlines():
      line = line.strip()
      if not line:
          continue
      # remove all LaTeX table commands anywhere in the line
      line = re.sub(r'\\(toprule|midrule|bottomrule)', '', line)
      # remove trailing \\ if present
      line = line.replace('\\\\', '').strip()
      if not line:
          continue
      cells = [clean_cell(c) for c in line.split('&')]
      rows.append(cells)
  header = rows[0]
  data = rows[1:]
  names = [row[0] for row in data]
  values = np.array([[float(x) for x in row[1:]] for row in data])

  # Calculate correlations with land cover spatial scale
  r_dicts.append({n: scipy.stats.spearmanr(v, np.stack([e for e in entropy_dist[4:]]), alternative='greater') for n, v in zip(names,values)})
  p_vals.append({n: v.pvalue for n,v in r_dicts[-1].items()})

# Correct p-vals for multiple comparisons
corr_p_vals = [{k: v for k, v in zip(ps.keys(), multipletests([v for v in ps.values()], method='fdr_bh')[1])} for ps in p_vals]
