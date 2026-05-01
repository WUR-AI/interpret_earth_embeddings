import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.stats import zscore
import os
from tqdm import tqdm
import data_utils as du


def get_overlap_matrix(df_all, col_names, regressor_list=None, target_list=None,
                       method='regression', kwargs_for_method={}, verbose=0):

    if regressor_list is None:
        regressor_list = list(col_names.keys())
    if target_list is None:
        target_list = list(col_names.keys())

    assert all(r in col_names for r in regressor_list), 'Some regressors not found in col_names.'
    assert all(t in col_names for t in target_list), 'Some targets not found in col_names.'

    overlap_matrix_r2 = np.zeros((len(regressor_list), len(target_list)))
    overlap_matrix_mse = np.zeros((len(regressor_list), len(target_list)))

    if verbose > 0:
        print(f'There are {len(regressor_list)} regressors and {len(target_list)} targets. Total combinations: {len(regressor_list) * len(target_list)}.')
    if method == 'regression':
        for i, regressor in enumerate(regressor_list):
            if verbose > 0:
                print(f"Regressor: {regressor}. {i+1}/{len(regressor_list)}")
            for j, target in enumerate(target_list):
                r2, mse, _, __ = get_r2_regression(df_all, col_names, regressor, target,
                                                               **kwargs_for_method)
                overlap_matrix_r2[i, j] = r2
                overlap_matrix_mse[i, j] = mse

    elif method == 'classification':
        for i, regressor in enumerate(regressor_list):
            if verbose > 0:
                print(f"Regressor: {regressor}. {i+1}/{len(regressor_list)}")
            for j, target in enumerate(target_list):
                acc, mse, _, __ = get_accuracy_classification(df_all, col_names, regressor, target,
                                                                     **kwargs_for_method)
                overlap_matrix_r2[i, j] = acc
                overlap_matrix_mse[i, j] = mse
    else:
        raise ValueError(f'Method {method} not supported.')
    
    # if metric == 'mse_normalised':
        # overlap_matrix_mse = overlap_matrix_mse / np.max(overlap_matrix_mse, axis=0, keepdims=True)  

    return overlap_matrix_r2, overlap_matrix_mse

def get_r2_regression(df_all, col_names, regressor, target, n_splits=4, equalize_ambient_dim=False,
                      regression_method='ridge', zscore_embeddings=False):
    df_all = df_all.copy()
    data_regressor = df_all[col_names[regressor]].values
    data_target = df_all[col_names[target]].values
    # print(f"Regressor: {regressor}, Target: {target}, Regressor shape: {data_regressor.shape}, Target shape: {data_target.shape}")
    assert data_regressor.shape[0] == data_target.shape[0]
    if regression_method == 'ridge' and not zscore_embeddings:
        print(f'Warning: It is recommended to z-score the embeddings when using ridge regression. Consider setting zscore_embeddings=True for better performance.')
    if zscore_embeddings:
        data_regressor = zscore(data_regressor, axis=0)
        data_target = zscore(data_target, axis=0)
    if equalize_ambient_dim and regressor != 'dynamicworld' and target != 'dynamicworld':
        if data_regressor.shape[1] > data_target.shape[1]:
            pca = PCA(n_components=data_target.shape[1])
            data_regressor = pca.fit_transform(data_regressor)
        elif data_target.shape[1] > data_regressor.shape[1]:
            pca = PCA(n_components=data_regressor.shape[1])
            data_target = pca.fit_transform(data_target)
        
    assert n_splits > 1

    rs = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_per_point = np.zeros(len(df_all))
    r2 = np.zeros(n_splits)
    Y_pred = np.zeros_like(data_target)
    var_target = np.var(data_target, axis=0)
    # print(f'Variance of target {target}: {var_target}')
    for i, (train_index, test_index) in enumerate(rs.split(df_all)):
        
        ## All samples x features
        X_train = data_regressor[train_index]
        Y_train = data_target[train_index]
        X_test = data_regressor[test_index]
        Y_test = data_target[test_index]
        if regression_method == 'linear':
            reg = LinearRegression().fit(X_train, Y_train)
            pred = reg.predict(X_test)
        elif regression_method == 'ridge':
            reg = Ridge(alpha=1.0).fit(X_train, Y_train)
            pred = reg.predict(X_test)
        elif regression_method == 'truncated_svd':
            pass
            # svd = TruncatedSVD(n_components=64).fit(X_train, Y_train)
            # weights = svd.transform(X_test)
            # components = svd.components_
            # X_test_pred = weights @ components
            # print(pred.shape, Y_pred[test_index].shape)
        if len(pred.shape) == 1:
            pred = pred[:, np.newaxis]
        Y_pred[test_index] = pred
        mse_per_point[test_index] = np.mean((Y_test - Y_pred[test_index]) ** 2 / var_target, axis=1)
        
    r2 = r2_score(data_target, Y_pred, multioutput='variance_weighted')
    mean_mse = np.mean(mse_per_point)
    residuals = data_target - Y_pred
    df_all[f'{regressor}_to_{target}_mse'] = mse_per_point
    return np.mean(r2), mean_mse, df_all, {'target': data_target, 'predictions': Y_pred, 'residuals': residuals, 'mse_per_point': mse_per_point}

def get_accuracy_classification(df_all, col_names, regressor, target: str, n_splits=4, 
                                zscore_embeddings=False, method='logistic_regression'):
    
    assert type(target) == str, 'Target should be a string representing the column name in col_names.'
    assert target in df_all.columns, f'Target {target} not found in df_all.'
    assert n_splits > 1, 'n_splits should be greater than 1 for classification to work properly.'
    data_regressor = df_all[col_names[regressor]].values
    # print(f'nans: {np.isnan(data_regressor).sum()} out of {data_regressor.size} values in regressor {regressor}.')
    # print(f'infs : {np.isinf(data_regressor).sum()} out of {data_regressor.size} values in regressor {regressor}.')
    if zscore_embeddings:
        data_regressor = zscore(data_regressor, axis=0)
        
    ## map target to integers
    unique_classes = df_all[target].unique()
    class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
    df_all[f'{target}_int'] = df_all[target].map(class_to_int)
    data_target = df_all[f'{target}_int'].values

    ## create stratified splits
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1 / n_splits, random_state=42)
    accuracies = []
    mse_per_point = np.zeros(len(df_all))
    for train_index, test_index in sss.split(data_regressor, data_target):
        X_train, X_test = data_regressor[train_index], data_regressor[test_index]
        y_train, y_test = data_target[train_index], data_target[test_index]
        if method == 'logistic_regression':
            clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            accuracies.append(acc)
            pred = clf.predict(X_test)
            mse_per_point[test_index] = np.nan
            # print(pred.shape, y_test.shape, pred[:10], y_test[:10])
        else:
            raise ValueError(f'Method {method} not supported.')
    df_all[f'{regressor}_to_{target}_mse'] = mse_per_point
    mean_mse = np.mean(mse_per_point)
    return np.mean(accuracies), mean_mse, df_all, {'target': data_target, 'predictions': pred, 'mse_per_point': mse_per_point}


def get_dim(im):
    assert im.shape[0] > im.shape[1], f'Number of samples {im.shape[0]} should be greater than number of features {im.shape[1]} for PCA to work properly.'
    if np.isnan(im).sum() > 0:
        print(f'Warning: There are {np.isnan(im).sum()} NaN values in the input data. PCA will not work properly with NaNs. Consider imputing or removing NaNs before calling get_dim.')
        return np.nan, None
    if np.isinf(im).sum() > 0:
        print(f'Warning: There are {np.isinf(im).sum()} Inf values in the input data. PCA will not work properly with Infs. Consider imputing or removing Infs before calling get_dim.')
        return np.nan, None
    pca = PCA(n_components=im.shape[1])
    pca.fit(im)
    sum_squares = np.sum(np.power(pca.explained_variance_, 2))
    square_sum = np.sum(pca.explained_variance_) ** 2
    dim = float(square_sum / sum_squares)
    return dim, pca

def get_list_dims(parent_folder, sample_type='lc_stratified_sample', modality='alphaearth',
                  save_results=False, dir_save=None):
    if save_results:
        assert dir_save is not None, 'If save_results is True, path_save should be provided.'
        assert os.path.exists(dir_save) and os.path.isdir(dir_save), f'Directory {dir_save} does not exist.'
    list_ids, modality_folders, gdf_points = du.get_list_complete_ids(parent_folder)
    assert modality in modality_folders, f'Modality {modality} not found in modality_folders. Available modalities: {list(modality_folders.keys())}.'
    if modality == 'alphaearth':
        suffix = '_alphaearth_y-2024.tif'
    elif modality == 'tessera':
        suffix = '_tessera_y-2024.tif'
    else:
        raise ValueError(f'Modality {modality} not supported.')
    dict_results = {x: [] for x in ['id', 'dim']}
    it = 0
    for id_patch in list_ids:
        it += 1
        if it % 100 == 0:
            print(f'Processing patch {id_patch}. {it}/{len(list_ids)} patches processed.')
        path_modality = os.path.join(modality_folders[modality], f'{id_patch}{suffix}')
        if os.path.exists(path_modality):
            im = du.load_tiff(path_modality, datatype='np')
            im = im.reshape(im.shape[0], -1).T
            dim, _ = get_dim(im)

            dict_results['id'].append(int(id_patch))
            dict_results['dim'].append(dim)
    df_results = pd.DataFrame(dict_results)
    if save_results:
        fname = f'{modality}_dims_{sample_type}.csv'
        path_save = os.path.join(dir_save, fname)
        df_results.to_csv(path_save, index=False)
    return df_results

def calculate_complementarity(df_scores, metric_type='mse', aggr='sum',
                              gfm_mod=['alphaearth', 'tessera', 'geoclip', 'satclip']):
    cols = list(df_scores.columns)

    df_compl = {col: [] for col in cols}
    models = df_scores.index

    if metric_type == 'mse':
        metric_best = 0
    elif metric_type == 'r2':
        metric_best = 1

    for col in cols:
        arr_scores = df_scores[col].values
        dict_scores = {} 
        name_list = []
        for model, score in zip(models, arr_scores):
            if model in gfm_mod:
                dict_scores[model] = score
                continue
            elif '+' in model:
                models_in_combination = model.split('+')
                models_in_combination = [m.strip() for m in models_in_combination]
            elif model == 'All GFMs':
                models_in_combination = gfm_mod
            else:
                raise ValueError(f'Model name {model} not recognized. Should be either a single model in gfm_mod, a combination of models separated by " +\n", or "All GFMs".')
            assert all(m in gfm_mod for m in models_in_combination), f'Models in combination {model} not all in gfm_mod.'
            if metric_type == 'mse':
                best_individual_score = min([dict_scores[m] for m in models_in_combination])
            elif metric_type == 'r2':
                best_individual_score = max([dict_scores[m] for m in models_in_combination])
            compl = (score - best_individual_score) / (metric_best - best_individual_score) if metric_best != best_individual_score else 0
            df_compl[col].append(compl)
            name_list.append(model)
    df_compl = pd.DataFrame(df_compl, index=name_list)
    if aggr == 'sum':
        df_compl['Sum'] = df_compl.sum(axis=1)
    elif aggr == 'mean':
        df_compl['Mean'] = df_compl.mean(axis=1)
    return df_compl
        
    