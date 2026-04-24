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
                       method='regression', metric='r2', kwargs_for_method={}, verbose=0):

    if regressor_list is None:
        regressor_list = list(col_names.keys())
    if target_list is None:
        target_list = list(col_names.keys())

    assert all(r in col_names for r in regressor_list), 'Some regressors not found in col_names.'
    assert all(t in col_names for t in target_list), 'Some targets not found in col_names.'

    overlap_matrix = np.zeros((len(regressor_list), len(target_list)))
    if verbose > 0:
        print(f'There are {len(regressor_list)} regressors and {len(target_list)} targets. Total combinations: {len(regressor_list) * len(target_list)}.')
    if method == 'regression':
        for i, regressor in enumerate(regressor_list):
            if verbose > 0:
                print(f"Regressor: {regressor}. {i+1}/{len(regressor_list)}")
            for j, target in enumerate(target_list):
                r2, mse, _, _ = get_r2_regression(df_all, col_names, regressor, target,
                                                               **kwargs_for_method)
                if metric == 'r2':
                    overlap_matrix[i, j] = r2
                elif 'mse' in metric:
                    overlap_matrix[i, j] = mse
                else:
                    raise ValueError(f'Metric {metric} not supported.')
    elif method == 'classification':
        for i, regressor in enumerate(regressor_list):
            if verbose > 0:
                print(f"Regressor: {regressor}. {i+1}/{len(regressor_list)}")
            for j, target in enumerate(target_list):
                acc, _ = get_accuracy_classification(df_all, col_names, regressor, target,
                                                     **kwargs_for_method)
                # print(f"Accuracy for regressor {regressor} and target {target}: {acc}. {kwargs_for_method}")
                overlap_matrix[i, j] = acc
    else:
        raise ValueError(f'Method {method} not supported.')
    
    if metric == 'mse_normalised':
        overlap_matrix = overlap_matrix / np.max(overlap_matrix, axis=0, keepdims=True)  

    return overlap_matrix

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
        
    if n_splits > 1:
        rs = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        mse_per_point = np.zeros(len(df_all))
        r2 = np.zeros(n_splits)
        Y_pred = np.zeros_like(data_target)
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
            mse_per_point[test_index] = np.mean((Y_test - Y_pred[test_index]) ** 2, axis=1)
        
    elif n_splits == 1:
        X = data_regressor
        Y = data_target
        if regression_method == 'linear':
            reg = LinearRegression().fit(X, Y)
        elif regression_method == 'ridge':
            reg = Ridge(alpha=1.0).fit(X, Y)
        elif regression_method == 'truncated_svd':
            reg = TruncatedSVD(n_components=64).fit(X, Y)
        Y_pred = reg.predict(X)
        mse_per_point = np.mean((Y - Y_pred) ** 2, axis=1)
    r2 = r2_score(data_target, Y_pred)
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
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1 / n_splits, random_state=0)
    accuracies = []
    for train_index, test_index in sss.split(data_regressor, data_target):
        X_train, X_test = data_regressor[train_index], data_regressor[test_index]
        y_train, y_test = data_target[train_index], data_target[test_index]
        if method == 'logistic_regression':
            clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            accuracies.append(acc)
        else:
            raise ValueError(f'Method {method} not supported.')
    return np.mean(accuracies), accuracies


def get_dim(im):
    assert im.shape[0] > im.shape[1], f'Number of samples {im.shape[0]} should be greater than number of features {im.shape[1]} for PCA to work properly.'
    pca = PCA(n_components=im.shape[1])
    pca.fit(im)
    sum_squares = np.sum(np.power(pca.explained_variance_, 2))
    square_sum = np.sum(pca.explained_variance_) ** 2
    dim = float(square_sum / sum_squares)
    return dim, pca

def get_list_dims(parent_folder, sample_type='lc_stratified_sample', modality='alphaearth'):
    list_ids, modality_folders, gdf_points = du.get_list_complete_ids(parent_folder)
    if modality == 'alphaearth':
        suffix = '_alphaearth_y-2024.tif'
    else:
        raise ValueError(f'Modality {modality} not supported.')
    dict_results = {x: [] for x in ['id', 'dim']}
    for id_patch in tqdm(list_ids):
        path_modality = os.path.join(modality_folders[modality], f'{id_patch}{suffix}')
        if os.path.exists(path_modality):
            im = du.load_tiff(path_modality, datatype='np')
            im = im.reshape(im.shape[0], -1).T
            dim, _ = get_dim(im)

            dict_results['id'].append(int(id_patch))
            dict_results['dim'].append(dim)
    return dict_results