import polars as pl
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier
from boruta import BorutaPy
from PyImpetus import PPIMBC

import os
import pickle
import time
from tqdm.notebook import tqdm

from pipeline.model import train_xgboost
from pipeline.utils import filter_module


def fs_mutual_information(X_fs, y_fs, seed):
    # mutual information
    mi_measures = mutual_info_classif(X_fs, y_fs, random_state=seed, n_jobs=-1)

    # plot mutual information distribution
    sns.histplot(mi_measures)
    plt.show()

    # determine threshold
    third_quartile = np.quantile(mi_measures, 0.75)

    # select features
    mask = (mi_measures > third_quartile)[:, None]
    col_idx = np.where(np.any(mask, axis=1))[0]

    selected_features = X_fs.iloc[:, col_idx].columns
    return selected_features


def fs_markov_blanket(X_train, y_train, protocols, seed, thresholds=[0.01, 0.001], val_size=0.1, verbose=0):
    predictor = DecisionTreeClassifier(random_state=seed)

    ### first, select features for each protocol ###
    selected_features = []
    for p in (pbar := tqdm(protocols)):
        cols = filter_module(X_train.columns, p)
        X_train_protocol = X_train[cols].reset_index(drop=True)

        pbar.set_postfix(protocol=p, num_feats=X_train_protocol.shape[1])
        pbar.refresh()

        model = PPIMBC(
            model=predictor,
            p_val_thresh=thresholds[0],
            num_simul=5,
            simul_size=val_size,  # validation set size
            simul_type=0,
            sig_test_type='parametric',
            cv=2,
            n_jobs=-1,
            random_state=seed,
            verbose=verbose,
        )
        model.fit_transform(X_train_protocol, y_train.values)
        selected_features.extend(model.MB)

    print('Intermediate number of selected features:', len(selected_features))
    print('Running final selection...')

    ### finally, select the features globally ###
    model = PPIMBC(
        model=predictor,
        p_val_thresh=thresholds[1],
        num_simul=5,
        simul_size=val_size,  # validation set size
        simul_type=0,
        sig_test_type='parametric',
        cv=2,
        n_jobs=-1,
        random_state=seed,
        verbose=verbose,
    )
    model.fit_transform(X_train[selected_features].T.reset_index(drop=True).T, y_train.values)

    # select features
    selected_features = X_train[selected_features].iloc[:, model.MB].columns

    # plot mutual information distribution
    model.MB = selected_features
    model.feature_importance()
    plt.show()

    return selected_features


def fs_boruta(X_fs, y_fs, seed, max_iter=30, n_estimators=100, verbose=2):
    # boruta
    clf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    boruta = BorutaPy(clf, max_iter=max_iter, n_estimators=n_estimators, random_state=seed, verbose=verbose)
    boruta.fit_transform(X_fs.values, y_fs.values)

    # select features
    cols_selected = boruta.support_.tolist()
    selected_features = X_fs.iloc[:, cols_selected].columns
    return selected_features


def fs_rfe(X_fs, y_fs, n_features_to_select, seed):
    # recursive feature elimination
    clf = DecisionTreeClassifier(random_state=seed)
    selector = RFE(clf, n_features_to_select=n_features_to_select, step=2)
    selector = selector.fit(X_fs, y_fs)

    # select features
    selected_features = X_fs.iloc[:, selector.support_].columns
    return selected_features


def random_features(X_train, y_train, X_valid, y_valid, n_features_to_select, model_save_dir, data_seed, ml_seed):
    random_results = []
    for idx, features_seed in enumerate(range(0, 300, 10)):
        random_cols = X_train.sample(n_features_to_select, axis=1, random_state=features_seed).columns
        print(f'\tRound: {idx} Selected features: {random_cols}')
        df_results = train_xgboost(
            X_train.loc[:, random_cols], y_train, 
            X_valid.loc[:, random_cols], y_valid, 
            seed=ml_seed,
            method_name=f'Random {n_features_to_select} features', 
            save_dir=model_save_dir,
            filename=f'random-{n_features_to_select}_dataseed{data_seed}.pkl',
            verbose=0
        )
        random_results.append(df_results)
    df_results = pd.concat(random_results, axis=0)
    return df_results