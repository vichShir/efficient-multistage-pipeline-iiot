import polars as pl
import pandas as pd
import numpy as np
import argparse
import time
import gc
import os
os.path.insert(0, '../')

from multistage_pipeline.preprocessing import (
    create_splits, 
    transform_labels, 
    get_labels_mapping,
    balance_labels_with_smote
)
from multistage_pipeline.feature_selection import (
    fs_mutual_information,
    fs_markov_blanket,
    fs_boruta,
    fs_rfe,
    random_features
)
from multistage_pipeline.model import train_xgboost
from multistage_pipeline.training_size import training_size
from multistage_pipeline.utils import create_data_dirs, get_protocols, filter_module

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


MODEL_SAVE_DIR_TS = '../data/models/training_size'  # training size models
MODEL_SAVE_DIR_FS = '../data/models/feature_selection'  # feature selection models


def load_data():
	# load dataset with Polars
	df = pl.read_csv('../data/datasets/BRUIIoT.csv')
	print('Original dataset size:', df.shape)
	print(f'There are {df.shape[0]} total observations.')

	# create splits
	print('\nSplits division:')
	df_train, df_fs, df_valid, df_test = create_splits(df, '../data/datasets/splits', seed=SEED)

	# clear memory
	del df
	gc.collect()

	return df_train, df_fs, df_valid, df_test


def preprocess_data(df_train, df_fs, df_valid, df_test):
	# transform labels to normal, mirai, icmp, syn and others
	df_train = transform_labels(df_train)
	df_fs = transform_labels(df_fs)
	df_valid = transform_labels(df_valid)
	df_test = transform_labels(df_test)

	# plotting
	df_labels_original = df_train['attack_label__most_frequent'].value_counts().to_pandas()
	df_labels_original['type'] = 'Original'
	df_labels_original = df_labels_original.sort_values(by='count', ascending=False)

	# balance training labels (SMOTE)
	start_time = time.perf_counter()
	df_train = balance_labels_with_smote(df_train, seed=SEED)
	end_time = time.perf_counter()
	smote_time = end_time - start_time

	# plotting
	df_labels_balanced = pd.DataFrame({
	    'attack_label__most_frequent': df_labels_original['attack_label__most_frequent'],
	    'count': df_train['attack_label_enc__most_frequent'].value_counts().to_list()
	})
	df_labels_balanced['type'] = 'Balanced'

	df_labels_plot = pd.concat([df_labels_original, df_labels_balanced], axis=0)
	
	ax = sns.barplot(df_labels_plot, x='attack_label__most_frequent', y='count', hue='type')
	ax.bar_label(ax.containers[0], fontsize=9)
	ax.bar_label(ax.containers[1], fontsize=9)
	ax.tick_params(left=False)
	ax.set(yticklabels=[])

	sns.despine(top=True, right=True, left=True, bottom=False)
	plt.title('Attack types distribution - Training set')
	plt.xlabel('')
	plt.ylabel('')
	plt.xticks(rotation=30)
	plt.tight_layout()
	plt.savefig(f"../data/results/attack_distribution_training.png", bbox_inches='tight')
	plt.close()

	# save preprocessed splits to disk
	print('Saving data splits to disk. This may take a while...')
	if not os.path.exists(f'../data/datasets/splits/BRUIIoT_train_preprocessed_seed{SEED}.csv'):
		df_train.to_csv(f'../data/datasets/splits/BRUIIoT_train_preprocessed_seed{SEED}.csv', index=False)
		df_fs.write_csv(f'../data/datasets/splits/BRUIIoT_fs_preprocessed_seed{SEED}.csv')
		df_valid.write_csv(f'../data/datasets/splits/BRUIIoT_valid_preprocessed_seed{SEED}.csv')
		df_test.write_csv(f'../data/datasets/splits/BRUIIoT_test_preprocessed_seed{SEED}.csv')

	del df_train, df_fs, df_valid, df_test
	gc.collect()

	return smote_time


def determine_training_size(df_train, df_valid, targets_cols, labels, n_jobs=2):
	# get features and targets from validation
	X_valid = df_valid.drop(columns=targets_cols+['index'])
	y_valid = df_valid.loc[:, 'attack_label_enc__most_frequent']

	print('Training size:', df_train.shape)
	print('Validation features size:', X_valid.shape)
	print('Validation labels size:', y_valid.shape)

	# run training size variation
	start_time = time.perf_counter()
	df_results = training_size(
	    df_train, 
	    X_valid, y_valid, 
	    seed=SEED,
	    model_save_dir=MODEL_SAVE_DIR_TS,
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	training_size_time = end_time - start_time

	# save results
	df_results.to_csv(f'../data/results/training_size/training_size_results_seed{SEED}.csv', index=False)

	# plotting
	left_color = 'dodgerblue'
	right_color = 'royalblue'

	plt.figure(figsize=(8, 4))

	sns.lineplot(df_results, y='accuracy', x='model', label='Accuracy', linewidth=2, marker='o', markersize=7, color=left_color)
	sns.lineplot(df_results, y='precision', x='model', label='Precision', linewidth=2, marker='D', color=left_color)
	sns.lineplot(df_results, y='recall', x='model', label='Recall', linewidth=2, marker='s', color=left_color)
	ax = sns.lineplot(df_results, y='f1-score', x='model', label='F1-Score', linewidth=2, marker='X', markersize=8, color=left_color)
	ax.tick_params(axis='y', colors=left_color)
	ax.yaxis.grid(True)

	plt.xlabel('')
	plt.ylabel('Performance', color=left_color)
	plt.xticks(rotation=90, color='darkblue')
	plt.ylim(top=1)
	# Get the legend object
	legend = plt.legend()
	handles = legend.legend_handles
	for h in handles:
	    h.set_color('darkgray')
	sns.move_legend(plt.gca(), loc='upper left', bbox_to_anchor=(0.78, 0.55))

	ax2 = plt.twinx()

	sns.lineplot(y=df_results['accuracy'].diff().values, x=df_results['model'].to_list(), linewidth=2, marker='o', markersize=7, color=right_color, ax=ax2)
	sns.lineplot(y=df_results['precision'].diff().values, x=df_results['model'].to_list(), linewidth=2, marker='D', color=right_color, ax=ax2)
	sns.lineplot(y=df_results['recall'].diff().values, x=df_results['model'].to_list(), linewidth=2, marker='s', color=right_color, ax=ax2)
	ax = sns.lineplot(y=df_results['f1-score'].diff().values, x=df_results['model'].to_list(), linewidth=2, marker='X', markersize=8, color=right_color, ax=ax2)

	ax.tick_params(axis='y', colors=right_color)
	ax.spines['left'].set_color(left_color)
	ax.spines['right'].set_color(right_color)
	ax.spines['bottom'].set_color('darkblue')
	ax.spines['left'].set_linewidth(1.1)
	ax.spines['right'].set_linewidth(1.1)
	ax.spines['bottom'].set_linewidth(1.1)
	ax.tick_params(width=1.1)

	sns.despine(top=True, right=False, left=False, bottom=False)
	plt.ylabel('Delta', color=right_color)
	plt.title(f'Performance metrics vs training size - Seed {SEED}')
	plt.savefig(f"../data/results/training_size/metrics_seed{SEED}.png", bbox_inches='tight')
	plt.close()
	

	plt.figure(figsize=(8, 4))

	ax = sns.lineplot(df_results, y='model_size', x='model', linewidth=2, marker='o', color=left_color)
	ax.tick_params(axis='y', colors=left_color)
	ax.yaxis.grid(True)

	plt.xticks(rotation=90, color='darkblue')
	plt.xlabel('')
	plt.ylabel('Model file size (megabytes)', color=left_color)

	ax2 = plt.twinx()

	ax = sns.lineplot(df_results, y='memory_usage', x='model', linewidth=2, marker='o', color=right_color, ax=ax2)
	ax.tick_params(axis='y', colors=right_color)
	ax.spines['left'].set_color(left_color)
	ax.spines['right'].set_color(right_color)
	ax.spines['bottom'].set_color('darkblue')
	ax.spines['left'].set_linewidth(1.1)
	ax.spines['right'].set_linewidth(1.1)
	ax.spines['bottom'].set_linewidth(1.1)
	ax.tick_params(width=1.1)

	sns.despine(top=True, right=False, left=False, bottom=False)
	plt.ylabel('Memory usage (megabytes)', color=right_color)
	plt.title(f'Model file size and memory usage - Seed {SEED}')
	plt.savefig(f"../data/results/training_size/filesize-memory_seed{SEED}.png", bbox_inches='tight')
	plt.close()

	return training_size_time


def feature_selection(X_train, y_train, X_fs, y_fs, X_valid, y_valid, X_test, y_test, labels, modules, smote_time, training_size_time, n_jobs=2):
	##########################
	### mutual information ###
	##########################
	# select features
	print('Executing Mutual Information...')
	start_time = time.perf_counter()
	mi_features = fs_mutual_information(X_fs, y_fs, seed=SEED, n_jobs=n_jobs)
	end_time = time.perf_counter()
	mi_decision_time = end_time - start_time
	print('\tNumber of selected features:', len(mi_features))

	# train model
	start_time = time.perf_counter()
	mi_results = train_xgboost(
	    X_train.loc[:, mi_features], y_train, 
	    X_valid.loc[:, mi_features], y_valid, 
	    seed=SEED,
	    method_name='Mutual Information', 
	    save_dir=MODEL_SAVE_DIR_FS,
	    filename=f'mi_seed{SEED}.pkl',
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	mi_training_time = end_time - start_time

	######################
	### markov blanket ###
	######################
	# select features
	print('Executing Markov Blanket...')
	start_time = time.perf_counter()
	causal_selected_features = fs_markov_blanket(X_fs, y_fs, modules, seed=SEED, thresholds=[0.001, 0.01], val_size=0.1, n_jobs=n_jobs)
	end_time = time.perf_counter()
	mb_decision_time = end_time - start_time
	print('\tNumber of selected features:', len(causal_selected_features))

	# train model
	start_time = time.perf_counter()
	mb_results = train_xgboost(
	    X_train.loc[:, causal_selected_features], y_train, 
	    X_valid.loc[:, causal_selected_features], y_valid, 
	    seed=SEED,
	    method_name='Markov Blanket', 
	    save_dir=MODEL_SAVE_DIR_FS,
	    filename=f'mb_seed{SEED}.pkl',
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	mb_training_time = end_time - start_time

	##############
	### boruta ###
	##############
	# select features
	print('Executing Boruta...')
	start_time = time.perf_counter()
	boruta_selected_features = fs_boruta(X_fs, y_fs, seed=SEED, max_iter=30, n_estimators=100, n_jobs=n_jobs)
	end_time = time.perf_counter()
	boruta_decision_time = end_time - start_time
	print('\tNumber of selected features:', len(boruta_selected_features))

	# train model
	start_time = time.perf_counter()
	boruta_results = train_xgboost(
	    X_train.loc[:, boruta_selected_features], y_train, 
	    X_valid.loc[:, boruta_selected_features], y_valid, 
	    seed=SEED,
	    method_name='Boruta', 
	    save_dir=MODEL_SAVE_DIR_FS,
	    filename=f'boruta_seed{SEED}.pkl',
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	boruta_training_time = end_time - start_time

	###########
	### RFE ###
	###########
	# select features
	print('Executing RFE...')
	start_time = time.perf_counter()
	rfe_selected_features = fs_rfe(X_fs, y_fs, n_features_to_select=len(causal_selected_features), seed=SEED)
	end_time = time.perf_counter()
	rfe_decision_time = end_time - start_time
	print('\tNumber of selected features:', len(rfe_selected_features))

	# train model
	start_time = time.perf_counter()
	rfe_results = train_xgboost(
	    X_train.loc[:, rfe_selected_features], y_train, 
	    X_valid.loc[:, rfe_selected_features], y_valid, 
	    seed=SEED,
	    method_name=f'RFE {len(rfe_selected_features)} features', 
	    save_dir=MODEL_SAVE_DIR_FS,
	    filename=f'rfe_seed{SEED}.pkl',
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	rfe_training_time = end_time - start_time

	#######################
	### random features ###
	#######################
	df_results_random = random_features(
	    X_train, y_train, 
	    X_test, y_test, 
	    n_features_to_select=len(causal_selected_features), 
	    model_save_dir=MODEL_SAVE_DIR_FS,
	    seed=SEED,
	    n_jobs=n_jobs
	)

	#######################
	### compile results ###
	#######################
	# number of features
	df_num_feats = pd.DataFrame({
	    'Mutual Information': [len(mi_features)],
	    'Markov Blanket': [len(causal_selected_features)],
	    'Boruta': [len(boruta_selected_features)],
	    'RFE': [len(rfe_selected_features)],
	    'Random Features': [len(causal_selected_features)]
	})
	df_num_feats.to_csv(f'../data/results/feature_selection/features/number_features_seed{SEED}.csv', index=False)

	# features
	df_feats = pd.DataFrame({
	    'Mutual Information': [list(mi_features)],
	    'Markov Blanket': [list(causal_selected_features)],
	    'Boruta': [list(boruta_selected_features)],
	    'RFE': [list(rfe_selected_features)],
	    'Random Features': [random_selected_features.to_list()]
	})
	df_feats.to_csv(f'../data/results/feature_selection/features/selected_features_seed{SEED}.csv', index=False)

	# performance metrics
	df_metrics = pd.concat([
	    mi_results,
	    mb_results,
	    boruta_results,
	    rfe_results
	])
	df_metrics.to_csv(f'../data/results/feature_selection/metrics/feature_selection_metrics_valid_seed{SEED}.csv', index=False)

	# random features
	df_results_random.to_csv(f'../data/results/feature_selection/metrics/random_features_metrics_test_seed{SEED}.csv', index=False)

	# total pipeline time
	df_pipeline_time = pd.DataFrame({
	    'Type': ['Preprocessing', 'Training size', 'Feature Selection Decision', 'Model Training & Eval'],
	    'Mutual Information': [smote_time, training_size_time, mi_decision_time, mi_training_time],
	    'Markov Blanket': [smote_time, training_size_time, mb_decision_time, mb_training_time],
	    'Boruta': [smote_time, training_size_time, boruta_decision_time, boruta_training_time],
	    'RFE': [smote_time, training_size_time, rfe_decision_time, rfe_training_time],
	})
	df_pipeline_time.to_csv(f'../data/results/feature_selection/pipeline/pipeline_times_seed{SEED}.csv', index=False)


def main(args):
	global N_JOBS
	global SEED

	N_JOBS = args.n_jobs
	SEED = args.seed

	print('Seed:', SEED)
	print('n_jobs:', N_JOBS)

	############
	### data ###
	############
	df_train, df_fs, df_valid, df_test = load_data()
	smote_time = preprocess_data(df_train, df_fs, df_valid, df_test)

	#########################
	### training set size ###
	#########################
	# restore preprocessed data splits
	df_train = pd.read_csv(f'../data/datasets/splits/BRUIIoT_train_preprocessed_seed{SEED}.csv')
	df_valid = pd.read_csv(f'../data/datasets/splits/BRUIIoT_valid_preprocessed_seed{SEED}.csv')

	# shuffle training set
	df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)

	# label columns
	targets_cols = ['attack_label__most_frequent', 'attack_label_enc__most_frequent', 'is_attack__most_frequent']

	# labels mapping
	labels = get_labels_mapping(df_valid)

	# keep only protocol features
	df_train = df_train.drop(columns=['frame.time__calculate_duration'])
	df_valid = df_valid.drop(columns=['frame.time__calculate_duration'])

	# perform training size
	training_size_time = determine_training_size(df_train, df_valid, targets_cols, labels, n_jobs=N_JOBS)

	# training proportion to sample
	try:
		training_size_prop = float(input('Which training size proportion to use? >>>'))
		print('Training size proportion:', training_size_prop)
	except Exception as e:
		print(e)

	# sufficient training set size
	df_train = df_train.sample(frac=training_size_prop, random_state=SEED)
	print('Defined training size:', df_train.shape[0])

	# save training set with all features
	df_train.to_csv(f'../data/datasets/splits/BRUIIoT_train_size{training_size_prop}_seed{SEED}.csv', index=False)

	#########################
	### feature selection ###
	#########################
	# restore remaining data splits (fs and test)
	df_fs = pd.read_csv(f'../data/datasets/splits/BRUIIoT_fs_preprocessed_seed{SEED}.csv')
	df_test = pd.read_csv(f'../data/datasets/splits/BRUIIoT_test_preprocessed_seed{SEED}.csv')

	# get procotols
	modules = get_protocols(df_fs, targets_cols)

	# keep only protocols
	df_fs = df_fs.drop(columns=['frame.time__calculate_duration'])
	df_test = df_test.drop(columns=['frame.time__calculate_duration'])

	# get features and targets
	X_train = df_train.drop(columns=['attack_label_enc__most_frequent'])
	X_fs = df_fs.drop(columns=targets_cols+['index'])
	X_valid = df_valid.drop(columns=targets_cols+['index'])
	X_test = df_test.drop(columns=targets_cols+['index'])

	y_train = df_train.loc[:, 'attack_label_enc__most_frequent']
	y_fs = df_fs.loc[:, 'attack_label_enc__most_frequent']
	y_valid = df_valid.loc[:, 'attack_label_enc__most_frequent']
	y_test = df_test.loc[:, 'attack_label_enc__most_frequent']

	print('Training features size:', X_train.shape)
	print('FS features size:', X_fs.shape)
	print('Validation features size:', X_valid.shape)
	print('Test features size:', X_test.shape)
	print('')
	print('Training labels size:', y_train.shape)
	print('FS labels size:', y_fs.shape)
	print('Validation labels size:', y_valid.shape)
	print('Test labels size:', y_test.shape)

	for m in modules:
	    print(f'**Processing module {m}...**')

	    # get module's columns
	    cols = filter_module(df_train.columns, m)
	    print(f'Columns length: {len(cols)}, columns:', cols)
	    print('')

	feature_selection(X_train, y_train, X_fs, y_fs, X_valid, y_valid, X_test, y_test, labels, modules, smote_time, training_size_time, n_jobs=N_JOBS)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, help="Random number generator.")
	parser.add_argument("--n_jobs", type=int, default=2, help="Number of parallel workers (CPU cores).")
	args = parser.parse_args()

	# create directories
	create_data_dirs()

	# run pipeline
	main(args)