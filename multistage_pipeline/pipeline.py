import polars as pl
import pandas as pd
import numpy as np
import argparse
import time
import gc
import os
import sys
sys.path.insert(0, '../')

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
from multistage_pipeline.utils import header, get_protocols, filter_module

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
plt.switch_backend('agg')


MODEL_SAVE_DIR_TS = '../data/models/training_size'  # training size models
MODEL_SAVE_DIR_FS = '../data/models/feature_selection'  # feature selection models


class MultiStagePipelineIIoT:

	def __init__(self,
				df_train,
				df_fs,
				df_valid,
				df_test,
				data_splits_path,
				labels,
				targets_cols=['attack_label__most_frequent', 'attack_label_enc__most_frequent', 'is_attack__most_frequent'],
				seed=0,
				n_jobs=2,
				**kwargs):
		# data splits
		self.df_train = df_train
		self.df_fs = df_fs
		self.df_valid = df_valid
		self.df_test = df_test

		self.data_splits_path = data_splits_path

		# label columns
		self.labels = labels
		self.targets_cols = targets_cols

		# balacing time
		self.smote_time = kwargs['smote_time']

		self.seed = seed
		self.n_jobs = n_jobs

	def start(self):
		training_size_time = self._first_stage()
		self._second_stage(training_size_time)

	def _first_stage(self):
		header('First Stage - Sufficient Training Size Estimation')

		training_size_time = determine_training_size(
			self.df_train, 
			self.df_valid, 
			self.targets_cols, 
			self.labels, 
			seed=self.seed,
			n_jobs=self.n_jobs
		)

		# determine sufficient training size by user
		try:
			training_size_prop = float(input('Which training size proportion to use? >>>'))
			print('Training size proportion:', training_size_prop)
		except Exception as e:
			print(e)

		# get sufficient training set
		df_new_train = self.df_train.sample(frac=training_size_prop, random_state=self.seed)
		df_new_train.to_csv(
			os.path.join(
				self.data_splits_path, 
				f'BRUIIoT_train_size{training_size_prop}_seed{self.seed}.csv'
			), 
			index=False
		)
		print('Defined training size:', df_new_train.shape[0])

		return training_size_time

	def _second_stage(self, training_size_time):
		header('Second Stage - Protocol Feature Analysis')

		# get procotols
		modules = get_protocols(self.df_fs, self.targets_cols)

		# get features and targets
		X_train = self.df_train.drop(columns=['attack_label_enc__most_frequent'])
		X_fs = self.df_fs.drop(columns=self.targets_cols+['index'])
		X_valid = self.df_valid.drop(columns=self.targets_cols+['index'])
		X_test = self.df_test.drop(columns=self.targets_cols+['index'])

		y_train = self.df_train.loc[:, 'attack_label_enc__most_frequent']
		y_fs = self.df_fs.loc[:, 'attack_label_enc__most_frequent']
		y_valid = self.df_valid.loc[:, 'attack_label_enc__most_frequent']
		y_test = self.df_test.loc[:, 'attack_label_enc__most_frequent']

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
		    cols = filter_module(self.df_train.columns, m)
		    print(f'Columns length: {len(cols)}, columns:', cols)
		    print('')

		feature_selection(
			X_train, y_train, 
			X_fs, y_fs, 
			X_valid, y_valid, 
			X_test, y_test, 
			self.labels, 
			modules, 
			self.smote_time, 
			training_size_time, 
			seed=self.seed,
			n_jobs=self.n_jobs
		)


def determine_training_size(df_train, df_valid, targets_cols, labels, n_jobs=2, seed=0):
	# get features and targets from validation
	X_valid = df_valid.drop(columns=targets_cols+['index'])
	y_valid = df_valid.loc[:, 'attack_label_enc__most_frequent']

	print('Training size:', df_train.shape)
	print('Validation features size:', X_valid.shape)
	print('Validation labels size:', y_valid.shape)

	# run training size variation
	print('\nDetermining proportions performances...')
	start_time = time.perf_counter()
	df_results = training_size(
	    df_train, 
	    X_valid, y_valid, 
	    seed=seed,
	    model_save_dir=MODEL_SAVE_DIR_TS,
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	training_size_time = end_time - start_time

	# save results
	df_results.to_csv(f'../data/results/training_size/training_size_results_seed{seed}.csv', index=False)

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
	plt.title(f'Performance metrics vs training size - Seed {seed}')
	plt.savefig(f"../data/results/training_size/metrics_seed{seed}.png", bbox_inches='tight')
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
	plt.title(f'Model file size and memory usage - Seed {seed}')
	plt.savefig(f"../data/results/training_size/filesize-memory_seed{seed}.png", bbox_inches='tight')
	plt.close()

	return training_size_time


def feature_selection(X_train, y_train, X_fs, y_fs, X_valid, y_valid, X_test, y_test, labels, modules, smote_time, training_size_time, n_jobs=2, seed=0):
	##########################
	### mutual information ###
	##########################
	# select features
	print('***Executing Mutual Information***')
	start_time = time.perf_counter()
	mi_features = fs_mutual_information(X_fs, y_fs, seed=seed, n_jobs=n_jobs)
	end_time = time.perf_counter()
	mi_decision_time = end_time - start_time
	print('\tNumber of selected features:', len(mi_features))

	# train model
	start_time = time.perf_counter()
	mi_results = train_xgboost(
	    X_train.loc[:, mi_features], y_train, 
	    X_valid.loc[:, mi_features], y_valid, 
	    seed=seed,
	    method_name='Mutual Information', 
	    save_dir=MODEL_SAVE_DIR_FS,
	    filename=f'mi_seed{seed}.pkl',
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	mi_training_time = end_time - start_time

	######################
	### markov blanket ###
	######################
	# select features
	print('***Executing Markov Blanket***')
	start_time = time.perf_counter()
	causal_selected_features = fs_markov_blanket(X_fs, y_fs, modules, seed=seed, thresholds=[0.001, 0.01], val_size=0.1, n_jobs=n_jobs)
	end_time = time.perf_counter()
	mb_decision_time = end_time - start_time
	print('\tNumber of selected features:', len(causal_selected_features))

	# train model
	start_time = time.perf_counter()
	mb_results = train_xgboost(
	    X_train.loc[:, causal_selected_features], y_train, 
	    X_valid.loc[:, causal_selected_features], y_valid, 
	    seed=seed,
	    method_name='Markov Blanket', 
	    save_dir=MODEL_SAVE_DIR_FS,
	    filename=f'mb_seed{seed}.pkl',
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	mb_training_time = end_time - start_time

	##############
	### boruta ###
	##############
	# select features
	print('***Executing Boruta***')
	start_time = time.perf_counter()
	boruta_selected_features = fs_boruta(X_fs, y_fs, seed=seed, max_iter=30, n_estimators=100, n_jobs=n_jobs)
	end_time = time.perf_counter()
	boruta_decision_time = end_time - start_time
	print('\tNumber of selected features:', len(boruta_selected_features))

	# train model
	start_time = time.perf_counter()
	boruta_results = train_xgboost(
	    X_train.loc[:, boruta_selected_features], y_train, 
	    X_valid.loc[:, boruta_selected_features], y_valid, 
	    seed=seed,
	    method_name='Boruta', 
	    save_dir=MODEL_SAVE_DIR_FS,
	    filename=f'boruta_seed{seed}.pkl',
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	boruta_training_time = end_time - start_time

	###########
	### RFE ###
	###########
	# select features
	print('***Executing RFE***')
	start_time = time.perf_counter()
	rfe_selected_features = fs_rfe(X_fs, y_fs, n_features_to_select=len(causal_selected_features), seed=seed)
	end_time = time.perf_counter()
	rfe_decision_time = end_time - start_time
	print('\tNumber of selected features:', len(rfe_selected_features))

	# train model
	start_time = time.perf_counter()
	rfe_results = train_xgboost(
	    X_train.loc[:, rfe_selected_features], y_train, 
	    X_valid.loc[:, rfe_selected_features], y_valid, 
	    seed=seed,
	    method_name=f'RFE {len(rfe_selected_features)} features', 
	    save_dir=MODEL_SAVE_DIR_FS,
	    filename=f'rfe_seed{seed}.pkl',
	    labels=labels,
	    n_jobs=n_jobs
	)
	end_time = time.perf_counter()
	rfe_training_time = end_time - start_time

	#######################
	### random features ###
	#######################
	print('***Executing Random Selection***')
	df_results_random, random_selected_features = random_features(
	    X_train, y_train, 
	    X_test, y_test, 
	    n_features_to_select=len(causal_selected_features), 
	    model_save_dir=MODEL_SAVE_DIR_FS,
	    seed=seed,
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
	df_num_feats.to_csv(f'../data/results/feature_selection/features/number_features_seed{seed}.csv', index=False)

	# features
	df_feats = pd.DataFrame({
	    'Mutual Information': [list(mi_features)],
	    'Markov Blanket': [list(causal_selected_features)],
	    'Boruta': [list(boruta_selected_features)],
	    'RFE': [list(rfe_selected_features)],
	    'Random Features': [random_selected_features.to_list()]
	})
	df_feats.to_csv(f'../data/results/feature_selection/features/selected_features_seed{seed}.csv', index=False)

	# performance metrics
	df_metrics = pd.concat([
	    mi_results,
	    mb_results,
	    boruta_results,
	    rfe_results
	])
	df_metrics.to_csv(f'../data/results/feature_selection/metrics/feature_selection_metrics_valid_seed{seed}.csv', index=False)

	# random features
	df_results_random.to_csv(f'../data/results/feature_selection/metrics/random_features_metrics_test_seed{seed}.csv', index=False)

	# total pipeline time
	df_pipeline_time = pd.DataFrame({
	    'Type': ['Preprocessing', 'Training size', 'Feature Selection Decision', 'Model Training & Eval'],
	    'Mutual Information': [smote_time, training_size_time, mi_decision_time, mi_training_time],
	    'Markov Blanket': [smote_time, training_size_time, mb_decision_time, mb_training_time],
	    'Boruta': [smote_time, training_size_time, boruta_decision_time, boruta_training_time],
	    'RFE': [smote_time, training_size_time, rfe_decision_time, rfe_training_time],
	})
	df_pipeline_time.to_csv(f'../data/results/feature_selection/pipeline/pipeline_times_seed{seed}.csv', index=False)