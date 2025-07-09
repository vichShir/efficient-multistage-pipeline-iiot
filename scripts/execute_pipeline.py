import polars as pl
import pandas as pd
import numpy as np
import argparse
import gc
import os
import sys
sys.path.insert(0, '../')

from multistage_pipeline.preprocessing import (
    get_labels_mapping,
    create_splits, 
    preprocess_data
)
from multistage_pipeline.pipeline import MultiStagePipelineIIoT
from multistage_pipeline.utils import header


def load_data(dataset_path, data_splits_path, seed=0):
	# load dataset with Polars
	df = pl.read_csv(dataset_path)
	print('Original dataset size:', df.shape)
	print(f'There are {df.shape[0]} total observations.')

	# create splits
	print('\nSplits division:')
	df_train, df_fs, df_valid, df_test = create_splits(df, data_splits_path, seed=seed)

	# clear memory
	del df
	gc.collect()

	return df_train, df_fs, df_valid, df_test


def main(args):
	N_JOBS = args.n_jobs
	SEED = args.seed

	header('Arguments')
	print('Seed:', SEED)
	print('n_jobs:', N_JOBS)

	############
	### data ###
	############
	header('Load data and splits')
	df_train, df_fs, df_valid, df_test = load_data(args.dataset_path, args.data_splits_path, SEED)

	header('Preprocess data')
	smote_time = preprocess_data(
		df_train, 
		df_fs, 
		df_valid, 
		df_test, 
		args.data_splits_path, 
		SEED
	)

	################
	### pipeline ###
	################
	header('Efficient Multi-Stage Pipeline')

	# load splits
	df_train = pd.read_csv(os.path.join(args.data_splits_path, f'BRUIIoT_train_preprocessed_seed{SEED}.csv'))
	df_fs = pd.read_csv(os.path.join(args.data_splits_path, f'BRUIIoT_fs_preprocessed_seed{SEED}.csv'))
	df_valid = pd.read_csv(os.path.join(args.data_splits_path, f'BRUIIoT_valid_preprocessed_seed{SEED}.csv'))
	df_test = pd.read_csv(os.path.join(args.data_splits_path, f'BRUIIoT_test_preprocessed_seed{SEED}.csv'))

	# shuffle training set
	df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)

	# keep only protocol features
	df_train = df_train.drop(columns=['frame.time__calculate_duration'])
	df_fs = df_fs.drop(columns=['frame.time__calculate_duration'])
	df_valid = df_valid.drop(columns=['frame.time__calculate_duration'])
	df_test = df_test.drop(columns=['frame.time__calculate_duration'])

	# labels mapping
	labels = get_labels_mapping(df_valid)

	pipeline = MultiStagePipelineIIoT(
		df_train,
		df_fs,
		df_valid,
		df_test,
		args.data_splits_path,
		labels,
		smote_time=smote_time,
		seed=SEED,
		n_jobs=N_JOBS
	)
	pipeline.start()

	print('Finished!')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, help="Random number generator.")
	parser.add_argument("--n_jobs", type=int, default=2, help="Number of parallel workers (CPU cores).")
	parser.add_argument("--dataset_path", type=str, default='../data/datasets/BRUIIoT.csv', help="Path to BRUIIoT dataset.")
	parser.add_argument("--data_splits_path", type=str, default='../data/datasets/splits/', help="Path to data splits.")
	args = parser.parse_args()

	# run pipeline
	main(args)