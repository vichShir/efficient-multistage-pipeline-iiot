import polars as pl
import pandas as pd
import numpy as np
import os
import gc
from imblearn.over_sampling import SMOTE


def create_splits(df, save_dir, seed):
	# add index column
	total_rows = df.shape[0]
	df = df.with_columns(pl.Series("index", list(range(total_rows))))

	### test ###
	# get 20% random sample for test set
	df_test = df.sample(n=int(total_rows*0.2), seed=seed)

	# remove test from dataset
	df = df.join(df_test, on=['index'], how='anti')

	### validation ###
	# get 20% random sample for validation set
	df_valid = df.sample(n=int(total_rows*0.2), seed=seed)

	# remove validation from dataset
	df = df.join(df_valid, on=['index'], how='anti')

	### feature selection ###
	# get 5% random sample for feature selection
	df_fs = df.sample(n=int(total_rows*0.05), seed=seed)

	### training ###
	# use the remaining (55%) for training
	df_train = df.join(df_fs, on=['index'], how='anti')

	### print split division ###
	train_prop = df_train.shape[0]/total_rows
	fs_prop = df_fs.shape[0]/total_rows
	valid_prop = df_valid.shape[0]/total_rows
	test_prop = df_test.shape[0]/total_rows
	print('Training division:', round(train_prop, 2))
	print('Feature Selection division:', round(fs_prop, 2))
	print('Validation division:', round(valid_prop, 2))
	print('Test division:', round(test_prop, 2))

	del df
	gc.collect()

	return df_train, df_fs, df_valid, df_test


def transform_labels(df):
	# Polars
	if isinstance(df, pl.dataframe.frame.DataFrame):
		df = df.with_columns(pl.col('attack_label__most_frequent').replace({
		    'ddos_http_flood': 'other_attack_type',
		    'ddos_slowloris': 'other_attack_type',
		    'ddos_smurf': 'other_attack_type',
		}))
		df = df.with_columns(pl.col('attack_label_enc__most_frequent').replace({
		    0: 1,  # Mirai-greeth_flood
		    1: 4,  # ddos_http_flood
		    2: 2,  # ddos_icmp_flood
		    3: 4,  # ddos_slowloris
		    4: 4,  # ddos_smurf
		    5: 3,  # ddos_syn_flood
		    6: 0,  # normal
		}))
	else:
		# Pandas
		df['attack_label__most_frequent'] = df['attack_label__most_frequent'].replace({
		    'ddos_http_flood': 'other_attack_type',
		    'ddos_slowloris': 'other_attack_type',
		    'ddos_smurf': 'other_attack_type',
		})
		df['attack_label_enc__most_frequent'] = df['attack_label_enc__most_frequent'].replace({
		    0: 1,  # Mirai-greeth_flood
		    1: 4,  # ddos_http_flood
		    2: 2,  # ddos_icmp_flood
		    3: 4,  # ddos_slowloris
		    4: 4,  # ddos_smurf
		    5: 3,  # ddos_syn_flood
		    6: 0,  # normal
		})
	return df


def get_labels_mapping(df):
	# map label to indices
	labels = list(set(zip(df['attack_label_enc__most_frequent'], df['attack_label__most_frequent'])))  # get encoding and label pairs
	labels = {x[0]: x[1] for x in labels}  # to dict
	labels = dict(sorted(labels.items()))  # order by key (enconding)
	return labels


def balance_labels_with_smote(df_train, seed):
	# determine the number of samples to balance classes
	n_classes = 5
	n_balance = int(df_train.shape[0]/n_classes)

	# filter each class
	df_train_normal = df_train.filter(pl.col('attack_label__most_frequent') == 'normal')
	df_train_mirai = df_train.filter(pl.col('attack_label__most_frequent') == 'Mirai-greeth_flood')
	df_train_icmp = df_train.filter(pl.col('attack_label__most_frequent') == 'ddos_icmp_flood')
	df_train_syn = df_train.filter(pl.col('attack_label__most_frequent') == 'ddos_syn_flood')
	df_train_others = df_train.filter(pl.col('attack_label__most_frequent') == 'other_attack_type')

	del df_train
	gc.collect()

	# are there any class has more than the number of samples to balance?
	more_samples_normal = df_train_normal.shape[0] > n_balance
	more_samples_mirai = df_train_mirai.shape[0] > n_balance
	more_samples_icmp = df_train_icmp.shape[0] > n_balance
	more_samples_syn = df_train_syn.shape[0] > n_balance
	more_samples_others = df_train_others.shape[0] > n_balance

	print(f'Has normal label more than {n_balance} to cut?', more_samples_normal)
	print(f'Has mirai label more than {n_balance} to cut?', more_samples_mirai)
	print(f'Has icmp label more than {n_balance} to cut?', more_samples_icmp)
	print(f'Has syn label more than {n_balance} to cut?', more_samples_syn)
	print(f'Has others label more than {n_balance} to cut?', more_samples_others)

	# remove extra samples randomly
	if more_samples_normal: df_train_normal = df_train_normal.sample(n_balance, seed=seed);
	if more_samples_mirai: more_samples_mirai = more_samples_mirai.sample(n_balance, seed=seed);
	if more_samples_icmp: more_samples_icmp = more_samples_icmp.sample(n_balance, seed=seed);
	if more_samples_syn: df_train_syn = df_train_syn.sample(n_balance, seed=seed);
	if more_samples_others: df_train_others = df_train_others.sample(n_balance, seed=seed);
	
	# concatenate labels
	df_train = pl.concat([
	    df_train_normal,
	    df_train_mirai,
	    df_train_icmp,
	    df_train_syn,
	    df_train_others
	], how='vertical')

	# get features and labels to pass to SMOTE
	X_train = df_train.to_pandas().drop(columns=['index', 'attack_label__most_frequent', 'attack_label_enc__most_frequent', 'is_attack__most_frequent'])
	y_train = df_train.to_pandas().loc[:, 'attack_label_enc__most_frequent']

	# create new samples with SMOTE (balancing)
	X_res, y_res = SMOTE(random_state=seed).fit_resample(X_train, y_train)
	df_train = pd.concat([X_res, y_res], axis=1)

	return df_train