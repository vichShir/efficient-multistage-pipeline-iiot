import pandas as pd
import numpy as np
import os
import gc
from pipeline.model import train_xgboost
from pipeline.utils import convert_bytes_to_megabytes
from tqdm import tqdm


def get_model_size(x, seed, path='./'):
    # prop = x.split(' ')[-1].replace('%', '')
    prop = x.replace('%', '')
    filename = f'training_size_{prop}_seed{seed}.pkl'
    model_size = os.path.getsize(os.path.join(path, filename))
    return convert_bytes_to_megabytes(model_size)


def training_size(df_train, X_valid, y_valid, seed, model_save_dir, labels, n_jobs=2):
	df_results = []
	size_proportions = np.linspace(0.04, 1.0, num=20)**2  # np.linspace(0.04, 1.0, num=20)**2
	for prop in (pbar := tqdm(size_proportions)):
	    prop = round(prop, 4)
	    
	    if prop < 1:
	        df_train_prop = df_train.sample(int(df_train.shape[0]*prop), random_state=seed)
	    else:
	        df_train_prop = df_train

	    pbar.set_postfix(prop=f'{round(prop*100, 2)}%', train_size=df_train_prop.shape[0])
	    pbar.refresh()
	    
	    X_train_prop = df_train_prop.drop(columns=['attack_label_enc__most_frequent'])
	    y_train_prop = df_train_prop.loc[:, 'attack_label_enc__most_frequent']
	    
	    res = train_xgboost(
	        X_train_prop, y_train_prop, 
	        X_valid, y_valid, 
	        seed=seed,
	        method_name=f'{round(prop*100, 2)}%', 
	        save_dir=model_save_dir,
	        filename=f'training_size_{round(prop*100, 2)}_seed{seed}.pkl',
	        verbose=0 if prop < 1 else 1,
	        labels=labels,
	        n_jobs=n_jobs
	    )
	    memory_usage = df_train_prop.memory_usage(index=True).sum()
	    res['memory_usage'] = convert_bytes_to_megabytes(memory_usage)
	    res['sample_size'] = df_train_prop.shape[0]
	    df_results.append(res)
	    
	    del df_train_prop
	    del X_train_prop
	    del y_train_prop
	    gc.collect()

	df_results = pd.concat(df_results, axis=0)
	df_results['model_size'] = df_results['model'].apply(lambda x: get_model_size(x, seed, path=model_save_dir))
	df_results

	return df_results