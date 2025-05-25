import pandas as pd
import numpy as np
import os
import gc
from pipeline.model import train_xgboost
from tqdm import tqdm


def get_model_size(x, path='./'):
    prop = x.split(' ')[-1]
    filename = f'training_size_{prop}.pkl'
    model_size = os.path.getsize(os.path.join(path, filename))
    return model_size


def training_size(df_train, X_valid, y_valid, data_seed, ml_seed, model_save_dir):
	df_results = []
	size_proportions = np.linspace(0.04, 1.0, num=20)**2  # np.linspace(0.04, 1.0, num=20)**2
	for prop in (pbar := tqdm(size_proportions)):
	    prop = round(prop, 4)
	    
	    if prop < 1:
	        df_train_prop = df_train.sample(int(df_train.shape[0]*prop), random_state=data_seed)
	    else:
	        df_train_prop = df_train

	    pbar.set_postfix(prop=round(prop*100, 2), train_size=df_train_prop.shape[0])
	    pbar.refresh()
	    
	    X_train_prop = df_train_prop.drop(columns=['attack_label_enc__most_frequent'])
	    y_train_prop = df_train_prop.loc[:, 'attack_label_enc__most_frequent']
	    
	    res = train_xgboost(
	        X_train_prop, y_train_prop, 
	        X_valid, y_valid, 
	        seed=ml_seed,
	        method_name=f'Training size {round(prop*100, 2)}', 
	        save_dir=model_save_dir,
	        filename=f'training_size_{round(prop*100, 2)}.pkl',
	        verbose=0
	    )
	    memory_usage = df_train_prop.memory_usage(index=True).sum()
	    res['memory_usage'] = memory_usage
	    res['sample_size'] = df_train_prop.shape[0]
	    df_results.append(res)
	    
	    del df_train_prop
	    del X_train_prop
	    del y_train_prop
	    gc.collect()

	df_results = pd.concat(df_results, axis=0)
	df_results['model_size'] = df_results['model'].apply(lambda x: get_model_size(x, path=model_save_dir))
	df_results

	return df_results