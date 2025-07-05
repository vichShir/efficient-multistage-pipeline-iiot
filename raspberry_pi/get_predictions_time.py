import pandas as pd
import pickle
import time
import os
import argparse
from tqdm import tqdm


def calculate_inference_time(predictor, X, num_exec=100):
    times = []
    print('Doing inference test...')
    for _ in tqdm(range(num_exec)):
        tic = time.time()
        predictor.predict(X)
        tac = time.time()

        inference_time = tac-tic
        times.append(inference_time)

    return pd.DataFrame({'inference_time': times})


def main(args):
    print('Starting inference time...')
    X = pd.read_csv(args.dataset)
    print('Data loaded!', X.shape)
    xgb_model_loaded = pickle.load(open(args.model, "rb"))

    times = calculate_inference_time(xgb_model_loaded, X, num_exec=args.num_exec)
    print(f'Test inference time for {X.shape[0]} samples in seconds:')
    print(times.describe())
    
    times.to_csv(os.path.join('./results', args.save_filename), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset in .csv")
    parser.add_argument("--model", help="XGBoost model")
    parser.add_argument("--num_exec", type=int, default=100, help="XGBoost model")
    parser.add_argument("--save_filename", help="Filename")
    args = parser.parse_args()
    main(args)
