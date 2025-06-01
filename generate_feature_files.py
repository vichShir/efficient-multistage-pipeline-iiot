import pandas as pd
import argparse
import ast
from tqdm import tqdm


def main(args):
    if args.split == 'test':
        df = pd.read_csv(f'./data/datasets/splits/BRUIIoT_{args.split}_preprocessed_seed{args.seed}.csv')
    else:
        df = pd.read_csv(f'./data/datasets/splits/{args.filename}')

    # feature selection
    all_features = pd.read_csv(f'./data/results/feature_selection/features/selected_features_seed{args.seed}.csv')
    fs_methods = all_features.columns.to_list()
    for fs in tqdm(fs_methods):
        selected_features = ast.literal_eval(all_features[fs].iloc[0])
        df_fs = df.loc[:, selected_features]
        df_fs.to_csv(f"./data/datasets/features/{args.split}_features_{fs.lower().replace(' ', '-')}_seed{args.seed}.csv", index=False)

    # all features
    X_test = df.drop(columns=['index', 'frame.time__calculate_duration', 'attack_label__most_frequent', 'attack_label_enc__most_frequent', 'is_attack__most_frequent'], errors='ignore')
    X_test.to_csv(f"./data/datasets/features/{args.split}_263-features_seed{args.seed}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random number generator.")
    parser.add_argument("--split", type=str, help="Data split: train or test.")
    parser.add_argument("--filename", type=str, default='', help="Specify the filename for training sets.")
    args = parser.parse_args()
    main(args)