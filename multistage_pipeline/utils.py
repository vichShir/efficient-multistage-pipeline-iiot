import os


def get_protocols(df, targets_cols):
    modules = list(set(map(lambda x: x.split('.')[0], df.columns)))  # get protocols from columns
    modules = [m for m in modules if m not in targets_cols]  # remove target columns
    modules.remove('frame')
    return modules


def filter_module(columns, module):
    match_module = map(lambda x: x if module+'.' in x else None, columns)  # match module columns else None
    clean_module = filter(lambda x: x is not None, match_module)  # remove None values
    return list(clean_module)


def convert_bytes_to_megabytes(size_bytes):
   file_size_kb = size_bytes / 1024
   file_size_mb = file_size_kb / 1024
   return file_size_mb


def create_data_dirs():
    os.makedirs('../data/datasets/', exist_ok=True)
    os.makedirs('../data/models/', exist_ok=True)
    os.makedirs('../data/results/', exist_ok=True)

    os.makedirs('../data/datasets/splits', exist_ok=True)
    os.makedirs('../data/datasets/features', exist_ok=True)

    os.makedirs('../data/models/training_size', exist_ok=True)
    os.makedirs('../data/models/feature_selection', exist_ok=True)

    os.makedirs('../data/results/confusion_matrix', exist_ok=True)
    os.makedirs('../data/results/feature_importance', exist_ok=True)
    os.makedirs('../data/results/feature_selection', exist_ok=True)
    os.makedirs('../data/results/inference_time', exist_ok=True)
    os.makedirs('../data/results/training_size', exist_ok=True)

    os.makedirs('../data/results/feature_selection/features', exist_ok=True)
    os.makedirs('../data/results/feature_selection/metrics', exist_ok=True)
    os.makedirs('../data/results/feature_selection/pipeline', exist_ok=True)