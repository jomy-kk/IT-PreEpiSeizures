from os.path import join

import numpy as np
import pandas as pd
from scipy.stats import zscore


datasets_paths = {
    "Newcastle": "/Volumes/MMIS-Saraiv/Datasets/Newcastle/EC/features_source_ind-bands",
    "Izmir": "/Volumes/MMIS-Saraiv/Datasets/Izmir/EC/features_source_ind-bands",
    "Istambul": "/Volumes/MMIS-Saraiv/Datasets/Istambul/features_source_ind-bands",
    "Miltiadous": "/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features_source_ind-bands",
}

datasets_metadata_paths = {
    "Izmir": "/Volumes/MMIS-Saraiv/Datasets/Izmir/metadata.csv",
    "Istambul": "/Volumes/MMIS-Saraiv/Datasets/Istambul/metadata.csv",
    "Newcastle": "/Volumes/MMIS-Saraiv/Datasets/Newcastle/metadata.csv",
}


def get_subject_code(identifier: str) -> int:
    return int(identifier.split("PARTICIPANT")[1])


def intra_dataset_norm(df: pd.DataFrame, method:str):
    if method == 'z-score':  # Z-score
        return df.apply(zscore)
    elif method == 'min-max':
        return (df - df.min()) / (df.max() - df.min())
    elif method == 'log':
        return np.log10(df)
    else:
        raise ValueError(f"Unknown method: {method}")


def inter_dataset_norm(df: pd.DataFrame):
    pass


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(join(path, "all_features.csv"), index_col=0)
    return df


def read_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def read_all_datasets():
    # Read datasets
    datasets = {}
    for dataset_name, path in datasets_paths.items():
        dataset = read_dataset(path)
        dataset.index = [(f"{dataset_name}-{get_subject_code(s)}" if isinstance(s, str) and dataset_name != 'Istambul' else f"{dataset_name}-{s}") for s in dataset.index]
        datasets[dataset_name] = dataset
    return datasets


def read_all_metadata():
    # Read metadata
    datasets_metadata = {}
    for dataset_name, path in datasets_metadata_paths.items():
        dataset = read_metadata(path)
        dataset.index = [(f"{dataset_name}-{get_subject_code(s)}" if isinstance(s, str) and dataset_name != 'Istambul' else f"{dataset_name}-{s}") for s in dataset.index]
        dataset['SITE'] = [dataset_name]*len(dataset)
        datasets_metadata[dataset_name] = dataset
    return datasets_metadata


def read_all_transformed_datasets(common_path: str):
    datasets = {}
    for dataset_name in datasets_paths.keys():
        dataset = pd.read_csv(join(common_path, f"{dataset_name}.csv"), index_col=0)
        datasets[dataset_name] = dataset
    return datasets

