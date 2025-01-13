from os.path import join

import numpy as np
import pandas as pd
from scipy.stats import zscore


datasets_paths = {
    "Newcastle": "/Volumes/MMIS-Saraiv/Datasets/Newcastle/EC/features_source_ind-bands",
    "Izmir": "/Volumes/MMIS-Saraiv/Datasets/Izmir/EC/features_source_ind-bands",
    "Istambul": "/Volumes/MMIS-Saraiv/Datasets/Istambul/features_source_ind-bands",
    "Miltiadous": "/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features_source_ind-bands",
    "BrainLat:CL": "/Volumes/MMIS-Saraiv/Datasets/BrainLat/features_source_ind-bands/CL",
    "BrainLat:AR": "/Volumes/MMIS-Saraiv/Datasets/BrainLat/features_source_ind-bands/AR",
}

datasets_metadata_paths = {
    "Newcastle": "/Volumes/MMIS-Saraiv/Datasets/Newcastle/metadata.csv",
    "Izmir": "/Volumes/MMIS-Saraiv/Datasets/Izmir/metadata.csv",
    "Istambul": "/Volumes/MMIS-Saraiv/Datasets/Istambul/metadata.csv",
    "Miltiadous": "/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/metadata.csv",
    "BrainLat:CL": "/Volumes/MMIS-Saraiv/Datasets/BrainLat/features_source_ind-bands/metadata_CL.csv",
    "BrainLat:AR": "/Volumes/MMIS-Saraiv/Datasets/BrainLat/features_source_ind-bands/metadata_AR.csv",
}


def get_subject_code(identifier: str) -> int:
    if 'PARTICIPANT' in identifier:
        return int(identifier.split("PARTICIPANT")[1])
    if 'sub-' in identifier:
        return int(identifier.split("sub-")[1])


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


def read_all_datasets(dataset_names):
    # Read datasets
    datasets = {}
    for dataset_name in dataset_names:
        path = datasets_paths[dataset_name]
        dataset = read_dataset(path)
        dataset.index = [f"{dataset_name}-{s}" for s in dataset.index]
        datasets[dataset_name] = dataset
    return datasets


def read_all_metadata(dataset_names):
    # Read metadata
    datasets_metadata = {}
    for dataset_name in dataset_names:
        path = datasets_metadata_paths[dataset_name]
        dataset = read_metadata(path)
        if dataset_name == 'Izmir' or dataset_name == 'Newcastle' or dataset_name == 'Miltiadous':
            dataset.index = [f"{dataset_name}-{get_subject_code(s)}" for s in dataset.index]
        else:
            dataset.index = [f"{dataset_name}-{s}" for s in dataset.index]
        dataset['SITE'] = [dataset_name]*len(dataset)

        # Keep only some columns, if Df has them: (AGE, GENDER, DIAGNOSIS, EDUCATION YEARS, MMSE, MoCA, SITE)
        to_keep = []
        if 'AGE' in dataset.columns:
            to_keep.append('AGE')
        if 'GENDER' in dataset.columns:
            to_keep.append('GENDER')
        if 'DIAGNOSIS' in dataset.columns:
            to_keep.append('DIAGNOSIS')
        if 'EDUCATION YEARS' in dataset.columns:
            to_keep.append('EDUCATION YEARS')
        if 'MMSE' in dataset.columns:
            to_keep.append('MMSE')
        if 'MoCA' in dataset.columns:
            to_keep.append('MoCA')
        if 'SITE' in dataset.columns:
            to_keep.append('SITE')
        dataset = dataset[to_keep]
        datasets_metadata[dataset_name] = dataset
    return datasets_metadata


def read_all_transformed_datasets(common_path: str, dataset_names):
    datasets = {}
    for dataset_name in dataset_names:
        dataset = pd.read_csv(join(common_path, f"{dataset_name}.csv"), index_col=0)
        datasets[dataset_name] = dataset
    return datasets

