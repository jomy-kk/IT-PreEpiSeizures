from os.path import join

import numpy as np
import pandas as pd
from scipy.stats import zscore


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

def read_dataset(path: str, label:str) -> pd.DataFrame:
    df = pd.read_csv(join(path, "all_features.csv"), index_col=0)
    df.index = [(f"{label}-{get_subject_code(s)}" if isinstance(s, str) else f"{label}-{s}") for s in df.index]
    return df

def read_metadata(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = [(f"{label}-{get_subject_code(s)}" if isinstance(s, str) else f"{label}-{s}")  for s in df.index]
    return df

