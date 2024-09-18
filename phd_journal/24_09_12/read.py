from os.path import join
from typing import Collection

import pandas as pd
from numpy import argmax
from pandas import DataFrame
from glob import glob

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline

common_datasets_path = '/Volumes/MMIS-Saraiv/Datasets'
features_dir = 'features'


def read_spectral_features(dataset: str, multiples: bool) -> DataFrame:
    if not multiples:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Spectral#Channels.csv')
    else:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Spectral#Channels$Multiple.csv')
    return pd.read_csv(path, index_col=0)


def read_hjorth_features(dataset: str, multiples) -> DataFrame:
    if not multiples:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Hjorth#Channels.csv')
    else:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Hjorth#Channels$Multiple.csv')
    return pd.read_csv(path, index_col=0)


def read_pli_features(dataset: str, multiples: bool) -> DataFrame:
    if not multiples:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Connectivity#Regions.csv')
    else:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Connectivity#Regions$Multiple.csv')
    return pd.read_csv(path, index_col=0)


def read_all_features(dataset, multiples=False) -> DataFrame:
    spectral = read_spectral_features(dataset, multiples)
    hjorth = read_hjorth_features(dataset, multiples)
    pli = read_pli_features(dataset, multiples)
    res = spectral.join(hjorth).join(pli)
    return res


def read_ages(dataset: str) -> dict[str|int, float|int]:
    if dataset == 'HBN':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Healthy Brain Network/participants.tsv', sep='\t')
        return {row['participant_id']: row['Age'] for _, row in df.iterrows()}
    if dataset == 'KJPP':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata.csv', sep=';')
        return {row['SESSION']: row['EEG AGE MONTHS'] / 12 for _, row in df.iterrows()}


def read_diagnoses(dataset: str) -> dict[str|int, list[str]]:
    if dataset == 'KJPP':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/curated_metadata.csv', sep=';')
        res = {}
        for _, row in df.iterrows():
            key = row['SESSION']
            value = row['ALL DIAGNOSES CODES']
            if '[' in value:
                value = eval(row['ALL DIAGNOSES CODES'])
            res[key] = value
        return res

