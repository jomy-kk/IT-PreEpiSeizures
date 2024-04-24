from os.path import join
from typing import Collection

import pandas as pd
from pandas import DataFrame
from glob import glob

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline

common_datasets_path = '/Volumes/MMIS-Saraiv/Datasets'
features_dir = 'features'


def read_spectral_features(dataset: str) -> DataFrame:
    if dataset == 'INSIGHT':
        dataset = 'DZNE/INSIGHT/EEG'
    path = join(common_datasets_path, dataset, features_dir, 'Cohort#Spectral#Channels.csv')
    return pd.read_csv(path, index_col=0)

def read_hjorth_features(dataset: str) -> DataFrame:
    if dataset == 'INSIGHT':
        dataset = 'DZNE/INSIGHT/EEG'
    path = join(common_datasets_path, dataset, features_dir, 'Cohort#Hjorth#Channels.csv')
    return pd.read_csv(path, index_col=0)


def read_pli_features(dataset: str, regions=True) -> DataFrame:
    if dataset == 'INSIGHT':
        dataset = 'DZNE/INSIGHT/EEG'
    if regions:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#PLI#Regions.csv')
    else:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#PLI#Channels.csv')
    return pd.read_csv(path, index_col=0)

def read_all_features(dataset) -> DataFrame:
    spectral = read_spectral_features(dataset)
    hjorth = read_hjorth_features(dataset)
    pli = read_pli_features(dataset)
    return spectral.join(hjorth).join(pli)

def read_ages(dataset: str) -> dict[str|int, float|int]:
    if dataset == 'KJPP':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata.csv', sep=';')
        return {row['SESSION']: row['EEG AGE MONTHS'] / 12 for _, row in df.iterrows()}
    if dataset == 'INSIGHT':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/SocioDemog.csv', sep=',')
        return {int(row['CODE']): row['AGE'] for _, row in df.iterrows()}


def read_mmse(dataset: str) -> dict[str, float|int]:
    if dataset == 'INSIGHT':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/cognition_m0.csv', sep=',')
        return {row['CODE']: int(row['MMSE']) for _, row in df.iterrows() if row['MMSE'] not in ('MD', 'NA')}
    if dataset == 'BrainLat':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/BrainLat/metadata.csv', sep=';')
        return {row['ID']: row['MMSE equivalent'] for _, row in df.iterrows()}
    if dataset == 'Miltiadous Dataset':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/participants.tsv', sep='\t')
        return {row['participant_id']: row['MMSE'] for _, row in df.iterrows()}


def read_brainage(dataset: str) -> dict[int, float]:
    if dataset == 'INSIGHT':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/brainage_scores.csv', sep=',')
        return {int(row['CODE']): row['BRAIN AGE'] for _, row in df.iterrows()}


def read_all_eeg(dataset: str, N=None) -> Collection[EEG]:
    all_biosignals = []
    if dataset == 'KJPP':
        all_files = glob(join(common_datasets_path, dataset, 'autopreprocessed_biosignal', '**/*.biosignal'), recursive=True)
        if N is not None:
            all_files = all_files[:N]
        for i, filepath in enumerate(all_files):
            if i % 100 == 0:
                print(f"Read {i/len(all_files)*100:.2f}% of files")
            filename = filepath.split('/')[-1].split('.')[0]
            file_dir = '/' + join(*filepath.split('/')[:-1])
            x = EEG.load(filepath)
            good = Timeline.load(join(file_dir, filename + '_good.timeline'))
            x = x[good]
            all_biosignals.append(x)
    return all_biosignals


def read_disorders(dataset: str) -> dict[str, list[str]]:
    if dataset == 'KJPP':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata.csv', sep=';')
        res = {}
        for _, row in df.iterrows():
            # get list of diagnosis (columns index 8 to 33 inclusive)
            diagnosis = [str(d) for d in row[8:34] if str(d) != 'nan']
            res[row['SESSION']] = diagnosis
        return res
    else:
        raise NotImplementedError(f"Disorders not implemented for dataset {dataset}.")
4