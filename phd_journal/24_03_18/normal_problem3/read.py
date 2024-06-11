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
    if dataset == 'INSIGHT':
        dataset = 'DZNE/INSIGHT/EEG'
    if not multiples:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Spectral#Channels.csv')
    else:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Spectral#Channels$Multiple.csv')
    return pd.read_csv(path, index_col=0)

def read_hjorth_features(dataset: str, multiples) -> DataFrame:
    if dataset == 'INSIGHT':
        dataset = 'DZNE/INSIGHT/EEG'
    if not multiples:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Hjorth#Channels.csv')
    else:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Hjorth#Channels$Multiple.csv')
    return pd.read_csv(path, index_col=0)


def read_pli_features(dataset: str, multiples: bool) -> DataFrame:
    if dataset == 'INSIGHT':
        dataset = 'DZNE/INSIGHT/EEG'
    if not multiples:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Connectivity#Regions.csv')
    else:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Connectivity#Regions$Multiple.csv')
    return pd.read_csv(path, index_col=0)


def select_safe(all_features: DataFrame, dataset: str) -> DataFrame:
    safe = pd.read_csv(join(common_datasets_path, dataset, features_dir, 'safe_multiples.csv'), index_col=0)
    if dataset == 'Miltiadous Dataset':
        safe.index = [format(n, '03') for n in safe.index]

    # Indexes are: 'subject$multiple'
    # Make two indices: 'subject' and 'multiple'
    all_features['subject'] = all_features.index.str.split('$').str[0]
    all_features['multiple'] = all_features.index.str.split('$').str[1]
    all_features = all_features.set_index(['subject', 'multiple'])

    res = pd.DataFrame(columns=all_features.columns)
    # Iterate by subject
    for subject in all_features.index.get_level_values('subject').unique():
        this_safe_multiples = eval(safe.loc[subject].iloc[0])
        this_subject_features = all_features.loc[subject]

        if len(this_safe_multiples) == 0:  # no tuples
            # keep only the middle multiple of this subject
            to_keep = this_subject_features.iloc[len(this_subject_features) // 2]
            to_keep = to_keep.to_frame().T
            to_keep.index = [subject]
            res = pd.concat([res, to_keep])
        elif len(this_safe_multiples) == 1:  # one tuple
            # keep all the multiples mentioned in that tuple
            to_keep = this_subject_features.iloc[list(this_safe_multiples[0])]
            to_keep.index = [f"{subject}${x}" for x in to_keep.index]
            res = pd.concat([res, to_keep])
        elif len(this_safe_multiples) > 1:  # more than one tuple
            # Heuristic: choose the tuple with the highest difference between its elements
            furthest_away_tuple = max(this_safe_multiples, key=lambda t: sum(t[i+1] - t[i] for i in range(len(t)-1)))
            to_keep = this_subject_features.iloc[list(furthest_away_tuple)]
            to_keep.index = [f"{subject}${x}" for x in to_keep.index]
            res = pd.concat([res, to_keep])

    return res


def read_all_features(dataset, multiples=False) -> DataFrame:
    spectral = read_spectral_features(dataset, multiples)
    hjorth = read_hjorth_features(dataset, multiples)
    pli = read_pli_features(dataset, multiples)
    res = spectral.join(hjorth).join(pli)

    if not multiples or dataset == 'INSIGHT' or dataset == 'KJPP':
        return res
    else:
        return select_safe(res, dataset)

def read_ages(dataset: str) -> dict[str|int, float|int]:
    if dataset == 'KJPP':
        #df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata_as_given.csv', sep=';')
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/KJPP/metadata.csv', sep=';')
        #return {row['EEG_GUID']: row['AgeMonthEEG'] / 12 for _, row in df.iterrows()}
        return {row['SESSION']: row['EEG AGE MONTHS'] / 12 for _, row in df.iterrows()}
    if dataset == 'INSIGHT':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/SocioDemog.csv', sep=',')
        return {int(row['CODE']): row['AGE'] for _, row in df.iterrows()}


def read_mmse(dataset: str) -> dict[str, float|int]:
    if dataset == 'INSIGHT':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/cognition_m0.csv', sep=',')
        return {row['CODE']: int(row['MMSE']) for _, row in df.iterrows() if row['MMSE'] not in ('MD', 'NA')}
    if dataset == 'BrainLat':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/BrainLat/metadata.csv', sep=',')
        return {row['ID']: row['MMSE equivalent'] for _, row in df.iterrows()}
    if dataset == 'Miltiadous Dataset':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/participants.tsv', sep='\t')
        return {row['participant_id']: row['MMSE'] for _, row in df.iterrows()}
    if dataset == 'Sapienza':
        df = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Sapienza/metadata.csv', sep=',')
        return {row['ID']: row['MMSE'] for _, row in df.iterrows()}


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
