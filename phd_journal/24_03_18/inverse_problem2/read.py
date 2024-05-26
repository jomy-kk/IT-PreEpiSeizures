from os.path import join
from typing import Collection

import numpy as np
import pandas as pd
from pandas import DataFrame
from glob import glob

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from utils import feature_wise_normalisation

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
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Connectivity#Regions.csv')
    else:
        path = join(common_datasets_path, dataset, features_dir, 'Cohort#Connectivity#Channels.csv')
    res = pd.read_csv(path, index_col=0)
    return res


def read_all_features(dataset) -> DataFrame:
    spectral = read_spectral_features(dataset)
    hjorth = read_hjorth_features(dataset)
    pli = read_pli_features(dataset)
    return spectral.join(hjorth).join(pli)


def read_all_features_multiples() -> DataFrame:
    connectivity = pd.read_csv(join(common_datasets_path, 'Miltiadous Dataset', features_dir, 'Cohort#Connectivity#Regions$Multiple.csv'), index_col=0)
    spectral = pd.read_csv(join(common_datasets_path, 'Miltiadous Dataset', features_dir, 'Cohort#Spectral#Channels$Multiple.csv'), index_col=0)
    hjorth = pd.read_csv(join(common_datasets_path, 'Miltiadous Dataset', features_dir, 'Cohort#Hjorth#Channels$Multiple.csv'), index_col=0)
    return pd.concat([connectivity, spectral, hjorth], axis=1)


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


def read_elders_cohorts(insight_cohort=True,
                        brainlat_cohort=True,
                        miltiadous_cohort=True, miltiadous_multiples=True,
                        sapienza_cohort=True,
                        select_features = None) -> tuple[pd.DataFrame, pd.Series]:

    all_cohorts = []
    discarded_total = 0

    if insight_cohort:
        insight = read_all_features('INSIGHT')
        if select_features:
            insight = insight[select_features]
        print("INSIGHT shape (all):", insight.shape)
        insight_before = insight.shape[0]
        insight = insight.dropna(axis=0)  # drop sessions with missing values
        insight_after = insight.shape[0]
        print("INSIGHT shape (sessions w/ required features):", insight.shape,
              f"({insight_before - insight_after} sessions dropped)")
        all_cohorts.append(insight)
        discarded_total += insight_before - insight_after

    if brainlat_cohort:
        brainlat = read_all_features('BrainLat')
        if select_features:
            brainlat = brainlat[select_features]
        print("BrainLat shape (all):", brainlat.shape)
        brainlat_before = brainlat.shape[0]
        brainlat = brainlat.dropna(axis=0)  # drop sessions with missing values
        brainlat_after = brainlat.shape[0]
        print("BrainLat shape (sessions w/ required features):", brainlat.shape,
              f"({brainlat_before - brainlat_after} sessions dropped)")
        all_cohorts.append(brainlat)
        discarded_total += brainlat_before - brainlat_after

    if miltiadous_cohort:
        miltiadous = read_all_features('Miltiadous Dataset')
        if select_features:
            miltiadous = miltiadous[select_features]
        print("Miltiadous shape (all):", miltiadous.shape)
        miltiadous_before = miltiadous.shape[0]
        miltiadous = miltiadous.dropna(axis=0)  # drop sessions with missing values
        miltiadous_after = miltiadous.shape[0]
        print("Miltiadous shape (sessions w/ required features):", miltiadous.shape,
              f"({miltiadous_before - miltiadous_after} sessions dropped)")
        all_cohorts.append(miltiadous)
        discarded_total += miltiadous_before - miltiadous_after

    if sapienza_cohort:
        sapienza = read_all_features('Sapienza')
        if select_features:
            sapienza = sapienza[select_features]
        print("Sapienza shape (all):", sapienza.shape)
        sapienza_before = sapienza.shape[0]
        sapienza = sapienza.dropna(axis=0)  # drop sessions with missing values
        sapienza_after = sapienza.shape[0]
        print("Sapienza shape (sessions w/ required features):", sapienza.shape,
              f"({sapienza_before - sapienza_after} sessions dropped)")
        all_cohorts.append(sapienza)
        discarded_total += sapienza_before - sapienza_after

    if miltiadous_multiples:
        # EXTRA: Read multiples examples (from fake subjects)
        multiples = read_all_features_multiples()
        if select_features:
            multiples = multiples[select_features]
        print("Multiples shape:", multiples.shape)
        multiples_before = multiples.shape[0]
        multiples = multiples.dropna(axis=0)  # drop sessions with missing values
        multiples_after = multiples.shape[0]
        print("Multiples shape (sessions w/ required features):", multiples.shape,
              f"({multiples_before - multiples_after} sessions dropped)")

        # Perturb the multiple features, so they are not identical to the original ones
        # These sigma values were defined based on similarity with the original features; the goal is to make them disimilar inasmuch as other examples from other subjects.
        jitter = lambda x: x + np.random.normal(0, 0.1, x.shape)
        scaling = lambda x: x * np.random.normal(1, 0.04, x.shape)
        print("Perturbing multiple examples...")
        for feature in multiples.columns:
            data = multiples[feature].values
            data = jitter(data)
            data = scaling(data)
            multiples[feature] = data

        all_cohorts.append(multiples)
        discarded_total += multiples_before - multiples_after

    features = pd.concat(all_cohorts, axis=0)
    print("Read all features. Final Shape:", features.shape)
    print(
        f"Discarded a total of {discarded_total} sessions with missing values")

    # 1.2) Normalise feature vectors
    features = feature_wise_normalisation(features, method='min-max')
    features = features.dropna(axis=1)
    print("Normalised features.")

    # 2) Read targets
    insight_targets = read_mmse('INSIGHT')
    brainlat_targets = read_mmse('BrainLat')
    miltiadous_targets = read_mmse('Miltiadous Dataset')
    sapienza_targets = read_mmse('Sapienza')
    targets = pd.Series()
    for index in features.index:
        if '_' in str(index):  # insight
            key = int(index.split('_')[0])
            if key in insight_targets:
                targets.loc[index] = insight_targets[key]
        elif '-' in str(index):  # brainlat
            if index in brainlat_targets:
                targets.loc[index] = brainlat_targets[index]
        elif 'PARTICIPANT' in str(index):  # sapienza
            if index in sapienza_targets:
                targets.loc[index] = sapienza_targets[index]
        else:  # miltiadous
            # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
            if '$' in str(
                    index):  # EXTRA: multiple examples, remove the $ and the number after it; the target is the same
                key = 'sub-' + str(str(index).split('$')[0]).zfill(3)
            else:
                key = 'sub-' + str(index).zfill(3)
            if key:
                targets.loc[index] = miltiadous_targets[key]

    print("Read targets. Shape:", targets.shape)

    # Drop subject_sessions with nans targets
    targets = targets.dropna()
    features_sessions_before = set(features.index)
    features = features.loc[targets.index]
    features_sessions_after = set(features.index)
    print("After Dropping sessions with no targets - Shape:", features.shape)
    print("Dropped sessions:", features_sessions_before - features_sessions_after)

    return features, targets


def read_children_cohorts(kjpp_cohort=True, select_features=None) -> tuple[pd.DataFrame, pd.Series]:
    # 1) Get all features
    features = read_all_features('KJPP')

    # 1.1) Select features
    if select_features:
        features = features[select_features]
        print("Number of features selected:", len(features.columns))

    # Drop sessions with missing values
    features = features.dropna()

    # 2) Get targerts
    targets = pd.Series()
    ages = read_ages('KJPP')
    for session in features.index:
        age = ages[session]
        targets.loc[session] = age

    # Drop targets with missing values
    targets = targets.dropna()
    features = features.loc[targets.index]

    # 3) Normalise features
    features = feature_wise_normalisation(features, 'min-max')

    return features, targets
