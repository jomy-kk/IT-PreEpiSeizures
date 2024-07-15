from os.path import join

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.viz import plot_topomap

from read import read_all_features, read_mmse, read_ages
from utils import feature_wise_normalisation_with_coeffs, feature_wise_normalisation


def read_elders():
    # 1) Read features
    # 1.1. Multiples = yes
    # 1.2. Which multiples = all
    # 1.3. Which features = all
    miltiadous = read_all_features('Miltiadous Dataset', multiples=True)
    brainlat = read_all_features('BrainLat', multiples=True)
    sapienza = read_all_features('Sapienza', multiples=True)
    insight = read_all_features('INSIGHT', multiples=True)
    features = pd.concat([brainlat, miltiadous, sapienza, insight], axis=0)
    print("Features Shape:", features.shape)

    # 2) Read targets
    insight_targets = read_mmse('INSIGHT')
    brainlat_targets = read_mmse('BrainLat')
    miltiadous_targets = read_mmse('Miltiadous Dataset')
    sapienza_targets = read_mmse('Sapienza')
    targets = pd.Series()
    batch = []
    for index in features.index:
        if '$' in str(index):  # Multiples
            key = str(index).split('$')[0]  # remove the multiple
        else:  # Original
            key = index

        if '_' in str(key):  # insight
            key = int(key.split('_')[0])
            if key in insight_targets:
                targets.loc[index] = insight_targets[key]
                batch.append(1)
        elif '-' in str(key):  # brainlat
            if key in brainlat_targets:
                targets.loc[index] = brainlat_targets[key]
                batch.append(2)
        elif 'PARTICIPANT' in str(key):  # sapienza
            if key in sapienza_targets:
                targets.loc[index] = sapienza_targets[key]
                batch.append(3)
        else:  # miltiadous
            # parse e.g. 24 -> 'sub-024'; 1 -> 'sub-001'
            key = 'sub-' + str(key).zfill(3)
            if key:
                targets.loc[index] = miltiadous_targets[key]
                batch.append(4)
    targets = targets.dropna()  # Drop subject_sessions with nans targets
    features = features.loc[targets.index]

    # 3) Normalize features min-max
    features = (features - features.min()) / (features.max() - features.min())

    return features, targets


def read_children():
    # 1) Read features
    # 1.1. Multiples = yes
    # 1.3. Which features = FEATURES_SELECTED
    features = read_all_features('KJPP', multiples=True)
    features.index = features.index.str.split('$').str[0]  # remove $ from the index

    # 1.2.1) Remove the ones with bad-diagnoses
    BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
    n_before = len(features)
    features = features.drop(BAD_DIAGNOSES, errors='ignore')
    print("Removed Bad diagnoses:", n_before - len(features))

    # 1.2.2) Remove others
    # 1.2.2) Remove others
    REMOVED_SESSIONS = np.loadtxt("/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_03_18/inverse_problem3/scheme57/removed_sessions.txt", dtype=str)
    n_before = len(features)
    features = features.drop(REMOVED_SESSIONS, errors='ignore')
    print("Removed:", n_before - len(features))

    # 2) Get targerts
    targets = pd.Series()
    ages = read_ages('KJPP')
    n_age_not_found = 0
    for session in features.index:
        if '$' in str(session):  # Multiples
            key = str(session).split('$')[0]  # remove the multiple
        else:
            key = session
        if key in ages:
            age = ages[key]
            targets.loc[session] = age
        else:
            print(f"Session {session} not found in ages")
            n_age_not_found += 1
    print(f"Number of sessions without age: {n_age_not_found}")
    targets = targets.dropna()  # Drop sessions without age
    features = features.loc[targets.index]

    # 3) Normalisation
    # 3.1. Normalisation method = min-max
    features = feature_wise_normalisation(features, 'min-max')

    return features, targets


########

# Channels Info
channel_order = ['F4', 'T3', ]
# Create an Info object, necessary for creating Evoked object
info = mne.create_info(ch_names=channel_order, sfreq=1, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

def distance(F, T, groups, label):
    for i, group in enumerate(groups):
        # Filter by target
        F_group = F.loc[T.between(*group)]

        print("Group:", group)
        #print("Number of examples:", F_group.shape[0])

        # Keep only one session per subject and discard remainder
        F_group = F_group.groupby(F_group.index.str.split('$').str[0]).mean()

        # Average all sessions
        F_group = F_group.mean(axis=0).values

        # Compute the distance between the both channels
        distance = np.linalg.norm(F_group[0] - F_group[1])
        print("Distance:", distance)



#######


feature_name = 'Hjorth#Complexity'
feature_names = ['{}#{}'.format(feature_name, channel) for channel in channel_order]

#"""
# Elders
features_elders, targets_elders = read_elders()
features_elders = features_elders[feature_names]
distance(features_elders, targets_elders, ((0, 12), (13, 19), (20, 24), (25, 30)), label='mmse')


#"""
# Children
features_children, targets_children = read_children()
features_children = features_children[feature_names]
distance(features_children, targets_children, ((0, 5), (5, 8), (8, 13), (13, 19)), label='age')
#"""
