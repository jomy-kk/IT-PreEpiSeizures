import numpy as np
import pandas as pd
from pandas import Series

from read import read_all_features, read_ages, read_patient_codes, read_gender

# 1) Read features
# 1.1. Multiples = yes
# 1.3. Which features = FEATURES_SELECTED
features = read_all_features('KJPP', multiples=True)
FEATURES_SELECTED = ['Hjorth#Complexity#T5', 'Hjorth#Complexity#F4',
                     'COH#Frontal(R)-Parietal(L)#delta', 'Hjorth#Complexity#T3',
                     'Spectral#RelativePower#F7#theta', 'COH#Frontal(R)-Temporal(L)#theta',
                     'Spectral#EdgeFrequency#O2#beta', 'COH#Frontal(L)-Temporal(R)#beta',
                     'COH#Temporal(L)-Parietal(L)#gamma', 'Spectral#EdgeFrequency#O1#beta',
                     'COH#Frontal(R)-Parietal(L)#theta', 'COH#Temporal(L)-Temporal(R)#alpha',
                     'COH#Frontal(R)-Temporal(L)#gamma', 'COH#Temporal(R)-Parietal(L)#beta',
                     'COH#Frontal(R)-Occipital(L)#theta', 'COH#Temporal(L)-Parietal(L)#beta',
                     'Hjorth#Activity#F7', 'COH#Occipital(L)-Occipital(R)#gamma',
                     'Spectral#Flatness#P3#beta', 'COH#Temporal(R)-Parietal(R)#alpha',
                     'Spectral#Entropy#P3#alpha', 'COH#Frontal(R)-Parietal(R)#theta',
                     'COH#Frontal(R)-Temporal(L)#delta', 'Spectral#Entropy#O2#alpha',
                     'Spectral#Entropy#T4#theta', 'Spectral#RelativePower#Cz#beta',
                     'Spectral#Diff#Pz#delta', 'COH#Parietal(R)-Occipital(L)#beta',
                     'Spectral#EdgeFrequency#Fz#beta', 'Spectral#Diff#Cz#gamma',
                     'Spectral#RelativePower#Fp1#gamma', 'COH#Frontal(R)-Parietal(L)#gamma',
                     'PLI#Frontal(R)-Parietal(L)#alpha', 'Spectral#Diff#F7#beta',
                     'Hjorth#Mobility#O1', 'Spectral#Flatness#T4#gamma',
                     'PLI#Parietal(L)-Occipital(L)#gamma', 'Spectral#Flatness#T6#delta',
                     'COH#Parietal(R)-Occipital(L)#alpha',
                     'COH#Parietal(R)-Occipital(R)#beta', 'Spectral#Diff#T4#delta',
                     'Spectral#Diff#F8#alpha', 'COH#Temporal(R)-Occipital(L)#beta',
                     'COH#Parietal(R)-Occipital(L)#gamma', 'Hjorth#Mobility#P4',
                     'COH#Frontal(L)-Temporal(L)#beta',
                     'COH#Occipital(L)-Occipital(R)#alpha', 'Spectral#Entropy#T3#theta',
                     'COH#Frontal(R)-Occipital(R)#alpha', 'Hjorth#Complexity#P3',
                     'COH#Frontal(L)-Occipital(L)#beta', 'Hjorth#Activity#C3',
                     'COH#Temporal(L)-Occipital(R)#theta', 'Spectral#Diff#F4#beta',
                     'COH#Frontal(L)-Frontal(R)#gamma', 'Spectral#Diff#C3#gamma',
                     'COH#Frontal(L)-Frontal(R)#theta', 'COH#Parietal(L)-Occipital(R)#theta',
                     'Spectral#RelativePower#F7#gamma', 'Spectral#RelativePower#F3#beta',
                     'PLI#Temporal(R)-Parietal(R)#beta', 'Spectral#Flatness#F7#beta',
                     'Hjorth#Complexity#O2', 'Spectral#Entropy#Cz#theta',
                     'PLI#Frontal(R)-Occipital(R)#beta', 'COH#Temporal(L)-Parietal(R)#beta',
                     'COH#Frontal(L)-Occipital(L)#delta', 'Spectral#Flatness#F8#delta',
                     'Spectral#Entropy#F4#delta', 'PLI#Temporal(R)-Parietal(R)#gamma',
                     'COH#Occipital(L)-Occipital(R)#delta',
                     'COH#Temporal(L)-Parietal(R)#delta', 'PLI#Frontal(L)-Temporal(R)#delta',
                     'Spectral#Flatness#P3#theta', 'Spectral#Entropy#F7#alpha',
                     'COH#Frontal(R)-Temporal(R)#delta', 'COH#Frontal(L)-Occipital(R)#gamma',
                     'COH#Frontal(L)-Frontal(R)#beta', 'Hjorth#Complexity#Cz',
                     'COH#Frontal(L)-Occipital(R)#beta']
features = features[FEATURES_SELECTED]
features = features.dropna()  # drop sessions with missing values
features.index = features.index.str.split('$').str[0]  # remove $ from the index

# 2) Get ages
targets = Series()
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

# Remove targets > 25
targets = targets[targets <= 25]

targets = targets.dropna()  # Drop sessions without age
features = features.loc[targets.index]

# 3) Get patient codes
session_patient_codes = Series()
codes_dict = read_patient_codes('KJPP')
n_codes_not_found = 0
for session in features.index:
    if '$' in str(session):  # Multiples
        key = str(session).split('$')[0]  # remove the multiple
    else:
        key = session
    if key in codes_dict:
        code = codes_dict[key]
        session_patient_codes.loc[session] = code
    else:
        n_codes_not_found += 1
print(f"Number of sessions without patient code: {n_codes_not_found}")
print("Total number of patients before exclusion:", len(session_patient_codes.unique()))

# 4) Get gender
genders_dict = read_gender('KJPP')


###########
# Remove the ones with bad-diagnoses
BAD_DIAGNOSES = np.loadtxt("/Volumes/MMIS-Saraiv/Datasets/KJPP/session_ids/bad_diagnoses.txt", dtype=str)
n_before = len(features)
#print("Number of sessions before removing bad-diagnoses:", n_before)
features = features.drop(BAD_DIAGNOSES, errors='ignore')
#print("Number of sessions after removing bad-diagnoses:", len(features))
targets = targets.drop(BAD_DIAGNOSES, errors='ignore')
session_patient_codes = session_patient_codes.drop(BAD_DIAGNOSES, errors='ignore')
print("Total number of patients after discarding bad-diagnosis:", len(session_patient_codes.unique()))

###########
# Get prediction-targets of scheme57
removed_sessions = np.loadtxt("/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_03_18/inverse_problem3/scheme57/removed_sessions.txt", dtype=str)
# Remove the sessions that were removed from the features
features = features.drop(removed_sessions)
targets = targets.drop(removed_sessions)
session_patient_codes = session_patient_codes.drop(removed_sessions)
#print("Number of sessions after batota removal:", len(features))
print("Total number of patients after batota:", len(session_patient_codes.unique()))


###########
# Update session_patient_codes {session: patient_code} and genders {patient_code: gender}; some of them might not exist
sessions = features.index
patients = session_patient_codes.unique()
not_found = 0
for patient in patients:
    patient_sessions = session_patient_codes[session_patient_codes == patient].index
    for session in patient_sessions:
        if session not in sessions:
            print(f"Session {session} not found in features")
            not_found += 1
            session_patient_codes = session_patient_codes.drop(session)

print("Number of sessions not found in features:", not_found)
patients = session_patient_codes.unique()

# Make/Update genders
genders_dict = {k: v for k, v in genders_dict.items() if k in patients}
genders = Series(genders_dict)

###########
# STATISTICS

# Total number of patients
print("\nTotal number of patients:", len(session_patient_codes.unique()))

# Average number of sessions per unique patient
print("Average number of sessions per unique patient:", len(features) / len(session_patient_codes.unique()))
# Min
print("Min:", session_patient_codes.value_counts().min())
# Max
print("Max:", session_patient_codes.value_counts().max())

# Total number of sessions
print("Total number of sessions:", len(features))


# Age (mean, std)
print("\nAge Statistics:")
print("Mean:", targets.mean())
print("Std:", targets.std())
print("Min:", targets.min())
print("Max:", targets.max())

# by group
age_groups = (
    (0, 8),
    (8, 13),
    (13, 25),
)
for group in age_groups:
    print(f"\nAge group {group}:")
    group_indices = targets[(targets >= group[0]) & (targets < group[1])].index
    print("Mean:", targets.loc[group_indices].mean())
    print("Std:", targets.loc[group_indices].std())
    print("Min:", targets.loc[group_indices].min())
    print("Max:", targets.loc[group_indices].max())
    # number of patients
    print("Number of patients:", len(session_patient_codes.loc[group_indices].unique())
            if len(group_indices) > 0 else 0)
    # number of sessions
    print("Number of sessions:", len(group_indices))


# Gender ('Male' count and %)
print("\nGender Statistics:")

# count "Male" in genders
n_males = 0
n_total = 0
for patient, gender in genders.items():
    if gender == 'Male':
        n_males += 1
        n_total += 1
    if gender == 'Female':
        n_total += 1

print("Number of males", n_males)
print("Percentage", n_males / n_total * 100)

# by group
for group in age_groups:
    print(f"\nAge group {group}:")
    group_indices = targets[(targets >= group[0]) & (targets < group[1])].index
    genders_group = genders.loc[session_patient_codes.loc[group_indices].unique()]
    # count
    n_males = 0
    n_total = 0
    for patient, gender in genders_group.items():
        if gender == 'Male':
            n_males += 1
            n_total += 1
        if gender == 'Female':
            n_total += 1
    print("Number of males", n_males)
    print("Percentage", n_males / n_total * 100)





