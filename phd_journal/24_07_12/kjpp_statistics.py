import numpy as np
import pandas as pd
from pandas import Series

from read import read_all_features, read_ages, read_patient_codes, read_gender

# 1) Which sessions where used (Scheme 59)
SESSIONS = np.loadtxt("/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_03_18/inverse_problem3/scheme59/used_sessions.txt", dtype=str)

# 2) Get ages
targets = Series()
ages = read_ages('KJPP')
n_age_not_found = 0
for session in SESSIONS:
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

# 3) Get patient codes
session_patient_codes = Series()
codes_dict = read_patient_codes('KJPP')
n_codes_not_found = 0
for session in SESSIONS:
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
# Update session_patient_codes {session: patient_code} and genders {patient_code: gender}; some of them might not exist
patients = session_patient_codes.unique()
not_found = 0
for patient in patients:
    patient_sessions = session_patient_codes[session_patient_codes == patient].index
    for session in patient_sessions:
        if session not in SESSIONS:
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
print("Average number of sessions per unique patient:", len(SESSIONS) / len(session_patient_codes.unique()))
# Min
print("Min:", session_patient_codes.value_counts().min())
# Max
print("Max:", session_patient_codes.value_counts().max())

# Total number of sessions
print("Total number of sessions:", len(SESSIONS))


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





