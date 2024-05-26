import pickle
from datetime import timedelta
from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np
import pandas as pd

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.clinical.Patient import Sex
from ltbio.processing.formaters import Segmenter, Normalizer

# FIXME: Change common_path and out_common_path to applicable paths
common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/denoised_biosignal'

REJECT = ['sub-30005', 'sub-30007', 'sub-30010', 'sub-30014', 'sub-30030', 'sub-30032', ]

#################################
# DO NOT CHANGE ANYTHING BELOW

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '**/*.biosignal'), recursive=True)


n_sessions = 0
ages, genders = np.array([]), np.array([])
total_durations = np.array([])
useful_durations = np.array([])


for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]

    if filename in REJECT:
        continue

    print(filename)

    # Load
    x = EEG.load(filepath)
    n_sessions += 1

    # Get demographics
    age, sex = x._Biosignal__patient._Patient__age, x._Biosignal__patient._Patient__sex
    ages = np.append(ages, float(age))
    genders = np.append(genders, 1 if sex is Sex.M else 0 if sex is Sex.F else None)

    # Get duration
    total_durations = np.append(total_durations, x.duration.total_seconds()/60)

    # Get useful duration
    good = Timeline.load(join(common_path, filename + '_good.timeline'))
    x = x[good]
    useful_durations = np.append(useful_durations, x.duration.total_seconds()/60)

    # Delete the object to free memory
    del x

# Print results
print(f"Total sessions: {n_sessions}")
print(f"Mean age: {np.mean(ages)}")
print(f"Std age: {np.std(ages)}")
print(f"Age range: {np.min(ages)} - {np.max(ages)}")
print(f"{np.sum(genders)/n_sessions*100}% male, {(1-(np.sum(genders)/n_sessions))*100}% female")
print(f"Total duration: {np.sum(total_durations)}")
print(f"Useful duration: {np.sum(useful_durations)}")
print(f"Duration range: {np.min(useful_durations)} - {np.max(useful_durations)}")
print(f"Useful Duration range: {np.min(useful_durations)} - {np.max(useful_durations)}")
