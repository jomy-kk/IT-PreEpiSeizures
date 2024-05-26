from glob import glob
from os import remove
from os.path import join
from pickle import load

from pandas import DataFrame

# FIXME: Change this path to the correct one. It works with all.
#datasets_path = "/Volumes/MMIS-Saraiv/Datasets/KJPP/features/1"
#datasets_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'
#datasets_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/features'
datasets_path = '/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/features'

# Hjorth Mobility
name = 'Hjorth#Mobility'
cohort_files = glob(join(datasets_path, '**/hjorth_mobility.pickle'), recursive=True)
for file in cohort_files:
    print(file)
    with open(file, 'rb') as f:
        x = load(f)
        subject_session = file.split('/')[-2]
        # save as CSV
        x = DataFrame(x, index=(subject_session, ))
        x.columns = [f'{name}#{channel}' for channel in x.columns]
        x.to_csv(join(datasets_path, subject_session, f"{name}.csv"))
    # delete pickle
    remove(file)


# Hjorth Complexity
name = 'Hjorth#Complexity'
cohort_files = glob(join(datasets_path, '**/hjorth_complexity.pickle'), recursive=True)
for file in cohort_files:
    print(file)
    with open(file, 'rb') as f:
        x = load(f)
        subject_session = file.split('/')[-2]
        # save as CSV
        x = DataFrame(x, index=(subject_session, ))
        x.columns = [f'{name}#{channel}' for channel in x.columns]
        x.to_csv(join(datasets_path, subject_session, f"{name}.csv"))
    # delete pickle
    remove(file)


# Hjorth Activity
name = 'Hjorth#Activity'
cohort_files = glob(join(datasets_path, '**/hjorth_activity.pickle'), recursive=True)
for file in cohort_files:
    print(file)
    with open(file, 'rb') as f:
        x = load(f)
        subject_session = file.split('/')[-2]
        # save as CSV
        x = DataFrame(x, index=(subject_session, ))
        x.columns = [f'{name}#{channel}' for channel in x.columns]
        x.to_csv(join(datasets_path, subject_session, f"{name}.csv"))
    # delete pickle
    remove(file)

