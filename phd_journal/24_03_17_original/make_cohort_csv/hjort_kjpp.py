from glob import glob
from os.path import join
from os import listdir
from pickle import load

import pandas as pd
from pandas import read_csv, DataFrame

# FIXME: Change this path to the correct one. It works with all.
datasets_path = "/Volumes/MMIS-Saraiv/Datasets/KJPP/features/1"
#datasets_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'
#datasets_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/features'
#datasets_path = '/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/features'


all_sessions = listdir(datasets_path)  # List all directories in 'features' directory with os.listdir
all_dataframes = []
for session in all_sessions:
    this_session_dataframe = None
    session_path = join(datasets_path, session)
    hjorth_files = glob(join(session_path, 'Hjorth#*.csv'))
    for file in hjorth_files:
        x = read_csv(file, index_col=0)
        if this_session_dataframe is None:
            this_session_dataframe = x
        else:
            this_session_dataframe = pd.concat([this_session_dataframe, x], axis=1)
    all_dataframes.append(this_session_dataframe)
res = pd.concat(all_dataframes, axis=0)

# Only for KJPP
datasets_path = "/Volumes/MMIS-Saraiv/Datasets/KJPP/features/2"
all_sessions = listdir(datasets_path)  # List all directories in 'features' directory with os.listdir
all_dataframes = []
for session in all_sessions:
    this_session_dataframe = None
    session_path = join(datasets_path, session)
    hjorth_files = glob(join(session_path, 'Hjorth#*.csv'))
    for file in hjorth_files:
        x = read_csv(file, index_col=0)
        if this_session_dataframe is None:
            this_session_dataframe = x
        else:
            this_session_dataframe = pd.concat([this_session_dataframe, x], axis=1)
    all_dataframes.append(this_session_dataframe)

all_dataframes.append(res)
res = pd.concat(all_dataframes, axis=0)

# Save to CSV
res.to_csv(join(datasets_path, 'Cohort#Hjorth#Channels.csv'))
