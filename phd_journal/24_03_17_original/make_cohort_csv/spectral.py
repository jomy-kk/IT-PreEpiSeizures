from glob import glob
from os.path import join
from os import listdir
from pickle import load

import pandas as pd
from pandas import read_csv, DataFrame

# FIXME: Change this path to the correct one. It works with all.
#datasets_path = "/Volumes/MMIS-Saraiv/Datasets/KJPP/features"
#datasets_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'
#datasets_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/features'
datasets_path = '/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/features'


# List all directories in 'features' directory with glob
all_sessions = glob(datasets_path + '/**/Spectral#Channels.csv', recursive=True)
all_dataframes = []
index = []
for session in all_sessions:
    print(session)
    x = read_csv(session)
    # check if it has index
    if 'Unnamed: 0' in x.columns:
        x.drop(columns='Unnamed: 0', inplace=True)
    all_dataframes.append(x)
    session_name = session.split('/')[-2]
    index.append(session_name)
res = pd.concat(all_dataframes, axis=0)
res.index = index

# Save to CSV
res.to_csv(join(datasets_path, 'Cohort#Spectral#Channels.csv'))
