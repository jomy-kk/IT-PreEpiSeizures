from glob import glob
from os import listdir
from os.path import join

import pandas as pd
from pandas import read_csv

datasets_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'

# SPECTRAL
# List all directories in 'features' directory with glob
all_sessions = glob(datasets_path + '/**/Spectral#Channels$Multiple.csv', recursive=True)
all_dataframes = []
for session in all_sessions:
    print(session)
    x = read_csv(session, index_col=0)
    all_dataframes.append(x)
    session_name = session.split('/')[-2]
res = pd.concat(all_dataframes, axis=0)

# Save to CSV
res.to_csv(join(datasets_path, 'Cohort#Spectral#Channels$Multiple.csv'))

# CONNECTIVITY
# Find all the files with the given name
all_files = glob(join(datasets_path, '**', 'Connectivity#Regions$Multiple.csv'), recursive=True)
# Load all and concatenate
df = pd.concat([pd.read_csv(f, index_col=0) for f in all_files])
# Save in root
df.to_csv(join(datasets_path, 'Cohort#Connectivity#Regions$Multiple.csv'))


# HJORTH
all_sessions = listdir(datasets_path)  # List all directories in 'features' directory with os.listdir
all_dataframes = []
for session in all_sessions:
    this_session_dataframe = None
    session_path = join(datasets_path, session)
    hjorth_files = glob(join(session_path, 'Hjorth#*$Multiple.csv'))
    for file in hjorth_files:
        x = read_csv(file, index_col=0)
        if this_session_dataframe is None:
            this_session_dataframe = x
        else:
            this_session_dataframe = pd.concat([this_session_dataframe, x], axis=1)
    all_dataframes.append(this_session_dataframe)
res = pd.concat(all_dataframes, axis=0)

# Save to CSV
res.to_csv(join(datasets_path, 'Cohort#Hjorth#Channels$Multiple.csv'))
