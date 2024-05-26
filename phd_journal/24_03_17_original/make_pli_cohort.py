from glob import glob
from os.path import join

import pandas as pd

# FIXME: Change the path to the dataset
#dataset_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'
#dataset_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/features'
dataset_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/features'
#dataset_path = '/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/features'

for filename in ('PLI#Channels.csv', 'PLI#Regions.csv'):
    # Find all the files with the given name
    all_files = glob(join(dataset_path, '**', filename), recursive=True)
    # Load all and concatenate
    df = pd.concat([pd.read_csv(f, index_col=0) for f in all_files])
    # Save in root
    df.to_csv(join(dataset_path, 'Cohort#'+filename))
