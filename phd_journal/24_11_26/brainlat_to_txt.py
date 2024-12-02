from datetime import timedelta
from glob import glob
from os import mkdir, remove
from os.path import join, exists

import numpy as np

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter

common_path = "/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_biosignal/"
out_common_path = "/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_txt_epochs/"
all_files = glob(join(common_path, "*.biosignal"))

metadata_path = "/Volumes/MMIS-Saraiv/Datasets/BrainLat/metadata.xlsx"

# Open xlsx file
import pandas as pd
metadata_ad = pd.read_excel(metadata_path, 'Demographics_AD_EEG_data', header=0, index_col=1, usecols=[0, 1])
metadata_hc = pd.read_excel(metadata_path, 'Demographics_HC_EEG_data', header=0, index_col=1, usecols=[0, 1])

channel_order = ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'T4', 'C3', 'Cz', 'C4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2')

for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    print(filename)

    # Check if subject is AR or CL
    if filename in metadata_ad.index:
        if 'AR' in str(metadata_ad.loc[filename]):
            subject_out_path = join(out_common_path, "AR")
        elif 'CL' in str(metadata_ad.loc[filename]):
            subject_out_path = join(out_common_path, "CL")
        else:
            raise ValueError(f"'AR' nor 'CL' found in metadata_ad for {filename}.")
    elif filename in metadata_hc.index:
        if 'AR' in str(metadata_hc.loc[filename]):
            subject_out_path = join(out_common_path, "AR")
        elif 'CL' in str(metadata_hc.loc[filename]):
            subject_out_path = join(out_common_path, "CL")
        else:
            raise ValueError(f"'AR' nor 'CL' found in metadata_hc for {filename}.")
    elif filename.startswith('sub-2'):  # FTD
        continue
    else:
        raise ValueError(f"Subject {filename} not found in any of the metadata excel sheets.")

    subject_out_path = join(subject_out_path, filename)
    if not exists(subject_out_path):
        mkdir(subject_out_path)
    else:
        print("Already exists.")
        continue
    # Load Biosignal
    x = EEG.load(filepath)

    # Get only signal with quality
    good = Timeline.load(join(common_path, filename + '_good.timeline'))
    x = x[good]

    print(x.channel_names)

    # Epochs
    print("Segmenting...")
    segmenter = Segmenter(timedelta(seconds=2))
    segmented_x = segmenter(x)
    print("Segmented")

    # Go by segment
    intervals = segmented_x['O2'].domain
    print("Writing...")
    for i, interval in enumerate(intervals):
        #print(f"Segment {i}")
        segment = segmented_x[interval]
        _array: np.ndarray = segment.to_array(channel_order).T
        # Make a txt in ASCII format, and put the array data there; write with maximum 4 decimal places
        with open(join(subject_out_path, f"{filename}_{i}.txt"), 'w', encoding='ascii') as f:
            np.savetxt(f, _array, fmt='%.4f', delimiter='\t')


