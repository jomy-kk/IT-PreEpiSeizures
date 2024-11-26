from datetime import timedelta
from glob import glob
from os import mkdir, remove
from os.path import join, exists

import numpy as np

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter

common_path = "/Volumes/MMIS-Saraiv/Datasets/Istambul/denoised_biosignal/"
out_common_path = "/Volumes/MMIS-Saraiv/Datasets/Istambul/denoised_txt_epochs/"
all_files = glob(join(common_path, "*.biosignal"))

channel_order = ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2')

for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    print(filename)
    subject_out_path = join(out_common_path, filename)
    if not exists(subject_out_path):
        mkdir(subject_out_path)
    # Load Biosignal
    x = EEG.load(filepath)

    # Get only signal with quality
    good = Timeline.load(join(common_path, filename + '_good.timeline'))
    x = x[good]

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


