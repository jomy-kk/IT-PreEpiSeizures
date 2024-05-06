import pickle
from datetime import timedelta
from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np
import pandas as pd

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer

# FIXME: Change common_path and out_common_path to applicable paths
common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/denoised_biosignal'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'

#################################
# DO NOT CHANGE ANYTHING BELOW

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '*.biosignal'))

# Processing tools
normalizer = Normalizer(method='minmax')
segmenter = Segmenter(timedelta(seconds=5))

# Read targets
mmse = pd.read_csv('/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/participants.tsv', sep='\t', index_col=0)
mmse = mmse['MMSE']

for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    print(filename)

    # Process only if MMSE is <= 15
    if mmse['sub-' + filename] > 15:
        print(f"Skipping because MMSE is > 15")
        continue

    # Load
    x = EEG.load(filepath)
    good = Timeline.load(join(common_path, filename + '_good.timeline'))
    x = x[good]
    domain = x['T5'].domain

    # Normalize
    x = normalizer(x)

    # Traverse segments
    all_activity, all_mobility, all_complexity = {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}
    for i, interval in enumerate(domain):
        z = x[interval]
        if z.duration < timedelta(seconds=5):
            continue
        print(f"Segment {i + 1} of {len(domain)}")

        this_seg_activity, this_seg_mobility, this_seg_complexity = {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}

        # Segment in windows of 5 seconds
        z = segmenter(z)
        z_domain = z['T5'].domain
        windows_less_than_5_seconds = [i for i, w in enumerate(z_domain) if w.timedelta < timedelta(seconds=5)]

        # Compute features for all windows
        activity = z.hjorth_activity()
        for k, v in activity.items():
            this_seg_activity[k].extend(score for i, score in enumerate(v) if i not in windows_less_than_5_seconds)
        mobility = z.hjorth_mobility()
        for k, v in mobility.items():
            this_seg_mobility[k].extend(score for i, score in enumerate(v) if i not in windows_less_than_5_seconds)
        complexity = z.hjorth_complexity()
        for k, v in complexity.items():
            this_seg_complexity[k].extend(score for i, score in enumerate(v) if i not in windows_less_than_5_seconds)

        # Average every 5 windows. Each average must contain exactly 5 windows, if not, we discard the last ones.
        this_seg_activity = {k: [np.mean(v[i:i + 5]) for i in range(0, len(v) - len(v) % 5, 5)] for k, v in
                             this_seg_activity.items()}
        this_seg_mobility = {k: [np.mean(v[i:i + 5]) for i in range(0, len(v) - len(v) % 5, 5)] for k, v in
                             this_seg_mobility.items()}
        this_seg_complexity = {k: [np.mean(v[i:i + 5]) for i in range(0, len(v) - len(v) % 5, 5)] for k, v in
                               this_seg_complexity.items()}
        print(f"Averaged {len(z_domain) - len(windows_less_than_5_seconds)} 5s windows in groups of 5. Hence, {len(this_seg_activity['T5'])} examples.")

        # Append to all
        for k, v in this_seg_activity.items():
            all_activity[k].extend(v)
        for k, v in this_seg_mobility.items():
            all_mobility[k].extend(v)
        for k, v in this_seg_complexity.items():
            all_complexity[k].extend(v)

        pass

    total_examples = len(all_activity['T5'])
    print(f"Total: {total_examples} examples.")

    # Save
    subject_out_path = join(out_common_path, filename)
    if not exists(subject_out_path):
        mkdir(subject_out_path)

    # Make DataFrames and save as CSV
    all_activity = pd.DataFrame(all_activity, index=[f'{filename}${i+1}' for i in range(total_examples)], columns=all_activity.keys())
    all_activity.columns = [f"Hjorth#Activity#{ch}" for ch in all_activity.columns]
    all_activity.to_csv(join(subject_out_path, 'Hjorth#Activity$Multiple.csv'))

    all_mobility = pd.DataFrame(all_mobility, index=[f'{filename}${i+1}' for i in range(total_examples)], columns=all_mobility.keys())
    all_mobility.columns = [f"Hjorth#Mobility#{ch}" for ch in all_mobility.columns]
    all_mobility.to_csv(join(subject_out_path, 'Hjorth#Mobility$Multiple.csv'))

    all_complexity = pd.DataFrame(all_complexity, index=[f'{filename}${i+1}' for i in range(total_examples)], columns=all_complexity.keys())
    all_complexity.columns = [f"Hjorth#Complexity#{ch}" for ch in all_complexity.columns]
    all_complexity.to_csv(join(subject_out_path, 'Hjorth#Complexity$Multiple.csv'))
