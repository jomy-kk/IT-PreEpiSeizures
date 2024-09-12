from datetime import timedelta
from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np
import pandas as pd

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer

# FIXME: Change this to the correct path
common_path = '/Volumes/MMIS-Saraiv/Datasets/Healthy Brain Network/biosignal'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Healthy Brain Network/features'

verify_quality = True
verify_subject_length = False

#################################
# DO NOT CHANGE ANYTHING BELOW

BLOCK_LENGTH = timedelta(seconds=4)
SUBJECT_LENGTH = timedelta(seconds=24)

# Processing tools
normalizer = Normalizer(method='minmax')

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '*.biosignal'))

for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    print(filename)

    # Load
    x = EEG.load(filepath)
    if verify_quality:
        good = Timeline.load(join(common_path, filename + '_good.timeline'))
        x = x[good]
    domain = x['T5'].domain

    # Normalize
    x = normalizer(x)

    # Go by existing discontiguous segments
    all_activity, all_mobility, all_complexity = {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}
    for i, interval in enumerate(domain):
        z = x[interval]
        if z.duration < BLOCK_LENGTH:
            continue
        print(f"Segment {i + 1} of {len(domain)}")

        this_seg_activity, this_seg_mobility, this_seg_complexity = {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}, {ch: [] for ch in x.channel_names}

        # Segment in blocks
        segmenter = Segmenter(BLOCK_LENGTH)
        z = segmenter(z)
        z_domain = z['T5'].domain
        short_blocks = [i for i, w in enumerate(z_domain) if w.timedelta < BLOCK_LENGTH]

        # Compute features for all blocks
        activity = z.hjorth_activity()
        for k, v in activity.items():
            this_seg_activity[k].extend(score for i, score in enumerate(v) if i not in short_blocks)
        mobility = z.hjorth_mobility()
        for k, v in mobility.items():
            this_seg_mobility[k].extend(score for i, score in enumerate(v) if i not in short_blocks)
        complexity = z.hjorth_complexity()
        for k, v in complexity.items():
            this_seg_complexity[k].extend(score for i, score in enumerate(v) if i not in short_blocks)

        if verify_subject_length:
            # Average every S windows. Each average must contain exactly S windows, if not, we discard the last ones.
            S = int(SUBJECT_LENGTH.total_seconds() / BLOCK_LENGTH.total_seconds())
            this_seg_activity = {k: [np.mean(v[i:i + S]) for i in range(0, len(v) - len(v) % S, S)] for k, v in
                                 this_seg_activity.items()}
            this_seg_mobility = {k: [np.mean(v[i:i + S]) for i in range(0, len(v) - len(v) % S, S)] for k, v in
                                 this_seg_mobility.items()}
            this_seg_complexity = {k: [np.mean(v[i:i + S]) for i in range(0, len(v) - len(v) % S, S)] for k, v in
                                   this_seg_complexity.items()}
        else:
            # Average all windows together in the end
            pass

        # Append to all
        for k, v in this_seg_activity.items():
            all_activity[k].extend(v)
        for k, v in this_seg_mobility.items():
            all_mobility[k].extend(v)
        for k, v in this_seg_complexity.items():
            all_complexity[k].extend(v)

        if not verify_subject_length:
            all_activity = {k: [np.mean(v)] for k, v in all_activity.items()}
            all_mobility = {k: [np.mean(v)] for k, v in all_mobility.items()}
            all_complexity = {k: [np.mean(v)] for k, v in all_complexity.items()}

        pass

    total_examples = len(all_activity['T5'])
    print(f"Total: {total_examples} examples.")

    # Save
    subject_out_path = join(out_common_path, filename)
    #if not exists(subject_out_path):
    #    mkdir(subject_out_path)

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
