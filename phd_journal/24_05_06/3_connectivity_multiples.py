from datetime import timedelta
from datetime import timedelta
from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np
import pandas as pd
from pandas import DataFrame

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer

# FIXME: Change this to the correct path
#common_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_biosignal'
#out_common_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/features'
#common_path = '/Volumes/MMIS-Saraiv/Datasets/Sapienza/denoised_biosignal'
#out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Sapienza/features'
#common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/denoised_biosignal'
#out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Miltiadous Dataset/features'
#common_path = '/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/autopreprocessed_biosignal'
#out_common_path = '/Volumes/MMIS-Saraiv/Datasets/DZNE/INSIGHT/EEG/features'
common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/autopreprocessed_biosignal/2'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/features/2'

verify_quality = True
verify_subject_length = False

#############################################
# DO NOT CHANGE ANYTHING BELOW THIS LINE

# Processing tools
normalizer = Normalizer(method='minmax')

# Channels and Bands
channel_order = ('C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6')  # without mastoids

regions = {
    'Frontal(L)': ('F3', 'F7', 'Fp1'),
    'Frontal(R)': ('F4', 'F8', 'Fp2'),
    'Temporal(L)': ('T3', 'T5'),
    'Temporal(R)': ('T4', 'T6'),
    'Parietal(L)': ('C3', 'P3', ),
    'Parietal(R)': ('C4', 'P4', ),
    'Occipital(L)': ('O2', ),
    'Occipital(R)': ('O1', ),
}

bands = {
    'delta': (1.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45),
}

# Window lengths (seconds) per band
window_lengths = {
    'delta': timedelta(seconds=4),
    'theta': timedelta(seconds=4),
    'alpha': timedelta(seconds=2),
    'beta': timedelta(seconds=1),
    'gamma': timedelta(seconds=0.5)
}

# Subject lengths
SUBJECT_LENGTH = timedelta(seconds=24)
BLOCK_LENGTH = timedelta(seconds=8)


def _get_region_of(channel: str) -> str:
    for region, channels in regions.items():
        if channel in channels:
            return region
    raise ValueError(f"Channel {channel} not found in any region")


def matrix_to_series(matrix, channel_order, conn_metric, band):
    # CONVERT FROM MATRICES TO SERIES
    features = DataFrame(matrix, columns=channel_order, index=channel_order)

    # ignore Future warnings
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # REGION-REGION FEATURES

    # 1. Drop mid-line channels (everything with 'z')
    midline_channels = [ch for ch in channel_order if 'z' in ch]
    features = features.drop(columns=midline_channels)
    features = features.drop(index=midline_channels)

    # 2. Convert features from matrix to series
    features.replace(0, np.nan, inplace=True)  # it's a triangular matrix, so we can discard 0s
    features = features.stack(dropna=True)

    # 3. Populate region pairs values in a list
    # We want to average the features within the same region. Every inter-region pair is discarded.
    region_pairs = {key: [] for key in region_pair_keys}  # empty list for each region pair
    for ch_pair, value in features.items():
        chA, chB = ch_pair
        # check the region of each channel
        regionA = _get_region_of(chA)
        regionB = _get_region_of(chB)
        # if they are the same region, discard
        if regionA == regionB:
            continue
        # if they are different regions, append to the region pair to later average
        region_pair = f"{regionA}-{regionB}"
        region_pair_rev = f"{regionB}-{regionA}"
        if region_pair in region_pairs:
            region_pairs[region_pair].append(value)
        elif region_pair_rev in region_pairs:
            region_pairs[region_pair_rev].append(value)
        else:
            raise ValueError(f"Region pair {region_pair} not found in region pairs.")

    # 4. Average region pairs
    avg_region_pairs = {}
    for region_pair, values in region_pairs.items():
        avg_region_pairs[f"{conn_metric.upper()}#{region_pair}#{band}"] = np.mean(values)
    avg_region_pairs = DataFrame(avg_region_pairs, dtype='float', index=[filename, ])

    return avg_region_pairs


# Initialize region pairs
region_pair_keys = []  # 28Cr2 = 28 region pairs
region_names = tuple(regions.keys())
for i in range(len(region_names)):
    for j in range(i + 1, len(region_names)):
        region_pair_keys.append(f"{region_names[i]}-{region_names[j]}")

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '**/*.biosignal'), recursive=True)

for filepath in all_files:
    # Identifiers
    filename = filepath.split('/')[-1].split('.')[0]
    subject_out_path = join(out_common_path, filename)
    if not exists(subject_out_path):
        mkdir(subject_out_path)
    #if exists(join(subject_out_path, 'Connectivity#Regions$Multiple.csv')):
    #    print('Already processed. Skipping.')
    #    continue
    print(filename)
    # Load Biosignal
    x = EEG.load(filepath)
    x = x[channel_order]  # get only channels of interest

    # Get only signal with quality
    if verify_quality:
        good = Timeline.load(join(common_path, filename + '_good.timeline'))
        x = x[good]

    # Normalize
    x = normalizer(x)

    # GET INDIVIDUAL MATRICES
    all_windows = []
    S = SUBJECT_LENGTH.total_seconds() / BLOCK_LENGTH.total_seconds()

    # Go by existing discontiguous segments
    domain = x['T5'].domain
    for i, interval in enumerate(domain):
        z = x[interval]
        if verify_subject_length and z.duration < SUBJECT_LENGTH:
            continue

        # Segment in blocks
        segmenter = Segmenter(BLOCK_LENGTH, timedelta(seconds=0))
        z = segmenter(z)
        z_domain = z['T5'].domain

        # remove last block if it is incomplete
        if z_domain[-1].timedelta < BLOCK_LENGTH:
            z_domain = z_domain[:-1]

        features_by_blocks = []
        for j, window in enumerate(z_domain):
            if window.timedelta < BLOCK_LENGTH:
                continue
            if verify_subject_length and len(z_domain) - j < S - len(features_by_blocks):
                break  # insufficient blocks to make a subject

            y = z[window]

            window_features = {}

            # Compute for each band
            for band, freqs in bands.items():
                #print(band)
                pli_matrix = y.pli(window_length=window_lengths[band], fmin=freqs[0], fmax=freqs[1], channel_order=channel_order)
                coh_matrix = y.coh(window_length=window_lengths[band], fmin=freqs[0], fmax=freqs[1], channel_order=channel_order)

                # Check if there is any NaN
                if np.isnan(pli_matrix).any():
                    print(f'A window with NaNs was found.')
                    print(f"Subject {filename} | Segment {i + 1} | Duration {z.duration} | Metric PLI")
                    continue
                if np.isnan(coh_matrix).any():
                    print(f'A window with NaNs was found.')
                    print(f"Subject {filename} | Segment {i + 1} | Duration {z.duration} | Metric PLI")
                    continue

                name = f"PLI#{band}"
                window_features[name] = pli_matrix

                name = f"COH#{band}"
                window_features[name] = coh_matrix

                pass

            if verify_subject_length:
                if len(features_by_blocks) < S - 1:
                    features_by_blocks.append(window_features)
                    pass
                else:  # Average matrices
                    features_by_blocks.append(window_features)
                    F = {}
                    for label in features_by_blocks[0].keys():
                        F[label] = np.mean([block[label] for block in features_by_blocks], axis=0)
                    all_windows.append(F)
                    features_by_blocks = []
            else:
                features_by_blocks.append(window_features)
                pass

        if not verify_subject_length and len(features_by_blocks) > 0: # Average matrices
            F = {}
            for label in features_by_blocks[0].keys():
                F[label] = np.mean([block[label] for block in features_by_blocks], axis=0)
            all_windows += [F, ]
            pass

    if len(all_windows) == 0:
        print(f"No windows found for subject {filename}. Skipping.")
        continue

    all_fake_subjects = []
    for f, fake_subject in enumerate(all_windows):
        all_region_pairs = []
        for label, matrix in fake_subject.items():
            conn_metric, band = label.split('#')
            region_pairs = matrix_to_series(matrix, channel_order, conn_metric, band)
            all_region_pairs.append(region_pairs)

        all_region_pairs = pd.concat(all_region_pairs, axis=1)
        all_fake_subjects.append(all_region_pairs)

    all_fake_subjects = pd.concat(all_fake_subjects, axis=0)
    all_fake_subjects.index = [f"{filename}${i+1}" for i in range(len(all_fake_subjects))]
    all_fake_subjects.to_csv(join(subject_out_path, f'Connectivity#Regions$Multiple.csv'))
