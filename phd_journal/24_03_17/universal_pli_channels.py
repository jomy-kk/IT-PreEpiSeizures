import pickle
from datetime import timedelta
from glob import glob
from os import mkdir, remove
from os.path import join, exists

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.processing.formaters import Segmenter, Normalizer

# FIXME: Change this to the correct path
common_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/denoised_biosignal'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/BrainLat/features'


#############################################
# DO NOT CHANGE ANYTHING BELOW THIS LINE

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '**/*.biosignal'), recursive=True)

# Processing tools
normalizer = Normalizer(method='minmax')
WINDOW_LENGTH = timedelta(seconds=4)
segmenter = Segmenter(WINDOW_LENGTH)

# Channels and Bands
channel_order = ('C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6')  # without mastoids
bands = {
    'delta': (1.5, 4),
    'theta': (4, 8),
    'alpha1': (8, 10),
    'alpha2': (10, 12),
    'beta1': (12, 15),
    'beta2': (15, 20),
    'beta3': (20, 30),
    'gamma': (30, 45),
}
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

def _get_region_of(channel: str) -> str:
    for region, channels in regions.items():
        if channel in channels:
            return region
    raise ValueError(f"Channel {channel} not found in any region")


# Initialize region pairs
region_pair_keys = []  # 28Cr2 = 28 region pairs
region_names = tuple(regions.keys())
for i in range(len(region_names)):
    for j in range(i + 1, len(region_names)):
        region_pair_keys.append(f"{region_names[i]}-{region_names[j]}")

for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    print(filename)

    subject_out_path = join(out_common_path, filename)
    if not exists(subject_out_path):
        mkdir(subject_out_path)

    if exists(join(subject_out_path, 'Connectivity#Regions.csv')):
        print('Already processed. Skipping.')
        continue

    # Load Biosignal
    x = EEG.load(filepath)
    x = x[channel_order]  # get only channels of interest

    # Get only signal with quality
    good = Timeline.load(join(common_path, filename + '_good.timeline'))
    x = x[good]

    # Normalize
    x = normalizer(x)

    all_metrics_channels = []
    all_metrics_regions = []
    for conn_metric in ('pli', 'coh', 'ppc'):  # appologies for the variables in this cycle being named with 'pli'
        print(conn_metric)

        # Do for each band
        all_channels_pli = []
        all_regions_pli = []
        for band, freqs in bands.items():
            print(band)

            # GET INDIVIDUAL MATRICES
            # Traverse segments
            all_pli = []
            durations = []
            domain = x['T5'].domain
            for i, interval in enumerate(domain):
                z = x[interval]
                if z.duration < WINDOW_LENGTH:
                    continue

                # Compute connectivity metric and keep only the values of windows of 4 seconds
                if conn_metric == 'pli':
                    pli = z.pli(window_length=WINDOW_LENGTH, fmin=freqs[0], fmax=freqs[1], channel_order=channel_order)
                elif conn_metric == 'coh':
                    pli = z.coh(window_length=WINDOW_LENGTH, fmin=freqs[0], fmax=freqs[1], channel_order=channel_order)
                elif conn_metric == 'ppc':
                    pli = z.ppc(window_length=WINDOW_LENGTH, fmin=freqs[0], fmax=freqs[1], channel_order=channel_order)
                elif conn_metric == 'gc':
                    pli = z.gc(window_length=WINDOW_LENGTH, fmin=freqs[0], fmax=freqs[1], channel_order=channel_order)
                else:
                    raise ValueError(f'Unknown connectivity metric {conn_metric}')

                # Check if there is any NaN
                if np.isnan(pli).any():
                    print(f'NaN was found. Discarding.')
                    continue
                all_pli.append(pli)
                durations.append(z.duration)

            # Average all windows
            if len(all_pli) == 0:
                print(f'No good windows. Not saving this subject-session-band.')
                continue
            all_pli = np.array(all_pli)
            all_pli = np.average(all_pli, axis=0, weights=durations)

            # CONVERT FROM MATRICES TO SERIES

            features = DataFrame(all_pli, columns=channel_order, index=channel_order)


            # CHANNEL-CHANNEL FEATURES
            ch_ch_features = features.copy()
            ch_ch_features.replace(0, np.nan, inplace=True)  # it's a triangular matrix, so we can discard 0s
            ch_ch_features = ch_ch_features.stack(dropna=True)
            ch_ch_features.index = [f"{conn_metric.upper()}#{ch1}-{ch2}#{band}" for ch1, ch2 in ch_ch_features.index]
            ch_ch_features = DataFrame(ch_ch_features, columns=[filename, ]).T
            ch_ch_features.index = [filename, ]
            # Save like this as CSV
            all_channels_pli.append(ch_ch_features)

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

            # 4. Average
            avg_region_pairs = {}
            for region_pair, values in region_pairs.items():
                avg_region_pairs[f"{conn_metric.upper()}#{region_pair}#{band}"] = np.mean(values)
            avg_region_pairs = DataFrame(avg_region_pairs, dtype='float', index=[filename, ])

            # Save to CSV
            all_regions_pli.append(avg_region_pairs)

        # Concatenate from all bands
        if len(all_channels_pli) == 0:
            print(f'No good windows. Not saving this subject-session-band.')
            # leave a txt file with this message
            with open(join(subject_out_path, f'{conn_metric.upper()}.txt'), 'w') as f:
                f.write(f'No good windows. Not saving this subject-session-band {conn_metric}.')
        else:
            all_channels_pli = pd.concat(all_channels_pli, axis=1)
            all_regions_pli = pd.concat(all_regions_pli, axis=1)
            all_metrics_channels.append(all_channels_pli)
            all_metrics_regions.append(all_regions_pli)

    if len(all_metrics_channels) != 0:
        # Concatenate all metrics in one DataFrame
        all_metrics_channels = pd.concat(all_metrics_channels, axis=1)
        all_metrics_regions = pd.concat(all_metrics_regions, axis=1)

        # Save final DataFrames to CSV
        all_metrics_regions.to_csv(join(subject_out_path, f'Connectivity#Regions.csv'))
        all_metrics_channels.to_csv(join(subject_out_path, f'Connectivity#Channels.csv'))


# get all old files with 'pli' on the name and delete them
all_pli_files = glob(join(out_common_path, '**/PLI*.csv'), recursive=True)
for file in all_pli_files:
    remove(file)
print('All old PLI files were deleted.')