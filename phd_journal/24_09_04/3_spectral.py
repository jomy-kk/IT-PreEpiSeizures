from datetime import timedelta
from glob import glob
from os import mkdir
from os.path import join, exists

import pandas as pd

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.features.Features import SpectralFeatures
from ltbio.processing.PSD import PSD
from ltbio.processing.formaters import Normalizer, Segmenter

# FIXME: Change this to the correct path
common_path = '/Volumes/MMIS-Saraiv/Datasets/Healthy Brain Network/biosignal'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/Healthy Brain Network/features'

verify_quality = True
verify_subject_length = False


#############################################
# DO NOT CHANGE ANYTHING BELOW THIS LINE

# Processing tools
normalizer = Normalizer(method='minmax')

# Channels and Bands
channel_order = ('C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6')  # without mastoids
bands = {
    'delta': (1.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45),
}

# Hamming window lengths (seconds) per band
window_lengths = {
    'delta': timedelta(seconds=8),
    'theta': timedelta(seconds=4),
    'alpha': timedelta(seconds=2),
    'beta': timedelta(seconds=1),
    'gamma': timedelta(seconds=0.5)
}

# Subject lengths
SUBJECT_LENGTH = timedelta(seconds=24)
BLOCK_LENGTH = timedelta(seconds=8)

def extract_spectral_features(eeg: EEG, fft_window_type: str, block_length: timedelta, subject_length: timedelta):
    """
    Extracts all spectral features from an EEG signal
    :param eeg: mne.Raw object
    :param fft_window_type: Window type for the FFT (e.g. 'hamming')
    :param block_length: Length of each block to average features across (e.g. 8 seconds)
    :param subject_length: Length of a subject (e.g. 24 seconds)
    """

    feature_names_functions = {
        'Spectral#RelativePower': SpectralFeatures.total_power,
        'Spectral#Entropy': SpectralFeatures.spectral_entropy,
        'Spectral#Flatness': SpectralFeatures.spectral_flatness,
        'Spectral#EdgeFrequency': SpectralFeatures.spectral_edge_frequency,
        'Spectral#PeakFrequency': SpectralFeatures.spectral_peak_freq,
        'Spectral#Diff': SpectralFeatures.spectral_diff,
    }

    # Go by existing discontiguous segments
    all_subjects = []
    total_segments_analised, total_segments = 0, 0
    S = subject_length.total_seconds() / block_length.total_seconds()

    domain = eeg['T5'].domain
    for i, interval in enumerate(domain):
        total_segments += 1
        z = eeg[interval]
        if verify_subject_length and z.duration < subject_length:
            continue
        print(f"Segment {i + 1} of {len(domain)}")

        # Segment in blocks
        segmenter = Segmenter(block_length, timedelta(seconds=0))
        z = segmenter(z)
        z_domain = z['T5'].domain

        # remove last block if it is incomplete
        if z_domain[-1].timedelta < block_length:
            z_domain = z_domain[:-1]

        features_by_blocks = []
        for j, window in enumerate(z_domain):
            if window.timedelta < block_length:
                continue  # incomplete block

            if verify_subject_length and len(z_domain) - j < S - len(features_by_blocks):
                break  # insufficient blocks to make a subject

            y = z[window]

            block_features = {}

            for channel_name in channel_order:
                channel = y._get_channel(channel_name)

                total_power = 0.0
                # Go by bands
                for band_name, (lower, upper) in bands.items():
                    # Get PSD
                    fft_window_length = window_lengths[band_name]
                    fft_window_overlap = fft_window_length / 2
                    psd = PSD.fromTimeseries(channel, fft_window_type, fft_window_length, fft_window_overlap)
                    psd_band = psd[lower:upper]
                    # Calculate features
                    for feature_name, feature_function in feature_names_functions.items():
                        res = feature_function(psd_band)
                        if feature_name == 'Spectral#RelativePower':
                            total_power += res
                        name = f"{feature_name}#{channel_name}#{band_name}"
                        block_features[name] = res
                # Afterward, divide all relative powers by the total power
                for name, value in block_features.items():
                    if f'Spectral#RelativePower#{channel_name}' in name:
                        block_features[name] = value / total_power

            block_features = pd.DataFrame(block_features, index=[f"{filename}${i+1}${j+1}"])

            if verify_subject_length:
                if len(features_by_blocks) < S - 1:
                    features_by_blocks.append(block_features)
                    pass
                else:  # Average features
                    features_by_blocks.append(block_features)
                    F = pd.concat(features_by_blocks, axis=0)
                    F = F.mean(axis=0)
                    F = F.to_frame().transpose()
                    all_subjects.append(F)
                    features_by_blocks = []

            else:
                features_by_blocks.append(block_features)

        if not verify_subject_length and len(features_by_blocks) > 0:
            F = pd.concat(features_by_blocks, axis=0)
            all_subjects += [F, ]

        if len(features_by_blocks) > 0:
            total_segments_analised += 1

    if not verify_subject_length:
        print(f"Contributing segments: {total_segments_analised} (out of {total_segments})")
        if len(all_subjects) > 0:
            all_subjects = pd.concat(all_subjects, axis=0)
            all_subjects = all_subjects.mean(axis=0)
            all_subjects = all_subjects.to_frame().transpose()
            return [all_subjects,]
        else:
            return []

    print(f"Contributing segments: {total_segments_analised} (out of {total_segments})")
    return all_subjects


# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '**/*.biosignal'), recursive=True)

for filepath in all_files:
    # Identifiers
    filename = filepath.split('/')[-1].split('.')[0]
    print(filename)
    subject_out_path = join(out_common_path, filename)
    if not exists(subject_out_path):
        mkdir(subject_out_path)
    subject_out_filepath = join(subject_out_path, 'Spectral#Channels$Multiple.csv')

    if True: #not exists(subject_out_filepath):
        # Load Biosignal
        x = EEG.load(filepath)
        x = x[channel_order]  # get only channels of interest, by the correct order

        # Get only signal with quality
        if verify_quality:
            good = Timeline.load(join(common_path, filename + '_good.timeline'))
            x = x[good]

        # Normalize
        x = normalizer(x)

        # Extract all spectral features
        window_type = 'hamming'
        all_windows = extract_spectral_features(x, window_type, BLOCK_LENGTH, SUBJECT_LENGTH)

        # Convert to dataframe
        if len(all_windows) == 0:
            continue
        df = pd.concat(all_windows, axis=0)
        df.index = [f"{filename}${i+1}" for i in range(len(df))]

        # Save
        df.to_csv(subject_out_filepath, index=True, header=True)

        # Delete "spectral.csv" if it exists
        #old_file = join(subject_out_path, 'spectral.csv')
        #if exists(old_file):
        #    remove(old_file)
