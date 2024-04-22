from datetime import timedelta
from glob import glob
from os import mkdir, remove
from os.path import join, exists

import numpy as np
from pandas import DataFrame

from ltbio.biosignals.modalities import EEG
from ltbio.biosignals.timeseries import Timeline
from ltbio.features.Features import SpectralFeatures
from ltbio.processing.PSD import PSD
from ltbio.processing.formaters import Normalizer

# FIXME: Change this to the correct path
common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/autopreprocessed_biosignal/2'
out_common_path = '/Volumes/MMIS-Saraiv/Datasets/KJPP/features/2'


#############################################
# DO NOT CHANGE ANYTHING BELOW THIS LINE

# Get recursively all .biosignal files in common_path
all_files = glob(join(common_path, '**/*.biosignal'), recursive=True)

# Processing tools
normalizer = Normalizer(method='minmax')

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


def extract_spectral_features(eeg: EEG, fft_window_type: str, fft_window_length: timedelta, fft_window_overlap: timedelta,
                              segment_length: timedelta = None, segment_overlap: timedelta = None, subject_out_path = None, normalise=False):
    """
    Extracts all spectral features from an EEG signal
    :param eeg: mne.Raw object
    :param fft_window_type: Window type for the FFT (e.g. 'hamming')
    :param fft_window_length: Window length for the FFT (e.g. 256 points)
    :param fft_window_overlap: Window overlap for the FFT (e.g. 128 points)
    :param segment_length: Segment length to average features across (e.g. 30 seconds)
    :param segment_overlap: Segment overlap to average features across (e.g. 15 seconds)
    :param normalise: Whether to normalise features to have zero mean and unit variance (e.g. True)
    :return: feature_names, features (e.g. (['F3_delta_relative_power', 'F3_delta_spectral_entropy', ...],
                                            [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]))
    """

    eeg_duration = eeg.duration.total_seconds()
    if segment_length is None:  # in seconds
        segment_length = eeg_duration  # no segmentation
    if segment_overlap is None:  # in seconds
        segment_overlap = segment_length  # no overlap

    segment_length, segment_overlap = segment_length.total_seconds(), segment_overlap.total_seconds()

    feature_names_functions = {
        'Spectral#RelativePower': SpectralFeatures.total_power,
        'Spectral#Entropy': SpectralFeatures.spectral_entropy,
        'Spectral#Flatness': SpectralFeatures.spectral_flatness,
        'Spectral#EdgeFrequency': SpectralFeatures.spectral_edge_frequency,
        'Spectral#PeakFrequency': SpectralFeatures.spectral_peak_freq,
        'Spectral#Diff': SpectralFeatures.spectral_diff,
    }

    # Go by segments with overlap
    feature_names, features = None, []
    total_segments_analised, total_segments = 0, 0
    for i in range(int(segment_length), int(eeg_duration), int(segment_overlap)):
        total_segments += 1
        #print(f'Window from {i-segment_length}s to {i}s')
        start = eeg.initial_datetime + timedelta(seconds=i-segment_length)
        end = eeg.initial_datetime + timedelta(seconds=i)
        try:
            eeg_segment = eeg[start: end]
        except IndexError:
            #print("\tWindow discarded for being out-of-bounds.")
            continue

        if eeg_segment._n_segments > 1:
            #print("\tWindow discarded for being in-between an interruption.")
            continue

        if eeg_segment.duration.total_seconds() < segment_length:
            #print("\tWindow discarded for being too short.")
            continue

        feature_names = []  # it's going to be overwritten in each iteration, because I'm lazy
        seg_features = []  # let's store features for this segment here

        for channel_name in channel_order:
            channel = eeg_segment._get_channel(channel_name)

            # Compute PSD and total power
            psd = PSD.fromTimeseries(channel, fft_window_type, fft_window_length, fft_window_overlap)
            total_power = SpectralFeatures.total_power(psd)

            # Go by bands
            for band_name, (lower, upper) in bands.items():
                psd_band = psd[lower:upper]

                for feature_name, feature_function in feature_names_functions.items():
                    feature_names.append(f"{feature_name}#{channel_name}#{band_name}")
                    res = feature_function(psd_band)
                    if feature_name == 'Spectral#RelativePower':
                        res /= total_power
                    seg_features.append(res)

        # Append features of this segment
        features.append(seg_features)
        total_segments_analised += 1

    print(f"Contributing segments: {total_segments_analised} (out of {total_segments})")
    # Write in txt file "{total_segments_analised}/{total_segments}"
    with open(join(subject_out_path, 'segments_used_for_spectral.txt'), 'w') as f:
        f.write(f"{total_segments_analised}/{total_segments}")

    # Average features across segments
    if len(features) > 1:
        print("=> Averaging features across segments")
        features = np.mean(features, axis=0)
    elif len(features) == 1:
        print("=> Only one segment was able to extract from this subject (not average)")
        features = features[0]
    else:
        print(f"=> No segments were proper for feature extraction. No extraction for subject-session {filename}.")
        return None, None

    features = np.array(features)

    # Normalise features?
    if normalise:
        print("=> Normalising feature vectors")
        features = (features - features.mean(axis=0)) / features.std(axis=0)

    return feature_names, features



for filepath in all_files:
    filename = filepath.split('/')[-1].split('.')[0]
    print(filename)
    subject_out_path = join(out_common_path, filename)
    if not exists(subject_out_path):
        mkdir(subject_out_path)

    subject_out_filepath = join(subject_out_path, 'Spectral#Channels.csv')
    if not exists(subject_out_filepath):

        # Load Biosignal
        x = EEG.load(filepath)
        x = x[channel_order]  # get only channels of interest

        # Get only signal with quality
        good = Timeline.load(join(common_path, filename + '_good.timeline'))
        x = x[good]

        # Normalize
        x = normalizer(x)

        # Extract all spectral features
        window_type = 'hamming'
        window_length = timedelta(seconds=4)  # 2 seconds
        window_overlap = window_length / 2  # 50% overlap

        # Segmentation parameters
        segment_length = timedelta(seconds=4)  # 4 seconds
        segment_overlap = segment_length / 2  # 50% overlap
        feature_names, features = extract_spectral_features(x, window_type, window_length, window_overlap,
                                                            segment_length, segment_overlap, subject_out_path, normalise=False)
        if features is None:
            continue  # no features extracted

        # Convert to dataframe
        df = DataFrame(features).T
        df.columns = feature_names
        df.index = [filename, ]

        # Save
        df.to_csv(subject_out_filepath, index=True, header=True)

        # Delete "spectral.csv" if it exists
        #old_file = join(subject_out_path, 'spectral.csv')
        #if exists(old_file):
        #    remove(old_file)
