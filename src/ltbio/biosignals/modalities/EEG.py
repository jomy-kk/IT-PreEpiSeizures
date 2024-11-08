# -*- encoding: utf-8 -*-
from datetime import timedelta

import mne.io
import numpy as np
from array import array

import pandas as pd
from mne import create_info, make_fixed_length_epochs, compute_covariance
from mne.minimum_norm import apply_inverse
from numpy import average

from ltbio.biosignals.modalities import ECG
# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: EEG
# Description: Class EEG, a type of Biosignal named Electroencephalogram.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 07/07/2022

# ===================================

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries import Timeline
from ltbio.biosignals.timeseries.Unit import Volt, Multiplier
from ltbio.clinical import BodyLocation
from ltbio.features.Features import HjorthParameters, ConnectivityFeatures

from biosppy.signals.tools import _filter_signal, power_spectrum

from ltbio.processing.PSD import PSD
from ltbio.processing.filters.TimeDomainFilter import *


class EEG(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None, **options):
        super(EEG, self).__init__(timeseries, source, patient, acquisition_location, name, **options)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

    # ===================================
    # Features

    def hjorth_activity(self) -> dict[str | BodyLocation, float]:
        """
        Computes Hjorth Activity per channel.
        :return: A dictionary of float values, one per each channel name.
        """
        return self.apply_operation_and_return(HjorthParameters.hjorth_activity)

    def hjorth_mobility(self) -> dict[str | BodyLocation, float]:
        """
        Computes Hjorth Mobility per channel.
        :return: A dictionary of float values, one per each channel name.
        """
        return self.apply_operation_and_return(HjorthParameters.hjorth_mobility)

    def hjorth_complexity(self) -> dict[str | BodyLocation, float]:
        """
        Computes Hjorth Complexity per channel.
        :return: A dictionary of float values, one per each channel name.
        """
        return self.apply_operation_and_return(HjorthParameters.hjorth_complexity)

    def pli(self, window_length: timedelta, fmin: float = None, fmax: float = None,
            channel_order: tuple[str | BodyLocation] = None) -> np.ndarray:
        """
        Computes Phase Lag Index per channel.
        :return: A matrix of float values, one cell per each pair of channels.
        """
        res = ConnectivityFeatures.pli(self, window_length, fmin, fmax, channel_order=channel_order)
        res_matrix = res.get_data('dense')[:, :, 0]
        assert res.names == channel_order
        return res_matrix

    def coh(self, window_length: timedelta, fmin: float = None, fmax: float = None,
            channel_order: tuple[str | BodyLocation] = None) -> np.ndarray:
        """
        Computes Coherence per channel.
        :return: A matrix of float values, one cell per each pair of channels.
        """
        res = ConnectivityFeatures.coh(self, window_length, fmin, fmax, channel_order=channel_order)
        res_matrix = res.get_data('dense')[:, :, 0]
        assert res.names == channel_order
        return res_matrix

    def gc(self, window_length: timedelta, fmin: float = None, fmax: float = None,
            channel_order: tuple[str | BodyLocation] = None) -> np.ndarray:
        """
        Computes Granger Causality per channel.
        :return: A matrix of float values, one cell per each pair of channels.
        """
        res = ConnectivityFeatures.gc(self, window_length, fmin, fmax, channel_order=channel_order)
        res_matrix = res.get_data('dense')[:, :, 0]
        assert res.names == channel_order
        return res_matrix

    def ppc(self, window_length: timedelta, fmin: float = None, fmax: float = None,
            channel_order: tuple[str | BodyLocation] = None) -> np.ndarray:
        """
        Computes Pairwise Phase Consistency per channel.
        :return: A matrix of float values, one cell per each pair of channels.
        """
        res = ConnectivityFeatures.ppc(self, window_length, fmin, fmax, channel_order=channel_order)
        res_matrix = res.get_data('dense')[:, :, 0]
        assert res.names == channel_order
        return res_matrix


    def psd(self, window_length: timedelta, fmin: float = None, fmax: float = None,
            channel_order: tuple[str | BodyLocation] = None) -> np.ndarray:
        """
        Returns the Power Spectrum Density per channel.
        :return: A PSD object.
        """
        res = ConnectivityFeatures.pli(self, 'pli', window_length, fmin, fmax, channel_order=channel_order)
        res_matrix = res.get_data('dense')[:, :, 0]
        assert res.names == channel_order
        return res_matrix

    # ===================================
    # Signal Quality Indexes

    def oha_sqi(self, threshold: float = 1, by_segment: bool = False):
        """
        Evaluates the ratio of data points that exceed the absolute value a certain voltage amplitude.
        EEG amplitude is usually in uV, but check the units of the Biosignal.

        Note: It is recommended beforehand to normalize the data by zero mean and unit variance. See 'Normalizer'.

        :param threshold: The threshold above which the signal is considered to be of poor quality.
            Recommended Value: 1 if data is normalized by zero mean and unit variance.
        :param by_segment: Calculate by segment (True) or average across all segments (False).

        :return: An all-channel OHA score. If `by_segment` is True, a list of values is returned for each
            contiguous uninterrupted segment (list[float]), otherwise the weighted average, by duration of each segment,
            is returned (float).

        :raises AssertionError: If all channels don't have the same domain.

        Recommended Interpretation:
            0%  < OHA < 10% => good quality

            10% < OHA < 20% => acceptable quality

            20% < OHA       => poor quality

        Adapted from: https://github.com/methlabUZH/automagic/wiki/Quality-Assessment-and-Rating (2023-11-30)
        """
        n_segments = self._n_segments
        if not isinstance(n_segments, int):
            raise AssertionError("All channels must have the same domain.")

        res = []
        for i in range(n_segments):
            eeg_data = self._vblock(i)
            res.append(np.sum(np.abs(eeg_data) > threshold) / (eeg_data.shape[0] * eeg_data.shape[1]))

        if by_segment:
            return res
        else:
            return average(res, weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), self.domain)))

    def thv_sqi(self, threshold: float = 1, by_segment: bool = False):
        """
        Evaluates the ratio of time points in which the % of standard deviation across all channels exceeds a threshold.
        EEG amplitude is usually in uV, but check the units of the Biosignal.

        Note: It is recommended beforehand to normalize the data by zero mean and unit variance. See 'Normalizer'.

        :param threshold: The threshold above which the signal is considered to be of poor quality.
            Recommended Value: 1 if data is normalized by zero mean and unit variance.
        :param by_segment: Calculate by segment (True) or average across all segments (False).

        :return: An all-channel THV score. If `by_segment` is True, a list of values is returned for each
            contiguous uninterrupted segment (list[float]), otherwise the weighted average, by duration of each segment,
            is returned (float).

        :raises AssertionError: If all channels don't have the same domain.

        Recommended Interpretation:
            0%  < THV < 10% => good quality

            10% < THV < 20% => acceptable quality

            20% < THV       => poor quality

        Adapted from: https://github.com/methlabUZH/automagic/wiki/Quality-Assessment-and-Rating (2023-11-30)
        """
        n_segments = self._n_segments
        if not isinstance(n_segments, int):
            raise AssertionError("All channels must have the same domain.")

        res = []
        for i in range(n_segments):
            eeg_data = self._vblock(i)
            res.append(np.sum(np.greater(np.std(eeg_data, axis=0, ddof=1), threshold)) / eeg_data.shape[1])

        if by_segment:
            return res
        else:
            return average(res, weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), self.domain)))

    def chv_sqi(self, threshold, rejection_cutoff=None, rejection_ratio=None, by_segment: bool = False):
        """
        Evaluates the ratio of channels in which the % of standard deviation across all timepoints exceeds a threshold.
        EEG amplitude is usually in uV, but check the units of the Biosignal.

        Note: It is recommended beforehand to normalize the data by zero mean and unit variance. See 'Normalizer'.

        :param threshold: The threshold above which the signal is considered to be of poor quality.
            Recommended Value: 1 if data is normalized by zero mean and unit variance.
        :param by_segment: Calculate by segment (True) or average across all segments (False).

        :return: An all-channel CHV score. If `by_segment` is True, a list of values is returned for each
            contiguous uninterrupted segment (list[float]), otherwise the weighted average, by duration of each segment,
            is returned (float).

        :raises AssertionError: If all channels don't have the same domain.

        Recommended Interpretation:
            0%  < CHV < 15% => good quality

            15% < CHV < 30% => acceptable quality

            30% < CHV       => poor quality

        Adapted from: https://github.com/methlabUZH/automagic/wiki/Quality-Assessment-and-Rating (2023-11-30)
        """

        def __chv(eeg_array, channel_threshold, rejection_cutoff=None, rejection_ratio=None):

            # 1. Remove timepoints of very high variance from channels
            if rejection_cutoff is not None:
                ignoreMask = np.logical_or(eeg_array > rejection_cutoff, eeg_array < -rejection_cutoff)
                onesPerChan = np.sum(ignoreMask, axis=1)
                onesPerChan = onesPerChan / eeg_array.shape[1]
                overRejRatio = onesPerChan > rejection_ratio
                ignoreMask[overRejRatio, :] = False
                eeg_array[ignoreMask] = np.nan

            # 2. Calculate CHV
            return np.sum(np.nanstd(eeg_array, axis=1) > channel_threshold) / eeg_array.shape[0]

        n_segments = self._n_segments
        if not isinstance(n_segments, int):
            raise AssertionError("All channels must have the same domain.")

        res = []
        for i in range(n_segments):
            eeg_data = self._vblock(i)
            res.append(__chv(eeg_data, threshold, rejection_cutoff, rejection_ratio))

        if by_segment:
            return res
        else:
            return average(res, weights=list(map(lambda subdomain: subdomain.timedelta.total_seconds(), self.domain)))


    def acceptable_quality(self, ecg: ECG = None):
        """
        Returns the Timeline of acceptable (good) EEG quality.
        Good EEG periods are absent of noise and noise contamination depends on a variety of factors,
        such as electrode placement, skin preparation, subject movements and surrounding electrical noise.

        Here 5 SQIs are evaluated:
        - Baseline wander (BW)
          Possible causes of high BW: Electrodermal noise, slow lateral eye movements, cardiobalistic artifacts, eye blinks, etc.
        - High-frequency noise (HFN):
          Possible causes of high HFN: Myogenic noise, hypoglossal and cheweing artifacts, bruxisms, electrode noise, etc.
        - Noisy Rhythmic Activity (RA):
          Possible causes of high RA: Head rubbing, chest tapping, etc.
        - Burst Drifts (BD):
          Possible causes of high BD: Eyes opening, closing and blinking; electrode pops, etc.
        - Opposing polarities slow waves (OPSW):
          Possible causes of high OPSW: Lateral eye movement (in frontal electrodes), head shaking (in back electrodes), etc.
        - Cardiac Artifacts (CA): [only applicable if 'ecg' is given]
          Possible causes of high CA: Peaks synchronised with the R wave of the ECG, etc.

        The lower, the better, for all SQIs.
        It is advisable to normalise the signal before computing the SQIs. See 'Normalizer'.
        """

        sf = self.sampling_frequency

        # Baseline wander (BW)  FIXME
        # Apply low-pass filter of 1.5 Hz
        #filter_design = TimeDomainFilter(ConvolutionOperation.MEDIAN, window_length=timedelta(seconds=5), overlap_length=timedelta(seconds=2.5))
        #baseline = self.__copy__()
        #baseline.filter(filter_design)
        # When the peak-to-peak amplitude of the baseline is greater than 0.2, it is considered high BW.
        #bw_good = baseline.when(lambda x: max(x) - min(x) < 0.2, window=timedelta(seconds=2))

        # High-frequency noise (HFN)
        def __hfn(x):  # for each window
            freqs, power = power_spectrum(x, sampling_rate=sf)  # Get power spectral density
            hf = np.sum(power[freqs >= 49])  # Sum power in the [49, +inf[ Hz range
            total = np.sum(power)
            return hf / total
        # When the high-frequency components dominate (>50% of spectrum), it is considered high HFN.
        hfn_good = self.when(lambda x: __hfn(x) < 0.5, window=timedelta(seconds=2))

        # Noisy Rhythmic Activity (RA)
        ra_good = self.domain  # TODO

        # Burst Drifts (BD)
        def __channel_std(x: np.ndarray):
            # Get all standard deviations of the whole channel, with a sliding window of 2 seconds
            res = []
            for i in range(0, len(x), int(sf * 2)):
                res.append(np.std(x[i:i + int(sf * 2)]))
            return res
        # Compute the standard deviations per channel
        all_stds = self.apply_operation_and_return(lambda x: __channel_std(x))
        # Get the median of the standard deviations per channel
        median_std = []
        for v in all_stds.values():
            for p in v:
                median_std.extend(p)
        median_std = np.median(median_std)

        # When the standard deviation of a channel is less than 2 median standard deviations, it is considered high BD.
        bd_good = self.when(lambda x: np.std(x) < 2*median_std, window=timedelta(seconds=2))

        # Opposing polarities slow waves (OPSW)
        opsw_good = self.domain  # TODO

        # Cardiac Artifacts (CA)
        if ecg is not None:
            ca_good = self.domain  # TODO
        else:
            ca_good = self.domain

        # Make unique timeline
        good = Timeline.union(bw_good, hfn_good, bd_good)
        return good

    def inverse_solution(self, method: str, epoch_duration: float, spacing: str, mri_space: str = "fsaverage", eeg_channel_locations:str=None):
        assert self._n_segments == 1, "The EEG should be contiguous."

        # 0. Convert to MNE Rawobject
        _channel_order = tuple(self.channel_names)
        _array = self.to_array(_channel_order)
        _info = create_info(_channel_order, self.sampling_frequency, ch_types='eeg')
        # channel locations from ´eeg_channel_locations´
        if eeg_channel_locations is not None:
            with open(eeg_channel_locations, 'r') as f:
                locs = [line.split('\t') for line in f]
                locs = {str(l[4].replace('\n', '').replace(' ', '')): (float(l[1]), float(l[2]), float(l[3])) for l in locs}
                montage = mne.channels.make_dig_montage(ch_pos=locs, coord_frame='head')
                _info.set_montage(montage)

        raw = mne.io.RawArray(_array, _info)
        raw.set_eeg_reference(projection=True)  # Set EEG reference

        # 1. Equally-spaced epochs
        epochs = make_fixed_length_epochs(raw, duration=epoch_duration,
                                          preload=True, baseline=(None, 0))  # with baseline correction

        # 2. Noise covariance
        noise_cov = compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)

        # 3. Compute forward solution
        src = mne.setup_source_space(mri_space, spacing='oct6', subjects_dir="/Users/saraiva/mne_data/MNE-fsaverage-data", add_dist=False)
        model = mne.make_bem_model(subject=mri_space, ico=4, subjects_dir="/Users/saraiva/mne_data/MNE-fsaverage-data")
        bem = mne.make_bem_solution(model)
        fwd = mne.make_forward_solution(raw.info, trans=mri_space, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)
        print(src)

        # 4. Compute inverse operator
        inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)
        del fwd, noise_cov

        # 5. Apply inverse operator
        stc = mne.minimum_norm.apply_inverse(epochs, inverse_operator, lambda2=1.0 / 9.0, method=method)

        data = stc.data  # The source time courses (n_sources x n_times)
        times = stc.times  # The time points corresponding to the data

        res = pd.DataFrame(data, columns=times).T
        return res
