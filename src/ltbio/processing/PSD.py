# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
#
# Package: 
# Module: 
# Description: Power Spectral Density (PSD) class
#
# Contributors: João Saraiva
# Created: 
# Last Updated: 
# ===================================
from datetime import timedelta

import numpy as np
from biosppy.signals.tools import welch_spectrum
from numpy import ndarray

import ltbio.biosignals.timeseries as ts
from ltbio.biosignals.timeseries import Frequency


class PSD:

    ###############################
    # Constructors

    def __init__(self, freqs: ndarray, powers: ndarray, sampling_frequency: Frequency | float):
        self.__freqs, self.__powers = freqs, powers
        self.__sampling_frequency = sampling_frequency

    @classmethod
    def fromTimeseries(cls, x: ts.Timeseries, window_type, window_length: timedelta, window_overlap: timedelta) -> tuple['PSD']:
        if x.n_segments > 1:
            raise NotImplementedError(
                'PSD.fromTimeseries() is not implemented for multi-segment timeseries. Please index when segment before calling this method.')

        window_length = int(window_length.total_seconds() * x.sampling_frequency)
        window_overlap = int(window_overlap.total_seconds() * x.sampling_frequency)

        psd_by_seg = x._apply_operation_and_return(welch_spectrum,
                                                   sampling_rate=x.sampling_frequency,
                                                   size=window_length, overlap=window_overlap, window=window_type,
                                                   decibel=False
                                                   )
        if len(psd_by_seg) == 1:
            return cls(psd_by_seg[0][0], psd_by_seg[0][1], x.sampling_frequency)
        else:
            return tuple([cls(freqs, powers[0], x.sampling_frequency) for freqs, powers in psd_by_seg])

    @classmethod
    def average(cls, *multiple_psds: tuple['PSD']) -> 'PSD':
        if len(multiple_psds) < 2:
            return multiple_psds[0]

        for i in range(1, len(multiple_psds)):
            if not np.array_equal(multiple_psds[i].freqs, multiple_psds[0].freqs):
                raise ValueError("The PSD objects do not have the same frequency bins.")
            if not np.array_equal(multiple_psds[i].sampling_frequency, multiple_psds[0].sampling_frequency):
                raise ValueError("The PSD objects do not have the same sample frequency attribute.")

        average_powers = np.mean([psd.powers for psd in multiple_psds], axis=0)

        return cls(multiple_psds[0].freqs, average_powers, multiple_psds[0].sampling_frequency)

    ###############################
    # Getters

    @property
    def freqs(self):
        return self.__freqs.view()

    @property
    def powers(self):
        return self.__powers.view()

    @property
    def sampling_frequency(self) -> float:
        return float(self.__sampling_frequency)

    @property
    def argmax(self) -> float:
        i = np.argmax(self.__powers)
        return self.freqs[i]

    @property
    def argmin(self) -> float:
        i = np.argmin(self.__powers)
        return self.freqs[i]

    def iaf(self):
        return self.get_band(8., 14.).argmax  # maxiumum of the extended alpha range

    def tf(self):
        return self.get_band(3., 8.).argmin  # minimum between 3-8 Hz

    ###############################
    # Bands

    def get_band(self, lower: Frequency, upper:Frequency) -> 'PSD':
        """
        Returns a new PSD object truncated to the specified band.
        Both f and Pxx_den are truncated.
        """
        f = self.__freqs[(self.__freqs >= lower) & (self.__freqs <= upper)]
        Pxx_den = self.__powers[(self.__freqs >= lower) & (self.__freqs <= upper)]
        return PSD(f, Pxx_den, self.sampling_frequency)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.get_band(item.start, item.stop)
        else:
            raise NotImplementedError('PSD.__getitem__ is only implemented for slices.')
