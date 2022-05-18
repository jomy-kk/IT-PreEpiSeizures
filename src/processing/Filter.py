###################################

# IT - PreEpiSeizures

# Package: processing
# File: Filter
# Description: Class representing a filter and its properties. It works with the design pattern 'Visitor' in order to apply itself to Biosignals.

# Contributors: João Saraiva
# Created: 17/05/2022

###################################

from typing import Tuple
from enum import unique, Enum

from biosppy.signals.tools import get_filter as get_coefficients, OnlineFilter as FilterExec
from numpy import array


@unique
class FrequencyResponse(str, Enum):
    FIR = 'Finite Impulse Response (FIR)'
    BUTTER = 'IIR Butterworth'
    CHEBY1 = 'IIR Chebyshev 1'
    CHEBY2 = 'IIR Chebyshev 2'
    ELLIP = 'IIR Elliptic'
    BESSEL = 'IIR Bessel'


@unique
class BandType(str, Enum):
    LOWPASS = 'Low-pass'
    HIGHPASS = 'High-pass'
    BANDPASS = 'Band-pass'
    BANDSTOP = 'Band-stop'


class Filter:
    """
    Describes the design of a digital filter and holds the ability to apply that filter to any array of samples.
    It acts as a visitor in the Visitor Design Pattern.

    To instantiate, give:
        - fresponse: The frequency response of the filter. Choose one from FrequencyResponse enumeration.
        - band_type: Choose whether it should low, high, or band pass or reject a band of the samples' spectrum. Choose one from BandType enumeration.
        - order: The order of the filter (in int).
        - cutoff: The cutoff frequency at 3 dB (for lowpass and highpass) or a tuple of two cutoffs (for bandpass or bandstop) (in Hertz, float).
    """

    def __init__(self, fresponse: FrequencyResponse, band_type: BandType, order: int, cutoff: float | Tuple[float]):
        # These properties can be changed as pleased:
        self.fresponse = fresponse
        self.band_type = band_type
        self.order = order
        self.cutoff = cutoff
        # These are private properties:
        self.__b, self.__a = None, None
        self.__exec = None

    def _compute_coefficients(self, sampling_frequency: float):
        """
        Computes the coefficients of the H function.
        They are stored as 'b' and 'a', respectively, the numerator and denominator coefficients.

        :param sampling_frequency: The sampling frequency of what should be filtered.
        """

        # Digital filter coefficients (from Biosppy)
        self.__b, self.__a = get_coefficients(ftype=self.fresponse.name.lower(), band=self.band_type.name.lower(),
                                          order=self.order,
                                          frequency=self.cutoff, sampling_rate=sampling_frequency)
        self.exex = FilterExec(b=self.__b, a=self.__a)

    def __are_coefficients_computed(self) -> bool:
        """
        :return: True if coefficients have already been computed, and the Filter is ready to be applied.
        """
        return self.__exec is not None  # or (self.b is not None and self.a is not None)

    def _visit(self, samples: array) -> array:
        """
        Applies the Filter to a sequence of samples.
        It acts as the visit method of the Visitor Design Pattern.

        :param samples: Sequence of samples to filter.
        :return: The filtered sequence of samples.
        """
        self.__exec.reset()
        return self.__exec.filter(samples)['filtered']
