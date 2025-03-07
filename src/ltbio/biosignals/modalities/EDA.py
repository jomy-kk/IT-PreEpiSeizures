# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: EDA
# Description: Class EDA, a type of Biosignal named Electrodermal Activity.

# Contributors: João Saraiva, Mariana Abreu
# Created: 12/05/2022
# Last Updated: 07/07/2022

# ===================================
from datetime import timedelta

from datetimerange import DateTimeRange
from numpy import mean

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.timeseries.Unit import Volt, Multiplier


class EDA(Biosignal):

    DEFAULT_UNIT = Volt(Multiplier.m)

    def __init__(self, timeseries, source=None, patient=None, acquisition_location=None, name=None):
        super(EDA, self).__init__(timeseries, source, patient, acquisition_location, name)

    def plot_summary(self, show: bool = True, save_to: str = None):
        pass

    @property
    def preview(self):
        """Returns 30 seconds of the middle of the signal."""
        domain = self.domain
        middle_of_domain: DateTimeRange = domain[len(domain) // 2]
        middle = middle_of_domain.start_datetime + (middle_of_domain.timedelta / 2)
        try:
            return self[middle - timedelta(seconds=2): middle + timedelta(seconds=28)]
        except IndexError:
            raise AssertionError(
                f"The middle segment of {self.name} from {self.patient_code} does not have at least 5 seconds to return a preview.")

    @staticmethod
    def racSQI(samples):
        """
        Rate of Amplitude change (RAC)
        It is recomended to be analysed in windows of 2 seconds.
        """
        max_, min_ = max(samples), min(samples)
        amplitude = max_ - min_
        return abs(amplitude / max_)

    def acceptable_quality(self):  # -> Timeline
        """
        Suggested by Böttcher et al. Scientific Reports, 2022, for wearable wrist EDA.
        """
        return self.when(lambda x: mean(x) > 0.05 and EDA.racSQI(x) < 0.2, window=timedelta(seconds=2))
