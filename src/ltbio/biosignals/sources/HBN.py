# -*- encoding: utf-8 -*-
import csv
import os
from datetime import datetime

import mat73
import mne.io

from .. import Timeseries
from ..modalities import EEG
from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Unit import Volt, Multiplier
from ...clinical import Patient, BodyLocation
from ...clinical.Patient import Sex
from ...clinical.conditions.AD import AD
from ...clinical.conditions.FTD import FTD
from ...clinical.conditions.MedicalCondition import MedicalCondition


class HealthyBrainNetwork(BiosignalSource):
    """This class represents the source of Healthy Brain Network (in EDF format) and includes methods to read and write
    biosignal files provided by them."""

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Healthy Brain Network"

    @staticmethod
    def __read_edf_file(filepath, metadata=False):
        """
        Reads one EDF file
        param: filepath points to the file to read.
        If metadata is False, only returns samples and initial datetime.
        If metadata is True, also returns list of channel names and sampling frequency.
        Else return arrays; one per each channel
        """

        edf = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        samples = edf.get_data()
        initial_datetime = datetime(2023, 1, 1, 0, 0, 0)

        if metadata:
            channel_names = edf.ch_names
            sf = edf.info['sfreq']
            return samples, initial_datetime, channel_names, sf

        else:
            return samples, initial_datetime

    @staticmethod
    def _timeseries(filepath, type=EEG, **options):
        """
        Reads the SET file specified and returns a dictionary with one Timeseries per channel name.
        Args:
            filepath (str): Path to the EDF file
            type (Biosignal): Type of biosignal to extract. Only EEG allowed.
        """

        samples, initial_datetime, channel_names, sf = HealthyBrainNetwork.__read_edf_file(filepath, metadata=True)
        units = Volt(Multiplier.u)  # micro-volts

        by_product_segments = None  # indexes
        timeseries = {}
        for ch in range(len(channel_names)):
            if channel_names[ch] == "Status":
                continue
            ch_samples = samples[ch, :]
            ts = Timeseries(ch_samples, initial_datetime, sf, units, name=f"{channel_names[ch]}")
            timeseries[channel_names[ch]] = ts

        return timeseries

    @staticmethod
    def _patient(path, **options):
        """
        Gets:
        - patient code from the filepath
        - age from the demographics file
        - gender from the demographics file
        and adds SMC diagnosis.
        """

        filename = os.path.split(path)[-1]
        patient_code = filename.split('.')[0]
        return Patient(patient_code)

    @staticmethod
    def _acquisition_location(path, type, **options):
        return BodyLocation.SCALP

    @staticmethod
    def _name(path, type, **options):
        """
        Gets the trial number from the filepath.
        """
        return f"Resting-state EEG"

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass
