# -*- encoding: utf-8 -*-
import csv
import os
from datetime import timedelta, datetime
from os import path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from .. import Timeseries
from ..modalities import EEG
from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Unit import Volt, Multiplier
from ...clinical import Patient, BodyLocation
from ...clinical.Patient import Sex
from ...clinical.conditions.AD import AD
from ...clinical.conditions.SMC import SMC


class BrainLat(BiosignalSource):
    """This class represents the source of BrainLat dataset (in SET format) and includes methods to read and write
    biosignal files provided by them."""

    def __init__(self, demographic_csv):
        super().__init__()
        BrainLat.demographic_csv = demographic_csv

    def __repr__(self):
        return "BrainLat dataset"

    @staticmethod
    def __read_set_file(filepath, metadata=False):
        """
        Reads one SET file
        param: filepath points to the file to read.
        If metadata is False, only returns samples and initial datetime.
        If metadata is True, also returns list of channel names and sampling frequency.
        Else return arrays; one per each channel
        """

        mat = loadmat(filepath)

        samples = mat['data']
        initial_datetime = datetime(2023, 1, 1, 0, 0, 0)

        if metadata:
            channel_names = [str(x[0]) for x in (mat['chanlocs'][0]['labels'])]
            sf = float(mat['srate'][0][0])
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

        samples, initial_datetime, channel_names, sf = BrainLat.__read_set_file(filepath, metadata=True)
        units = Volt(Multiplier.u)  # micro-volts

        by_product_segments = None  # indexes
        timeseries = {}
        for ch in range(len(channel_names)):
            if channel_names[ch] == "Status":
                continue
            ch_samples = samples[ch, :]
            ts = Timeseries(ch_samples, initial_datetime, sf, units, name=f"{channel_names[ch]} equiv. of Biosemi 128")
            timeseries[channel_names[ch]] = ts

        return timeseries

    @staticmethod
    def __find_sex_age(patient_code) -> tuple[Sex, int]:
        with open(BrainLat.demographic_csv) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['ID'] == patient_code:
                    return Sex.M if int(row['Sex']) == 1 else Sex.F, row['Age']
        raise ValueError(f"Patient code {patient_code} not found in demographics file '{BrainLat.demographic_csv}'.")

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
        sex, age = BrainLat.__find_sex_age(patient_code)
        ad = AD()

        return Patient(patient_code, age=age, sex=sex, conditions=(ad, ))

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
