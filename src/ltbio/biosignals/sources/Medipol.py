# -*- encoding: utf-8 -*-
import csv
import glob
import os
from datetime import datetime, timedelta
from os.path import join, exists

import numpy as np
from scipy.io import loadmat

from .. import Timeseries
from ..modalities import EEG
from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Unit import Volt, Multiplier
from ...clinical import Patient, BodyLocation
from ...clinical.Patient import Sex
from ...clinical.conditions.AD import AD
from ...clinical.conditions.MedicalCondition import MedicalCondition


class Medipol(BiosignalSource):
    """This class represents the source of Istambul Medipol Univerisity (in SET format) and includes methods to read and write
    biosignal files provided by them."""

    DELTA_BETWEEN_SEGMENTS = 1  # seconds
    BOUNDARIES_FILENAME = "_asr_boundaries.txt"  # as produced by EEGLAB after cleaning artifacts

    def __init__(self, demographic_csv):
        super().__init__()
        Medipol.demographic_csv = demographic_csv

    def __repr__(self):
        return "Istambul Medipol Univerisity"


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
    def __read_boundaries_file(filepath) -> tuple[int]:
        """
        Reads the boundaries file as TSV and returns a list of interruptions indexes.
        """
        with open(filepath) as f:
            interruptions = csv.DictReader(f, delimiter='\t')
            interruptions_ixs = []
            for row in interruptions:
                if row['type'] == 'boundary':
                    interruptions_ixs.append(int(float(row['latency'])))
        print("Interruptions indexes:", interruptions_ixs)
        return tuple(interruptions_ixs)

    @staticmethod
    def _timeseries(path, type=EEG, **options):
        """
        Reads all SET files below the given path and makes discontiguous Timeseries with all of them, according to the
           numeric specified in each file name. Returns a dictionary with one Timeseries per channel name.
        Args:
            path (str): Path to the SET files
            type (Biosignal): Type of biosignal to extract. Only EEG allowed.
        """

        filepaths = glob.glob(join(path, '**/*.set'), recursive=True)
        filepaths = tuple(sorted(filepaths, key=lambda x: int(x.split('/')[-2])))  # sort by session number

        if len(filepaths) == 0:
            raise FileNotFoundError(f"No SET files found in '{path}'.")

        timeseries = {}
        segment_shift = 0  # in seconds
        for filepath in filepaths:  # for each SET segment
            samples, initial_datetime, channel_names, sf = Medipol.__read_set_file(filepath, metadata=True)
            units = Volt(Multiplier.u)  # micro-volts

            # Get interruptions, if any
            boundaries_filepath = join(os.path.split(filepath)[0], Medipol.BOUNDARIES_FILENAME)
            interruptions_exist = exists(boundaries_filepath)
            print("Interruptions file looked for:", boundaries_filepath)
            if interruptions_exist:
                interruptions_ixs = np.array(Medipol.__read_boundaries_file(boundaries_filepath))
                # Convert indexes to seconds
                interruptions_times = interruptions_ixs / sf
                segments_initial_times = [0] + interruptions_times.tolist()

            by_product_segments = None  # indexes
            for ch in range(len(channel_names)):
                if channel_names[ch] == "Status":
                    continue
                ch_samples = samples[ch, :]

                # Split samples by interruptions, if any
                if interruptions_exist:
                    samples_by_segment = np.split(ch_samples, interruptions_ixs)

                    # Check for segments with 0 or 1 samples => they are a by-product of MATLAB
                    if by_product_segments is None:  # find them if not before
                        by_product_segments = []
                        for i, seg in enumerate(samples_by_segment):
                            if len(seg) <= 1:
                                by_product_segments.append(i)
                        segments_initial_times = [seg for i, seg in enumerate(segments_initial_times) if i-1 not in by_product_segments]  # i-1 because of the initial 0 added before
                        pass

                    # Discard by-product segments
                    samples_by_segment = [seg for i, seg in enumerate(samples_by_segment) if i not in by_product_segments]

                    # Create Timeseries
                    samples_by_segment_with_time = {}
                    for i in range(len(samples_by_segment)):
                        seg_initial_datetime = initial_datetime + timedelta(seconds=segments_initial_times[i] + i*Medipol.DELTA_BETWEEN_SEGMENTS)
                        samples_by_segment_with_time[seg_initial_datetime] = samples_by_segment[i]
                    ts = Timeseries.withDiscontiguousSegments(samples_by_segment_with_time, sf, units,
                                                              name=f"{channel_names[ch]}")

                else:  # No interruptions
                    ts = Timeseries(ch_samples, initial_datetime, sf, units, name=f"{channel_names[ch]}")

                # Assign or concatenate?
                if channel_names[ch] not in timeseries:
                    timeseries[channel_names[ch]] = ts
                else:
                    ts.timeshift(timeseries[channel_names[ch]].duration + timedelta(seconds=segment_shift+Medipol.DELTA_BETWEEN_SEGMENTS))
                    timeseries[channel_names[ch]] = timeseries[channel_names[ch]] >> ts

            # Update segment shift
            segment_shift = list(timeseries.values())[0].duration.total_seconds()

        return timeseries

    @staticmethod
    def __find_sex_age_diagnosis(patient_code) -> tuple[Sex, int, MedicalCondition | None]:
        with open(Medipol.demographic_csv) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            for row in reader:
                if row['CODE'] == patient_code:
                    sex =  Sex.M if row['GENDER'] == 'M' else Sex.F
                    age = row['AGE']
                    diagnosis = AD() if row['DIAGNOSIS'] == 'AD' else None
                    return sex, age, diagnosis
        raise ValueError(f"Patient code {patient_code} not found in demographics file '{Medipol.demographic_csv}'.")

    @staticmethod
    def _patient(path, **options):
        """
        Gets:
        - patient code from the filepath
        - age from the demographics file
        - gender from the demographics file
        and adds SMC diagnosis.
        """
        # Split path by OS separator
        patient_code = path.split(os.path.sep)[-1]
        sex, age, diagnosis = Medipol.__find_sex_age_diagnosis(patient_code)

        if diagnosis is not None:
            return Patient(patient_code, age=age, sex=sex, conditions=(diagnosis, ))
        else:
            return Patient(patient_code, age=age, sex=sex)

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
