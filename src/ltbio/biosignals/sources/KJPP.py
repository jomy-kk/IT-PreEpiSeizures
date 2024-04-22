# -*- encoding: utf-8 -*-
import csv
import glob
import os
from datetime import timedelta, datetime
from os.path import join, exists

import numpy as np
from pandas import read_csv
from scipy.io import loadmat
from simple_icd_10 import is_valid_item

from .. import Timeseries
from ..modalities import EEG
from ..sources.BiosignalSource import BiosignalSource
from ..timeseries.Unit import Volt, Multiplier
from ...clinical import Patient, BodyLocation
from ...clinical.Patient import Sex
from ...clinical.conditions.ICDCode import ICDCode
from ...clinical.conditions.MedicalCondition import MedicalCondition
from ...clinical.medications import Medication
from ...clinical.medications.UnstructuredNotes import UnstructuredNotes


class KJPP(BiosignalSource):
    """This class represents the source of KJPP EEG files and includes methods to read and write
    biosignal files provided by them after denoised by the team with EEGLAB (SET files)."""

    DELTA_BETWEEN_SEGMENTS = 1  # seconds
    BOUNDARIES_FILENAME = "_asr_boundaries.txt"  # as produced by EEGLAB after cleaning artifacts

    def __init__(self, demographic_csv):
        super().__init__()
        KJPP.demographic_csv = demographic_csv

    def __repr__(self):
        return "KJPP Klinik"

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
        for filepath in filepaths:
            samples, initial_datetime, channel_names, sf = KJPP.__read_set_file(filepath, metadata=True)
            units = Volt(Multiplier.u)  # micro-volts

            # Get interruptions, if any
            boundaries_filepath = join(*path.split(filepath)[:-1], KJPP.BOUNDARIES_FILENAME)
            interruptions_exist = exists(boundaries_filepath)
            if interruptions_exist:
                interruptions_ixs = np.array(KJPP.__read_boundaries_file(boundaries_filepath))
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
                        seg_initial_datetime = initial_datetime + timedelta(seconds=segments_initial_times[i] + i*KJPP.DELTA_BETWEEN_SEGMENTS)
                        samples_by_segment_with_time[seg_initial_datetime] = samples_by_segment[i]
                    ts = Timeseries.withDiscontiguousSegments(samples_by_segment_with_time, sf, units,
                                                              name=f"{channel_names[ch]}")

                else:  # No interruptions
                    ts = Timeseries(ch_samples, initial_datetime, sf, units, name=f"{channel_names[ch]}")

                # Assign or concatenate?
                if channel_names[ch] not in timeseries:
                    timeseries[channel_names[ch]] = ts
                else:
                    ts.timeshift(timeseries[channel_names[ch]].duration + timedelta(seconds=KJPP.DELTA_BETWEEN_SEGMENTS))
                    timeseries[channel_names[ch]] = timeseries[channel_names[ch]] >> ts

        return timeseries

    @staticmethod
    def __find_code_sex_age_diagnoses_medication(session_code) -> tuple[str, Sex, float, list[MedicalCondition], Medication]:
        metadata = read_csv(KJPP.demographic_csv, sep=';')
        row = metadata[metadata['SESSION'] == session_code].iloc[0]  # try to find the session code in the demographics file
        if row is not None:
            """ # For future, when the demographics file is updated
            sex = Sex.M if row['SEX'] == 'Male' else Sex.F if row['SEX'] == 'Female' else Sex._
            return (row['PATIENT CODE'], row['PATIENT CODE (short)'], sex, row['AGE']*12)  # age was in months
            """
            # SEX
            sex = Sex.M if row['GENDER'] == 'Male' else Sex.F if row['GENDER'] == 'Female' else Sex._

            # AGE
            age = row['EEG AGE MONTHS']
            if np.isnan(age):
                raise LookupError(f"Age not found for session code {session_code}.")
            else:
                age = row['EEG AGE MONTHS'] / 12  # age was in months

            # DIAGNOSES
            # They are ICD-10 codes, but they are grouped by the Multi-Axial System (MAS) of the DSM-V.
            axis1 = [str(d).replace(',', '.') for d in row['MAS11':'MAS15'] if str(d) != 'nan']  # add 'F' to make them ICD-10 codes
            axis1 = ['F0' + str(d) if float(d) < 10 else 'F'+str(d) for d in axis1]  # add leading 0 to single-digit codes
            axis2 = [str(d).replace(',', '.') for d in row['MAS21':'MAS25'] if str(d) != 'nan']  # add 'F' to make them ICD-10 codes
            axis2 = ['F0' + str(d) if float(d) < 10 else 'F'+str(d) for d in axis2]  # add leading 0 to single-digit codes

            axis3 = str(row['MAS3']).replace(',', '.')
            if str(axis3) == 'nan':
                axis3 = []
            else:
                axis3 = float(axis3)

                if 0 <= axis3 <= 9:  # physician meant the levels 1-9
                    if axis3.is_integer():  # it should be an integer, if not it's an IQ score
                        if axis3 >= 5:
                            match axis3:
                                case 5:
                                    axis3 = ['F70', ]
                                case 6:
                                    axis3 = ['F71', ]
                                case 7:
                                    axis3 = ['F72', ]
                                case 8:
                                    axis3 = ['F73', ]
                                case 9:
                                    axis3 = ['F78', ]
                        else:
                            axis3 = []  # IQ score >= 70 => no diagnosis
                    else:
                        axis3 = ['F73', ]  # IQ score < 20


                elif 70 <= axis3 <= 74 or 78 <= axis3 <= 79:  # physician meant the F-letter codes
                    axis3 = ['F' + str(axis3), ]

                elif 10 <= axis3 < 70:
                    if axis3 < 20:
                        axis3 = ['F73', ]
                    elif axis3 < 35:
                        axis3 = ['F72', ]
                    elif axis3 < 50:
                        axis3 = ['F71', ]
                    elif axis3 < 70:
                        axis3 = ['F70', ]

                else:
                    axis3 = []

            axis4 = [str(d.replace(',', '.')) for d in row['MAS41':'MAS45'] if str(d) != 'nan']
            for i, d in enumerate(axis4):
                if float(d) < 10:
                    d = '0' + str(d)
                assigned = False
                for letter in ('G', 'Q', 'R', 'C', 'D', 'Z', 'H', 'I', 'E', 'J', 'L', 'O', 'M', 'N', 'K', 'Y',):  # from the most statistically likely to the least
                    if is_valid_item(letter + d):
                        axis4[i] = letter + d
                        assigned = True
                        break
                if not assigned:
                    axis4[i] = 'Axis4:' + d

            # We'll not include axes 5 and 6, as they are usually not relevant for biosignal analysis
            diagnoses_codes = axis1 + axis2 + axis3 + axis4

            diagnoses = []
            for d in diagnoses_codes:
                try:
                    # Is there a diagnosis age?
                    diagnosis_age = str(row['DIAGNOSIS AGE MONTHS'])
                    if diagnosis_age != 'nan':
                        diagnosis_age = float(diagnosis_age)/12  # age was in months
                        diagnoses.append(ICDCode(d, diagnosis_age-age))
                    else:
                        diagnoses.append(ICDCode(d))
                except ValueError:
                    print(f"ICD code {d} not found.")
                    diagnoses.append(d)
            medication = UnstructuredNotes(row['MEDICATION'])

            return (row['PATIENT'], sex, age, diagnoses, medication)
        else:
            raise FileNotFoundError(f"Session code {session_code} not found in demographics file '{KJPP.demographic_csv}'.")

    @staticmethod
    def _patient(path, **options):
        """
        With the:
        - session code from the filepath
        Gets the:
        - patient code from the demographics file
        - short patient code from the demographics file
        - age from the demographics file
        - gender from the demographics file
        """

        session_code = os.path.split(path)[-1]
        patient_code, sex, age, diagnoses, medication = KJPP.__find_code_sex_age_diagnoses_medication(session_code)

        return Patient(code=patient_code,
                       age=age,
                       sex=sex,
                       conditions=tuple(diagnoses),
                       medications=(medication, ),
                       name=patient_code)  # keep the long patient code as name for backwards compatibility

    @staticmethod
    def _acquisition_location(path, type, **options):
        return BodyLocation.SCALP

    @staticmethod
    def _name(path, type, **options):
        """
        Gets the short session code from the demographics file.
        """
        session_code = os.path.split(path)[-1]
        """ # For future, when the demographics file is updated
        #with open(KJPP.demographic_csv) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['SESSION CODE'] == session_code:
                    return row['SESSION CODE (short)']
        raise ValueError(f"Session code {session_code} not found in demographics file '{KJPP.demographic_csv}'.")
        """
        return session_code

    @staticmethod
    def _write(path:str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples, to_unit):
        pass
