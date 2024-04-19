# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: AFDB
# Description: Class AFDB, a type of BiosignalSource, with static procedures to read and write datafiles from the
# MIT-BIH Atrial Fibrillation at https://physionet.org/content/afdb/1.0.0/.

# Contributors: Rafael Silva
# Created: 27/03/2023
# Last Updated: 28/03/2023

# ===================================

import os
import datetime

import wfdb
import wget
from dateutil.parser import parse as to_datetime

from .. import timeseries
from .. import modalities
from ..sources.BiosignalSource import BiosignalSource
from ltbio.biosignals.timeseries.Unit import *
from ...clinical import Patient, BodyLocation


class AFDB(BiosignalSource):
    '''This class represents the source of MIT-BIH Atrial Fibrillation Database and includes methods to read and write
    biosignal files provided by them.'''

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "MIT-BIH Atrial Fibrillation Database"

    @staticmethod
    def __read_data(dirfile):
        """
        Reads the header, record and annotation files of a record.
        param: dirfile (str) path to one file
        return: record (wfdb.Record) record object
        return: header (wfdb.Header) header object
        return: annotation (wfdb.Annotation) annotation object
        """

        # Read header
        header = wfdb.rdheader(dirfile)
        # Read record
        record = wfdb.rdrecord(dirfile)
        # Read annotation
        annotation = wfdb.rdann(dirfile, 'atr')

        return record, header, annotation

    @staticmethod
    def __aux_date(header):
        """
        Get starting time from header.
        param: header (wfdb.Header)
        """

        # get hour from header
        time = str(header.base_time) if header.base_time is not None else '00:00:00'

        # get date from header
        date = str(header.base_date) if header.base_date is not None else '2000-01-01'

        return to_datetime(date + ' ' + time)

    @staticmethod
    def __aux_sample_to_datetime(sample, start_time, fs):
        """
        Get datetime from sample.
        param: sample (int) sample number
        param: start_time (datetime) starting time of the record
        param: fs (int) sampling frequency
        return: time (datetime) datetime of the sample
        """

        # compute sample timedelta
        time_delta = datetime.timedelta(seconds=sample / fs)

        # compute sample time by adding the time to the starting time
        return start_time + time_delta

    @staticmethod
    def _name(path, type, **options):
        """
        Defines the name of the biosignal file.
        param: path (str) path to the file
        param: type (modalities) modality of the biosignal
        return: name (str) name of the biosignal file
        """

        return 'ECG'

    @staticmethod
    def _patient(path, **options):
        """
        Defines the patient of the biosignal file.
        param: path (str) path to the file
        return: patient (Patient) patient object
        """

        code = path.split(os.sep)[-1]
        patient = Patient(code=code)

        return patient

    @staticmethod
    def _acquisition_location(path, type, **options):
        """
        Defines the acquisition location of the biosignal file.
        param: path (str) path to the file
        param: type (modalities) modality of the biosignal
        return: location (BodyLocation) location of the biosignal file
        """

        return BodyLocation.CHEST

    @staticmethod
    def _timeseries(path: str, type, **options):
        """
        Defines the timeseries of the biosignal file.
        param: path (str) path to the file
        param: type (modalities) modality of the biosignal
        return: data (dict) dictionary with the timeseries of each channel
        """

        # check if modality is ECG
        if type != modalities.ECG:
            raise Exception("This source only supports ECG modality.")

        # read data
        record, header, _ = AFDB.__read_data(path)

        data = dict()

        # one key for each channel
        for ind, channel in enumerate(record.sig_name):

            # set data
            signal = record.p_signal[:, ind]
            unit = Volt(Multiplier.m) if 'mV' in header.units[ind] else None
            fs = header.fs if header.fs is not None else 250
            start_time = AFDB.__aux_date(header)

            # create timeseries
            data[channel] = timeseries.Timeseries(
                samples=signal,
                units=unit,
                sampling_frequency=fs,
                initial_datetime=start_time,
                name=channel
            )

        return data

    @staticmethod
    def _events(path: str, **options):
        """
        Defines the events of the biosignal file.
        param: path (str) path to the file
        return: events (list) list of events
        """

        # read data
        record, header, annotation = AFDB.__read_data(path)
        start_time = AFDB.__aux_date(header)
        fs = header.fs if header.fs is not None else 250

        # create events
        events = []
        for ind, sample in enumerate(annotation.sample):
            # get event
            name = annotation.aux_note[ind]

            # add sequential number to event name
            name = name + '_' + str(ind)

            # get sample time
            onset_time = AFDB.__aux_sample_to_datetime(sample, start_time, fs)

            # get offset
            sample_offset = annotation.sample[ind + 1] - 1 if ind < len(annotation.sample) - 1 else record.p_signal.shape[0]
            offset_time = AFDB.__aux_sample_to_datetime(sample_offset, start_time, fs)

            # create event
            events.append(timeseries.Event(name=name,
                                           onset=onset_time,
                                           offset=offset_time))

        return events

    @staticmethod
    def fetch(patient_code: str):
        """
        Fetch one patient from the wfdb database.
        param: patient_code (str) patient code
        """

        # check if patient code is valid
        if patient_code is None:
            raise Exception("Patient code is required.")

        # create directory for patient in folder
        path = '.cache/afdb/'
        if not os.path.exists(path):
            os.mkdir(path)

        # download files
        for file in ['atr', 'dat', 'hea']:
            file_path = patient_code + '.' + file
            # check if file exists
            if not os.path.exists(path + file_path):
                # download file
                wget.download('https://physionet.org/files/afdb/1.0.0/' + file_path, path)
            else:
                print('File ' + file_path + ' already exists.')

    @staticmethod
    def _write(path: str, timeseries: dict):
        pass

    @staticmethod
    def _transfer(samples: array, type) -> array:
        pass
