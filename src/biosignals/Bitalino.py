import ast
from datetime import datetime
from os import listdir, path

import numpy as np
from dateutil.parser import parse as to_datetime

import numpy as np

from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.Timeseries import Timeseries
from src.biosignals.ECG import ECG


class Bitalino(BiosignalSource):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Bitalino"

    def __aux_date(header):
        """
        Get starting time from header
        """
        time_key = [key for key in header.keys() if 'time' in key][0]
        try:
            start_time = to_datetime(header['date'].strip('\"') + ' ' + header[time_key].strip('\"'))
        except Exception as e:
            print(e)
        return start_time

    def __check_empty(len_, type=''):
        """
        Confirm if the length is acceptable and return the desired output
        """
        if type == 'file_size':
            if len_ <= 50:
                return True
        else:
            if len_ < 1:
                return True
        return False

    def __change_sens_list(sens, device, channels):
        """
        Confirm if the list of sensors has only RAW as labels, and ask the user for new labels in that case.
        """
        if list(set(sens)) == ['RAW']:
            print(f'Please update sens according to the sensors used:')
            analogs = channels[-len(sens):]
            for se in range(len(sens)):
                new_se = str(input(f'{device} -- {sens[se]} -- {analogs[se]}')).upper()
                sens[se] = new_se
        return sens

    def __analog_idx(header, sensor, **options):
        """
        From a header choose analog sensor key idx that correspond to a specific sensor.
        This also runs read json to save configurations to facilitate implementation
        This function leads with several devices and it returns a list that may contain one or several integers
        """
        sensor_idx, sensor_names, json_bool, chosen_device = [], [], False, ''
        if options:
            if 'json' in options.keys():
                json_bool = options['json']
                json_dir = options['json_dir'] if 'json_dir' in options.keys() \
                    else path.join(os.getcwd(), 'bitalino.json')
        len_ch = 0
        for device in header.keys():
            chosen_device = device
            sens_id = ''
            # iterate over each device
            if json_bool:
                sens, ch, location = Bitalino.__read_json(json_dir, header[device])
            else:
                sens = header[device][str(input(f'What is the header key of sensor names? {header}\n ')).strip().lower()]
                ch = header[device][str(input(f'What is the header key for analog channels? {header}\n ')).strip().lower()]
                location = str(input(f'What is the body location of this device {device}? \n'))
                sens = Bitalino.__change_sens_list(sens, device, ch)
            analogs = ch[-len(sens):]

            if sensor in str(sens):
                # add other column devices as offset to the column to retrieve
                sens_id = [lab + '_' + location for lab in sens if sensor in lab.upper()]
                sensor_idx += [len_ch + ch.index(analogs[sens.index(sid.split('_')[0])]) for sid in sens_id]
            if sens_id != '':
                chosen_device = device
            len_ch = len(ch)
            sensor_names += sens_id

        return sensor_idx, sensor_names, chosen_device

    def __read_json(dir_, header):
        # check if bitalino json exists and returns the channels and labels and location
        if os.path.isfile(dir_) and os.access(dir_, os.R_OK):
            # checks if file exists
            with open(dir_, 'r') as json_file:
                json_string = json.load(json_file)
        else:
            print("Either file is missing or is not readable, creating file...")
            json_string = {}
        if 'device connection' in header.keys():
            device = header['device connection']
        else:
            device = input('Enter device id (string): ')
        if device not in json_string.keys():
            json_string[device] = {}

        for key in ['column', 'label', 'firmware version', 'device', 'resolution', 'channels', 'sensor', 'location']:
            if key not in json_string[device].keys():
                if key in header.keys():
                    json_string[device][key] = header[key]
                else:
                    print(header['device connection'], header['label'])
                    new_info = str(input(f'{key}: ')).lower()
                    json_string[device][key] = new_info
            if key == 'label':
                sens = Bitalino.__change_sens_list(json_string[device]['label'], device, header['column'])
                json_string[device][key] = sens
        with open(dir_, 'w') as db_file:
            json.dump(json_string, db_file, indent=2)
        return json_string[device]['label'], json_string[device]['column'], json_string[device]['location']

    # @staticmethod
    def __read_bit(list_, metadata=False, sensor_idx=[], sensor_names=[], device='', **options):
        """
        Reads one edf file
        If metadata is True - returns list of channels and sampling frequency and initial datetime
        Else return arrays one for each channel
        """
        dirfile = list_[0]
        sensor = list_[1]
        # size of bitalino file
        file_size = path.getsize(dirfile)
        if file_size <= 50:
            if metadata:
                return {}
            else:
                return '', []
        with open(dirfile) as fh:
            next(fh)
            header = next(fh)[2:]
            next(fh)
            # signal
            data = np.array([line.strip().split() for line in fh], float)
        # if file is empty, return
        if Bitalino.__check_empty(len(data)):
            return None

        header = ast.literal_eval(header)
        if len(sensor_idx) < 1:
            sensor_idx, sensor_names, device = Bitalino.__analog_idx(header, sensor, **options)
        if metadata:
            return sensor_idx, sensor_names, device, header[device]
        if len(sensor_names) > 0:
            sensor_data = data[:, sensor_idx]
            date = Bitalino.__aux_date(header[device])
            return sensor_data, date
        else:
            raise IOError(f"Sensor {sensor} was not found in this acquisition, please insert another")

    @staticmethod
    def _read(dir, type, startkey='A20', **options):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        """
        """
        # sensor = 'ECG' if type is ECG else 'EDA' if type is EDA else 'PPG' if type is PPG else 'ACC' if type is ACC
        sensor = 'ECG' if type is ECG else type
        # TODO create sensor if other types are used besides ECG
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        all_files = sorted([[path.join(dir, file), sensor] for file in listdir(dir) if file.startswith(startkey)])
        # get header and sensor positions by running the bitalino files until a header is found
        if not all_files:
            raise IOError(f'No files in dir="{dir}" that start with {startkey}')
        header, h = {}, 0
        while len(header) < 1:
            ch_idx, channels, device, header = Bitalino.__read_bit(all_files[h], metadata=True, **options)
            h += 1
        if header == {}:
            raise IOError(f'The files in {dir} did not contain a bitalino type {header}')
        new_dict = {}
        segments = [Bitalino.__read_bit(file, sensor_idx=ch_idx, sensor_names=channels, device=device, **options)
                    for file in all_files[h-1:h+2]]
        for ch, channel in enumerate(channels):
            samples = [Timeseries.Segment(segment[0][:, ch], initial_datetime=segment[1],
                                          sampling_frequency=header['sampling rate'])
                       for segment in segments if segment]
            new_timeseries = Timeseries(samples, sampling_frequency=header['sampling rate'], ordered=True)
            new_dict[channel] = new_timeseries
        return new_dict

    @staticmethod
    def _write(dir):
        '''Writes multiple TXT files on the directory 'path' so they can be opened in Opensignals.'''
        # TODO


import time
for pat in ['Patient Test']: # listdir('G:\\PreEpiSeizures\\Patients_HSM'):
    pat_dir = ('G:\\PreEpiSeizures\\Patients_HSM\\'+pat)

    start_time = time.time()
    bit_dir = 'C:\\Users\\Mariana\\PycharmProjects\\IT-PreEpiSeizures\\resources\\BIT_TXT_tests'
    options = {'json': True,
               'json_dir': 'C:\\Users\\Mariana\\PycharmProjects\\IT-PreEpiSeizures\\src\\biosignals\\bitalino.json'}
    vals = Bitalino._read(bit_dir, 'PPG', **options)
    for valey in vals.keys():
        print(f'Pat {pat} files {len(vals[valey])}')
    print(f'Pat {pat} run in {time.time() - start_time}s')
