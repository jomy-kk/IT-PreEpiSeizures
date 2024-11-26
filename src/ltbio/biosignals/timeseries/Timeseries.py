# -*- encoding: utf-8 -*-
import os
# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: Timeseries
# Description: Class Timeseries, which mathematically conceptualizes timeseries and their behaviour.
# Class OverlappingTimeseries, a special kind of Timeseries for signal processing purposes.

# Contributors: João Saraiva, Mariana Abreu
# Created: 20/04/2022
# Last Updated: 22/07/2022

# ===================================

from datetime import datetime, timedelta
from math import ceil
from os.path import join
from tempfile import mkstemp
from typing import List, Iterable, Collection, Dict, Tuple, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from biosppy.signals.tools import power_spectrum
from datetimerange import DateTimeRange
from dateutil.parser import parse as to_datetime
from numpy import array, append, ndarray, divide, concatenate, tile, memmap
from scipy.signal import resample

from ltbio.biosignals.timeseries import Timeline
from ltbio.biosignals.timeseries.Event import Event
from ltbio.biosignals.timeseries.Frequency import Frequency
from ltbio.biosignals.timeseries.Unit import Unit


#from ltbio.processing.filters.Filter import Filter

class Timeseries():
    """
    A Timeseries is a sequence of data points that occur in successive order over some period of time.
    In a Biosignal, one Timeseries' data points are the measurement of a biological variable, in some unit, taken from a
    sensor or channel. This data points are often called samples, and are acquired at fixed sampling frequency.

    To each time point of a Timeseries' domain corresponds one and only one sample. However, a Timeseries might be
    contiguous if a sample was acquired at every sampling time point, or discontiguous if there were interruptions. Each
    interval/sequence of contiguous samples is called a Segment, but those are managed internally.

    Constructors / Initializers
    ______________

    Timeseries: default
        Instantiates a Timeseries with a contiguous sequence of samples.

    Timeseries.withDiscontiguousSegments
        Instantiates a Timeseries with discontiguous sequences of samples.


    Properties:
    ______________

    name: str
        The name of the Timeseries, if any.

    samples: array  # FIXME
        Contiguous or discontiguous sequence of samples.

    sampling_frequency: float
        The frequency at which the samples were acquired, in Hz.

    units: Unit
        The physical unit at which the samples should be interpreted.

    events: tuple[Event]
        The events timely associated to the Timeseries.

    initial_datetime: datetime
        The date and time of the first sample.

    final_datetime: datetime
        The date and time of the last sample.

    duration: timedelta
        The total time of acquired samples, excluding interruptions.

    domain: tuple[DateTimeRange]
        The intervals of date and time in which the Timeseries is defined, i.e., samples were acquired.

    is_equally_segmented: bool
        The logic value stating if each interval in the domain has the same duration.

    segment_duration: timedelta:
        Duration of all segments, if is_equally_segmented is True.


    Built-ins:
    ______________

    len: Returns the total number of samples.

    copy: Copies all Timeseries' content.

    iter: Returns an iterator over the samples of all Timeseries' Segments.

    in: Returns True if a date, time or event is contained in the Timeseries.

    [] : Indexes by date, time or events.

    + : Adds Timeseries.

    += : Appends more samples to the last Timeseries' Segment.

    Methods:
    ______________

    append(datetime, array):
        Appends a new sequence of samples in a separate Segment.

    associate(Event):
        Timely associates a given Event to the Timeseries.

    dissociate(str):
        Removes any association the Timeseries has with an Event with the given name.

    filter(Filter):
        Filters the Timeseries with the given design.

    undo_filters():
        Reverts the effect of all filters.

    plot():
        Plots the Timeseries amplitude over time, with all its interruptions, if any.

    plot():
        Plots the Timeseries frequency spectrum.

    ______________

    Full documentation in:
    https://github.com/jomy-kk/IT-LongTermBiosignals/wiki/%5BClass%5D-Timeseries
    """

    __SERIALVERSION: int = 2

    # ===================================
    # Class: Segment

    class __Segment():
        """
        A Segment is an interrupted sequence of samples.
        This is an internal class of Timeseries, for its internal management, and should not be used outside Timeseries.

        Properties:
        ______________

        samples: array
            Sequence of samples, acquired at a fixed sampling rate.

        raw_samples: array
            Original samples, without application of filters or other operations in-place.

        initial_datetime: datetime
            The date and time of the first sample.

        final_datetime: datetime
            The date and time of the last sample.

        duration: timedeltao
            The total time of acquired samples, i.e., final_datetime - initial_datetime.

        is_filtered: bool
            The logic value stating if the samples have been filtered.


        Built-ins:
        ______________

        len: Returns the total number of samples.

        copy: Copies all Timeseries' content.

        in: Returns True if a date, time or event is contained in the Timeseries.

        [] : Indexes by sample index.

        += : Appends more samples.

        <, >, <=, >= : State if onw Timeseries come before another.


        Methods:
        ______________

        adjacent(Segment):
            Returns whether two Segments are adjacent in time.

        overlap(Segment):
            Returns whether two Segments overlap in time.

        """

        __SERIALVERSION: int = 2

        def __init__(self, samples: ndarray, initial_datetime: datetime, sampling_frequency: Frequency,
                     is_filtered: bool = False):
            """
            A Segment is an uninterrupted sequence of samples.

            Parameters
            ------------
            samples: ndarray
                The samples to store.

            initial_datetime: datetime
                The date and time of the first sample.

            sampling_frequency: Frequency
                Reference to the sampling frequency object of the respective Timeseries.

            is_filtered: bool
                If samples have been filtered.
            """

            # Save metadata
            self.__initial_datetime = initial_datetime
            self.__final_datetime = self.initial_datetime + timedelta(seconds=len(samples) / sampling_frequency)
            self.__raw_samples = samples  # if some filter is applied to a Timeseries, the raw version of each Segment should be saved here
            self.__is_filtered = is_filtered
            self.__sampling_frequency = sampling_frequency

            # Save samples
            self.__samples = samples
            """
            if not isinstance(samples, memmap):
                # Create a memory map for the array
                file_name = str(hash(self.__initial_datetime) * hash(self.__final_datetime) * hash(len(samples)))
                self.__filepath = join(__temp__.name, file_name)
                self.__samples = memmap(self.__filepath, dtype='float32', mode='w+', shape=samples.shape)
                self.__samples[:] = samples[:]
                self.__samples.flush()  # release memory in RAM; don't know if this is actually helping
                del samples  # delete np.array
            else:
                self.__samples = samples
            """

        # ===================================
        # Properties

        @property
        def samples(self) -> array:
            return self.__samples

        @property
        def raw_samples(self) -> array:
            return self.__raw_samples

        @property
        def initial_datetime(self) -> datetime:
            return self.__initial_datetime

        @property
        def final_datetime(self) -> datetime:
            return self.__final_datetime

        @property
        def domain(self) -> DateTimeRange:
            return DateTimeRange(self.__initial_datetime, self.__final_datetime)

        @property
        def duration(self) -> timedelta:
            return self.__final_datetime - self.__initial_datetime

        @property
        def is_filtered(self) -> bool:
            return self.__is_filtered

        # ===================================
        # Built-ins

        def __len__(self):
            return len(self.__samples)

        def __rshift__(self, other: ndarray | list):
            if isinstance(other, type(self)):  # if it's Segment
                if other.final_datetime == self.initial_datetime and other._Segment__sampling_frequency == self.__sampling_frequency:
                    self.__samples = append(self.__samples, other.samples)
                    if self.is_filtered and other.is_filtered:
                        self.__raw_samples = append(self.__samples, other.raw_samples)
                    self.__final_datetime = other.final_datetime
                else:
                    raise AssertionError("Segments cannot be concatenated.")
            else:
                self.__samples = append(self.__samples, other)
                self.__final_datetime += timedelta(seconds= len(other)/self.__sampling_frequency)

        def __add__(self, other):
            if isinstance(other, type(self)):  # if it is a Segment
                return self._new(samples = self.samples + other.samples, is_filtered=False)  # raw is lost

        def __contains__(self, item):  # Operand 'in' === belongs to
            if isinstance(item, datetime):
                return self.initial_datetime <= item < self.final_datetime
            if isinstance(item, DateTimeRange):
                return item in self.domain
            if isinstance(item, type(self)):  # item is a Segment
                # A Segment contains other Segment if its start is less than the other's and its end is greater than the other's.
                return self.initial_datetime < item.initial_datetime and self.final_datetime > item.final_datetime

        def __getitem__(self, position):
            '''The built-in slicing and indexing (segment[x:y]) operations.'''
            if isinstance(position, (int, tuple)):
                return self.__samples[position]
            elif isinstance(position, slice):
                if position.start is None:
                    new_initial_datetime = self.__initial_datetime
                else:
                    new_initial_datetime = self.__initial_datetime + timedelta(
                        seconds=position.start / self.__sampling_frequency.value)
                return self._new(samples=self.__samples[position], initial_datetime=new_initial_datetime,
                                 raw_samples=self.__raw_samples[position])

        # ===================================
        # Amplitude methods

        def max(self):
            return np.max(self.__samples)

        def min(self):
            return np.min(self.__samples)

        # ===================================
        # Binary Logic using Time and Conditions

        def __lt__(self, other):
            """A Segment comes before other Segment if its end is less than the other's start."""
            if isinstance(other, Timeseries._Timeseries__Segment):
                return self.final_datetime < other.initial_datetime
            else:
                return tuple(self.__when(self.__samples < other))

        def __le__(self, other):
            if isinstance(other, Timeseries._Timeseries__Segment):
                return self.final_datetime <= other.initial_datetime
            else:
                return tuple(self.__when(self.__samples <= other))

        def __gt__(self, other):
            """A Segment comes after other Segment if its start is greater than the other's end."""
            if isinstance(other, Timeseries._Timeseries__Segment):
                return self.initial_datetime > other.final_datetime
            else:
                return tuple(self.__when(self.__samples > other))

        def __ge__(self, other):
            if isinstance(other, Timeseries._Timeseries__Segment):
                return self.initial_datetime >= other.final_datetime
            else:
                return tuple(self.__when(self.__samples >= other))

        def __eq__(self, other):
            """A Segment corresponds to the same time period than other Segment if their start and end are equal."""
            if isinstance(other, Timeseries._Timeseries__Segment):
                return self.initial_datetime == other.initial_datetime and self.final_datetime == other.final_datetime
            else:
                return tuple(self.__when(self.__samples == other))

        def __ne__(self, other):
            if isinstance(other, Timeseries._Timeseries__Segment):
                return not self.__eq__(other)
            else:
                return tuple(self.__when(self.__samples != other))

        def overlaps(self, other):
            """A Segment overlaps other Segment if its end comes after the other's start, or its start comes before the others' end, or vice versa."""
            if self <= other:
                return self.final_datetime > other.initial_datetime
            else:
                return self.initial_datetime < other.final_datetime

        def adjacent(self, other):
            """Returns True if the Segments' start or end touch."""
            return self.final_datetime == other.initial_datetime or self.initial_datetime == other.final_datetime

        def __when(self, condition):
            intervals = []
            true_interval = False
            start, end = None, None

            for i, x in enumerate(condition):
                if x:
                    if not true_interval:  # not open
                        true_interval = True  # then open
                        start = i
                else:
                    if true_interval:  # is open
                        true_interval = False
                        end = i
                        intervals.append((start, end))  # close interval

            if true_interval:  # is open
                intervals.append((start, i+1))  # then close

            return intervals

        def _when(self, condition, window_length: int = 1):
            assert  window_length > 0
            if window_length == 1:
                evaluated = [condition(x) for x in self.__samples]
            else:
                evaluated = []
                for i in range(0, len(self.__samples), window_length):
                    x = self.__samples[i: i+window_length]
                    evaluated += [condition(x), ] * len(x)
            return self.__when(evaluated)

        def sliding_window(self, window_length: int, overlap_length: int = 0) -> Iterable[ndarray]:
            """
            Yields windows of length window_length, sliding by overlap_length.
            If the last window is smaller than window_length, these last samples are not yielded.
            """
            assert window_length > 0
            assert overlap_length > 0
            assert window_length > overlap_length

            for i in range(0, len(self.__samples), window_length - overlap_length):
                yield self.__samples[i: i+window_length]

        def timeshift(self, delta: timedelta):
            self.__initial_datetime = self.__initial_datetime + delta
            self.__final_datetime = self.__final_datetime + delta

        # ===================================
        # INTERNAL USAGE - Accept Methods

        # General-purpose

        def _apply_operation(self, operation, **kwargs):
            """
            Protected Access: For use of this module.
            Applies operation in-place to its samples.
            """
            self.__samples = operation(self.__samples, **kwargs)

        def _apply_operation_and_return(self, operation, **kwargs):
            """
            Protected Access: For use of this module.
            Applies operation to a copy of its samples and returns the output.
            """
            return operation(self.__samples.copy(), **kwargs)

        # Purpose-specific

        def _accept_filtering(self, filter_design):
            """
            Protected Access: For use of this module.
            Applies a filter to its samples, given a design.
            """
            res = filter_design._visit(self.__samples)  # replace with filtered samples
            self.__samples = res
            self.__is_filtered = True

        def _restore_raw(self):
            """
            Protected Access: For use of this module.
            Restores the raw samples.
            """
            if self.is_filtered:
                self.__samples = self.__raw_samples
                self.__is_filtered = False

        def _resample(self, new_frequency: Frequency):
            """
            Protected Access: For use of this module.
            Resamples the samples to a new sampling frequency.
            """
            n_samples = int(new_frequency * len(self) / self.__sampling_frequency)
            self.__samples = resample(self.__samples, num=n_samples)
            self.__raw_samples = resample(self.__raw_samples, num=n_samples)
            self.__sampling_frequency = new_frequency
            self.__final_datetime = self.initial_datetime + timedelta(seconds=len(self) / new_frequency.value)

        # ===================================
        # INTERNAL USAGE - Make similar copies or itself

        def __copy__(self):
            """ Creates an exact copy of the Segment contents and returns the new object. """
            new = type(self)(self.samples.copy(), self.initial_datetime, self.__sampling_frequency.__copy__(),
                             self.is_filtered)
            new._Segment__raw_samples = self.__raw_samples
            return new

        def _new(self, samples: array = None, initial_datetime: datetime = None, sampling_frequency: Frequency = None,
                 is_filtered: bool = False, raw_samples: array = None):
            """
            Protected Access: For use of this module.

            Creates a similar copy of the Segment's contents and returns the new object.
            The value of any field can be changed, when explicitly given a new value for it. All others will be copied.

            :param samples: Different samples. Optional.
            :param initial_datetime: A different date and time of the first sample. Optional.
            :param sampling_frequency: A different sampling frequency of the samples. Optional.
            :param is_filtered: Alter the filtered state. Optional.
            :param raw_samples: Different raw samples. Optional.

            Note: If none of these parameters is given, this method is equivalent to '__copy__'.

            :return: A new Segment with the given fields changed. All other contents shall remain the same.
            :rtype: Segment
            """
            samples = self.__samples if samples is None else samples
            initial_datetime = self.__initial_datetime if initial_datetime is None else initial_datetime
            sampling_frequency = self.__sampling_frequency if sampling_frequency is None else sampling_frequency
            is_filtered = self.__is_filtered if is_filtered is None else is_filtered
            raw_samples = self.__raw_samples if raw_samples is None else raw_samples

            new = type(self)(samples, initial_datetime, sampling_frequency, is_filtered)
            new._Segment__raw_samples = raw_samples
            return new

        def _apply_operation_and_new(self, operation: Callable, initial_datetime: datetime = None,
                                     sampling_frequency: Frequency = None, **kwargs):
            """
            Protected Access: For use of this module.

            Similarly to '_apply_operation', it applies 'operation' but saves the resulting samples in a new Segment,
            which is returned.

            :param operation: A procedures to be executed over the samples. Its First argument must expect a ndarray.
            :param initial_datetime: A different date and time the first sample might have after the operation.
            :param sampling_frequency: A different sampling frequency the samples might have after the operation.
            :param kwargs: Additional arguments to pass when calling 'operation'.

            :return: A new Segment with the samples outputted by the operation. All other contents shall remain the same,
            except for initial_datetime and sampling_frequency if new values were given.
            :rtype: Segment
            """
            samples = operation(self.__samples.copy(), **kwargs)
            return self._new(samples, initial_datetime=initial_datetime, sampling_frequency=sampling_frequency)

        def _partition(self, individual_length: int, overlap_length: int = 0):
            """
            Protected Access: For use of this module.

            Splits one Segment into many Segments of equal length. Overlaps can be enforced.

            :param individual_length: Length of each resulting Segment.
            :param overlap_length: Overlap length between the resulting Segments.

            :return: A list of new Segments with the original samples distributed. All other properties shall remain the same,
            except for initial_datetime which changes for each new Segment.
            :rtype: list
            """

            res = []

            step = individual_length - overlap_length
            for i in range(0, len(self), step):
                trimmed_samples = self.__samples[i: i + individual_length]
                trimmed_raw_samples = self.__raw_samples[i: i + individual_length]
                if len(trimmed_samples) == individual_length:
                    res.append(self._new(samples=trimmed_samples, raw_samples=trimmed_raw_samples,
                                         initial_datetime=self.__initial_datetime + timedelta(seconds=i/self.__sampling_frequency)))

            return res

        # ===================================
        # SERIALIZATION

        def _memory_map(self, path):
            if len(self.__samples) == 0:  # Empty Segment, let it go, don't save. This is a very "engineering" solution to a problem that should not exist in the first place. But it works, so we'll keep it.
                return
            if not isinstance(self.__samples, memmap):  # Create a memory map for the array
                _, file_name = mkstemp(dir=path, suffix='.segment')
                filepath = join(path, file_name)
                self.__memory_map = memmap(filepath, dtype='float32', mode='r+', shape=self.__samples.shape)
                self.__memory_map[:] = self.__samples[:]
                self.__memory_map.flush()  # release memory in RAM; don't know if this is actually helping

        def __hash__(self):
            return hash(self.__initial_datetime) * hash(self.__final_datetime) * hash(self.__samples)

        def __getstate__(self):
            """
            1: __initial_datetime (datetime)
            2: __samples (ndarray)
            """
            if isinstance(self.__samples, memmap):  # Case: has been saved as .biosignal before
                return (Timeseries._Timeseries__Segment._Segment__SERIALVERSION, self.__initial_datetime, self.__samples)
            elif hasattr(self, '_Segment__memory_map'):  # Case: being saved as .biosignal for the first time
                return (Timeseries._Timeseries__Segment._Segment__SERIALVERSION, self.__initial_datetime, self.__memory_map)
            else:  # Case: being called by deepcopy
                return (Timeseries._Timeseries__Segment._Segment__SERIALVERSION, self.__initial_datetime, self.__samples)

        def __setstate__(self, state):
            """
            Version 1 and 2:
            1: __initial_datetime (datetime)
            2: __samples (ndarray)
            3: __sampling_frequency (Frequency)
            """
            if state[0] == 1 or state[0] == 2:
                self.__initial_datetime, self.__samples, self.__sampling_frequency = state[1], state[2], state[3]
                self.__final_datetime = self.initial_datetime + timedelta(seconds=len(self.__samples) / self.__sampling_frequency)
                self.__is_filtered = False
                self.__raw_samples = self.__samples
            else:
                raise IOError(
                    f'Version of Segment object not supported. Serialized version: {state[0]}; '
                    f'Supported versions: 1, 2.')


    # ===================================
    # Class: Timeseries

    def __init__(self, samples: ndarray | list | tuple, initial_datetime: datetime, sampling_frequency: float,
                 units: Unit = None, name: str = None):
        """
        Give a sequence of contiguous samples, i.e. without interruptions, and the datetime of the first sample.
        If there are interruptions, append the remaining segments using the 'append' method.
        It also receives the sampling frequency of the samples.
        Additionally, it can receive the samples' units and a name, if needed.

        Parameters
        ------------
        samples: ndarray | list | tuple
            The samples to store, without interruptions.

        initial_datetime: datetime
            The date and time of the first sample.

        sampling_frequency: float | Frequency
            The frequency at which the samples where sampled.

        units: Unit
            The physical units of the variable measured.

        name: str
            A symbolic name for the Timeseries. It is mentioned in plots, reports, error messages, etc.
        """
        _sampling_frequency = sampling_frequency if isinstance(sampling_frequency, Frequency) else Frequency(
            sampling_frequency)

        # Shortcut: Check if being copied
        if isinstance(samples, list) and isinstance(samples[0], Timeseries.__Segment):
            self.__segments = samples

        else:
            # Creat first Segment
            samples = array(samples) if not isinstance(samples, ndarray) else samples
            segment = Timeseries.__Segment(samples, initial_datetime, _sampling_frequency)
            self.__segments = [segment, ]

        # Metadata
        self.__sampling_frequency = _sampling_frequency
        self.__units = units
        self.__name = name
        self.__associated_events = {}
        self.__tags:set[str] = set()


        # Control Flags
        self.__is_equally_segmented = True  # Because there's only 1 Segment

    @classmethod
    def withDiscontiguousSegments(cls, segments_by_time: Dict[datetime, ndarray | list | tuple],
                                  sampling_frequency: float, units: Unit = None, name: str = None):
        """
        Give a dictionary of discontiguous sequences of samples, keyed by their initial date and time.
        It also receives the sampling frequency of the samples.
        Additionally, it can receive the samples' units and a name, if needed.

        Parameters
        ------------
        samples: dict [datetime, ndarray | list | tuple]
            The sequence of samples to store as separate Segments, in the format { datetime: [, ... ], ... }.

        initial_datetime: datetime
            The date and time of the first sample.

        sampling_frequency: float | Frequency
            The frequency at which the samples where sampled.

        units: Unit
            The physical units of the variable measured.

        name: str
            A symbolic name for the Timeseries. It is mentioned in plots, reports, error messages, etc.
        """

        #if len(segments_by_time) < 2:
        #    raise TypeError("Use the regular initializer to instantiate a Timeseries with 1 contiguous segment.")

        # Sort the segments
        ordered_arrays = sorted(segments_by_time.items())  # E.g. [ (datetime, array), (.., ..), .. ]

        # Create Timeseries with the first Segment
        initial_datetime, first_array = ordered_arrays[0]
        new = cls(first_array, initial_datetime, sampling_frequency, units, name)

        # Append the remaining Segments
        for datetime, array in ordered_arrays[1:]:
            new.append(datetime, array)

        return new

    # ===================================
    # Properties

    @property
    def segments(self) -> list:  # FIXME: deprecate this
        return self.__segments

    @property
    def samples(self) -> list | ndarray:
        if len(self.__segments) == 1:
            return  self.__segments[0].samples#.copy()
        else:
            #return [segment.samples.copy() for segment in self.__segments]
            return [segment.samples for segment in self.__segments]

    @property
    def initial_datetime(self) -> datetime:
        """The date and time of the first sample."""
        return self.__segments[0].initial_datetime  # Is the initial datetime of the first Segment.

    @property
    def final_datetime(self) -> datetime:
        """The date and time of the last sample."""
        return self.__segments[-1].final_datetime  # Is the final datetime of the last Segment.

    @property
    def domain(self) -> Tuple[DateTimeRange]:
        """The intervals of date and time in which the Timeseries is defined, i.e., samples were acquired."""
        return tuple([DateTimeRange(segment.initial_datetime, segment.final_datetime) for segment in self])

    @property
    def subdomains(self) -> Tuple[DateTimeRange]:
        return self.domain

    @property
    def duration(self) -> timedelta:
        """ returns actual recorded time without interruptions
        """
        total_time = timedelta(seconds=0)
        for segment in self:
            total_time += segment.duration
        return total_time

    @property
    def sampling_frequency(self) -> float:
        """The frequency at which the samples were acquired, in Hz."""
        return float(self.__sampling_frequency)

    @property
    def units(self):
        """The physical unit at which the samples should be interpreted."""
        return self.__units

    @property
    def name(self):
        """The name of the Timeseries, if any."""
        return self.__name if self.__name != None else "No Name"

    @name.setter
    def name(self, name: str):
        """Set or reset a name for the Timeseries."""
        self.__name = name

    @property
    def is_contiguous(self) -> bool:
        """The logic value stating if there are no interruptions in time."""
        return len(self.__segments) == 1

    @property
    def is_equally_segmented(self) -> bool:
        """The logic value stating if each interval in the domain has the same duration."""
        return self.__is_equally_segmented

    @property
    def segment_duration(self) -> timedelta:
        """Duration of segments, if equally segmented."""
        if not self.is_equally_segmented:
            raise AttributeError("There is no segment duration because this Timeseries is not equally segmented.")
        else:
            return self.__segments[0].duration

    @property
    def segment_length(self) -> int:
        """Number of samples of segments, if equally segmented."""
        if not self.is_equally_segmented:
            raise AttributeError("There is no segment length because this Timeseries is not equally segmented.")
        else:
            return len(self.__segments[0])

    @property
    def n_segments(self) -> int:
        """The number of uninterrupted segments."""
        return len(self.__segments)

    @property
    def events(self) -> Tuple[Event]:
        """The events timely associated to the Timeseries, timely ordered."""
        return tuple(sorted(self.__associated_events.values()))

    @property
    def tags(self) -> tuple[str]:
        return tuple(self.__tags)

    # ===================================
    # Built-ins

    def __len__(self):
        return sum([len(seg) for seg in self.__segments])

    def __iter__(self) -> Iterable:
        return self.__segments.__iter__()

    def __contains__(self, item):
        '''Checks if event occurs in Timeseries.'''
        if isinstance(item, str):
            return item in self.__associated_events
        elif isinstance(item, (datetime, DateTimeRange)):
            return any([item in segment for segment in self.__segments])

    def __getitem__(self, item):
        '''The built-in slicing and indexing ([x:y]) operations.'''
        if isinstance(item, int):
            return self.__segments[item].samples

        if isinstance(item, datetime):
            return self.__get_sample(item)

        if isinstance(item, str):
            return self.__get_sample(to_datetime(item))

        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Indexing with step is not allowed for Timeseries. Try resampling it first.")
            initial = to_datetime(item.start) if isinstance(item.start,
                                                            str) else self.initial_datetime if item.start is None else item.start
            final = to_datetime(item.stop) if isinstance(item.stop,
                                                         str) else self.final_datetime if item.stop is None else item.stop
            if isinstance(initial, datetime) and isinstance(final, datetime):
                return self.__new(segments=self.__get_samples(initial, final))
            else:
                raise IndexError("Index types not supported. Give a slice of datetimes (can be in string format).")

        if isinstance(item, tuple):
            res = list()
            for timepoint in item:
                if isinstance(timepoint, datetime):
                    res.append(self.__get_sample(timepoint))
                elif isinstance(timepoint, str):
                    res.append(self.__get_sample(to_datetime(timepoint)))
                else:
                    raise IndexError("Index types not supported. Give a tuple of datetimes (can be in string format).")
            return tuple(res)

        if isinstance(item,
                      DateTimeRange):  # This is not publicly documented. Only Biosignal sends DateTimeRanges, when it is dealing with Events.
            # First, trim the start and end limits of the interval.
            start, end = None, None
            for subdomain in self.domain:  # ordered subdomains
                if subdomain.is_intersection(item):
                    intersection = subdomain.intersection(item)
                    if start is None:
                        start = intersection.start_datetime
                    end = intersection.end_datetime
                elif start is not None:  # if there's no intersection with further subdomains and start was already found...
                    break  # ... then, the end was already reached
            if start is None and end is None:
                return None
            else:
                return self[start:end]

        raise IndexError(
            "Index types not supported. Give a datetime (can be in string format), a slice or a tuple of those.")

    def __add__(self, other):
        """The built-in + operation that adds sample-by-sample two Timeseries."""
        # Check errors
        if not isinstance(other, Timeseries):
            raise TypeError("Trying to add an object of type {}. Expected type: Timeseries.".format(type(other)))
        if other.sampling_frequency != self.__sampling_frequency:
            raise ArithmeticError("Both Timeseries must have the same sampling frequency ({} and {}).".format(
                self.__sampling_frequency, other.sampling_frequency))
        if other.units is not None and self.__units is not None and other.units != self.__units:
            raise ArithmeticError(
                "Both Timeseries must have the same units ({} and {}).".format(self.__units, other.units))
        if self.domain != other.domain:
            raise ArithmeticError("Timeseries to add must have the same domain.")

        # Perform addition
        new_segments = []
        for x, y in zip(self.__segments, other.segments):
            new_segments.append(x + y)

        return self.__new(segments=new_segments, units=self.units if self.__units is not None else other.units,
                          name=self.name + ' + ' + other.name if self.name != other.name else self.name)

    def __rshift__(self, other):
        """The built-in >> operation that concatenates two Timeseries."""
        if isinstance(other, Timeseries):
            if other.initial_datetime < self.final_datetime:
                raise ArithmeticError(
                    "The second Timeseries must start after the first one ends ({} + {}).".format(self.initial_datetime,
                                                                                                  other.final_datetime))
            if other.sampling_frequency != self.__sampling_frequency:
                raise ArithmeticError("Both Timeseries must have the same sampling frequency ({} and {}).".format(
                    self.__sampling_frequency, other.sampling_frequency))
            if other.units is not None and self.__units is not None and other.units != self.__units:
                raise ArithmeticError(
                    "Both Timeseries must have the same units ({} and {}).".format(self.__units, other.units))
            new_segments = self.__segments + other.segments  # concatenate lists
            return self.__new(segments=new_segments, units=self.units if self.__units is not None else other.units,
                              name=self.name + ' >> ' + other.name if self.name != other.name else self.name)

        raise TypeError("Trying to concatenate an object of type {}. Expected type: Timeseries.".format(type(other)))

    # ===================================
    # Binary Logic using Time and Conditions

    def __lt__(self, other):
        if isinstance(other, Timeseries):
            return self.final_datetime < other.initial_datetime
        else:
            return self._indices_to_timepoints([seg < other for seg in self.__segments])

    def __le__(self, other):
        if isinstance(other, Timeseries):
            return self.final_datetime <= other.initial_datetime
        else:
            return self._indices_to_timepoints(np.concatenate([seg <= other for seg in self.__segments]), by_segment=False)

    def __gt__(self, other):
        if isinstance(other, Timeseries):
            return self.initial_datetime > other.final_datetime
        else:
            return self._indices_to_timepoints([seg > other for seg in self.__segments])

    def __ge__(self, other):
        if isinstance(other, Timeseries):
            return self.initial_datetime >= other.final_datetime
        else:
            return self._indices_to_timepoints(np.concatenate([seg >= other for seg in self.__segments]), by_segment=False)

    def __eq__(self, other):
        if isinstance(other, Timeseries):
            return self.initial_datetime == other.initial_datetime and self.final_datetime == other.final_datetime
        else:
            return self._indices_to_timepoints(np.concatenate([seg == other for seg in self.__segments]), by_segment=False)

    def __ne__(self, other):
        if isinstance(other, Timeseries):
            return not self.__eq__(other)
        else:
            return self._indices_to_timepoints(np.concatenate([seg != other for seg in self.__segments]), by_segment=False)

    def _when(self, condition, window: timedelta):
        if window is not None:
            window_length = int(window.total_seconds() * self.__sampling_frequency)
            x = [seg._when(condition, window_length) for seg in self.__segments]
        else:
            x = [seg._when(condition) for seg in self.__segments]
        return self._indices_to_timepoints(x, by_segment=False)

    # ===================================
    # Methods

    def max(self):
        """Returns the maximum aplitude value of the Timeseries."""
        return max([seg.max() for seg in self.__segments])

    def min(self):
        """Returns the minimum aplitude value of the Timeseries."""
        return min([seg.min() for seg in self.__segments])

    def overlap(self, other) -> Tuple[DateTimeRange]:
        if isinstance(other, Timeseries):
            domain1:Tuple[DateTimeRange] = self.domain
            domain2:Tuple[DateTimeRange] = other.domain

        elif isinstance(other, tuple) and all(isinstance(x, DateTimeRange) for x in other):
            domain1: Tuple[DateTimeRange] = self.domain
            domain2: Tuple[DateTimeRange] = other

        else:
            raise TypeError("Overlap method must be used with another Timeseries or a union of intervals (Tuple[DateTimeRange]).")

        intersections = []  # Union of the intervals fom both Timeseries that intersect
        for interval1 in domain1:
            for interval2 in domain2:
                if interval1.is_intersection(interval2):
                    intersections.append(interval1.intersection(interval2))

        return tuple(intersections)

    def append(self, initial_datetime: datetime, samples: ndarray | list | tuple):
        """
        Appends a new sequence of samples in a separate Segment.
        :param initial_datetime: The date and time of the first sample in 'samples'.
        :param samples: The sequence of samples to add as a separate Segment.
        :return: None
        """
        assert len(self.__segments) > 0
        if self.__segments[-1].final_datetime > initial_datetime:  # Check for order and overlaps
            raise AssertionError("Cannot append more samples starting before the ones already existing.")
        """ # FROM THESIS #
            print("!! Cutting the begining of the new segment")
            segment = Timeseries.__Segment(array(samples) if not isinstance(samples, ndarray) else samples,
                                           initial_datetime, self.__sampling_frequency)
            segment = segment[ceil((self.__segments[-1].final_datetime - initial_datetime).total_seconds() * self.sampling_frequency): ]
            self.__segments.append(segment)
        else:
        """
        segment = Timeseries.__Segment(array(samples) if not isinstance(samples, ndarray) else samples,
                                       initial_datetime, self.__sampling_frequency)
        self.__segments.append(segment)

        # Check if equally segmented
        if self.__is_equally_segmented and len(samples) != len(self.__segments[0]):
            self.__is_equally_segmented = False

    def associate(self, events: Event | Collection[Event] | Dict[str, Event]):
        """
        Associates an Event with the Timeseries. Events have names that serve as keys. If keys are given,
        i.e. if 'events' is a dict, then the Event names are override.
        :param events: One or multiple Event objects.
        :return: None
        """

        def __add_event(event: Event):
            try:
                if event.has_onset and not event.has_offset:
                    self.__check_boundaries(event.onset)  # raises IndexError
                if event.has_offset and not event.has_onset:
                    self.__check_boundaries(event.offset)  # raises IndexError
                if event.has_onset and event.has_offset:
                    self.__check_boundaries(event.domain)
            except IndexError:
                raise ValueError(
                    f"Event '{event.name}' is outside of Timeseries domain, {' U '.join([f'[{subdomain.start_datetime}, {subdomain.end_datetime}[' for subdomain in self.domain])}.")
            if event.name in self.__associated_events and event != self.__associated_events[event.name]:
                raise NameError(
                    f"There is already another Event named with '{events.name}'. Cannot have two Events with the same name.")
            else:
                self.__associated_events[event.name] = event

        if isinstance(events, dict):
            for event_key in events:
                event = events[event_key]
                __add_event(Event(event_key, event._Event__onset, event._Event__offset))  # rename with given key
        elif isinstance(events, (tuple, set, list)):
            for event in events:
                __add_event(event)
        else:  # single Event
            __add_event(events)

    def disassociate(self, event_name: str):
        """
        Dissociate the event named after the given name.
        :param event_name: The name of the event to dissociate.
        :return: None
        :raise NameError: If there is no associated Event with the given name.
        """
        if event_name in self.__associated_events:
            del self.__associated_events[event_name]
        else:
            raise NameError(f"There's no Event '{event_name}' associated to this Timeseries.")

    def delete_events(self):
        self.__associated_events = {}

    def timeshift(self, delta: timedelta):
        for segment in self:
            segment.timeshift(delta)

    def tag(self, tags: str | tuple[str]):
        """
        Mark the Timeseries with a tag. Useful to mark machine learning targets.
        :param tags: The label or labels to tag the Timeseries.
        :return: None
        """
        if isinstance(tags, str):
            self.__tags.add(tags)
        elif isinstance(tags, tuple) and all(isinstance(x, str) for x in tags):
            for x in tags:
                self.__tags.add(x)
        else:
            raise TypeError("Give one or multiple string labels to tag the Timeseries.")

    def sliding_window(self, window_length: timedelta, overlap_length: timedelta = timedelta(0)):
        for segment in self:
            segment.sliding_window(int(window_length.total_seconds() * self.__sampling_frequency),
                                   int(overlap_length.total_seconds() * self.__sampling_frequency))

    def convert(self, to_unit: Unit, transfer_function: Callable[[ndarray], ndarray] = None):
        """
        Converts the Timeseries from the current units to 'to_unit'.
        If the current units are None (raw signal), then the transfer function must be given.
        :param to_unit: The unit to convert to.
        :param transfer_function: The transfer function to apply, if raw signal.
        :return: None (applied in-place)
        """
        if self.__units != to_unit:
            if self.__units is None:
                if transfer_function is not None:
                    try:
                        self._apply_operation(transfer_function)
                        self.__units = to_unit
                    except Exception as e:
                        raise RuntimeError(f"Conversion failed: {e}")
                else:
                    raise ValueError("Cannot convert from raw signal without a transfer function. Give one as argument.")
            else:
                raise NotImplementedError("Conversion between units is not implemented yet.")
        else:
            print("This Timeseries is already on the given unit.")

    # ===================================
    # INTERNAL USAGE - Convert indexes <-> timepoints && Get Samples

    def __get_sample(self, datetime: datetime) -> float:
        self.__check_boundaries(datetime)
        for segment in self.__segments:  # finding the first Segment
            if datetime in segment:
                return segment[int((datetime - segment.initial_datetime).total_seconds() * self.sampling_frequency)]
        raise IndexError("Datetime given is in not defined in this Timeseries.")

    def __get_samples(self, initial_datetime: datetime, final_datetime: datetime) -> List[__Segment]:
        '''Returns the samples between the given initial and end datetimes.'''
        self.__check_boundaries(initial_datetime)
        self.__check_boundaries(final_datetime)
        res_segments = []
        for i in range(len(self.__segments)):  # finding the first Segment
            segment = self.__segments[i]
            if segment.initial_datetime <= initial_datetime <= segment.final_datetime:
                if final_datetime <= segment.final_datetime:
                    trimmed_segment = segment[int((
                                                          initial_datetime - segment.initial_datetime).total_seconds() * self.sampling_frequency):int(
                        (final_datetime - segment.initial_datetime).total_seconds() * self.sampling_frequency)]
                    res_segments.append(trimmed_segment)
                    return res_segments
                else:
                    if not initial_datetime == segment.final_datetime: # skip what would be an empty set
                        trimmed_segment = segment[int((initial_datetime - segment.initial_datetime).total_seconds() * self.sampling_frequency):]
                        res_segments.append(trimmed_segment)
                    for j in range(i + 1,
                                   len(self.__segments)):  # adding the remaining samples, until the last Segment is found
                        segment = self.__segments[j]
                        if final_datetime <= segment.final_datetime:
                            trimmed_segment = segment[:int(
                                (final_datetime - segment.initial_datetime).total_seconds() * self.sampling_frequency)]
                            res_segments.append(trimmed_segment)
                            return res_segments
                        else:
                            trimmed_segment = segment[:]
                            res_segments.append(trimmed_segment)

    def __check_boundaries(self, datetime_or_range: datetime | DateTimeRange) -> None:
        intersects = False
        if isinstance(datetime_or_range, datetime):
            for subdomain in self.domain:
                if datetime_or_range in subdomain:
                    intersects = True
                    break
            if not intersects:
                raise IndexError(
                    f"Datetime given is outside of Timeseries domain, {' U '.join([f'[{subdomain.start_datetime}, {subdomain.end_datetime}[' for subdomain in self.domain])}.")

        elif isinstance(datetime_or_range, DateTimeRange):
            for subdomain in self.domain:
                if subdomain.is_intersection(datetime_or_range) and datetime_or_range.start_datetime != subdomain.end_datetime:
                    intersects = True
                    break
            if not intersects:
                raise IndexError(
                    f"Interval given is outside of Timeseries domain, {' U '.join([f'[{subdomain.start_datetime}, {subdomain.end_datetime}[' for subdomain in self.domain])}.")

    def _indices_to_timepoints(self, indices: Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]], by_segment=False) -> Sequence[datetime] | Sequence[Sequence[datetime]] | Sequence[DateTimeRange] | Sequence[Sequence[DateTimeRange]]:
        all_timepoints = []
        for index, segment in zip(indices, self.__segments):
            timepoints = divide(index, self.__sampling_frequency)  # Transform to seconds
            if isinstance(timepoints, ndarray) and len(timepoints.shape) == 2 and timepoints.shape[1] == 2:  # Intervals
                x = [DateTimeRange(segment.initial_datetime + timedelta(seconds=tp[0]), segment.initial_datetime + timedelta(seconds=tp[1])) for tp in timepoints]
            else:  # Timepoints
                x = [segment.initial_datetime + timedelta(seconds=tp) for tp in timepoints]
            if by_segment:
                all_timepoints.append(x)  # Append as list
            else:
                all_timepoints += x  # Join them all
        return tuple(all_timepoints)

    def to_array(self) -> np.ndarray:
        """
        Converts a Timeseries into a numpy array. If the Timeseries is composed of multiple Segments, the interruptions are filled with NaNs.
        :return: A 1D numpy array with its samples.
        """
        res = np.array(self.__segments[0].samples)
        for i in range(1, len(self.__segments)):
            segment = self.__segments[i]
            # get the time between the end of the current segment and the start of the next one
            time_between_segments = self.__segments[i].initial_datetime - self.__segments[i - 1].final_datetime
            # number of NaNs to fill the gap
            n_nans = round(self.__sampling_frequency * time_between_segments.total_seconds())
            # fill the gap with NaNs
            res = np.concatenate((res, [np.nan] * n_nans))
            # add the samples of the current segment
            res = np.concatenate((res, segment.samples))
        return res

    # ===================================
    # INTERNAL USAGE - Plots

    def _plot_spectrum(self):
        colors = ('blue', 'green', 'red')
        n_columns = len(self.__segments)
        for i in range(n_columns):
            segment = self.__segments[i]
            x, y = power_spectrum(signal=segment.samples)
            plt.plot(x, y, alpha=0.6, linewidth=0.5,
                     label='From {0} to {1}'.format(segment.initial_datetime, segment.final_datetime))

    def _plot(self, label:str = None):
        xticks, xticks_labels = [], []  # to store the initial and final ticks of each Segment
        SPACE = int(self.__sampling_frequency) * 2  # the empty space between each Segment

        for i in range(len(self.__segments)):
            segment = self.__segments[i]
            x, y = range(len(segment)), segment.samples
            if i > 0:  # except for the first Segment
                x = array(x) + (xticks[-1] + SPACE)  # shift right in time
                plt.gca().axvspan(x[0] - SPACE, x[0], alpha=0.05, color='black')  # add empty space in between Segments
            plt.gca().plot(x, y, linewidth=0.5, alpha=0.7, label=label)

            xticks += [x[0], x[-1]]  # add positions of the first and last samples of this Segment

            # add datetimes of the first and last samples of this Segment
            if segment.duration > timedelta(days=1):  # if greater that a day, include dates
                time_format = "%d-%m-%Y %H:%M:%S"
            else:  # otherwise, just the time
                time_format = "%H:%M:%S"
            xticks_labels += [segment.initial_datetime.strftime(time_format),
                              segment.final_datetime.strftime(time_format)]

        plt.gca().set_xticks(xticks, xticks_labels)
        plt.tick_params(axis='x', direction='in')

        if self.units is not None:  # override ylabel
            plt.gca().set_ylabel("Amplitude ({})".format(str(self.units)))

    # ===================================
    # INTERNAL USAGE - Accept methods

    # General-purpose

    def _apply_operation(self, operation, **kwargs):
        """
        Applies operation in-place to every Segment's samples.
        """
        for segment in self.__segments:
            segment._apply_operation(operation, **kwargs)

    def _apply_operation_and_return(self, operation, iterate_along_segments_key: [str] = None, **kwargs) -> list:
        """
        Applies operation out-of-place to every Segment's samples and returns the ordered output of each in a list.

        Procedure 'operation' must receive a ndarray of samples as first argument.
        It can receive other arguments, which should be passed in '**kwargs'.
        Procedure output can return whatever, which shall be returned.
        """
        res = []

        if isinstance(iterate_along_segments_key, str):
            items = kwargs[iterate_along_segments_key]
            for segment, item in zip(self, items):
                kwargs[iterate_along_segments_key] = item
                new_segment = segment._apply_operation_and_return(operation, **kwargs)
                res.append(new_segment)
        elif isinstance(iterate_along_segments_key, list) and all(isinstance(x, str) for x in iterate_along_segments_key):
            items = [kwargs[it] for it in iterate_along_segments_key]
            for segment, item in zip(self, *items):
                for it in iterate_along_segments_key:
                    kwargs[it] = item
                new_segment = segment._apply_operation_and_return(operation, *items, **kwargs)
                res.append(new_segment)

        else:
            for segment in self.__segments:
                res.append(segment._apply_operation_and_return(operation, **kwargs))
        return res

    # Purpose-specific

    def _accept_filtering(self, filter_design):
        filter_design._setup(self.__sampling_frequency)
        for segment in self.__segments:
            segment._accept_filtering(filter_design)

    def _undo_filters(self):
        for segment in self.__segments:
            segment._restore_raw()

    def _resample(self, frequency: float):
        frequency = frequency if isinstance(frequency, Frequency) else Frequency(frequency)
        for segment in self.__segments:
            segment._resample(frequency)
        self.__sampling_frequency = frequency  # The sf of all Segments points to this property in Timeseries. So, this is only changed here.

    # ===================================
    # INTERNAL USAGE - Make similar copies or itself

    def __copy__(self):
        """ Creates an exact copy of the Timeseries' contents and returns the new object. """
        new = type(self)([seg.__copy__() for seg in self.__segments], self.initial_datetime,
                         self.__sampling_frequency.__copy__(), self.__units,
                         str(self.name))  # Uses shortcut in __init__
        new._Timeseries__is_equally_segmented = self.__is_equally_segmented
        new.associate(self.events)
        new.tag(self.tags)
        return new

    def _new_samples(self, samples_by_segment: List[ndarray] = None):
        """
        Protected Access: For use of this module, since who uses is not aware of Segment.

        Creates a similar copy of the Timeseries' contents and returns the new object, with the samples of each segment changed.

        :return: A new Timeseries with the samples changed. All other fields shall remain the same.
        :rtype: Timeseries | OverlappingTimeseries
        """

        assert len(samples_by_segment) == len(self.__segments)

        segments = []
        for segment, samples in zip(self, samples_by_segment):
            segments.append(segment._new(samples=samples, raw_samples=samples, is_filtered=False))

        return self.__new(segments=segments)

    def __new(self, segments: List[__Segment] = None, sampling_frequency: float = None, units: Unit = None,
              name: str = None, equally_segmented: bool = None, overlapping_segments: bool = None,
              events: Collection[Event] = None):
        """
        Private Access: For in-class usage, since who uses is aware of Segment.

        Creates a similar copy of the Timeseries' contents and returns the new object.
        The value of any field can be changed, when explicitly given a new value for it. All others will be copied.

        :param segments: A list of new Segments to substitute. Optional.
        :param sampling_frequency: A different sampling frequency. Optional.
        :param units: Different units. Optional.
        :param name: A different name. Optional.
        :param equally_segmented: Alter the is_equally_segmented state. Optional.
        :param overlapping_segments: Opt to instantiate a Timeseries or an OverlappingTimeseries. Optional.
        :param events: A collections of different Events. Optional.

        :return: A new Timeseries with the given fields changed. All other contents shall remain the same.
        :rtype: Timeseries | biosignals.timeseries.OverlappingTimeseries.OverlappingTimeseries
        """

        initial_datetime = self.initial_datetime if segments is None else segments[0].initial_datetime
        segments = self.__segments if segments is None else segments  # Uses shortcut in __init__
        sampling_frequency = self.__sampling_frequency if sampling_frequency is None else sampling_frequency if isinstance(
            sampling_frequency,
            Frequency) else Frequency(sampling_frequency)
        units = self.__units if units is None else units
        name = str(self.__name) if name is None else name
        equally_segmented = self.__is_equally_segmented if equally_segmented is None else equally_segmented
        events = self.__associated_events if events is None else events

        if overlapping_segments is None:
            new = type(self)(segments, initial_datetime, sampling_frequency, units, name)
        elif overlapping_segments is True:
            new = OverlappingTimeseries(segments, initial_datetime, sampling_frequency, units, name)
        else:
            new = Timeseries(segments, initial_datetime, sampling_frequency, units, name)

        new._Timeseries__is_equally_segmented = equally_segmented

        events = events.values() if isinstance(events, dict) else events
        for event in events:
            try:
                new.associate(event)
            except ValueError:
                pass  # it's outside the new boundaries

        new.tag(self.tags)

        return new

    def _new(self, segments_by_time: Dict[datetime, ndarray | list | tuple] = None,
             sampling_frequency: float = None,
             units: Unit = None, name: str = None, equally_segmented: bool = None,
             overlapping_segments: bool = None,
             events: Collection[Event] = None, rawsegments_by_time: Dict[datetime, ndarray | list | tuple] = None):
        """
        Protected Access: For use of this module, since who uses is not aware of Segment.

        Creates a similar copy of the Timeseries' contents and returns the new object.
        The value of any field can be changed, when explicitly given a new value for it. All others will be copied.

        :param segments_by_time: The sequence of samples to store as separate Segments, keyed by their initial date and time. Optional.
        :param sampling_frequency: A different sampling frequency. Optional.
        :param units: Different units. Optional.
        :param name: A different name. Optional.
        :param equally_segmented: Alter the is_equally_segmented state. Optional.
        :param overlapping_segments: Opt to instantiate a Timeseries or an OverlappingTimeseries. Optional.
        :param events: A collections of different Events. Optional.
        :param rawsegments_by_time: The sequence of raw samples to associate to the Segments, keyed by their initial date and time. Optional.

        Note: If both 'segments_by_time' and 'rawsegments_by_time' are given, their key sets must be identical.

        :return: A new Timeseries with the given fields changed. All other contents shall remain the same.
        :rtype: Timeseries | OverlappingTimeseries
        """

        # Sampling frequency
        sampling_frequency = self.__sampling_frequency if sampling_frequency is None else sampling_frequency

        if segments_by_time is not None:
            # Transform dict into Segments
            segments = []
            for initial_datetime, samples in segments_by_time.items():
                seg = Timeseries.__Segment(samples, initial_datetime, sampling_frequency,
                                           is_filtered=rawsegments_by_time is not None)
                if rawsegments_by_time is not None:
                    seg._Segment__raw_samples = rawsegments_by_time[initial_datetime]
                segments.append(seg)
        else:
            # Send nothing
            segments = None

        return self.__new(segments=segments, sampling_frequency=sampling_frequency, units=units, name=name,
                          equally_segmented=equally_segmented, overlapping_segments=overlapping_segments,
                          events=events)

    def _apply_operation_and_new(self, operation, sampling_frequency: float = None, units: Unit = None,
                                 name: str = None, equally_segmented: bool = None,
                                 overlapping_segments: bool = None,
                                 events: Collection[Event] = None,
                                 iterate_over_each_segment_key: str = None, **kwargs):
        """
        For outside usage. Who uses is not aware of Segment.
        Creates new Segments from the existing ones, using Segment._new().

        If there is one item in '**kwargs' that has input to be iteratively passed to 'method',
        indicate its key in 'iterate_over_each_segment_key'.
        """

        # Sampling frequency
        sampling_frequency = self.__sampling_frequency if sampling_frequency is None else sampling_frequency

        # Apply operation
        all_new_segments = []
        if iterate_over_each_segment_key is not None:
            items = kwargs[iterate_over_each_segment_key]
            for segment, item in zip(self, items):
                kwargs[iterate_over_each_segment_key] = item
                new_segment = segment._apply_operation_and_new(operation, sampling_frequency=sampling_frequency, **kwargs)
                all_new_segments.append(new_segment)
        else:
            for segment in self:
                new_segment = segment._apply_operation_and_new(operation, sampling_frequency=sampling_frequency, **kwargs)
                all_new_segments.append(new_segment)

        # Get new Timeseries
        return self.__new(all_new_segments, sampling_frequency=sampling_frequency, units=units, name=name,
                          equally_segmented=equally_segmented, overlapping_segments=overlapping_segments,
                          events=events)

    def _equally_segment_and_new(self, window_length: timedelta, overlap_length: timedelta = timedelta(seconds=0)):
        """
        For internal usage.

        Segments the Timseries in equal junks.

        :param window_length:
        :param overlap_length:
        :return:
        """

        n_window_length = int(window_length.total_seconds() * self.sampling_frequency)
        n_overlap_length = int(overlap_length.total_seconds() * self.sampling_frequency)

        res_trimmed_segments = []
        for segment in self.__segments:
            partitioned_segments = segment._partition(n_window_length, n_overlap_length)
            for s in partitioned_segments:
                assert len(s) == n_window_length
            res_trimmed_segments += partitioned_segments

        return self.__new(segments=res_trimmed_segments, equally_segmented=True, overlapping_segments=n_overlap_length != 0)

    def _segment_and_new(self, method: Callable,
                         samples_rkey: str, indexes_rkey: str,
                         iterate_over_each_segment_key: str = None,
                         initial_datetimes_shift: timedelta = None,
                         equally_segmented: bool = True,
                         overlapping_segments: bool = False,
                         **kwargs):
        """
        For internal usage.

        Segments the Timeseries into smaller portions, using any 'method' that follows the following signature.

        Procedure 'method' should receive as first argument the array of samples to partition. It can receive other
        arguments after that, which should be passed in' **kwargs'.
        If there is one item in '**kwargs' that has input to be iteratively passed to 'method',
        indicate its key in 'iterate_over_each_segment_key'.

        Procedure 'method' should return a dictionary of objects, and at least two of them must be:
        - The arrays of samples destined to be the smaller Segments. Indicate their key in the dict using 'samples_rkey'.
        - The start indexes of each corresponding smaller Segment. Indicate their key in the dict using 'indexes_rkey'.

        If what 'indexes_rkey' contains are shifted initial indexes, indicate that offset in 'initial_datetimes_shift'.

        If procedures 'method' will return equally segmented partitions, pass equally_segmented=True.
        If procedures 'method' will return overlapping partitions, pass overlapping_segments=True.
        """

        def __patition(segment: Timeseries.__Segment, indices=None) -> list:
            """
            Indices should be an array of timepoint where to cut, if 'method' does not find them.
            """
            load = kwargs
            if indices is not None:
                load[iterate_over_each_segment_key] = indices
            res = method(segment.samples, **load)
            _, raw_values = method(segment.raw_samples, **load)
            assert len(res) >= 2
            assert samples_rkey in res.keys()
            assert indexes_rkey in res.keys()
            indexes, values = res[indexes_rkey], res[samples_rkey]
            assert len(indexes) == len(values)
            initial_datetimes = [timedelta(seconds=index / self.__sampling_frequency) + segment.initial_datetime for
                                 index in indexes]

            if initial_datetimes_shift is not None:
                initial_datetimes = [idt + initial_datetimes_shift for idt in initial_datetimes]

            trimmed_segments = [segment._new(samples=values[i], initial_datetime=initial_datetimes[i],
                                             raw_samples=raw_values[i]) for i in range(len(values))]

            return trimmed_segments

        res_trimmed_segments = []
        if iterate_over_each_segment_key is not None:
            for segment, indices in zip(self.__segments, kwargs[iterate_over_each_segment_key]):
                res_trimmed_segments += __patition(segment, indices)
        else:
            for segment in self.__segments:
                res_trimmed_segments += __patition(segment)

        return self.__new(segments=res_trimmed_segments, equally_segmented=equally_segmented,
                          overlapping_segments=overlapping_segments)

    # ===================================
    # INTERNAL USAGE - Reshape

    def _concatenate_segments(self):
        if len(self.__segments) > 1:
            self.__segments = [Timeseries.__Segment(concatenate(self.samples), self.initial_datetime, self.__sampling_frequency, False), ]
            self.__is_equally_segmented = True
        else:
            pass  # no need

    def _partition(self, time_intervals:tuple[DateTimeRange]):
        assert len(self.__segments) == 1
        samples = self.__segments[0]
        partitions = []
        i = 0
        for x in time_intervals:
            n_samples_required = ceil(x.timedelta.total_seconds() * self.__sampling_frequency)
            if n_samples_required > len(samples):
                samples = tile(samples, ceil(n_samples_required/len(samples)))  # repeat
                samples = samples[:n_samples_required]  # cut where it is enough
                partitions.append(Timeseries.__Segment(samples, x.start_datetime, self.__sampling_frequency))
                i = 0
            else:
                f = i + n_samples_required
                partitions.append(Timeseries.__Segment(samples[i: f], x.start_datetime, self.__sampling_frequency))
                i += f

        self.__segments = partitions

    def _delete_segments(self, selection_function: Callable[[ndarray], bool]):
        self.__segments = list(filter(lambda seg: selection_function(seg.samples), self.__segments))

    # ===================================
    # SERIALIZATION

    def _memory_map(self, path):
        # Create a memory map for the array
        for seg in self:
            seg._memory_map(path)

    def __getstate__(self):
        """
        Version 1:
        1: __name (str)
        2: __sampling_frequency (Frequency)
        3: __units (Unit)
        4: __is_equally_segmented (bool)
        5: segments_state (list)

        Version 2:
        1: __name (str)
        2: __sampling_frequency (Frequency)
        3: __units (Unit)
        4: __is_equally_segmented (bool)
        5: __tags (set)
        6: segments_state (list)
        """
        segments_state = [segment.__getstate__() for segment in self.__segments]
        return (self.__SERIALVERSION, self.__name, self.__sampling_frequency, self.__units, self.__is_equally_segmented, self.__tags,
                segments_state)

    def __setstate__(self, state):
        if state[0] == 1:
            self.__name, self.__sampling_frequency, self.__units = state[1], state[2], state[3]
            self.__is_equally_segmented = state[4]
            self.__segments = []
            for segment_state in state[5]:
                segment_state = list(segment_state)
                segment_state.append(self.__sampling_frequency)
                segment = object.__new__(Timeseries.__Segment)
                segment.__setstate__(segment_state)
                self.__segments.append(segment)
            self.__associated_events = {}  # empty; to be populated by Biosignal
            self.__tags = set()  # In version 1, tags were not a possibility, so none existed.
        elif state[0] == 2:
            self.__name, self.__sampling_frequency, self.__units = state[1], state[2], state[3]
            self.__is_equally_segmented = state[4]
            self.__segments = []
            for segment_state in state[6]:
                segment_state = list(segment_state)
                segment_state.append(self.__sampling_frequency)
                segment = object.__new__(Timeseries.__Segment)
                segment.__setstate__(segment_state)
                self.__segments.append(segment)
            self.__associated_events = {}  # empty; to be populated by Biosignal
            self.__tags = state[5]
        else:
            raise IOError(f'Version of {self.__class__.__name__} object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1 and 2.')


class OverlappingTimeseries(Timeseries):
    """
    An OverlappingTimeseries is a Timeseries that violates the rule that to each time point of its domain it must
    correspond one and only one sample. This special kind of Timeseries allows overlapping Segments, although it looses
    all its interpretational meaning in the context of being successive data points in time. This kind is useful to
    extract features from modalities or to train machine learning models.

    It inherits all properties of Timeseries and most of its behaviour.
    In order to have overlapping Segments, indexing an exact timepoint is no longer possible; Although it is legal to
    index slices. # FIXME
    """

    def __init__(self, samples: ndarray | list | tuple, initial_datetime: datetime, sampling_frequency: float,
                 units: Unit = None, name: str = None):
        super().__init__(samples, initial_datetime, sampling_frequency, units, name)

    def append(self, initial_datetime: datetime, samples: ndarray | list | tuple):
        assert len(self.__segments) > 0
        segment = Timeseries._Timeseries__Segment(array(samples) if not isinstance(samples, ndarray) else samples,
                                       initial_datetime, self._Timeseries__sampling_frequency)
        self._Timeseries__segments.append(segment)

    def _concatenate_segments(self):
        raise NotImplementedError("")

    def __getitem__(self, item):
        """
        The built-in slicing ([x:y]) operation.
        A segment is assumed to belong to the interval [x,y] if and only if its final datetime belongs to the interval.
        """

        if isinstance(item, int):
            return self._Timeseries__segments[item].samples

        if isinstance(item, datetime):
            raise IndexError('OverlappingTimeseries cannot return a unique value correpondent of a datetime.')

        if isinstance(item, tuple) and all(isinstance(dt, datetime) for dt in item):
            raise IndexError('OverlappingTimeseries cannot return a unique value correpondent of datetimes.')

        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Indexing with step is not allowed for OverlappingTimeseries. Try resampling it first.")
            initial = to_datetime(item.start) if isinstance(item.start, str) else self.initial_datetime if item.start is None else item.start
            final = to_datetime(item.stop) if isinstance(item.stop, str) else self.final_datetime if item.stop is None else item.stop
            #self.__check_boundaries(initial)
            #self.__check_boundaries(final)
            if isinstance(initial, datetime) and isinstance(final, datetime):
                return self._Timeseries__new(segments=self.__get_samples(initial, final))
            else:
                raise IndexError("Index types not supported. Give a slice of datetimes (can be in string format).")

        if isinstance(item, DateTimeRange):  # Not publicly documented. Only Biosignal sends DateTimeRanges, when it is dealing with Events.
            #self.__check_boundaries(item)
            return self._Timeseries__new(segments=self.__get_samples(item.start_datetime, item.end_datetime))

        raise IndexError(
            "Index types not supported. Give a datetime (can be in string format), a slice of those.")

    def __get_samples(self, initial_datetime: datetime, final_datetime: datetime):
        '''Returns the segemnts between the given initial and end datetimes, acording to the final datetime of each.'''
        res_segments = []
        for i in range(len(self._Timeseries__segments)):
            segment = self._Timeseries__segments[i]
            if initial_datetime <= segment.final_datetime < final_datetime:  # if the last timepoint of the Segment is inside the interval
                res_segments.append(segment)  # keep the whole Segment
            elif initial_datetime <= segment.final_datetime == final_datetime == self.final_datetime:  # if the last timepoint of the Segment is the end of the Timeseries
                res_segments.append(segment)  # keep that Segment as well

        return res_segments

    def __check_boundaries(self, datetime_or_range: datetime | DateTimeRange) -> None:
        intersects = False
        if isinstance(datetime_or_range, datetime):
            for subdomain in self.domain:
                if datetime_or_range in subdomain:
                    intersects = True
                    break
            if not intersects:
                raise IndexError(
                    f"Datetime given is outside of Timeseries domain, {' U '.join([f'[{subdomain.start_datetime}, {subdomain.end_datetime}[' for subdomain in self.domain])}.")

        elif isinstance(datetime_or_range, DateTimeRange):
            for subdomain in self.domain:
                if subdomain.is_intersection(datetime_or_range) and datetime_or_range.start_datetime != subdomain.end_datetime:
                    intersects = True
                    break
            if not intersects:
                raise IndexError(
                    f"Interval given is outside of Timeseries domain, {' U '.join([f'[{subdomain.start_datetime}, {subdomain.end_datetime}[' for subdomain in self.domain])}.")

    @property
    def domain(self) -> Tuple[DateTimeRange]:
        """The intervals of date and time in which the Timeseries is defined, i.e., samples were acquired."""
        domain = [DateTimeRange(self._Timeseries__segments[0].initial_datetime, self._Timeseries__segments[0].final_datetime)]
        for i in range(1, len(self._Timeseries__segments)):
            if self._Timeseries__segments[i].overlaps(self._Timeseries__segments[i-1]):
                domain[-1].set_end_datetime(self._Timeseries__segments[i].final_datetime)
            else:
                domain.append(DateTimeRange(self._Timeseries__segments[i].initial_datetime, self._Timeseries__segments[i].final_datetime))

        return tuple(domain)

    @property
    def domain_timeline(self) -> Timeline:  # TODO: mmerge with domain
        return Timeline(Timeline.Group(self.domain), name=self.name + ' Domain')

    @property
    def subdomains(self) -> Tuple[DateTimeRange]:
        return tuple([DateTimeRange(segment.initial_datetime, segment.final_datetime) for segment in self])

    def _block_subdomain(self, i) -> DateTimeRange:
        return DateTimeRange(self._Timeseries__segments[i].initial_datetime, self._Timeseries__segments[i].final_datetime)

    @property
    def duration(self) -> timedelta:
        """Returns real time passed from start to end, without overlaps."""
        domain = self.domain
        res = domain[0].timedelta
        for i in range(1, len(domain)):
            res += domain[i].timedelta
        return res
