# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: biosignals
# Module: MultimodalBiosignal
# Description: Class MultimodalBiosignal that can hold multiple modalities of Biosignals.

# Contributors: João Saraiva, Mariana Abreu
# Created: 08/07/2022

# ===================================

from typing import Set

from ltbio.biosignals.modalities.Biosignal import Biosignal
from ltbio.biosignals.sources.BiosignalSource import BiosignalSource
from ltbio.biosignals.timeseries import Timeseries
from ltbio.clinical import Patient
from ltbio.clinical.BodyLocation import BodyLocation


class MultimodalBiosignal(Biosignal):

    def __init__(self, timeseries: dict[str | BodyLocation, Timeseries], source=None, patient=None, acquisition_location=None, name=None):
        super(MultimodalBiosignal, self).__init__(timeseries, source, patient, acquisition_location, name)
        self.__biosignals = None

    @classmethod
    def from_biosignals(cls, **biosignals: Biosignal):
        timeseries = {}
        source = []
        patient = []
        location = []
        name = "Union of"
        events = {}

        for label, biosignal in biosignals.items():

            for channel_label, ts in biosignal._to_dict().items():
                timeseries[label + ':' + channel_label] = ts  # Join Timeseries in a single dictionary

            source.append(biosignal.source)
            patient.append(biosignal._Biosignal__patient)
            location.append(biosignal.acquisition_location)

            name += f" '{biosignal.name}'," if biosignal.name != "No Name" else f" '{label}',"

            # Check if there are no incompatible events
            if any(e != events[e.name] for e in biosignal.events if e.name in events.keys()):
               raise ValueError(f"There is an Event in Biosignal '{biosignal.name}' with the same name, but with different onset and/or offset, of one Event contained in the other Biosignals.")

            # Add new events
            remaining_events = set(biosignal.events) - set(events.values())
            for e in remaining_events:
                events[e.name] = e

        # Check if all sources, patients and locations are equal
        if len(set(source)) == 1:
            source = source[0]
        else:
            source = None
        if len(set(patient)) == 1:
            patient = patient[0]
        else:
            patient = None
        if len(set(location)) == 1:
            location = location[0]
        else:
            location = None

        x = MultimodalBiosignal(timeseries, source, patient, location, name[:-1])
        x.associate(events)
        x._MultimodalBiosignal__biosignals = biosignals
        return x

        # if (len(self.type)) == 1:
        #    raise TypeError("Cannot create Multimodal Biosignal of just 1 modality.")

    @property
    def type(self):
        return {biosignal.type for biosignal in self.__biosignals.values()}

    @property
    def source(self) -> BiosignalSource | Set[BiosignalSource]:
        if super().source is not None:
            return super().source
        else:
            return {biosignal.source for biosignal in self.__biosignals.values()}

    @property
    def acquisition_location(self) -> Set[BodyLocation]:
        if self.__acquision_location is not None:
            return self.__acquision_location
        else:
            return {biosignal.acquisition_location for biosignal in self.__biosignals.values()}

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == 2:
                biosignal = self.__biosignals[item[0]]
                return biosignal[item[1]]

        elif isinstance(item, str) and item in self.__biosignals.keys():
            return self.__biosignals[item]

        raise IndexError("Indexing a Multimodal Biosignal should have two arguments, like 'multisignal['ecg'][V5],"
                         "where 'ecg' is the Biosignal to address and 'V5' is the channel to get.")

    def __contains__(self, item):
        if isinstance(item, Biosignal) and item in self.__biosignals.values():
            return True
        super().__contains__(item)

    def __str__(self):
        '''Returns a textual description of the MultimodalBiosignal.'''
        res = f"MultimodalBiosignal containing {len(self.__biosignals)}:\n"
        for i, biosignal in enumerate(self.__biosignals):
            res += "({})\n{}".format(i, str(biosignal))
        return res

    def plot_summary(self, show: bool = True, save_to: str = None):
        raise TypeError("Functionality not available for Multimodal Biosignals.")
