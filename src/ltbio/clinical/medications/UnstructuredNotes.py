# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Medication
# Description: Abstract class Medication, to describe any medications taken.

# Contributors: João Saraiva
# Created: 23/04/2022
# Last Updated: 29/04/2022

# ===================================

from abc import ABC, abstractmethod

from ltbio.biosignals.timeseries.Unit import Unit
from ltbio.clinical.medications import Medication


class UnstructuredNotes(Medication):

    __SERIALVERSION: int = 1

    def __init__(self, text: str):
        self.notes = text
        super().__init__()

    @property
    def name(self):
        return "Unstructured Notes"

    def __str__(self):
        return f"{self.name}: {self.notes}"

    def __repr__(self):
        return f"{self.name}: {self.notes}"
