# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Alzheimer's Disease (AD)

# Contributors: João Saraiva
# Created: 23/04/2022
# Last update: 09/07/2022

# ===================================

from datetime import datetime, timedelta
from enum import Enum, unique
from typing import Sequence

from .. import BodyLocation, Semiology
from .MedicalCondition import MedicalCondition
from ...biosignals.timeseries.Event import Event

import simple_icd_10 as icd


class ICDCode(MedicalCondition):
    """
    Alzheimer's Disease (AD) is a neurodegenerative condition.
    """

    def __init__(self, code:str, years_since_diagnosis: float = None,):
        if not isinstance(code, str):
            code = str(code)
        if icd.is_valid_item(code):
            self.__code = code
        else:
        #    raise ValueError(f"Invalid ICD-10 code: {code}")  # FIXME: Uncomment this block
            print(f"ICD code {code} not found.")
        super(ICDCode, self).__init__(years_since_diagnosis)

    @property
    def code(self):
        return self.__code

    def __str__(self):
        return icd.get_description(self.__code)
