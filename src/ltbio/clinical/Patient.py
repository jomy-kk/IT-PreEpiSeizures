# -*- encoding: utf-8 -*-
from datetime import timedelta
# ===================================

# IT - LongTermBiosignals

# Package: clinical
# Module: Patient
# Description: Enumeration Sex and class Patient, to describe patients' clinical metadata.

# Contributors: João Saraiva
# Created: 22/04/2022
# Last update: 09/07/2022

# ===================================

from enum import unique, Enum
from typing import Tuple

from ltbio.clinical.conditions.MedicalCondition import MedicalCondition
from ltbio.clinical.medications.Medication import Medication
from ltbio.clinical.procedures.SurgicalProcedure import SurgicalProcedure


@unique
class Sex(str, Enum):
    """
    Biological sex of human beings.
    """
    M = "Male"
    F = "Female"
    _ = "n.d."


class Patient():

    __SERIALVERSION: int = 1

    def __init__(self, code, name:str=None, age:int=None, sex:Sex=Sex._, conditions:Tuple[MedicalCondition]=(), medications:Tuple[Medication]=(), procedures:Tuple[SurgicalProcedure]=()):
        """
        Instantiates a patient.
        :param code: The patient's anonymous code. This should be defined by the team.
        :param name: The patient's legal name.
        :param age: The patient's age in years.
        :param sex: The patient's biological sex.
        :param conditions: A tuple of objects of type MedicalCondition, e.g. Epilepsy. These objects contain information about the condition.
        :param medications: A tuple of objects of type Medication, e.g. Benzodiazepine.
        :param procedures: A tuple of objects of type SurgicalProcedure, e.g. EpilepsySurgery. These objects contain information about the procedures.
        """
        self.__procedures = procedures
        self.__medications = medications
        self.__conditions = conditions
        self.__age = age
        self.__sex = sex
        self.__name = name
        self.__code = code
        self.__notes = []
        self.__locked = True

    @property
    def code(self):
        return self.__code

    @property
    def name(self):
        return self.__name

    @property
    def age(self):
        return self.__age

    @property
    def sex(self):
        return self.__sex

    @property
    def conditions(self):
        return self.__conditions

    @property
    def medications(self):
        return self.__medications

    @property
    def notes(self):
        return self.__notes

    def add_note(self, description:str):
        self.__notes.append(description)

    def get_protected_info(self):
        """Returns a dictionary of the private/sensible information: Name, Age, Medications, and Surgical Procedures."""
        # TODO: implement digestion with an RSA key
        pass
        """
        password = input("Password: ")
        digest()
        if (not self.__locked):
            return
        """

    def timeshift(self, delta: timedelta):
        """
        Time shifts all clinical information.
        """
        for procedure in self.__procedures:
            procedure.date += delta

    def __eq__(self, other):
        return self.code == other.code

    def __hash__(self):
        return hash(self.__code) * hash(self.__name) * hash(self.__age) * hash(self.__sex)

    def __getstate__(self):
        """
        1: code
        2: name (str)
        3: age (int)
        4: sex (Sex)
        5: conditions (tuple)
        6: medications (tuple)
        7: procedures (tuple)
        8: notes (list)
        """
        return (self.__SERIALVERSION, self.__code, self.__name, self.__age, self.__sex,
                self.__conditions, self.__medications, self.__procedures, self.__notes)

    def __setstate__(self, state):
        if state[0] == 1:
            self.__code, self.__name, self.__age, self.__sex = state[1], state[2], state[3], state[4]
            self.__conditions, self.__medications, self.__procedures, self.__notes = state[5], state[6], state[7], state[8]
        else:
            raise IOError(f'Version of Patient object not supported. Serialized version: {state[0]};'
                          f'Supported versions: 1.')
