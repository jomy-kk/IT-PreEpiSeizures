# -*- encoding: utf-8 -*-

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

from datetime import datetime
from typing import Sequence, Union, Callable, Any

from multimethod import multimethod
from numpy import ndarray


class Segment():
    # INITIALIZERS
    def __init__(self, samples: ndarray | Sequence[float]): ...

    # GETTERS
    @property
    def samples(self) -> ndarray: ...

    # BUILT-INS (Basics)
    def __copy__(self) -> Segment: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

    # BUILT-INS (Joining Segments)
    def append(self, samples: ndarray | Sequence[float]) -> None: ...

    @classmethod
    def concatenate(cls, *other: 'Segment') -> 'Segment': ...

    # BUILT-INS (Arithmetic)
    def __add__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __iadd__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __sub__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __isub__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __mul__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __imul__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __truediv__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __itruediv__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __floordiv__(self, other: 'Segment' | float | int) -> 'Segment': ...

    def __ifloordiv__(self, other: 'Segment' | float | int) -> 'Segment': ...

    # BUILT-INS (Indexing)
    def __getitem__(self, index: int | slice | tuple) -> float | Segment: ...

    def __iter__(self) -> iter: ...

    # BUILT-INS (Binary Logic)
    @multimethod
    def __eq__(self, other: Segment) -> bool: ...

    @multimethod
    def __eq__(self, other: Union[int, float]) -> bool: ...

    @multimethod
    def __ne__(self, other: Segment) -> bool: ...

    @multimethod
    def __ne__(self, other: Union[int, float]) -> bool: ...

    # PROCESSING
    def apply(self, operation: Callable, inplace: bool = True, **kwargs): ...

    def extract(self, operation: Callable, **kwargs) -> Any: ...

    # SHORTCUT STATISTICS
    def max(self) -> float: ...
    def argmax(self) -> int: ...
    def min(self) -> float: ...
    def argmin(self) -> int: ...
    def mean(self) -> float: ...
    def median(self) -> float: ...
    def std(self) -> float: ...
    def var(self) -> float: ...
    def abs(self) -> Segment: ...
    def diff(self) -> Segment: ...

    # SERIALIZATION
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple) -> None: ...
