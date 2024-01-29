from __future__ import annotations
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

def is_proper(dist: pd.Series) -> bool:
    """
    Checks if `dist` really corresponds to probab. distribution,
    i.e., it is 1-summing and nonnegative.

    ## Example

    >>> import pandas as pd
    >>> import qify
    >>> dist = pd.Series([1/3, 1/3, 1/3])
    >>> qify.probab_dist.is_proper(dist)
    True

    >>> dist = pd.Series([1/3, 1/3, 1/4])
    >>> qify.probab_dist.is_proper(dist)
    False

    >>> dist = pd.Series([1/3, 1/3, 2/3])
    >>> qify.probab_dist.is_proper(dist)
    False
    """
    return np.isclose(dist.sum(), 1) and dist.ge(0).all()


def uniform(values: set[Any], name: str) -> ProbabDist:
    """
    Constructs a uniform distribution for the given values.

    ## Example

    >>> import qify
    >>> qify.probab_dist.uniform(["00", "01", "10", "11"],  "password")
    password
    00    0.25
    01    0.25
    10    0.25
    11    0.25
    dtype: float64
    """
    n = len(values)
    dist = [1/n] * n
    return ProbabDist(dist, values, name)


class ProbabDist:
    """
    Probability distributions are represented as a pandas Series
    with a single level (i.e., a marginal distribution). The
    level must be named, so that the distribution can be used
    in other computations (merging, etc).
    """
    def __init__(
        self,
        dist: list[float | Decimal],
        values: list[Any],
        name: str
    ):
        index = pd.Index(values, name=name)
        self._dist = pd.Series(dist, index=index)
        
        if not is_proper(self._dist):
            raise ValueError("Invalid probability distribution!")

        # Increment the class with some methods from pandas Series,
        # so that it works kinda like a wrapper for Series:
        methods = ["min", "max", "sum", "mul"]
        for method in methods:
            setattr(self, method, getattr(self._dist, method))
        

    @property
    def secrets(self) -> pd.Index:
        return self._dist.index

    
    @property
    def secret_name(self) -> str:
        return self.secrets.name


    def __repr__(self) -> str:
        return repr(self._dist)

    
    def inspect(self) -> pd.Series:
        return self._dist.copy()
