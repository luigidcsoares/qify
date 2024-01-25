from collections.abc import Callable, Iterable
from multimethod import multimethod
from typing import Any

import pandas as pd

from qify.channel.core import Channel
from qify.probab_dist.core import ProbabDist

@multimethod
def prior(pi: ProbabDist, g_series: pd.Series) -> float:
    """
    Computes the prior g-vulnerability of a distribution.

    :param g_series: pandas Series indexed by actions and secrets.
    
    ## Example

    Consider a 2-bit password and say that the user is allowed two
    attempts before getting locked. The set of possible actions is
    W = {(00, -), (01, -), ..., (00, 01), ..., (11, 10)}, where a
    dash in the second attempt indicates that the user got it
    right in the first try, and noting that repeating a wrong
    password makes no sense.

    >>> from qify.probab_dist import uniform
    >>> secrets = ["00", "01", "10", "11"]
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> index = pd.MultiIndex.from_tuples(
    ...   [(w, x) for w in actions for x in secrets],
    ...   names=["action", "secret"]
    ... )
    >>> g_series = pd.Series(
    ...   {(w, x): 1 if x in w else 0 for w, x in index},
    ...   index=index
    ... )
    >>> pi = uniform(secrets,  "password")
    >>> prior(pi, g_series)
    0.5

    The cells in which gain is 0 are not really necessary, though:
    
    >>> index = pd.MultiIndex.from_tuples(
    ...   [(w, x) for w in actions for x in secrets if x in w],
    ...   names=["action", "secret"]
    ... )
    >>> g_series = pd.Series([1] * len(index), index=index)
    >>> pi = uniform(secrets,  "password")
    >>> prior(pi, g_series)
    0.5
    """
    return (
        pi
        .mul(g_series, level="secret")
        .groupby("action")
        .sum()
        .max()
    )

@multimethod
def prior(
    pi: ProbabDist,
    g_func: Callable,
    actions: Iterable
) -> float:
    """
    Computes the prior g-vulnerability of a distribution.

    ## Example

    Instead of a pandas Series, you can provide a gain function:

    >>> from qify.probab_dist import uniform
    >>> secrets = ["00", "01", "10", "11"]
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> pi = uniform(secrets,  "password")
    >>> prior(pi, lambda w, x: 1 if x in w else 0, actions)
    0.5
    """
    index = pd.MultiIndex.from_tuples(
        [(w, x) for w in actions for x in pi.input_values],
        names=["action", "secret"]
    )

    g_series = pd.Series(
        {(w, x): g_func(w, x) for w, x in index},
        index=index
    )

    return prior(pi, g_series)

@multimethod
def prior(pi: ProbabDist, g_func: Callable) -> float:
    """
    Computes the prior g-vulnerability of a distribution.
    Assumes that the set of actions is the same of the secrets.

    ## Example

    Consider a 2-bit password and say that the user is allowed only
    one attempt before getting locked. Then, the user's possible
    actions are to guess one of the four possible passwords, and
    thus the set of actions is exactly the set of secrets:

    >>> from qify.probab_dist import uniform
    >>> secrets = ["00", "01", "10", "11"]
    >>> prior(pi, lambda w, x: 1 if x in w else 0)
    0.25
    """
    return prior(pi, g_func, pi.input_values)
