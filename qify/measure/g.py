from collections.abc import Callable, Iterable
from multimethod import multimethod
from typing import Any

import pandas as pd

from qify.channel.core import Channel
from qify.probab_dist.core import ProbabDist

def _from_func(
    g_func: Callable,
    secret_name: str,
    secrets: Iterable,
    actions: Iterable
) -> pd.Series:
    """
    Takes a callable and converts into a pandas Series representing
    a gain function, indexed by secrets and actions.
    """
    index = pd.MultiIndex.from_tuples(
        [(w, x) for w in actions for x in secrets],
        names=["action", secret_name]
    )

    return pd.Series(
        {(w, x): g_func(w, x) for w, x in index},
        index=index
    )


@multimethod
def prior(pi: ProbabDist, g_series: pd.Series) -> float:
    """
    Computes the prior g-vulnerability of a distribution.

    :param g_series: pandas Series indexed by actions and secrets.
    	The index level that refers to the secret values must match
    	the name of the index in the prior `pi`, while the index
    	level that refers to the actions must be named "action".
    
    ## Example

    Consider a 2-bit password and say that the user is allowed two
    attempts before getting locked. The set of possible actions is
    W = {(00, -), (01, -), ..., (00, 01), ..., (11, 10)}, where a
    dash in the second attempt indicates that the user got it
    right in the first try, and noting that repeating a wrong 
    password makes no sense. Furthermore, say it is known that
    passwords with distincts bits are more common:

    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> index = pd.MultiIndex.from_tuples(
    ...   [(w, x) for w in actions for x in secrets],
    ...   names=["action", "password"]
    ... )
    >>> g_series = pd.Series(
    ...   {(w, x): 1 if x in w else 0 for w, x in index},
    ...   index=index
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6],secrets, "password"
    ... )
    >>> qify.measure.g.prior(pi, g_series)
    0.6666666666666666

    The cells in which gain is 0 are not really necessary, though:
    
    >>> index = pd.MultiIndex.from_tuples(
    ...   [(w, x) for w in actions for x in secrets if x in w],
    ...   names=["action", "password"]
    ... )
    >>> g_series = pd.Series([1] * len(index), index=index)
    >>> qify.measure.g.prior(pi, g_series)
    0.6666666666666666
    """
    return pi.mul(g_series).groupby("action").sum().max()


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

    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6],secrets, "password"
    ... )
    >>> qify.measure.g.prior(pi, lambda w, x: 1 if x in w else 0, actions)
    0.6666666666666666
    """
    return prior(pi, _from_func(
        g_func, pi.secret_name, pi.secrets, actions
    ))


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

    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6],secrets, "password"
    ... )
    >>> qify.measure.g.prior(pi, lambda w, x: 1 if x in w else 0)
    0.3333333333333333
    """
    return prior(pi, g_func, pi.secrets)


@multimethod
def posterior(
    pi: ProbabDist,
    ch: Channel,
    g_series: pd.Series,
    max_case: bool = False
) -> float:
    """
    Computes the posterior g-vulnerability of a channel `ch`,
    given a prior distribution `pi`.

    :param g_series: pandas Series indexed by actions and secrets.
    	The index level that refers to the secret values must match
    	the name of the index in the prior `pi`, while the index
    	level that refers to the actions must be named "action".
    
    ## Example
    
    Say we are modelling a 2-bit password checker and consider a
    particular password 10. What is the chance of someone guessing
    this password correctly, given that they have observed the
    behavior of the system: reject immediately the first wrong bit,
    and they are allowed two attempts before getting locked out?

    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> g_index = pd.MultiIndex.from_tuples(
    ...   [(w, x) for w in actions for x in secrets],
    ...   names=["action", "password"]
    ... )
    >>> g_series = pd.Series(
    ...   {(w, x): 1 if x in w else 0 for w, x in g_index},
    ...   index=g_index
    ... )

    We may be interested in the expected case:

    >>> qify.measure.g.posterior(pi, ch, g_series)
    0.8333333333333333

    Or in the worst scenario possible:
    
    >>> qify.measure.g.posterior(pi, ch, g_series, max_case=True)
    1.0
    """
    joint = ch.push_prior(pi)
    outer_dist = joint.groupby(ch.output_names).sum()
    hyper = joint.div(outer_dist)
    vulns = hyper.mul(g_series).groupby(ch.output_names).max()
    return vulns.max() if max_case else vulns.mul(outer_dist).sum()


@multimethod
def posterior(
    pi: ProbabDist,
    ch: Channel,
    g_func: Callable,
    actions: Iterable,
    max_case: bool = False
) -> float:
    """
    Computes the posterior g-vulnerability of a channel `ch`,
    given a prior distribution `pi`.

    ## Example

    Instead of a pandas Series, you can provide a gain function:

    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> qify.measure.g.posterior(
    ...   pi, ch, lambda w, x: 1 if x in w else 0, actions
    ... )
    0.8333333333333333

    Or in the worst scenario possible:
    
    >>> qify.measure.g.posterior(
    ...   pi, ch,
    ...   lambda w, x: 1 if x in w else 0,
    ...   actions, max_case=True
    ... )
    1.0
    """
    return posterior(pi, ch, _from_func(
        g_func, pi.secret_name, pi.secrets, actions
    ), max_case)


@multimethod
def posterior(
    pi: ProbabDist,
    ch: Channel,
    g_func: Callable,
    max_case: bool = False
) -> float:
    """
    Computes the posterior g-vulnerability of a channel `ch`,
    given a prior distribution `pi`. Assumes that the set of
    actions is the same of the secrets.

    ## Example

    Consider a 2-bit password and say that the user is allowed only
    one attempt before getting locked. Then, the user's possible
    actions are to guess one of the four possible passwords, and
    thus the set of actions is exactly the set of secrets:

    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> qify.measure.g.posterior(
    ...   pi, ch, lambda w, x: 1 if x in w else 0
    ... )
    0.8333333333333333

    Or in the worst scenario possible:
    
    >>> qify.measure.g.posterior(
    ...   pi, ch,
    ...   lambda w, x: 1 if x in w else 0,
    ...   max_case=True
    ... )
    1.0
    """
    return posterior(pi, ch, g_func, pi.secrets, max_case)


@multimethod
def mult_leakage(
    pi: ProbabDist,
    ch: Channel,
    g_series: pd.Series,
    max_case: bool = False
) -> float:
    """
    Computes the multiplicative g-leakage of a channel `ch`,
    given a prior distribution `pi`.

    ## Example
    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> g_index = pd.MultiIndex.from_tuples(
    ...   [(w, x) for w in actions for x in secrets],
    ...   names=["action", "password"]
    ... )
    >>> g_series = pd.Series(
    ...   {(w, x): 1 if x in w else 0 for w, x in g_index},
    ...   index=g_index
    ... )
    >>> qify.measure.g.mult_leakage(pi, ch, g_series)
    1.25
    >>> qify.measure.g.mult_leakage(pi, ch, g_series, max_case=True)
    1.5
    """
    return (
        posterior(pi, ch, g_series, max_case) /
        prior(pi, g_series)
    )


@multimethod
def mult_leakage(
    pi: ProbabDist,
    ch: Channel,
    g_func: Callable,
    actions: Iterable,
    max_case: bool = False
) -> float:
    """
    Computes the multiplicative g-leakage of a channel `ch`,
    given a prior distribution `pi`.

    ## Example
    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> qify.measure.g.mult_leakage(
    ...   pi, ch, lambda w, x: 1 if x in w else 0, actions
    ... )
    1.25
    >>> qify.measure.g.mult_leakage(
    ...   pi, ch, lambda w, x: 1 if x in w else 0,
    ...   actions, max_case=True
    ... )
    1.5
    """
    return mult_leakage(pi, ch, _from_func(
        g_func, pi.secret_name, pi.secrets, actions
    ), max_case)


@multimethod
def mult_leakage(
    pi: ProbabDist,
    ch: Channel,
    g_func: Callable,
    max_case: bool = False
) -> float:
    """
    Computes the multiplicative g-leakage of a channel `ch`,
    given a prior distribution `pi`. Assumes that the set of
    actions is the same of the secrets.

    ## Example
    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> qify.measure.g.mult_leakage(
    ...   pi, ch, lambda w, x: 1 if x in w else 0, actions
    ... )
    1.25
    >>> qify.measure.g.mult_leakage(
    ...   pi, ch, lambda w, x: 1 if x in w else 0,
    ...   actions, max_case=True
    ... )
    1.5
    """
    return mult_leakage(pi, ch, g_func, pi.secrets, max_case)


@multimethod
def add_leakage(
    pi: ProbabDist,
    ch: Channel,
    g_series: pd.Series,
    max_case: bool = False
) -> float:
    """
    Computes the additive g-leakage of a channel `ch`,
    given a prior distribution `pi`.

    ## Example
    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> g_index = pd.MultiIndex.from_tuples(
    ...   [(w, x) for w in actions for x in secrets],
    ...   names=["action", "password"]
    ... )
    >>> g_series = pd.Series(
    ...   {(w, x): 1 if x in w else 0 for w, x in g_index},
    ...   index=g_index
    ... )
    >>> qify.measure.g.add_leakage(pi, ch, g_series)
    0.16666666666666663
    >>> qify.measure.g.add_leakage(pi, ch, g_series, max_case=True)
    0.33333333333333337
    """
    return (
        posterior(pi, ch, g_series, max_case) -
        prior(pi, g_series)
    )


@multimethod
def add_leakage(
    pi: ProbabDist,
    ch: Channel,
    g_func: Callable,
    actions: Iterable,
    max_case: bool = False
) -> float:
    """
    Computes the additive g-leakage of a channel `ch`,
    given a prior distribution `pi`.

    ## Example
    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> qify.measure.g.add_leakage(
    ...   pi, ch, lambda w, x: 1 if x in w else 0, actions
    ... )
    0.16666666666666663
    >>> qify.measure.g.add_leakage(
    ...   pi, ch, lambda w, x: 1 if x in w else 0,
    ...   actions, max_case=True
    ... )
    0.33333333333333337
    """
    return add_leakage(pi, ch, _from_func(
        g_func, pi.secret_name, pi.secrets, actions
    ), max_case)


@multimethod
def add_leakage(
    pi: ProbabDist,
    ch: Channel,
    g_func: Callable,
    max_case: bool = False
) -> float:
    """
    Computes the additive g-leakage of a channel `ch`,
    given a prior distribution `pi`. Assumes that the set of
    actions is the same of the secrets.

    ## Example
    >>> import pandas as pd
    >>> import qify
    >>> secrets = ["00", "01", "10", "11"]
    >>> ch_index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = qify.Channel(
    ...   pd.Series([1, 1, 1, 1], index=ch_index),
    ...   "password", ["obs"]
    ... )
    >>> pi = qify.ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6], secrets, "password"
    ... )
    >>> actions = (
    ...   [(x, "-") for x in secrets] +
    ...   [(x, y) for x in secrets for y in secrets if x != y]
    ... )
    >>> qify.measure.g.add_leakage(
    ...   pi, ch, lambda w, x: 1 if x in w else 0, actions
    ... )
    0.16666666666666663
    >>> qify.measure.g.add_leakage(
    ...   pi, ch, lambda w, x: 1 if x in w else 0,
    ...   actions, max_case=True
    ... )
    0.33333333333333337
    """
    return add_leakage(pi, ch, g_func, pi.secrets, max_case)
