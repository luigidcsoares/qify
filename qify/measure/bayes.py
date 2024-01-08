from qify.channel.core import Channel
from qify.probab_dist.core import ProbabDist

def prior(pi: ProbabDist) -> float:
    """
    Computes the prior Bayes vulnerability of a distribution.

    ## Example

    Consider, for instance, a 2-bit password:
    
    >>> from qify.probab_dist import uniform
    >>> pi = uniform(["00", "01", "10", "11"],  "password")
    >>> prior(pi)
    0.25

    Maybe we know that passwords with distinct bits are more common:
    
    >>> from qify import ProbabDist
    >>> pi = ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6],
    ...   ["00", "01", "10", "11"],
    ...   "password"
    ... )
    >>> prior(pi)
    0.3333333333333333
    """
    return pi.max();


def posterior(
    pi: ProbabDist,
    ch: Channel,
    max_case: bool = False
) -> float:
    """
    Computes the posterior Bayes vulnerability of a channel `ch`,
    given a prior distribution `pi`. Bayes is essentially the
    sum of the column maximums of the joint matrix.

    ## Example

    Say we are modelling a 2-bit password checker and consider a
    particular password 10. What is the chance of someone guessing
    this password correctly, given that they have observed the
    behavior of the system: reject immediately the first wrong bit?

    >>> import pandas as pd
    >>> from qify import Channel
    >>> from qify.probab_dist import uniform
    >>> index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = Channel(
    ...   pd.Series([1, 1, 1, 1], index=index),
    ...   "password",
    ...   ["obs"]
    ... )
    >>> pi = ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6],
    ...   ["00", "01", "10", "11"],
    ...   "password"
    ... )

    We may be interested in the expected case:
    >>> posterior(pi, ch)
    0.8333333333333333

    Or in the worst scenario possible:
    >>> posterior(pi, ch, max_case=True)
    1.0
    """
    joint = ch.push_prior(pi)
    joint_grouped = joint.groupby(ch.output_names)
    
    if not max_case: return joint_grouped.max().sum()

    outer_dist = joint_grouped.sum()
    return joint.div(outer_dist).max()


def mult_leakage(
    pi: ProbabDist,
    ch: Channel,
    max_case: bool = False
) -> float:
    """
    Computes the multiplicative Bayes leakage of a channel `ch`,
    given a prior distribution `pi`.

    ## Example

    >>> import pandas as pd
    >>> from qify import Channel
    >>> from qify.probab_dist import uniform
    >>> index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = Channel(
    ...   pd.Series([1, 1, 1, 1], index=index),
    ...   "password",
    ...   ["obs"]
    ... )
    >>> pi = ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6],
    ...   ["00", "01", "10", "11"],
    ...   "password"
    ... )
    >>> mult_leakage(pi, ch)
    2.5
    >>> mult_leakage(pi, ch, max_case=True)
    3.0
    """
    return posterior(pi, ch, max_case) / prior(pi)


def add_leakage(
    pi: ProbabDist,
    ch: Channel,
    max_case: bool = False
) -> float:
    """
    Computes the additive Bayes leakage of a channel `ch`,
    given a prior distribution `pi`.

    ## Example

    >>> import pandas as pd
    >>> from qify import Channel
    >>> from qify.probab_dist import uniform
    >>> index = pd.MultiIndex.from_tuples(
    ...   [("00", "Reject 1st"), ("01", "Reject 1st"),
    ...    ("10", "Accept"), ("11", "Reject 2nd")],
    ...   names=["password", "obs"]
    ... )
    >>> ch = Channel(
    ...   pd.Series([1, 1, 1, 1], index=index),
    ...   "password",
    ...   ["obs"]
    ... )
    >>> pi = ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6],
    ...   ["00", "01", "10", "11"],
    ...   "password"
    ... )
    >>> add_leakage(pi, ch)
    0.49999999999999994
    >>> add_leakage(pi, ch, max_case=True)
    0.6666666666666667
    """
    return posterior(pi, ch, max_case) - prior(pi)
