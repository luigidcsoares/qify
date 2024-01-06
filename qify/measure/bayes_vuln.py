import qify

def prior(dist: qify.ProbabDist) -> float:
    """
    Computes the prior Bayes vulnerability of a distribution.

    ## Example
    
    >>> from qify.probab_dist import uniform
    >>> dist = uniform(["00", "01", "10", "11"],  "password")
    >>> prior(dist)
    0.25
    
    >>> from qify import ProbabDist
    >>> dist = ProbabDist(
    ...   [1/6, 1/3, 1/3, 1/6],
    ...   ["00", "01", "10", "11"],
    ...   "password"
    ... )
    >>> prior(dist)
    0.3333333333333333
    """
    return dist.max();
