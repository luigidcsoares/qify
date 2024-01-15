from __future__ import annotations

import numpy as np
import pandas as pd

from qify.probab_dist.core import ProbabDist

def is_proper(ch: pd.Series, input_level: int | str = 0) -> bool:
    """
    Checks if `ch` really corresponds to a channel. That is,
    it's "rows" are 1-summing and all entries are nonnegative.

    By default, assumes that the input of the channel corresponds
    to the first level of the Series index. This behavior can be
    overwritten through the parameter `input_level`.

    ## Example

    >>> index = pd.MultiIndex.from_tuples([
    ...   ("x0", "y0"), ("x0", "y1"),
    ...   ("x1", "y0"), ("x1", "y1")
    ... ])
    >>> channel = pd.Series([1/2, 1/2, 1/3, 2/3], index=index)
    >>> is_proper(channel)
    True

    >>> index = pd.MultiIndex.from_tuples([
    ...   ("x0", "y0"), ("x0", "y1"),
    ...   ("x1", "y0"), ("x1", "y1")
    ... ])
    >>> channel = pd.Series([1/2, 1/2, 1/3, 1/3], index=index)
    >>> is_proper(channel)
    False
    """
    row_sums = ch.groupby(level=input_level).sum()
    return np.isclose(row_sums, 1).all() and ch.ge(0).all()


def from_pandas(
        df: pd.DataFrame,
        input_name: str,
        output_names: list[str]
) -> Channel:
    """
    Constructs a channel from a dataset.
    
    ## Example
        
    Consider a dataset that stores demographic data about people.
    Maybe we are interested in trying to learn someone's
    age, and the only thing that we can do is to query the
    database and observe a partition of the dataset induced by
    their target's gender. The corresponding channel is

    >>> df = pd.DataFrame({
    ...   "age":  [46, 58, 46, 23, 23],
    ...   "gender": ["m", "m", "f", "f", "f"]
    ... })
    >>> from_pandas(df, "age", ["gender"])
    age  gender
    23   f         1.0
    46   f         0.5
         m         0.5
    58   m         1.0
    dtype: float64
    """
    freq_input = df[input_name].value_counts()
    dists = (
        df.groupby([input_name] + output_names)
        .size()
        .div(freq_input, level=input_name)
    )
    return Channel(dists, input_name, output_names)


class Channel:
    """
    Channels are represented as pandas Series whose index is a
    MultiIndex. The purpose of the Series representation is to
    reduce memory usage when channel matrices are sparse.
    """
    def __init__(
            self,
            dists: pd.Series,
            input_name: str,
            output_names: list[str]
    ):
        if not is_proper(dists):
            raise ValueError("Input is not a valid channel!")
        
        self._dists = dists
        self._input_name = input_name
        self._output_names = output_names

    
    @property
    def input_name(self) -> str:
        return self._input_name
    
    
    @property
    def output_names(self) -> list[str]:
        return self._output_names


    def __repr__(self) -> str:
        return repr(self._dists)


    def inspect(self) -> pd.Series:
        return self._dists.copy()


    def push_prior(self, pi: ProbabDist) -> pd.Series:
        """
        Pushes a prior distribution pi through the channel to
        construct a joint. The result is a pandas Series,
        following the same format of the channel.

        ## Example

        Suppose we want to learn someone's age and we know a 
        that the population is ageing, so the distribution of
        young vs old people (say, > 40) is such that 4/5 are old:
        
        >>> df = pd.DataFrame({
        ...   "age":  [46, 58, 46, 23, 23],
        ...   "gender": ["m", "m", "f", "f", "f"]
        ... })
        >>> pi = ProbabDist(
        ...   [2/5, 2/5, 1/5],
        ...   df["age"].drop_duplicates(),
        ...   "age"
        ... )
        >>> ch = from_pandas(df, "age", ["gender"])
        >>> ch.push_prior(pi)
        age  gender
        23   f         0.2
        46   f         0.2
             m         0.2
        58   m         0.4
        dtype: float64
        """
        return pi.mul(self._dists, level=self.input_name)

    
    def cascade(self, other: Channel) -> Channel:
        """
        Computes the cascading BC as matrix multiplication,
        where B is the current object and C is `other`.

        ## Limitations

        Currently requires output of B to be a single attribute
        (since input of rhs must be a single attribute).

        ## Example
    
        Suppose there is a correlation between the number of
        transactions a person makes on average and its age
        (say young people buy things more often).

        We could have a dataset relating a person's transaction
        count to its age, in addition to a demographic dataset.
        The first dataset gives us a channel B : #Tr. -> Age,
        while the demographic dataset gives us C : Age -> Gender.

        The cascading BC maps #Tr. to Gender:

        >>> from qify.channel import from_pandas
        >>> correlation_df = pd.DataFrame({
        ...   "age": [46, 58, 46, 23, 23],
        ...   "n_tr": [4, 4, 5, 10, 10]
        ... })
        >>> demographic_df = pd.DataFrame({
        ...   "age":  [46, 58, 46, 23, 23],
        ...   "gender": ["m", "m", "f", "f", "f"]
        ... })
        >>> ch_b = from_pandas(correlation_df, "n_tr", ["age"])
        >>> ch_c = from_pandas(demographic_df, "age", ["gender"])
        >>> ch_b.cascade(ch_c)
        n_tr  gender
        4     f         0.25
              m         0.75
        5     f         0.50
              m         0.50
        10    f         1.00
        Name: bc, dtype: float64
        """
        merge_on = other.input_name
        group_on = [self.input_name] + other.output_names
        
        dists = (
            self._dists.reset_index(name="b").merge(
                other._dists.reset_index(name="c"),
                on=merge_on
            ).assign(bc=lambda df: df["b"] * df["c"])
            .groupby(group_on)["bc"]
            .sum()
        )

        return Channel(dists, self.input_name, other.output_names)


    def parallel(self, other: Channel) -> Channel:
        """
        Computes the parallel composition B || C,
        where B is the current object and C is `other`.

        ## Example

        >>> import pandas as pd
        >>> import qify
        >>> index_b = pd.MultiIndex.from_tuples(
        ...   [("x1", "y1"), ("x1", "y2"), ("x2", "y1"), ("x2", "y2")],
        ...   names=["X", "Y"]
        ... )
        >>> ch_b = qify.Channel(
        ...   pd.Series([0.4, 0.6, 0.8, 0.2], index=index_b),
        ...   "X", ["Y"]
        ... )
        >>> index_c = pd.MultiIndex.from_tuples(
        ...   [("x1", "y1"), ("x1", "y3"), ("x2", "y1"), ("x2", "y3")],
        ...   names=["X", "Y"]
        ... )
        >>> ch_c = qify.Channel(
        ...   pd.Series([1, 0, 0.3, 0.7], index=index_c),
        ...   "X", ["Y"]
        ... )
        >>> ch_b.parallel(ch_c)
        X   Yb  Yc
        x1  y1  y1    0.40
                y3    0.00
            y2  y1    0.60
                y3    0.00
        x2  y1  y1    0.24
                y3    0.56
            y2  y1    0.06
                y3    0.14
        dtype: float64
        """
        if self.input_name != other.input_name:
            raise ValueError("Incompatible channels: input is not the same!")

        dists = self._dists.reset_index(name="b").merge(
            other._dists.reset_index(name="c"),
            on=self.input_name,
            suffixes=("b", "c")
        )

        output_names = dists.columns[
            ~dists.columns.isin([self.input_name, "b", "c"])
        ].tolist()

        dists = dists.set_index([self.input_name] + output_names)
        return Channel(dists["b"] * dists["c"], self.input_name, output_names)
