from __future__ import annotations

import numpy as np
import pandas as pd

def is_proper(ch: pd.Series, input_level: int | str = 0) -> bool:
    """
    Checks if `ch` really corresponds to a channel. That is,
    it's "rows" are 1-summing and all entries are nonnegative.

    By default, assumes that the input of the channel corresponds
    to the first level of the Series index. This behavior can be
    overwritten through the parameter `input_name`.
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
    Maybe the adversary is interested in trying to learn someone's
    age, and the only thing that they can do is to query the
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
