from __future__ import annotations
from collections.abc import Collection

import numpy as np
import pandas as pd

from qify.probab_dist.core import ProbabDist, uniform
from qify.typing import AttrName, AttrValue

def is_proper(ch: pd.Series, secret_name: AttrName = 0) -> bool:
    """
    Checks if `ch` really corresponds to a channel. That is,
    it's "rows" are 1-summing and all entries are nonnegative.

    By default, assumes that the input of the channel corresponds
    to the first level of the Series index. This behavior can be
    overwritten through the parameter `input_level`.

    ## Example

    >>> import qify
    >>> import pandas as pd
    >>> index = pd.MultiIndex.from_tuples([
    ...   ("x0", "y0"), ("x0", "y1"),
    ...   ("x1", "y0"), ("x1", "y1")
    ... ])
    >>> channel = pd.Series([1/2, 1/2, 1/3, 2/3], index=index)
    >>> qify.channel.is_proper(channel)
    True

    >>> index = pd.MultiIndex.from_tuples([
    ...   ("x0", "y0"), ("x0", "y1"),
    ...   ("x1", "y0"), ("x1", "y1")
    ... ])
    >>> channel = pd.Series([1/2, 1/2, 1/3, 1/3], index=index)
    >>> qify.channel.is_proper(channel)
    False
    """
    row_sums = ch.groupby(level=secret_name).sum()
    return bool(np.isclose(row_sums, 1).all() and ch.ge(0).all())


def from_pandas(
    df: pd.DataFrame,
    secret_name: AttrName,
    output_names: Collection[AttrName]
) -> Channel:
    """
    Constructs a channel from a dataset.
    
    ## Example
        
    Consider a dataset that stores demographic data about people.
    Maybe we are interested in trying to learn someone's
    age, and the only thing that we can do is to query the
    database and observe a partition of the dataset induced by
    their target's gender. The corresponding channel is

    >>> import qify
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...   "age":  [46, 58, 46, 23, 23],
    ...   "gender": ["m", "m", "f", "f", "f"]
    ... })
    >>> qify.channel.from_pandas(df, "age", ["gender"])
    age  gender
    23   f         1.0
    46   f         0.5
         m         0.5
    58   m         1.0
    dtype: float64
    """
    freq_input = df[secret_name].value_counts()
    dists = pd.Series(
        df.groupby([secret_name, *output_names])
        .size()
        .div(freq_input, level=secret_name)
    )
    return Channel(dists, secret_name, output_names)


class Channel:
    """
    Channels are represented as pandas Series whose index is a
    MultiIndex. The purpose of the Series representation is to
    reduce memory usage when channel matrices are sparse.
    """
    def __init__(
        self,
        dists: pd.Series,
        secret_name: AttrName,
        output_names: Collection[AttrName],
        bypass_check: bool = False,
    ):
        if not bypass_check and not is_proper(dists):
            raise ValueError("Input does not form a valid channel!")

        self._dists = dists
        self._secret_name = secret_name
        self._output_names = output_names
        self._bypass_check = bypass_check


    def __repr__(self) -> str:
        return repr(self._dists)


    @property
    def secret_name(self) -> AttrName:
        return self._secret_name
    

    @property
    def output_names(self) -> Collection[AttrName]:
        return self._output_names


    @property
    def secrets(self) -> pd.Index:
        return self._dists.index.droplevel(self.output_names)


    @property
    def outputs(self) -> pd.Index:
        return self._dists.index.droplevel(self.secret_name)

    
    @property
    def reduced(self) -> Channel:
        """
        Returns the reduced form of the channel. Column labels are
        transformed to tuples, to indicate which columns were combined.

        ## Example

        >>> import pandas as pd
        >>> import qify
        >>> index = pd.MultiIndex.from_tuples([
        ...   ("x1", "y1"), ("x2", "y1"), ("x3", "y1"), 
        ...   ("x2", "y2"), ("x3", "y2"),
        ...   ("x2", "y3"), ("x3", "y3")
        ...  ], names=["X", "Y"])
        >>> ch = qify.channel.core.Channel(pd.Series(
        ...   [1, 1/4, 1/2, 1/2, 1/3, 1/4, 1/6], index=index
        ... ), "X", ["Y"])
        >>> ch.reduced
        X   Y       
        x1  (y1,)       1.00
        x2  (y1,)       0.25
        x3  (y1,)       0.50
        x2  (y2, y3)    0.75
        x3  (y2, y3)    0.50
        dtype: float64
        """
        # We build a hyper by pushing a uniform prior (hypers are easier
        # to reduce, as we just need to group columns that are equal)
        secrets = pd.Series(self.secrets.drop_duplicates())
        pi = uniform(secrets, self.secret_name)
        joint = self.push_prior(pi)
        outer = joint.groupby(self.output_names).sum()
        hyper = joint.div(outer)

        # This dict has entries of the form 
        # ((x1, prob), ..., (xn, prob)): [outer probab, (y1, ..., yk)]
        reduced_hyper = {}
        for label, group in hyper.groupby(self.output_names):
            indexed_probabs = group.droplevel(self.output_names).items()
            key = tuple(sorted(indexed_probabs))
            reduced_hyper[key] = reduced_hyper.get(key, [0, []])
            reduced_hyper[key][0] += outer.loc[label]
            reduced_hyper[key][1] += (label,)

        reduced_index = []
        reduced_hyper_values = []
        reduced_outer_index = []
        reduced_outer_values = []
        for secret_probabs, output_probabs in reduced_hyper.items():
            outer_probab = output_probabs[0]
            merged_output_name = tuple(zip(*output_probabs[1]))

            for secret, probab in secret_probabs:
                reduced_hyper_values.append(probab)
                reduced_index.append((secret, *merged_output_name))

            reduced_outer_values.append(outer_probab)
            reduced_outer_index.append(merged_output_name)

        attr_names = [self.secret_name, *(self.output_names)]
        reduced_hyper_series = pd.Series(
            reduced_hyper_values,
            index=pd.MultiIndex.from_tuples(
                reduced_index,
                names=attr_names
            )
        )

        reduced_outer_series = pd.Series(
            reduced_outer_values,
            index=pd.MultiIndex.from_tuples(
                reduced_outer_index,
                names=self.output_names
            )
        )

        reduced_joint = reduced_hyper_series.mul(reduced_outer_series)
        reduced_channel = pi.pow(-1).mul(reduced_joint)

        return Channel(
            reduced_channel.reorder_levels(attr_names), 
            self.secret_name, self.output_names
        )


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

        >>> import pandas as pd
        >>> import qify
        >>> df = pd.DataFrame({
        ...   "age":  [46, 58, 46, 23, 23],
        ...   "gender": ["m", "m", "f", "f", "f"]
        ... })
        >>> pi = qify.ProbabDist(
        ...   [2/5, 2/5, 1/5],
        ...   df["age"].drop_duplicates(),
        ...   "age"
        ... )
        >>> ch = qify.channel.from_pandas(df, "age", ["gender"])
        >>> ch.push_prior(pi)
        age  gender
        23   f         0.2
        46   f         0.2
             m         0.2
        58   m         0.4
        dtype: float64
        """
        return pi.mul(self._dists)

    
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

        >>> import pandas as pd
        >>> import qify
        >>> correlation_df = pd.DataFrame({
        ...   "age": [46, 58, 46, 23, 23],
        ...   "n_tr": [4, 4, 5, 10, 10]
        ... })
        >>> demographic_df = pd.DataFrame({
        ...   "age":  [46, 58, 46, 23, 23],
        ...   "gender": ["m", "m", "f", "f", "f"]
        ... })
        >>> ch_b = qify.channel.from_pandas(
        ...   correlation_df, "n_tr", ["age"]
        ... )
        >>> ch_c = qify.channel.from_pandas(
        ...   demographic_df, "age", ["gender"]
        ... )
        >>> ch_b.cascade(ch_c)
        n_tr  gender
        4     f         0.25
              m         0.75
        5     f         0.50
              m         0.50
        10    f         1.00
        Name: bc, dtype: float64
        """
        merge_on = other.secret_name
        group_on = [self.secret_name, *other.output_names]
        
        dists = (
            self._dists.reset_index(name="b").merge(
                other._dists.reset_index(name="c"),
                on=merge_on
            ).assign(bc=lambda df: df["b"] * df["c"])
            .groupby(group_on)["bc"]
            .sum()
        )

        return Channel(
            pd.Series(dists),
            self.secret_name,
            other.output_names,
            bypass_check=self._bypass_check
        )


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
        if self.secret_name != other.secret_name:
            raise ValueError("Incompatible channels: input is not the same!")

        dists = self._dists.reset_index(name="b").merge(
            other._dists.reset_index(name="c"),
            on=self.secret_name,
            suffixes=("b", "c")
        )

        output_names = dists.columns[
            ~dists.columns.isin([self.secret_name, "b", "c"])
        ].tolist()

        dists = dists.set_index([self.secret_name] + output_names)
        return Channel(
            dists["b"] * dists["c"],
            self.secret_name,
            output_names,
            bypass_check=self._bypass_check
        )
