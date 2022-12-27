import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

logger = logging.getLogger(__name__)


MAX_DRAW_SIZE: int = int(1e7)

# custom spiir distributions
distributions: dict = {}


def estimate_n_redraw(n: int, ratio: float) -> int:
    """Use ratio of samples satisfying constraints to increare redraw size and
    speed up parameter re-sampling from distributions.

    See: pycbc.distributions.joint.py#L331
    """
    return int(min(MAX_DRAW_SIZE, np.ceil(n / ratio)))


class Distribution(ABC):
    def __init__(
        self,
        variables: Union[str, Tuple[str, ...]],
        distribution: str,
        pipe: Optional[Tuple[Callable]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        *args,
        **kwargs,
    ):
        self.variables = (variables,) if isinstance(variables, str) else variables
        self.distribution = distribution
        self.pipe = pipe
        self.rng = np.random.default_rng(seed)
        self.args = args
        self.kwargs = kwargs

    def redraw(
        self, samples: pd.DataFrame, n: int, verbose: bool = False
    ) -> pd.DataFrame:
        n_samples = len(samples)
        n_remain = n - n_samples
        df = pd.DataFrame(
            np.zeros((n_remain, len(samples.columns))), columns=samples.columns
        )

        n_draw = estimate_n_redraw(n, n_samples / (n + 1))

        while n_remain:
            desc = f"Running data pipeline on remaining {n_remain}/{n} redrawn samples"
            resamples = self.draw(n_draw, False, verbose, desc)
            n_resamples = len(resamples)

            # update iteration tracking variables
            i = n - n_samples - n_remain  # start index to assign resamples to
            j = min(n_resamples, n_remain)  # cap resamples to prevent overflow
            n_remain = max(0, n_remain - n_resamples)
            n_draw = estimate_n_redraw(n_draw, n_resamples / (n_draw + 1))

            # assign sample subset to dataframe in memory
            df.iloc[i : i + j] = resamples.iloc[:j].values

        df.index += n_samples
        return pd.concat([samples, df], axis=0)

    def __repr__(self):
        """Overrides string representation of cls when printed."""
        repr_attrs = ["variables", "distribution"]
        if len(self.args) > 0:
            repr_attrs.append("args")
        if len(self.kwargs) > 0:
            repr_attrs.append("kwargs")

        # TODO: Add self.pipe to __repr__
        kws = [f"{key}={self.__dict__[key]!r}" for key in repr_attrs]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    @abstractmethod
    def draw(
        self,
        n: int,
        redraw: bool = True,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> pd.DataFrame:
        pass


class NumPyDistribution(Distribution):
    package = "numpy"

    def __init__(
        self,
        variables: Union[str, Tuple[str]],
        distribution: str,
        pipe: Optional[Tuple[Callable]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(variables, distribution, pipe, seed, *args, **kwargs)
        self._distribution = getattr(self.rng, self.distribution)

    def draw(
        self,
        n: int,
        redraw: bool = True,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> pd.DataFrame:
        size = (n, len(self.variables))
        samples = self._distribution(size=size, *self.args, **self.kwargs)
        samples = pd.DataFrame(samples, columns=self.variables)

        if self.pipe is not None:
            variables = ", ".join(self.variables)
            desc = desc or f"Running data pipeline on {variables}"
            for pipe in tqdm(self.pipe, disable=not verbose, desc=desc):
                samples = samples.pipe(pipe)

            if redraw and len(samples) != n:
                samples = self.redraw(samples, n, verbose)

        return samples.reset_index(drop=True)


class SciPyDistribution(Distribution):
    package = "scipy"

    def __init__(
        self,
        variables: Union[str, Tuple[str]],
        distribution: str,
        pipe: Optional[Tuple[Callable]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(variables, distribution, pipe, seed, *args, **kwargs)
        self._distribution = getattr(scipy.stats, self.distribution)

    def draw(
        self,
        n: int,
        redraw: bool = True,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> pd.DataFrame:
        size = (n, len(self.variables))
        samples = self._distribution.rvs(
            size=size, random_state=self.rng, *self.args, **self.kwargs
        )
        samples = pd.DataFrame(samples, columns=self.variables)

        if self.pipe is not None:
            variables = ", ".join(self.variables)
            desc = desc or f"Running data pipeline on {variables}"
            for pipe in tqdm(self.pipe, disable=not verbose, desc=desc):
                samples = samples.pipe(pipe)

            if redraw and len(samples) != n:
                samples = self.redraw(samples, n, verbose)

        return samples.reset_index(drop=True)


class PyCBCDistribution(Distribution):
    package = "pycbc"

    def __init__(
        self,
        variables: Union[str, Tuple[str]],
        distribution: str,
        pipe: Optional[Tuple[Callable]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(variables, distribution, pipe, seed, *args, **kwargs)
        # TODO: Work out a cleaner solution to handle PyCBC distributions
        #   i.e. pycbc requires a variable name as keyword argument when instantiating
        #   some (but not all) distributions without bounds, so pass as None
        from pycbc.distributions import distribs

        distribs_without_variable_kwargs = ["uniform_sky", "uniform_solidangle"]
        if self.distribution not in distribs_without_variable_kwargs:
            for variable in self.variables:
                if variable not in self.kwargs:
                    kwargs[variable] = None

        self._distribution = distribs[self.distribution](**kwargs)

    def draw(
        self,
        n: int,
        redraw: bool = True,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> pd.DataFrame:
        samples = self._distribution.rvs(size=n)
        samples = pd.DataFrame(samples, columns=self.variables)

        if self.pipe is not None:
            variables = ", ".join(self.variables)
            desc = desc or f"Running data pipeline on {variables}"
            for pipe in tqdm(self.pipe, disable=not verbose, desc=desc):
                samples = samples.pipe(pipe)

            if redraw and len(samples) != n:
                samples = self.redraw(samples, n, verbose)

        return samples.reset_index(drop=True)


class SPIIRDistribution(Distribution):
    package = "spiir"

    def __init__(
        self,
        variables: Union[str, Tuple[str]],
        distribution: str,
        pipe: Optional[Tuple[Callable]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(variables, distribution, pipe, seed, *args, **kwargs)
        self._distribution = distributions[self.distribution]

    def draw(
        self,
        n: int,
        redraw: bool = True,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


class JointDistribution:
    def __init__(
        self,
        distributions: Tuple[Distribution],
        pipe: Optional[Tuple[Callable]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        *args,
        **kwargs,
    ):
        self.distributions = distributions
        self.pipe = pipe
        self.rng = np.random.default_rng(seed)
        self.args = args
        self.kwargs = kwargs

        if self.rng is not None:
            for distribution in self.distributions:
                distribution.rng = self.rng

    def __repr__(self):
        """Overrides string representation of cls when printed."""
        distributions = ",\n    ".join([str(dist) for dist in self.distributions])
        str_repr = (
            f"{type(self).__name__}(\n  distributions=(\n    {distributions}\n  )"
        )

        # TODO: Review self.pipe __repr__
        if self.pipe is not None:
            pipeline = ",\n    ".join([str(pipe) for pipe in self.pipe])
            str_repr += f",\n  pipeline=(\n    {pipeline}"
            str_repr += "\n  )"

        str_repr += "\n)"
        return str_repr

    def redraw(
        self, samples: pd.DataFrame, n: int, verbose: bool = False
    ) -> pd.DataFrame:
        n_samples = len(samples)
        n_remain = n - n_samples
        df = pd.DataFrame(
            np.zeros((n_remain, len(samples.columns))), columns=samples.columns
        )

        n_draw = estimate_n_redraw(n, n_samples / (n + 1))

        while n_remain:
            desc = f"Running data pipeline on remaining {n_remain}/{n} samples"
            resamples = self.draw(n_draw, False, verbose, desc)
            n_resamples = len(resamples)

            # update iteration tracking variables
            i = n - n_samples - n_remain  # start index to assign resamples to
            j = min(n_resamples, n_remain)  # cap resamples to prevent overflow
            n_remain = max(0, n_remain - n_resamples)
            n_draw = estimate_n_redraw(n_draw, n_resamples / (n_draw + 1))

            # assign sample subset to dataframe in memory
            df.iloc[i : i + j] = resamples.iloc[:j].values

        df.index += n_samples
        return pd.concat([samples, df], axis=0)

    def draw(
        self,
        n: int,
        redraw: bool = True,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> pd.DataFrame:
        samples = pd.concat(
            [dist.draw(n, redraw, verbose) for dist in self.distributions], axis=1
        ).dropna(axis=0)

        if self.pipe is not None:
            desc = "Running data pipeline for joint distribution"
            for pipe in tqdm(self.pipe, disable=not verbose, desc=desc):
                samples = samples.pipe(pipe)

            if redraw and len(samples) != n:
                samples = self.redraw(samples, n, verbose)

        return samples.reset_index(drop=True)
