import logging
from copy import deepcopy
from os import PathLike
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import scipy.stats

from .config import parse_config
from .constraint import Constraint, load_constraints_from_config
from .transform import Transform, load_transforms_from_config

logger = logging.getLogger(__name__)


# TODO:
#   - Allow JointDistribution to take JointDistributions as arguments?
#       - or: refactor a single class for both Distribution and JointDistribution
#       - or: prior = JointDistribution((*masses.distributions, *spins.distributions))
#   - Add constraints to Distribution objects, rather than only JointDistributions
#   - Move location of _check_constraints method on JointDistribution to Constraint
#   - Consider necessity of tuples vs. lists for class arguments (collections.Iterable)?
#   - Review NumPy random number generation in distributed contexts:
#       - see: https://albertcthomas.github.io/good-practices-random-number-generators/


class Distribution:
    """Abstract Base Class for Distribution classes"""

    package: Optional[str] = None

    def __new__(
        cls,
        package: str = "numpy",
        *args,
        **kwargs,
    ):
        if cls is Distribution:
            # return a Distribution subclass
            subclass_package_map = {
                subclass.package: subclass for subclass in cls.__subclasses__()
            }
            try:
                subclass = subclass_package_map[package.lower()]
            except KeyError as exc:
                raise KeyError(
                    f"Invalid package '{package.lower()}', \
                        try one of {set(subclass_package_map.keys())}"
                ) from exc
            return super(Distribution, subclass).__new__(subclass)
        elif cls in Distribution.__subclasses__():
            # cls is already a Distribution subclass, continue
            return super(Distribution, cls).__new__(cls)
        else:
            raise TypeError(f"{cls} not one of valid {Distribution.__subclasses__()}")

    def __init__(
        self,
        distribution: str,
        variable: Optional[str] = None,
        name: Optional[str] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        package: str = "numpy",
        **kwargs,
    ):
        self.name = name
        self.distribution = distribution
        self.variable = variable
        self.kwargs = kwargs

    def __repr__(self):
        """Overrides string representation of cls when printed."""
        repr_attrs = ["variable", "distribution", "kwargs"]
        if self.name is not None:
            repr_attrs.insert(0, "name")

        kws = [f"{key}={self.__dict__[key]!r}" for key in repr_attrs]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def __call__(
        self,
        size: Union[int, Tuple[int, ...]],
        series: bool = True,
    ):
        raise self.draw(size=size, series=series)

    def draw(
        self,
        size: Union[int, Tuple[int, ...]],
        series: bool = True,
    ):
        raise NotImplementedError


class NumPyDistribution(Distribution):
    package = "numpy"

    def __init__(
        self,
        distribution: str,
        variable: Optional[str] = None,
        name: Optional[str] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        package: str = "numpy",
        **kwargs,
    ):
        assert package == getattr(NumPyDistribution, "package")
        super().__init__(distribution, variable, name, seed, package, **kwargs)

        # Note: identical seeds and distribution objects will yield identical results
        self._rng = np.random.default_rng(seed)
        self._distribution = self._load_distribution()

    def _load_distribution(self):
        return getattr(self._rng, self.distribution)

    def draw(
        self,
        size: Union[int, Tuple[int, ...]],
        series: bool = True,
    ) -> Union[np.ndarray, pd.Series]:
        samples = self._distribution(size=size, **self.kwargs)
        if series:
            return pd.Series(samples, name=self.variable)
        return samples

    def __call__(
        self,
        size: Union[int, Tuple[int, ...]],
        series: bool = True,
    ) -> Union[np.ndarray, pd.Series]:
        return self.draw(size=size, series=series)


class SciPyDistribution(Distribution):
    package = "scipy"

    def __init__(
        self,
        distribution: str,
        variable: Optional[str] = None,
        name: Optional[str] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        package: str = "scipy",
        **kwargs,
    ):
        assert package == getattr(SciPyDistribution, "package")
        super().__init__(distribution, variable, name, seed, package, **kwargs)

        # Note: identical seeds and distribution objects will yield identical results
        self._rng = np.random.default_rng(seed)
        self._distribution = self._load_distribution()

    def _load_distribution(self):
        return getattr(scipy.stats, self.distribution)

    def draw(
        self,
        size: Union[int, Tuple[int, ...]],
        series: bool = True,
    ) -> Union[np.ndarray, pd.Series]:
        samples = self._distribution.rvs(
            size=size, random_state=self._rng, **self.kwargs
        )
        if series:
            return pd.Series(samples, name=self.variable)
        return samples

    def __call__(
        self,
        size: Union[int, Tuple[int, ...]],
        series: bool = True,
    ) -> Union[np.ndarray, pd.Series]:
        return self.draw(size=size, series=series)


class PyCBCDistribution(Distribution):
    package = "pycbc"

    def __init__(
        self,
        distribution: str,
        variable: Optional[str] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        package: str = "pycbc",
        **kwargs,
    ):
        assert package == getattr(PyCBCDistribution, "package")
        super().__init__(distribution, variable, name, seed, package, **kwargs)
        self._rng = seed  # unused
        self._distribution = self._load_distribution()

    def _load_distribution(self):
        from pycbc.distributions import distribs

        kwargs = deepcopy(self.kwargs)

        # TODO: Work out a cleaner solution to handle PyCBC distribution particularities
        #   i.e. pycbc requires a variable name as keyword argument when instantiating
        #   some (but not all) distributions without bounds, so pass as None
        distribs_without_variable_kwargs = ["uniform_sky", "uniform_solidangle"]
        if self.distribution not in distribs_without_variable_kwargs:
            if self.variable not in self.kwargs:
                kwargs[self.variable] = None

        return distribs[self.distribution](**kwargs)

    def draw(
        self,
        size: Union[int, Tuple[int, ...]],
        series: bool = True,
    ) -> Union[np.ndarray, pd.Series]:
        samples = self._distribution.rvs(size=size)
        if series:
            return pd.DataFrame(samples).squeeze()
        return samples

    def __call__(
        self,
        size: Union[int, Tuple[int, ...]],
        series: bool = True,
    ) -> Union[np.ndarray, pd.Series]:
        return self.draw(size=size, series=series)


class JointDistribution:
    def __init__(
        self,
        distributions: Union[Distribution, Tuple[Distribution, ...]],
        transforms: Optional[Union[Transform, Tuple[Transform, ...]]] = None,
        constraints: Optional[Union[Constraint, Tuple[Constraint, ...]]] = None,
    ):
        self.distributions: Tuple[Distribution, ...] = (
            (distributions,)
            if isinstance(distributions, Distribution)
            else distributions
        )
        self.variables = tuple(dist.variable for dist in self.distributions)

        # TODO: test whether this handles all iterable cases
        self.transforms = (
            (transforms,) if isinstance(transforms, Transform) else transforms
        )
        self.constraints = (
            (constraints,) if isinstance(constraints, Constraint) else constraints
        )

    

    def draw(self, n: int, redraw: bool = True) -> pd.DataFrame:
        """Draws samples from all stored Distribution objects.

        If a Constraint is present, the constraint is applied and samples are rejected
        until n valid samples have been drawn. Follows a similar method from PyCBC:
        https://github.com/gwastro/pycbc/blob/master/pycbc/distributions/joint.py#L305.

        Parameters
        ----------
        n: int
            The number of samples to return
        redraw: bool
            Whether or not to redraw samples if a constraint is not satisfied.

        Returns
        -------
        np.ndarray
            Parameter samples
        """

        if n < 1 or not isinstance(n, int):
            raise ValueError(f"n must be a positive integer, not {n}.")

        if self.constraints is None:
            samples = pd.concat(
                [dist.draw(n, series=True) for dist in self.distributions], axis=1
            )
            if self.transforms is not None:
                for transform in self.transforms:
                    samples = transform.apply(samples)
            return samples

        else:
            if redraw:
                # re-draw samples until we have n samples that satisfy constraints
                df = pd.DataFrame(
                    np.zeros((n, len(self.distributions))), columns=self.variables
                )

                # NOTE: Implementation from PyCBC, see: pycbc.distributions.joint.py
                first_pass = True
                remaining = n
                ndraw = n
                while remaining:
                    # applying constraints requires named columns (i.e. pandas)
                    samples = pd.concat(
                        [dist.draw(ndraw, series=True) for dist in self.distributions],
                        axis=1,
                    )

                    if self.transforms:
                        for transform in self.transforms:
                            samples = transform.apply(samples)

                    if first_pass:
                        # after potentially running all self.transforms, we can now
                        # accurately count the number of random variable columns.
                        df = pd.DataFrame(
                            np.zeros((n, len(samples.columns))), columns=samples.columns
                        )

                    # get all drawn samples that satisfy constraints
                    mask = np.all(
                        [constraint.apply(samples) for constraint in self.constraints],
                        axis=0,
                    )
                    n_samples = mask.sum()

                    # update iteration tracking variables
                    i = n - remaining  # start index to assign valid samples from
                    # cap at remaining samples to prevent overflow
                    j = min(n_samples, remaining)
                    remaining = max(0, remaining - n_samples)
                    first_pass = False

                    # PyCBC: to try to speed up next go around, we'll increase the draw
                    # size by the fraction of values that were kept, but cap at n ~1e6~.
                    ndraw = int(min(n, ndraw * np.ceil(ndraw / (n_samples + 1.0))))

                    # assign sample subset to dataframe in memory
                    df[i : i + j] = samples[mask][:j].to_numpy()
                return df
            else:
                # apply constraints but do not resample to return n values
                samples = pd.concat(
                    [dist.draw(n, series=True) for dist in self.distributions], axis=1
                )
                mask = np.all(
                    [constraint(samples) for constraint in self.constraints], axis=0
                )

                samples = samples[mask]
                if self.transforms:
                    for transform in self.transforms:
                        samples = transform.apply(samples)
                return samples

    def __call__(self, n: int, redraw: bool = True) -> pd.DataFrame:
        return self.draw(n, redraw=redraw)

    def __repr__(self):
        """Overrides string representation of cls when printed."""
        distributions = ",\n    ".join([str(dist) for dist in self.distributions])
        str_repr = (
            f"{type(self).__name__}(\n  distributions=(\n    {distributions}\n  )"
        )
        if self.constraints is not None:
            constraints = ",\n    ".join([str(const) for const in self.constraints])
            str_repr += f",\n  constraints=(\n    {constraints}"
            str_repr += "\n  )"
        if self.transforms is not None:
            transforms = ",\n    ".join([str(trans) for trans in self.transforms])
            str_repr += f",\n  transforms=(\n    {transforms}"
            str_repr += "\n  )"
        str_repr += "\n)"
        return str_repr

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, bytes, PathLike],
        distributions: Optional[str] = "distributions",
        transforms: Optional[str] = "transforms",
        constraints: Optional[str] = "constraints",
    ):
        import yaml
        with open(path) as file:
            config = yaml.safe_load(file)

        _distributions = load_distributions_from_config(config, key=distributions)
        if _distributions is None:
            raise KeyError(f"No distributions found in config[{distributions}]")
        _transforms = load_transforms_from_config(config, key=transforms)
        _constraints = load_constraints_from_config(config, key=constraints)

        return cls(
            distributions=tuple(_distributions),
            transforms=tuple(_transforms) if _transforms is not None else None,
            constraints=tuple(_constraints) if _constraints is not None else None,
        )


def load_distributions_from_config(
    config: dict, key: Optional[str] = "distributions"
) -> Optional[List[Distribution]]:
    if key in config:
        distribution_config = deepcopy(config[key])
        for kwargs in distribution_config:
            # assign shared random seed if not present in each distribution config
            if "seed" not in kwargs:
                kwargs["seed"] = int(config["seed"]) if "seed" in config else None

        return [Distribution(**parse_config(kwargs)) for kwargs in distribution_config]
    else:
        logger.info(f"No distributions found in config[{key}].")
        return None


def load_joint_distribution_from_config(
    config: Dict[str, Any],
    distributions: Optional[str] = "distributions",
    transforms: Optional[str] = "transforms",
    constraints: Optional[str] = "constraints",
) -> JointDistribution:
    _distributions = load_distributions_from_config(config, key=distributions)
    if _distributions is None:
        raise KeyError(f"No distributions found in config[{distributions}]")
    assert _distributions is not None  # to prevent mypy linting warning
    _transforms = load_transforms_from_config(config, key=transforms)
    _constraints = load_constraints_from_config(config, key=constraints)

    # we prefer immutable tuples for our JointDistribution to preserve ordering
    return JointDistribution(
        distributions=tuple(_distributions),
        transforms=tuple(_transforms) if _transforms is not None else None,
        constraints=tuple(_constraints) if _constraints is not None else None,
    )
