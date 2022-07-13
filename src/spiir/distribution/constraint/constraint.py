import logging
from functools import partial
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# TODO:
#   - PyCBC defines constraint variable in class
#   - NumPY gets constraint variable at run-time
#   - We should reconsider this mis-match in approach and how _load_constraint works


class Constraint:
    """Abstract Base Class for Constraint classes"""

    package: Optional[str] = None

    def __new__(
        cls,
        package: str = "numpy",
        *args,
        **kwargs,
    ):
        if cls is Constraint:
            # return a Constraint subclass instance
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
            return super(Constraint, subclass).__new__(subclass)
        elif cls in Constraint.__subclasses__():
            # cls is already a Constraint subclass, continue
            return super(Constraint, cls).__new__(cls)
        else:
            raise TypeError(f"{cls} not one of valid {Constraint.__subclasses__()}")

    def __init__(
        self,
        constraint: str,
        variables: Optional[Union[str, Tuple[str, ...]]] = None,
        name: Optional[str] = None,
        package: str = "numpy",
        **kwargs,
    ):
        self.name = name
        self.variables = (variables,) if isinstance(variables, str) else variables
        self.constraint = constraint
        self.kwargs = kwargs

    def __repr__(self):
        """Overrides string representation of cls when printed."""
        repr_attrs = ["constraint", "kwargs"]
        if self.variables is not None:
            repr_attrs.insert(0, "variables")
        if self.name is not None:
            repr_attrs.insert(0, "name")

        kws = [f"{key}={self.__dict__[key]!r}" for key in repr_attrs]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def __call__(self, samples: pd.DataFrame):
        return self.apply(samples)

    def apply(self, samples: pd.DataFrame):
        raise NotImplementedError


class NumPyConstraint(Constraint):
    package = "numpy"

    def __init__(
        self,
        constraint: str,
        variables: Optional[Union[str, Tuple[str, ...]]] = None,
        name: Optional[str] = None,
        package: str = "numpy",
        **kwargs,
    ):
        assert package == getattr(NumPyConstraint, "package")
        super().__init__(constraint, variables, name, package, **kwargs)

        self._constraint = self._load_constraint()

    def _load_constraint(self):
        # return partial(getattr(np, self.constraint), **self.kwargs)
        return getattr(np, self.constraint)

    def __call__(self, samples: pd.DataFrame) -> pd.DataFrame:
        return self.apply(samples)

    def apply(self, samples: pd.DataFrame) -> pd.DataFrame:
        constraint_func = partial(self._constraint, **self.kwargs)
        if self.variables is not None:
            return constraint_func(
                *tuple(
                    samples[var] if isinstance(var, str) else var
                    for var in self.variables
                )
            )
        else:
            logger.warning(
                f"self.variables is None but column indexing variable expected."
            )
            return constraint_func(samples)


# Template
class SPIIRConstraint(Constraint):
    package = "spiir"

    def __init__(
        self,
        constraint: str,
        variables: Optional[Union[str, Tuple[str, ...]]] = None,
        name: Optional[str] = None,
        package: str = "spiir",
        **kwargs,
    ):
        assert package == getattr(SPIIRConstraint, "package")
        super().__init__(constraint, variables, name, package, **kwargs)

        self._load_constraint()

    def _load_constraint(self):
        try:
            return spiir.probability.constraints.constraints[self.constraint]
        except Exception as exc:
            raise exc

    def __call__(self, samples: pd.DataFrame):
        return self.apply(samples)

    def apply(self, samples: pd.DataFrame):
        raise NotImplementedError
