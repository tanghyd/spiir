import logging
from functools import partial
from typing import Optional, Union, Any, Dict, Tuple

import numpy as np
import pandas as pd

from .sigmasq import SigmaSqTransform

logger = logging.getLogger(__name__)

# custom transforms dictionary
# transforms = {SigmaSqTransform.transform: SigmaSqTransform}
transforms: Dict[str, Any] = {}

class Transform:
    """Abstract Base Class for Transform classes"""

    package: Optional[str] = None
    transform: Optional[str] = None

    def __new__(
        cls,
        transform: str,
        variables: Optional[Union[str, Tuple[str, ...]]] = None,
        name: Optional[str] = None,
        package: str = "spiir",
        *args,
        **kwargs,
    ):
        if cls is Transform:
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
            return super(Transform, subclass).__new__(subclass)
        elif cls in Transform.__subclasses__():
            # cls is already a Transform subclass, continue
            return super(Transform, cls).__new__(cls, *args, **kwargs)
        else:
            raise TypeError(f"{cls} is invalid Transform subclass or subsubclass.")

    def __init__(
        self,
        transform: str,
        variables: Optional[Union[str, Tuple[str, ...]]] = None,
        name: Optional[str] = None,
        package: str = "spiir",
        **kwargs,
    ):
        self.name = name
        self.variables = (variables,) if isinstance(variables, str) else variables
        self.transform = transform
        self.kwargs = kwargs

    def __repr__(self):
        """Overrides string representation of cls when printed."""
        repr_attrs = ["transform", "kwargs"]
        if self.variables is not None:
            repr_attrs.insert(0, "variables")
        if self.name is not None:
            repr_attrs.insert(0, "name")

        kws = [f"{key}={self.__dict__[key]!r}" for key in repr_attrs]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def __call__(self, data):
        return self.apply(data)

    def apply(self, data):
        raise NotImplementedError


class NumPyTransform(Transform):
    package = "numpy"

    def __init__(
        self,
        transform: str,
        variables: Optional[Union[str, Tuple[str, ...]]] = None,
        name: Optional[str] = None,
        package: str = "numpy",
        **kwargs,
    ):
        assert package == getattr(NumPyTransform, "package")
        super().__init__(transform, variables, name, package, **kwargs)
        self._transform = self._load_transform()

    def _load_transform(self):
        # return partial(getattr(np, self.transform), **self.kwargs)
        return getattr(np, self.transform)

    def __call__(self, samples: pd.DataFrame) -> pd.DataFrame:
        return self.apply(samples)

    def apply(self, samples: pd.DataFrame) -> pd.DataFrame:
        transform_func = partial(self._transform, **self.kwargs)
        if self.variables is not None:
            # NOTE: this may not work for multiple variables and numpy operations.
            #   We would need some test cases that take multiple arrays.
            transformed_samples = transform_func(
                *tuple(
                    samples[var] if isinstance(var, str) else var
                    for var in self.variables
                )
            )

            # TODO: Add tests to verify each of these conditional logic branches.
            # NOTE: Appending to dataframes is slow - can we assign memory first?
            if isinstance(transformed_samples, pd.Series):
                # replace previous variable with transformed samples, or append new col
                samples.loc[:, transformed_samples.name] = transformed_samples

            # NOTE: this branch has not been thoroughly evaluated - we need
            #   a proper numpy test case that would output multiple arrays
            elif isinstance(transformed_samples, pd.DataFrame):
                if any([var in samples.columns for var in transformed_samples.columns]):
                    # if any transformed variable already exists in the sample columns,
                    # then we loop through separately, either overwriting or appending.
                    for var in transformed_samples.columns:
                        samples.loc[:var] = transformed_samples[var]
                else:
                    # none of the multiple new columns are present in samples so concat
                    samples = pd.concat([samples, transformed_samples], axis=1)

                raise TypeError("Transformed samples should be pd.DataFrame or Series.")

            return samples
        else:
            logger.warning(
                f"self.variables is None but column indexing variable expected."
            )
            return transform_func(samples)


class SPIIRTransform(Transform):
    package = "spiir"

    def __init__(
        self,
        transform: str,
        variables: Optional[Union[str, Tuple[str, ...]]] = None,
        name: Optional[str] = None,
        package: str = "spiir",
        **kwargs,
    ):
        assert package == getattr(SPIIRTransform, "package")
        super().__init__(transform, variables, name, package, **kwargs)
        self._transform = self._load_transform()

    def _load_transform(self):
        try:
            # initialise custom SPIIR transformation with keyword arguments
            return transforms[self.transform](**self.kwargs)
        except Exception as exc:
            raise exc

    def __call__(self, samples: pd.DataFrame) -> pd.DataFrame:
        self.apply(samples)

    def apply(self, samples: pd.DataFrame) -> pd.DataFrame:
        return self._transform.apply(samples)
