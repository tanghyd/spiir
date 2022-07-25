import copy
import json
import logging
import os
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

from .constraint import Constraint

logger = logging.getLogger(__name__)


class PyCBCConstraint(Constraint):
    package = "pycbc"

    def __init__(
        self,
        constraint: str,
        variables: Optional[Union[str, Tuple[str, ...]]] = None,
        name: Optional[str] = None,
        package: str = "pycbc",
        **kwargs,
    ):
        """Some description.
        
        Parameters
        ----------
        constraint: str
            A constraint.
        variables: str | tuple[str, ...] | None
            Some variables
        name: str | None
            A name
        package: str
            A package - should be "pycbc".
        
        Methods
        -------
        _load_constraint:
            Some description.
        __call__:
            Some description.
        apply:
            Some description.
        """
        from pycbc.io import record
        from pycbc.transforms import apply_transforms
        from pycbc.distributions.constraints import constraints

        assert package == getattr(PyCBCConstraint, "package")
        super().__init__(constraint, variables, name, package, **kwargs)

        self._constraint = self._load_constraint()

    def _load_constraint(self):
        from spiir.distribution.constraint.pycbc import constraints

        return constraints.constraints[self.constraint](**self.kwargs)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.apply(data)

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return self._constraint(data.to_dict("list"))
        elif isinstance(data, np.recarray):
            from pycbc.io.record import FieldArray

            return self._constraint(FieldArray.from_arrays(data.to_records()))
        else:
            return self._constraint(data)


class MassConstraint:
    constraint = "mass"

    def __init__(self, constraint_arg=None, transforms=None, **kwargs):
        """
        Custom pycbc.distributions.constraints.Constraint object that evaluates
        to True if mass parameters (mass1 and mass2) obey the conventional notation
        where mass1 >= mass2.

        Methods
        -------
        __call__:
            Sample description.
        _constraint:
            Sample description.
        """
        self.constraint_arg = constraint_arg
        self.transforms = transforms
        for kwarg in kwargs.keys():
            setattr(self, kwarg, kwargs[kwarg])

    def __call__(self, params):
        """Evaluates constraint."""
        # cast to FieldArray
        if isinstance(params, dict):
            params = record.FieldArray.from_kwargs(**params)
        elif not isinstance(params, record.FieldArray):
            raise ValueError("params must be dict or FieldArray instance")

        # try to evaluate;
        # this assumes all of the needed parameters
        # for the constraint exists in params
        try:
            out = self._constraint(params)
        except NameError:
            # one or more needed parameters don't exist; try applying the transforms
            params = (
                apply_transforms(params, self.transforms) if self.transforms else params
            )
            out = self._constraint(params)
        if isinstance(out, record.FieldArray):
            out = out.item() if params.size == 1 else out
        return out

    def _constraint(self, params):
        """
        Evaluates constraint function.

        Warning: Requires priors to be specified as mass_1 and mass_2 in the PyCBC ini config file.
        """
        return params["mass_1"] >= params["mass_2"]


# add custom mass constraint to be read in by
# pycbc.distributions.read_constraints_from_config
constraints = {MassConstraint.constraint: MassConstraint}


# from pycbc.distributions import (
#     JointDistribution,
#     constraints,
#     read_constraints_from_config,
#     read_distributions_from_config,
#     read_params_from_config,
# )
# from pycbc.transforms import read_transforms_from_config
# from pycbc.workflow import WorkflowConfigParser

# from spiir.utils.types import StrPath

# class PyCBCParameterGenerator:
#     def __init__(
#         self,
#         config: StrPath | list[StrPath],
#         seed: int | None = None,
#     ):
#         """Class to generate CBC waveform parameters using PyCBC workflow and distribution packages."""

#         if seed is not None:
#             raise NotImplementedError("Reproducible random seed not yet implemented.")

#         self.load_config(config)

#     def load_config(self, config: StrPath | list[StrPath]):
#         """Loads a PyCBC JointDistribution object from a PyCBC style config.ini file."""

#         self.config = WorkflowConfigParser(configFiles=config)
#         self.parameters, self.static_args = read_params_from_config(self.config)
#         self._constraints = read_constraints_from_config(self.config)
#         self._transforms = read_transforms_from_config(self.config)
#         self.distribution = JointDistribution(
#             self.parameters,
#             *read_distributions_from_config(self.config),
#             **{"constraints": self._constraints},
#         )
#         self._func = lambda n: apply_transforms(self.distribution.rvs(size=n), self._transforms)

#     def draw(self, n: int = 1) -> np.record:
#         """
#         Draw a sample from the joint distribution and constructs a dictionary that maps
#         the parameternames to the values generated for them.

#         Parameters
#         ----------
#         n: int
#             The number of samples to draw.

#         Returns
#         -------
#         dict
#             A dictionary where keys are variable names and values are arrays of samples.
#         """
#         assert n >= 1, "n must be a positive integer."
#         return self._func(n)


# def amend_static_args(static_args: dict[str, str]) -> dict:
#     """
#     Amend the static_args from the `*.ini` configuration file by adding
#     the parameters that can be computed directly from others (more
#     intuitive ones). Note that the static_args should have been
#     properly typecast first; see :func:`typecast_static_args()`.

#     Parameters
#     ----------
#     static_args: dict
#         The static_args dict after it has been typecast by :func:`typecast_static_args()`.

#     Returns
#     -------
#         The amended `static_args`, where implicitly defined variables have been added.
#     """

#     # to do - automatic values if seconds before/after aren't provided?
#     # to do - handle sample_length vs. waveform_length

#     # Create a copy of the original static_args
#     args: dict = copy.deepcopy(static_args)

#     # If necessary, compute the sample length
#     if "sample_length" not in args.keys():
#         args["sample_length"] = args["seconds_before_event"] + args["seconds_after_event"]

#     # If necessary, add delta_t = 1 / target_sampling_rate
#     if "delta_t" not in args.keys():
#         args["delta_t"] = 1.0 / float(args["target_sampling_rate"])

#     # If necessary, add delta_f = 1 / waveform_length
#     if "delta_f" not in args.keys():
#         args["delta_f"] = 1.0 / float(args["waveform_length"])

#     # If necessary, add td_length = waveform_length * target_sampling_rate
#     if "td_length" not in args.keys():
#         args["td_length"] = int(args["waveform_length"] * float(args["target_sampling_rate"]))

#     # If necessary, add fd_length = td_length / 2 + 1
#     if "fd_length" not in args.keys():
#         if "f_final" in args.keys():
#             args["fd_length"] = int(float(args["f_final"]) / float(args["delta_f"])) + 1
#         else:
#             args["fd_length"] = int(float(args["td_length"]) / 2.0 + 1)

#     return args


# def typecast_static_args(static_args: dict[str, str]) -> dict:
#     """
#     Take the `static_args` dictionary as it is read in from the PyCBC
#     configuration file (i.e., all values are strings) and cast the
#     values to the correct types (`float` or `int`).

#     Args:
#         static_args (dict): The raw `static_args` dictionary as it is
#             read from the `*.ini` configuration file.

#     Returns:
#         The `static_args` dictionary with proper types for all values.
#     """

#     # list variables that must be casted to integers
#     # note: whitening_segment_duration was previously a float
#     int_args = [
#         "bandpass_lower",
#         "bandpass_upper",
#         "waveform_length",
#         "noise_interval_width",
#         "original_sampling_rate",
#         "target_sampling_rate",
#         "whitening_segment_duration",
#         "whitening_max_filter_duration",
#     ]

#     # list variables that must be casted to floats
#     float_args = [
#         "distance",
#         "f_lower",
#         "seconds_before_event",
#         "seconds_after_event",
#         "whitening_segment_duration",
#     ]

#     # copy dictionary with type cast conversions
#     args: dict = copy.deepcopy(static_args)

#     for float_arg in float_args:
#         if float_arg in args:
#             args[float_arg] = float(args[float_arg])

#     for int_arg in int_args:
#         if int_arg in args:
#             args[int_arg] = float(args[int_arg])

#     return args


# def read_ini_config(file_path: StrPath) -> tuple[dict, dict]:
#     """
#     Read in a `*.ini` config file, which is used mostly to specify the
#     waveform simulation (for example, the waveform model, the parameter
#     space for the binary black holes, etc.) and return its contents.

#     Returns a tuple `(variable_arguments, static_arguments)` where
#         - `variable_arguments` should simply be a list of all the
#             parameters which get randomly sampled from the specified
#             distributions, usually using an instance of
#             :class:`spiir.probability.pycbc.PyCBCParameterGenerator`.
#         - `static_arguments` should be a dictionary containing the keys
#             and values of the parameters that are the same for each
#             example that is generated (i.e., the non-physical parameters
#             such as the waveform model and the sampling rate).

#     Parameters
#     ----------
#     file_path: str
#         Path to the `*.ini` config file to be read in.

#     Returns
#     -------
#     tuple[dict, dict]
#     """

#     from pycbc.distributions import read_params_from_config
#     from pycbc.workflow import WorkflowConfigParser

#     # Make sure the config file actually exists
#     if not os.path.exists(file_path):
#         raise IOError(f"Specified configuration file does not exist: {file_path}")

#     # Set up a parser for the PyCBC config file
#     config_parser = WorkflowConfigParser(configFiles=[file_path])

#     # Read the variable_arguments and static_arguments using the parser
#     variable_arguments, static_arguments = read_params_from_config(config_parser)

#     # Typecast and amend the static arguments
#     static_arguments = typecast_static_args(static_arguments)
#     static_arguments = amend_static_args(static_arguments)

#     return variable_arguments, static_arguments


# def read_json_config(file_path: StrPath) -> tuple[dict, dict]:
#     """
#     Read in a `*.json` config file, which is used to specify the
#     sample generation process itself (for example, the number of
#     samples to generate, the number of concurrent processes to use,
#     etc.) and return its contents.

#     Args:
#         file_path: Union[str, os.PathLike]
#             Path to the `*.json` config file to be read in.

#     Returns:
#         A `dict` containing the contents of the given JSON file.
#     """

#     # Make sure the config file actually exists
#     if not os.path.exists(file_path):
#         raise IOError(f"Specified configuration file does not exist: {file_path}")

#     # Open the config while and load the JSON contents as a dict
#     with open(file_path, "r") as json_file:
#         config = json.load(json_file)

#     # Define the required keys for the config file in a set
#     required_keys = {
#         "background_data_directory",
#         "dq_bits",
#         "inj_bits",
#         "waveform_params_file_name",
#         "max_runtime",
#         "n_injection_samples",
#         "n_noise_samples",
#         "n_processes",
#         "random_seed",
#         "output_file_name",
#         "n_template_samples",
#     }

#     # Make sure no required keys are missing
#     missing_keys = required_keys.difference(set(config.keys()))
#     if missing_keys:
#         raise KeyError(
#             "Missing required key(s) in JSON configuration file: "
#             "{}".format(", ".join(list(missing_keys)))
#         )

#     return config
