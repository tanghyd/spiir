import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


config_parser_map = {
    "np.pi": np.pi,
    "np.inf": np.inf,
    "-np.inf": -np.inf,
}


def parse_config(
    kwargs: Dict[str, Any], config_parser: Dict[str, Any] = config_parser_map
) -> Dict[str, Any]:
    """Function converts values in dictionary to corresponding function
    representations according to a specified parser mapping in-place.

    NOTE: Only works at one level of depth for a given config dictionary.
    TODO: Enable parsing for non-string types (i.e. list) and arbitrary depths.
    """
    for key in kwargs:
        try:
            if kwargs[key] in config_parser:
                # parse strings to function representations for pi, inf, etc.
                kwargs[key] = config_parser[kwargs[key]]
        except TypeError as exc:
            logging.debug(f"key {key} ({type(kwargs[key])}) not parsed, see: {exc}.")
    return kwargs


<<<<<<< HEAD
# def load_distributions_from_config(
#     config: dict, key: Optional[str] = "distributions"
# ) -> Optional[List[Distribution]]:
#     if key in config:
#         distribution_config = deepcopy(config[key])
#         for kwargs in distribution_config:
#             # assign shared random seed if not present in each distribution config
#             if "seed" not in kwargs:
#                 kwargs["seed"] = int(config["seed"]) if "seed" in config else None

#         return [Distribution(**parse_config(kwargs)) for kwargs in distribution_config]
#     else:
#         logger.info(f"No distributions found in config[{key}].")
#         return None


# def load_transforms_from_config(
#     config: Dict[str, Any], key: Optional[str] = "transforms"
# ) -> Optional[List[Transform]]:
#     if key in config:
#         return [Transform(**parse_config(kwargs)) for kwargs in config[key]]
#     else:
#         logger.info(f"No transforms found in config[{key}].")
#         return None


# def load_constraints_from_config(
#     config: Dict[str, Any], key: Optional[str] = "constraints"
# ) -> Optional[List[Constraint]]:
#     if key in config:
#         return [Constraint(**parse_config(kwargs)) for kwargs in config[key]]
#     else:
#         logger.info(f"No constraints found in config[{key}].")
#         return None


# def load_joint_distribution_from_config(
#     config: Dict[str, Any],
#     distributions: Optional[str] = "distributions",
#     transforms: Optional[str] = "transforms",
#     constraints: Optional[str] = "constraints",
# ) -> JointDistribution:
#     _distributions = load_distributions_from_config(config, key=distributions)
#     logger.debug(f"_distributions: {_distributions}")
#     if _distributions is None:
#         raise KeyError(f"No distributions found in config[{distributions}]")
#     assert _distributions is not None  # to prevent mypy linting warning
#     _transforms = load_transforms_from_config(config, key=transforms)
#     _constraints = load_constraints_from_config(config, key=constraints)

#     # we prefer immutable tuples for our JointDistribution to preserve ordering
#     return JointDistribution(
#         distributions=tuple(_distributions),
#         transforms=tuple(_transforms) if _transforms is not None else None,
#         constraints=tuple(_constraints) if _constraints is not None else None,
#     )


# def parse_pycbc_config(path: Union[str, bytes, PathLike]) -> Dict[str, Any]:
#     raise NotImplementedError
=======
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


def load_transforms_from_config(
    config: Dict[str, Any], key: Optional[str] = "transforms"
) -> Optional[List[Transform]]:
    if key in config:
        return [Transform(**parse_config(kwargs)) for kwargs in config[key]]
    else:
        logger.info(f"No transforms found in config[{key}].")
        return None


def load_constraints_from_config(
    config: Dict[str, Any], key: Optional[str] = "constraints"
) -> Optional[List[Constraint]]:
    if key in config:
        return [Constraint(**parse_config(kwargs)) for kwargs in config[key]]
    else:
        logger.info(f"No constraints found in config[{key}].")
        return None


def load_joint_distribution_from_config(
    config: Dict[str, Any],
    distributions: Optional[str] = "distributions",
    transforms: Optional[str] = "transforms",
    constraints: Optional[str] = "constraints",
) -> JointDistribution:
    _distributions = load_distributions_from_config(config, key=distributions)
    logger.debug(f"_distributions: {_distributions}")
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


def parse_pycbc_config(path: Union[str, bytes, PathLike]) -> Dict[str, Any]:
    raise NotImplementedError
>>>>>>> 9c2888a... Expand project structure template
