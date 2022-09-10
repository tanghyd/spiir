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