"""Utilities for NumPy array and pandas Series processing."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_unique_index_diff(index: pd.Index, precision: Optional[int] = None):
    """Convenience function to retrieve a unique index diff value from a pd.Index."""
    if not isinstance(index, pd.Series):
        index = index.to_series()
    diff = index.diff().dropna()
    if precision is not None:
        diff = diff.round(precision)
    diff = diff.unique()
    if len(diff) == 1:
        return diff[0]
    else:
        raise RuntimeError(
            "Cannot automatically determine unique index diff from index, "
            "maybe due to varied delta values, NA values, or precision errors."
        )
