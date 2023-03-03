import functools
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

import click
import numpy as np
import pandas as pd

from spiir.io.ligolw import load_table_from_xmls
from spiir.logging import configure_logger
from spiir.cli import click_logger_options

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)


### Utility Functions ###


def load_table(p: Union[str, Path], table: str, glob: str = "*") -> pd.DataFrame:
    logger.info(f"Reading {table} table data from {p}...")
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"File or directory at {p} not found.")

    paths = sorted(path.glob(glob)) if path.is_dir() else [path]
    df = load_table_from_xmls(paths, table=table, legacy_postcoh_compat=False)
    n, m = len(df), len(df.columns)
    logger.info(f"Loaded {n} rows and {m} columns from {len(paths)} path(s).")
    return df


### Tests ###


def test_df_row_count(a: pd.DataFrame, b: pd.DataFrame):
    assert len(a) == len(b), f"Number of rows do not match."


def test_df_col_count(a: pd.DataFrame, b: pd.DataFrame):
    assert len(a.columns) == len(b.columns), f"Number of columns do not match."


def test_df_col_order(a: pd.DataFrame, b: pd.DataFrame):
    try:
        assert (a.columns == b.columns).all(), f"Columns do not exactly match in order."
    except ValueError as exc:
        raise AssertionError(f"Columns do not match - {exc}")


def test_df_col_exists(a: pd.DataFrame, b: pd.DataFrame):
    a_in_b = np.all([col in b.columns for col in a.columns])
    b_in_a = np.all([col in a.columns for col in b.columns])
    assert a_in_b and b_in_a, f"Columns from A in B? {a_in_b}; from B in A? {b_in_a}"


def test_not_na(a: pd.Series, b: pd.Series):
    assert not a.notna().all() and not b.notna().all()


def test_dtypes_equal(a: pd.Series, b: pd.Series):
    assert a.dtype == b.dtype, f"Data types between columns must match."


def test_diff(a: pd.Series, b: pd.Series):
    if (a != b).any():
        if a.dtype in DTYPES["float"] or b.dtype in DTYPES["float"]:
            # calculate order of magnitude for matching decimal places between a and b
            with np.errstate(divide="ignore", invalid="ignore"):
                decimals = -1 * np.floor(np.log10((a - b).abs()))
            decimals = decimals.replace(np.inf, np.nan).dropna().astype(int)

            if len(decimals) > 0:
                stats = {k: getattr(decimals, k)() for k in ("min", "max", "median")}
                stats_summary = " | ".join([f"{k}: {v}" for k, v in stats.items()])
                err = f"Values do not match up to n decimal places: {stats_summary}"
                raise AssertionError(err)
            else:
                logger.warning(f"[DEBUG] Unknown behaviour: {stats}")
        else:
            raise AssertionError("Values do not match.")


def test_str_equal(a: pd.Series, b: pd.Series):
    assert (a == b).all()


def test_str_equal_case_insensitive(a: pd.Series, b: pd.Series):
    assert (a.str.lower() == b.str.lower()).all()


def test_dtypes(a: pd.Series, b: pd.Series, dtype: str):
    assert a.dtype in DTYPES[dtype] and b.dtypes in DTYPES[dtype]


def test_float_dtypes(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, "float")


def test_int_dtype(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, "int")


def test_str_dtype(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, "str")


### CONFIGURATION ##

DTYPES: Dict[str, Set[np.dtype]] = {
    "float": {np.dtype(np.float32), np.dtype(np.float64)},
    "int": {np.dtype(np.int32), np.dtype(np.int64)},
    "str": {np.dtype(object)},
}

TESTS: Dict[str, List[Callable]] = {
    "df": [test_df_col_count, test_df_col_exists, test_df_col_order],
    "required_df": [test_df_row_count],
    "columns": [test_dtypes_equal],
    "float": [test_diff],
    "int": [test_diff],
    "str": [test_str_equal_case_insensitive, test_str_equal],
}


### Command Line Interface ###


@click.command
@click.argument("a", type=str)
@click.argument("b", type=str)
@click.option("--table", type=str, default="postcoh")
@click.option("-t", "--tests", type=str, multiple=True)
@click.option("-g", "--glob", type=str, default="*zerolag_*.xml*")
@click_logger_options
def main(
    a: str,
    b: str,
    table: str = "postcoh",
    tests: Optional[Union[str, List[str]]] = None,
    glob: str = "*",
    log_level: int = logging.WARNING,
    log_file: str = None,
):
    configure_logger(logger, log_level, log_file)
    duration = time.perf_counter()

    # load two sets of tables from .xml files to compare against eachother
    df_a = load_table(a, table=table, glob=glob)
    df_b = load_table(b, table=table, glob=glob)

    tests = tests or list(TESTS.keys())
    logger.info(f"Running zerolag tests: {tests}")
    required_test_fail = False  # we check required tests before column-wise tests

    for key in tests if isinstance(tests, list) else [tests]:
        if required_test_fail:
            logger.error(f"Required test failed. Aborting...")
            break

        for test in TESTS[key]:
            prefix = f"{key} - {test.__name__}"  # for logging messages

            # dataframe-wide tests
            if "df" in key:
                try:
                    test(df_a, df_b)
                except AssertionError as err:
                    if "required" in key:
                        logger.error(f"{prefix}: critical failure! {err}")
                        required_test_fail = True
                    else:
                        logger.warning(f"{prefix}: failure! {err}")
                else:
                    logger.info(f"{prefix} success!")

            # columnnar tests
            else:
                if key == "columns":
                    cols = df_a.columns.tolist() + df_b.columns.tolist()
                    cols = list(set(cols))  # drop duplicates
                elif key in df_a.columns or key in df_b.columns:
                    cols = [key]
                else:
                    a_cols = df_a.dtypes.loc[df_a.dtypes.isin(DTYPES[key])].index
                    b_cols = df_b.dtypes.loc[df_b.dtypes.isin(DTYPES[key])].index
                    cols = a_cols.tolist() + b_cols.tolist()
                    cols = list(set(cols))  # drop duplicates

                for col in cols:
                    # get column values from dataframe
                    column_values = []
                    for path, df in zip([a, b], [df_a, df_b]):
                        try:
                            column_values.append(df[col])
                        except KeyError:
                            logger.warning(f"{prefix}: {col} not found in {path}!")
                            break
                    else:
                        # run tests for retrieved column values
                        try:
                            test(*column_values)
                        except AssertionError as err:
                            if "required" in key:
                                logger.error(f"{prefix}: {col} critical failure! {err}")
                                required_test_fail = True
                            else:
                                logger.warning(f"{prefix}: {col} failure! {err}")
                        else:
                            logger.info(f"{prefix}: {col} success!")

    duration = time.perf_counter() - duration
    logger.info(f"{Path(__file__).stem} script ran in {duration:.4f} seconds.")


if __name__ == "__main__":
    main()
