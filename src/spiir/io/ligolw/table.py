import concurrent.futures
import logging
from functools import partial
from os import PathLike
from typing import Callable, Dict, Optional, Sequence, Union

import ligo.lw.table
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..mp import validate_cpu_count

# import after postcoh.py for PostcohInspiralTable compatibility
from .ligolw import _NUMPY_TYPE_MAP, get_ligolw_element, load_ligolw_xmldoc

logger = logging.getLogger(__name__)


def build_dataframe_from_table(
    table: ligo.lw.table.Table,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Reads a ligo.lw.table.Table object and returns as its corresponding pd.DataFrame.

    Parameters
    ----------
    table: ligo.lw.table.Table
        A LIGO_LW Table object instance loaded from a LIGO_LW XML Document.

    Returns
    -------
    pd.DataFrame
        The Table as a pd.DataFrame with appropriate data, types, and column names.

    """

    def col_to_array(col: ligo.lw.table.Column) -> np.ndarray:
        try:
            dtype = _NUMPY_TYPE_MAP[col.Type]
        except KeyError as e:
            raise TypeError(f"Cannot determine numpy dtype for Column {col.Name}: {e}")
        try:
            return np.fromiter(col, dtype=dtype)
        except TypeError as exc:
            logger.debug(f"Set {col.Name} dtype as object (None) for TypeError: {exc}.")
            return np.fromiter(col, dtype=object)

    cols = columns or table.columnnames
    return pd.DataFrame({col: col_to_array(table.getColumnByName(col)) for col in cols})


def get_table_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Document,
    table: str,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Convenience function to get one table from one LIGO_LW xmldoc element.

    This is a essentially a combined wrapper around the ligo.lw.table.Table.get_table
    class method and our dataframe conversion function for convenience.

    Parameters
    ----------
    xmldoc: ligo.lw.ligolw.Document
        A LIGO_LW XML Document element that contains Table element(s).
    table: str
        The name of the Table element to retrieve.
    columns: Sequence[str] | None
        If provided, retrieve the specified columns by their name.
        Otherwise if columns is None, retrieve all columns from the Table element.

    Returns
    -------
    pd.DataFrame
        The Table as a pd.DataFrame with appropriate data, types, and column names.

    """
    table_element = ligo.lw.table.Table.get_table(xmldoc, table)
    return build_dataframe_from_table(table_element, columns)


def load_table_from_xml(
    path: Union[str, bytes, PathLike],
    table: str,
    columns: Optional[Sequence[str]] = None,
    ilwdchar_compat: bool = True,
    legacy_postcoh_compat: bool = True,
    nullable: bool = False,
) -> pd.DataFrame:
    """Loads one table from one LIGO_LW XML file path and returns a pd.DataFrame.

    Parameters
    ----------
    path: str | bytes | PathLike | Sequence[str | bytes | PathLike]
        A path or list of paths to LIGO_LW XML Document(s) each with a postcoh table.
    table: str
        The name of the Table element to retrieve.
    columns: Sequence[str] | None = None
        A optional list of column names to filter and read in from each postcoh table.
    ilwdchar_compat: bool
        Whether to add ilwdchar conversion compatibility.
    legacy_postcoh_compat: bool
        Whether to handle compatibility for legacy postcoh table formats.
    nullable: bool
        If True, sets the values for missing postcoh columns to NoneType,
        otherwise it is set to an appropriate default value given the column type.

    Returns
    -------
    pd.DataFrame
        The Table as a pd.DataFrame with appropriate data, types, and column names.
    """
    xmldoc = load_ligolw_xmldoc(
        path,
        ilwdchar_compat=ilwdchar_compat,
        legacy_postcoh_compat=legacy_postcoh_compat,
        nullable=nullable,
    )

    return get_table_from_xmldoc(xmldoc, table, columns)


def load_table_from_xmls(
    paths: Union[str, bytes, PathLike, Sequence[Union[str, bytes, PathLike]]],
    table: str,
    columns: Optional[Sequence[str]] = None,
    ilwdchar_compat: bool = True,
    legacy_postcoh_compat: bool = True,
    nullable: bool = False,
    concat: bool = True,
    skip_exceptions: bool = False,
    nproc: int = 1,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> pd.DataFrame:
    """Loads a table from one or multiple LIGO_LW XML and returns a pandas DataFrame.

    Note that each XML file must have the same table present in each document.

    Parameters
    ----------
    paths: str | bytes | PathLike | Sequence[str | bytes | PathLike]
        A path or list of paths to LIGO_LW XML Document(s) each with a postcoh table.
    table: str
        The name of the Table element to retrieve.
    columns: Sequence[str] | None = None
        A optional list of column names to filter and read in from each postcoh table.
    ilwdchar_compat: bool
        Whether to add ilwdchar conversion compatibility.
    legacy_postcoh_compat: bool
        Whether to handle compatibility for legacy postcoh table formats.
    nullable: bool
        If True, sets the values for missing postcoh columns to NoneType,
        otherwise it is set to an appropriate default value given the column type.
    concat: bool
        If True, concatenates DataFrames together ignoring the index, else returns a
        list of DataFrames for each table in order.
    skip_exceptions: bool
        If True, exceptions raised by a process during mulitprocessing will be ignored.
    nproc: int
        Number of CPU processes to use for multiprocessing. Default: 1, recommended: 4.
    verbose: bool
        If True, displays a loading progress bar.
    desc: str | None
        A string description for the progress bar.

    Returns
    -------
    pd.DataFrame
        The Table as a pd.DataFrame with appropriate data, types, and column names.
    """
    if isinstance(paths, (str, bytes, PathLike)):
        return load_table_from_xml(
            paths,
            table=table,
            columns=columns,
            ilwdchar_compat=ilwdchar_compat,
            legacy_postcoh_compat=legacy_postcoh_compat,
            nullable=nullable,
        )
    else:
        if len(paths) == 0:
            raise IndexError("No paths provided as len(paths) == 0.")
        nproc = min(validate_cpu_count(nproc), len(paths))
        _load_table_from_xml: Callable = partial(
            load_table_from_xml,
            table=table,
            columns=columns,
            ilwdchar_compat=ilwdchar_compat,
            legacy_postcoh_compat=legacy_postcoh_compat,
            nullable=nullable,
        )

        desc = desc or f"Loading {table} tables from LIGOLW XML files"
        with tqdm(total=len(paths), desc=desc, disable=not verbose) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
                futures = [executor.submit(_load_table_from_xml, p) for p in paths]

                results = []
                for future in concurrent.futures.as_completed(futures):
                    if future.exception() is not None:
                        if skip_exceptions:
                            logger.debug(f"Exception raised: {future.exception()}")
                        else:
                            raise Exception from future.exception()
                    else:
                        results.append(future.result())
                    pbar.update(1)

        if concat:
            return pd.concat(results, ignore_index=True)
        return results


def get_tables_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Document,
    tables: Optional[
        Union[str, Sequence[str], Dict[str, Optional[Sequence[str]]]]
    ] = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience function to get one or more tables from one LIGO_LW xmldoc element.

    This function can retrieve multiple tables from a loaded LIGO_LW xmldoc element if
    the tables parameter is specified as either a sequence (e.g. list) or dictionary,
    where specific columns can be loaded by specifying them in the dictionary values.

    The function return signature is a dictionary of DataFrames, where each key
    corresponds to the table name(s) specified by the input.

    Parameters
    ----------
    xmldoc: ligo.lw.ligolw.Document
        A LIGO_LW XML Document element that contains Table element(s).
    tables: str | Sequence[str] | dict[str, Sequence[str] | None] | None
        The name(s) of the Table element to retrieve. If tables is a str, we retrieve
        one table with that given name if it exists in the xmldoc. If tables is a
        Sequence[str] (i.e. a list or tuple of strings), then we load each of the
        specified tables. If tables is a dict, we retrieve each table name specified in
        the dictionary keys where each value corresponds to the list of column names we
        wish to load for each table, loading all table if the value is None. If tables
        itself is None, then we load all table elements present in the xmldoc.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary where keys are table names and values are tables as a pd.DataFrame.
    """
    # return all tables in xmldoc
    if tables is None:
        return {
            tbl.Name: build_dataframe_from_table(tbl)
            for tbl in xmldoc.getElements(
                lambda elem: elem.tagName == ligo.lw.table.Table.tagName
            )
        }

    # load one table with all columns
    elif isinstance(tables, str):
        return {tables: get_table_from_xmldoc(xmldoc, tables)}

    # load one or more tables with user-specified columns, or all columns (None)
    elif isinstance(tables, dict):
        return {tbl: get_table_from_xmldoc(xmldoc, tbl, c) for tbl, c in tables.items()}

    # load multiple tables with all columns
    else:
        return {tbl: get_table_from_xmldoc(xmldoc, tbl) for tbl in tables}


def load_tables_from_xml(
    path: Union[str, bytes, PathLike],
    tables: Optional[
        Union[str, Sequence[str], Dict[str, Optional[Sequence[str]]]]
    ] = None,
    ilwdchar_compat: bool = True,
    legacy_postcoh_compat: bool = True,
    nullable: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Convenience function to get one or more tables from one LIGO_LW xmldoc element.

    This function can retrieve multiple tables from a loaded LIGO_LW xmldoc element if
    the tables parameter is specified as either a sequence (e.g. list) or dictionary,
    where specific columns can be loaded by specifying them in the dictionary values.

    The function return signature is a dictionary of DataFrames, where each key
    corresponds to the table name(s) specified by the input.

    Parameters
    ----------
    path: str | bytes | PathLike | Sequence[str | bytes | PathLike]
        A path or list of paths to LIGO_LW XML Document(s) each with a postcoh table.
    tables: str | Sequence[str] | dict[str, Sequence[str] | None] | None
        The name(s) of the Table element to retrieve. If tables is a str, we retrieve
        one table with that given name if it exists in the xmldoc. If tables is a
        Sequence[str] (i.e. a list or tuple of strings), then we load each of the
        specified tables. If tables is a dict, we retrieve each table name specified in
        the dictionary keys where each value corresponds to the list of column names we
        wish to load for each table, loading all table if the value is None. If tables
        itself is None, then we load all table elements present in the xmldoc.
    ilwdchar_compat: bool
        Whether to add ilwdchar conversion compatibility.
    legacy_postcoh_compat: bool
        Whether to handle compatibility for legacy postcoh table formats.
    nullable: bool
        If True, sets the values for missing postcoh columns to NoneType,
        otherwise it is set to an appropriate default value given the column type.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary where keys are table names and values are tables as a pd.DataFrame.
    """
    xmldoc = load_ligolw_xmldoc(
        path,
        ilwdchar_compat=ilwdchar_compat,
        legacy_postcoh_compat=legacy_postcoh_compat,
        nullable=nullable,
    )

    return get_tables_from_xmldoc(xmldoc, tables)


def load_tables_from_xmls(
    paths: Union[str, bytes, PathLike, Sequence[Union[str, bytes, PathLike]]],
    tables: Optional[
        Union[str, Sequence[str], Dict[str, Optional[Sequence[str]]]]
    ] = None,
    ilwdchar_compat: bool = True,
    legacy_postcoh_compat: bool = True,
    nullable: bool = False,
    concat: bool = True,
    skip_exceptions: bool = False,
    nproc: int = 1,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience function to get one or more tables from one LIGO_LW xmldoc element.

    This function can retrieve multiple tables from a loaded LIGO_LW xmldoc element if
    the tables parameter is specified as either a sequence (e.g. list) or dictionary,
    where specific columns can be loaded by specifying them in the dictionary values.

    The function return signature is a dictionary of DataFrames, where each key
    corresponds to the table name(s) specified by the input.

    Note that each XML file must have the same set of tables present in each document.

    Parameters
    ----------
    paths: str | bytes | PathLike | Sequence[str | bytes | PathLike]
        A path or list of paths to LIGO_LW XML Document(s) each with a postcoh table.
    tables: str | Sequence[str] | dict[str, Sequence[str] | None] | None
        The name(s) of the Table element to retrieve. If tables is a str, we retrieve
        one table with that given name if it exists in the xmldoc. If tables is a
        Sequence[str] (i.e. a list or tuple of strings), then we load each of the
        specified tables. If tables is a dict, we retrieve each table name specified in
        the dictionary keys where each value corresponds to the list of column names we
        wish to load for each table, loading all table if the value is None. If tables
        itself is None, then we load all table elements present in the xmldoc.
    ilwdchar_compat: bool
        Whether to add ilwdchar conversion compatibility.
    legacy_postcoh_compat: bool
        Whether to handle compatibility for legacy postcoh table formats.
    nullable: bool
        If True, sets the values for missing postcoh columns to NoneType,
        otherwise it is set to an appropriate default value given the column type.
    concat: bool
        If True, concatenates DataFrames together ignoring the index, else returns a
        list of DataFrames for each table in order.
    skip_exceptions: bool
        If True, exceptions raised by a process during mulitprocessing will be ignored.
    nproc: int
        Number of CPU processes to use for multiprocessing. Default: 1, recommended: 4.
    verbose: bool
        If True, displays a loading progress bar.
    desc: str | None
        A string description for the progress bar.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary where keys are table names and values are tables as a pd.DataFrame.
    """
    if isinstance(paths, (str, bytes, PathLike)):
        return load_tables_from_xml(
            paths,
            tables=tables,
            ilwdchar_compat=ilwdchar_compat,
            legacy_postcoh_compat=legacy_postcoh_compat,
            nullable=nullable,
        )
    else:
        if len(paths) == 0:
            raise IndexError("No paths provided as len(paths) == 0.")
        nproc = min(validate_cpu_count(nproc), len(paths))
        _load_tables_from_xml: Callable = partial(
            load_tables_from_xml,
            tables=tables,
            ilwdchar_compat=ilwdchar_compat,
            legacy_postcoh_compat=legacy_postcoh_compat,
            nullable=nullable,
        )

        desc = desc or f"Loading {tables} tables from LIGOLW XML files"
        with tqdm(total=len(paths), desc=desc, disable=not verbose) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
                futures = [executor.submit(_load_tables_from_xml, p) for p in paths]

                results = []
                for future in concurrent.futures.as_completed(futures):
                    if future.exception() is not None:
                        if skip_exceptions:
                            logger.debug(f"Exception raised: {future.exception()}")
                        else:
                            raise Exception from future.exception()
                    else:
                        results.append(future.result())
                    pbar.update(1)

                # every dict in results should have the same keys so this should work
                if concat:
                    return {
                        k: pd.concat((data[k] for data in results), ignore_index=True)
                        for k in results[0].keys()
                    }
                else:
                    return {k: [data[k] for data in results] for k in results[0].keys()}


def build_table_element(df: pd.DataFrame, table: str):
    """Builds an LIGO_LW lsctable XML element by table name from a DataFrame."""
    tbl = ligo.lw.lsctables.New(ligo.lw.lsctables.TableByName[table])
    for _, row in df.sort_index(axis=1).iterrows():
        tbl.append(ligo.lw.table.Table.RowType(**row.to_dict()))
    return tbl


def append_table_to_ligolw(
    xmldoc: ligo.lw.ligolw.Document,
    table: ligo.lw.table.Table,
    overwrite: bool = False,
) -> ligo.lw.ligolw.Document:
    """Writes the given LIGO_LW table into a LIGO_LW XML document.

    This code is sourced from gwpy.io.table.py.

    Parameters
    ----------
    xmldoc : :class:`~ligo.lw.ligolw.Document`
        the document to write into
    tables : `list` of :class:`~ligo.lw.table.Table`
        the set of tables to write
    overwrite : `bool`, optional, default: `False`
        if `True`, delete an existing instance of the table type, otherwise
        append new rows.
    """
    try:
        llw = get_ligolw_element(xmldoc)
    except ValueError:
        llw = ligo.lw.ligolw.LIGO_LW()
        xmldoc.appendChild(llw)

    try:  # append new data to existing table
        lsctable = ligo.lw.lsctables.TableByName[table.TableName(table.Name)]
        existing_table = lsctable.get_table(xmldoc)
    except ValueError:  # or create a new table
        llw.appendChild(table)
    else:
        if overwrite:
            llw.removeChild(existing_table)
            existing_table.unlink()
            llw.appendChild(table)
        else:
            existing_table.extend(table)

    return xmldoc


def load_ligolw_tables(
    paths: Union[str, bytes, PathLike, Sequence[Union[str, bytes, PathLike]]],
    table: str,
    columns: Optional[Sequence[str]] = None,
    ilwdchar_compat: bool = True,
    legacy_postcoh_compat: bool = True,
    nullable: bool = False,
    nproc: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    logger.warning(
        "spiir.io.ligolw.table deprecation warning: load_ligolw_tables will be "
        "deprecated in a future release. Please use load_table_from_xmls instead."
    )
    load_table_from_xmls(
        paths,
        table,
        columns,
        ilwdchar_compat,
        legacy_postcoh_compat,
        nullable,
        nproc=nproc,
        verbose=verbose,
    )
