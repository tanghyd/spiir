import concurrent.futures
import logging
import tempfile
import os
import shutil
from collections.abc import Iterable
from functools import partial
from os import PathLike
from typing import Optional, Union, List

import pandas as pd

# import after postcoh.py for PostcohInspiralTable compatibility
from . import postcoh
from .ligolw import load_ligolw_xmldoc
from ..mp import validate_cpu_count

import ligo.lw.lsctables
from gwpy.table import EventTable


logger = logging.getLogger(__name__)


def load_ligolw_tables(
    paths: Union[str, bytes, PathLike, List[Union[str, bytes, PathLike]]],
    table: str,
    columns: Optional[List[str]] = None,
    ilwdchar_compat: bool = True,
    legacy_postcoh_compat: bool = True,
    nullable: bool = True,
    verbose: bool = False,
    nproc: int = 1,
    df: bool = True,
) -> Union[EventTable, pd.DataFrame]:
    """Loads one or multiple LIGO_LW XML Documents each containing PostcohInspiralTables
    using GWPy and returns a pandas DataFrame object.

    Parameters
    ----------
    paths: str | bytes | PathLike | Iterable[str | bytes | PathLike]
        A path or list of paths to LIGO_LW XML Document(s) each with a postcoh table.
    columns: list[str] | None = None
        A optional list of column names to filter and read in from each postcoh table.
    ilwdchar_compat: bool
        Whether to add ilwdchar conversion compatibility.
    legacy_postcoh_compat: bool
        Whether to handle compatibility for legacy postcoh table formats.
    nullable: bool
        If True, sets the values for missing postcoh columns to NoneType,
        otherwise it is set to an appropriate default value given the column type.
    verbose: bool
        Whether to enable verbose output.
    nproc: int
        Number of CPU processes to use for multiprocessing. Default: 1, recommended: 4.
    df: bool
        If True returns a pd.DataFrame, else returns the original gwpy.table.EventTable.

    Returns
    -------
    pd.DataFrame
    """
    # TODO: Add column filtering in load_ligolw_xmldoc with LIGOLWContentHandler

    # construct keyword arguments for EventTable.read
    nproc = validate_cpu_count(nproc)
    event_table_read_kwargs = dict(
        format="ligolw",
        tablename=table,
        verbose=verbose,
        nproc=nproc,
    )

    # NOTE: if columns is None, EventTable.read fails
    if columns is not None:
        event_table_read_kwargs["columns"] = columns

    if (ilwdchar_compat or legacy_postcoh_compat) and table == "postcoh":
        # strip ilwdchar types and store output in tempfiles, with multiprocessing
        if not isinstance(paths, Iterable):
            paths = [paths]

        try:
            # create temporary files to strip xmldocs before being read in by gwpy
            temp_dir = tempfile.mkdtemp(prefix="temp_postcoh_tables_")
            logger.debug(
                f"Creating temp directory {temp_dir} for ilwd:char conversion."
            )
            with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
                # return objects must be pickle-able so we use file names not file objs
                load_ligolw_xmldoc_into_tempfile = partial(
                    _load_ligolw_xmldoc_into_tempfile,
                    temp_dir=temp_dir,
                    ilwdchar_compat=ilwdchar_compat,
                    legacy_postcoh_compat=legacy_postcoh_compat,
                    nullable=nullable,
                    delete=False,
                    return_filename=True,
                    verbose=verbose,
                )
                files = list(executor.map(load_ligolw_xmldoc_into_tempfile, paths))

        except KeyboardInterrupt as exc:
            shutil.rmtree(temp_dir)
            logging.debug(
                f"Process interrupted - removing temp files under {temp_dir}."
            )
            raise exc

        else:
            # load each temp file into dataframe via gwpy
            event_table = EventTable.read(files, **event_table_read_kwargs)
            shutil.rmtree(temp_dir)
            logger.debug(f"Removing temp files under {temp_dir}.")

    else:
        # no need to do tempfile processing on custom tables
        if table == "postcoh":
            warning_msg = f"Warning! Loading postcoh tables usually requires backward \
                compatibility arguments, but ilwdchar_compat={ilwdchar_compat} and \
                legacy_postcoh_compat={legacy_postcoh_compat}."
            logger.warning(warning_msg)
        event_table = EventTable.read(paths, **event_table_read_kwargs)

    if df:
        return event_table.to_pandas()
    return event_table


def _load_ligolw_xmldoc_into_tempfile(
    path: Union[str, bytes, PathLike],
    temp_dir: Optional[str] = None,
    ilwdchar_compat: bool = True,
    legacy_postcoh_compat: bool = True,
    nullable: bool = True,
    delete: bool = True,
    return_filename: bool = False,
    verbose: bool = True,
) -> Union[tempfile._TemporaryFileWrapper, str]:
    """Loads a file from path to a tempfile.NamedTemporaryFile object. This function
    returns an open file object and most likely must be closed manually after use.

    Parameters
    ----------
    path: str | bytes | PathLike
        Path to file object to be read into a tempfile.
    temp_dir: str | None
        Optional temporary directory to pass to tempfile.NamedTemporaryFile.
    ilwdchar_compat: bool
        Whether to pass ilwdchar_compat forward to the load_ligolw_xmldoc function.
    legacy_postcoh_compat: bool
        Whether to handle compatibility for legacy postcoh table formats.
    nullable: bool
        If True, sets the values for missing postcoh columns to NoneType,
        otherwise it is set to an appropriate default value given the column type.
    delete: bool
        Whether to trigger temporary file deletion when the file is closed.
    return_filename
        If True, returns the name (str) of the tempfile path, else returns the file.
    verbose: bool
        Whether to enable verbose output.

    Returns
    -------
    tempfile._TemporaryFileWrapper | str
        An open temporary file object or a path to a named file object.

    Notes
    -----
    If return_filename = True and delete = True, it may be the case that the temporary
    file object is closed when this function scope ends. This is because we only return
    the file name as a string, and then the delete input argument to the
    tempfile.NamedTemporaryFile class will automatically delete the file when it is
    closed exiting scope. This behaviour has not been thoroughly tested and we warn
    users to check that their temporary file handling is functioning as expected.

    """
    file = tempfile.NamedTemporaryFile(mode="w", dir=temp_dir, delete=delete)
    xmldoc = load_ligolw_xmldoc(
        path, ilwdchar_compat, legacy_postcoh_compat, nullable, verbose
    )
    xmldoc.write(file)
    file.seek(0)
    if return_filename:
        return file.name
    return file
