import logging
from os import PathLike
from typing import Optional, Union
from tqdm import tqdm

import lal.series
import pandas as pd
import numpy as np

from . import postcoh

import ligo.lw.ligolw
import ligo.lw.array
import ligo.lw.param
import ligo.lw.table
import ligo.lw.lsctables
import ligo.lw.utils

logger = logging.getLogger(__name__)


@ligo.lw.array.use_in
@ligo.lw.param.use_in
@ligo.lw.table.use_in
@ligo.lw.lsctables.use_in
class LIGOLWContentHandler(ligo.lw.ligolw.LIGOLWContentHandler):
    pass


def load_ligolw_xmldoc(
    path: Union[str, bytes, PathLike],
    ilwdchar_compat: bool = True,
    legacy_postcoh_compat: bool = True,
    nullable: bool = False,
    verbose: bool = False,
    contenthandler: Optional[ligo.lw.ligolw.LIGOLWContentHandler] = None,
) -> ligo.lw.ligolw.Document:
    """Reads a valid LIGO_LW XML Document from a file path and returns a dictionary containing
    the complex SNR timeseries arrays associated with each interferometer ('ifo').

    Parameters
    ----------
    path: str | bytes | PathLike
        A path-like to a file containing a valid LIGO_LW XML Document.
    add_epoch_time: bool
        Whether to add the epoch time to each SNR series array for correct timestamps.
    ilwdchar_compat: bool
        Whether to add ilwdchar conversion compatibility.
    legacy_postcoh_compat: bool
        Whether to handle compatibility for legacy postcoh table formats.
    nullable: bool
        If True, sets the values for missing postcoh columns to NoneType,
        otherwise it is set to an appropriate default value given the column type.
    verbose: bool
        Whether to enable verbose output for ligo.lw.utils.load_filename.

    Returns
    -------
    ligo.lw.ligolw.Document
        The loaded LIGO_LW Document object.
    """
    # define XML document parser
    if contenthandler is None:
        contenthandler = LIGOLWContentHandler

    xmldoc = ligo.lw.utils.load_filename(
        path, verbose=verbose, contenthandler=contenthandler
    )

    if ilwdchar_compat:
        xmldoc = strip_ilwdchar(xmldoc)

    if legacy_postcoh_compat:
        xmldoc = postcoh.rename_legacy_postcoh_columns(xmldoc)
        xmldoc = postcoh.include_missing_postcoh_columns(xmldoc, nullable=nullable)

    return xmldoc


def strip_ilwdchar(xmldoc: ligo.lw.ligolw.Element) -> ligo.lw.ligolw.Element:
    """Transforms a document containing tabular data using ilwd:char style row
    IDs to plain integer row IDs. This is used to translate documents in the
    older format for compatibility with the modern version of the LIGO Light
    Weight XML Python library.

    This is a refactor from ligo.lw to handle any ligo.lw.param.Param instances
    as well as ligo.lw.table.Table instances.

    Parameters
    ----------
    xmldoc: ligo.lw.ligo.lw.ligolw.Element
        A valid LIGO_LW XML Document or Element with the required LIGO_LW elements.

    Returns
    -------
    ligo.lw.ligolw.Element
        The same LIGO_LW Document object passed as input with ilwd:char types
        converted to integers.

    Notes
    -----
    The transformation is lossy, and can only be inverted with specific
    knowledge of the structure of the document being processed.  Therefore,
    there is no general implementation of the reverse transformation.
    Applications that require the inverse transformation must implement their
    own algorithm for doing so, specifically for their needs.
    """
    for elem in xmldoc.getElements(
        lambda e: (
            (e.tagName == ligo.lw.table.Table.tagName)
            or (e.tagName == ligo.lw.param.Param.tagName)
        )
    ):
        if elem.tagName == ligo.lw.table.Table.tagName:
            # first strip table names from column names that shouldn't have them
            if elem.Name in ligo.lw.lsctables.TableByName:
                valid_columns = ligo.lw.lsctables.TableByName[elem.Name].validcolumns
                valid_column_map = {
                    ligo.lw.table.Column.ColumnName(name): name
                    for name in valid_columns
                }
                for column in elem.getElementsByTagName(ligo.lw.ligolw.Column.tagName):
                    if column.getAttribute("Name") not in valid_columns:
                        column.setAttribute("Name", valid_column_map[column.Name])

            # convert ilwd:char ids to integers
            idattrs = tuple(
                elem.columnnames[i]
                for i, coltype in enumerate(elem.columntypes)
                if coltype == "ilwd:char"
            )

            if not idattrs:
                continue

            # convert table ilwd:char column values to integers
            for row in elem:
                for attr in idattrs:
                    new_value = getattr(row, attr)
                    if new_value is not None:
                        setattr(row, attr, int(new_value.split(":")[-1]))

            # update the column types
            for attr in idattrs:
                elem.getColumnByName(attr).Type = "int_8s"

        # convert param ilwd:char values to integers
        if elem.tagName == ligo.lw.param.Param.tagName:
            if elem.Type == "ilwd:char":
                value = getattr(elem, "value", None)
                setattr(elem, "Type", "int_8s")
                if value is not None:
                    new_value = int(value.split(":")[-1])
                    setattr(elem, "value", new_value)

    return xmldoc
