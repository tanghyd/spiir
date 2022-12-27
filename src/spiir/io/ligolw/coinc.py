"""Module to handle reading and writing standard LIGO_LW candidate event files."""

import logging
from os import PathLike
from typing import Dict, Optional, Union

import ligo.lw.ligolw
import pandas as pd

from .array import (
    append_psd_series_to_ligolw,
    append_snr_series_to_ligolw,
    build_psd_series_from_xmldoc,
    build_snr_series_from_xmldoc,
)
from .ligolw import get_ligolw_element, load_ligolw_xmldoc
from .param import append_p_astro_to_ligolw, get_p_astro_from_xmldoc
from .table import append_table_to_ligolw, build_table_element, get_tables_from_xmldoc

logger = logging.getLogger(__name__)


def save_coinc_xml(
    path: Union[str, bytes, PathLike],
    tables: Dict[str, pd.DataFrame],
    psds: Optional[Dict[str, pd.Series]] = None,
    snrs: Optional[Dict[str, pd.Series]] = None,
    p_astro: Optional[Dict[str, float]] = None,
):
    """Loads the data from a standard coinc.xml LIGO_LW XML file.

    This function returns the standard tables, arrays, and parameters that are
    expected in the coinc.xml and returns them as either a pd.DataFrame (tables), a
    pd.Series (LIGO_LW Series arrays, such as PSDs and SNRs), or a dictionary (p_astro
    probabilites), if they exist in the file.

    Parameters
    ----------
    path: str | bytes | os.PathLike
        The path to write the coinc.xml file.
    tables: dict[str, pd.DataFrame]
        A dictionary containing each of the LIGO_LW XML table elements.
    psds: dict[str, pd.Series] | None
        An optional dictionary of pd.Series corresponding to the PSDs for each ifo.
    snrs: dict[str, pd.Series] | None
        An optional dictionary of pd.Series corresponding to the SNRs for each ifo.
    p_astro: dict[str, float] | None
        An optional dictionary of source probabilities for the coincident inspiral.

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
        A dictionary corresponding to the LIGO_LW data elements present in the file.
    """
    # TODO: Check if all required tables are present
    xmldoc = ligo.lw.ligolw.Document()
    for table in tables:
        tbl = build_table_element(tables[table], table)
        append_table_to_ligolw(xmldoc, tbl)

    llw = get_ligolw_element(xmldoc)
    if psds is not None:
        append_psd_series_to_ligolw(llw, psds)
    if snrs is not None:
        for ifo, snr in snrs.items():
            append_snr_series_to_ligolw(llw, snr, ifo)
    if p_astro is not None:
        append_p_astro_to_ligolw(llw, p_astro)

    with open(path, mode="w") as f:
        xmldoc.write(f)


def load_coinc_xml(
    path: Union[str, bytes, PathLike],
) -> Dict[str, Union[pd.DataFrame, pd.Series, Dict[str, float]]]:
    """Loads the data from a standard coinc.xml LIGO_LW XML file.

    This function returns the standard tables, arrays, and parameters that are
    expected in the coinc.xml and returns them as either a pd.DataFrame (tables), a
    pd.Series (LIGO_LW Series arrays, such as PSDs and SNRs), or a dictionary (p_astro
    probabilites), if they exist in the file.

    Parameters
    ----------
    path: str | bytes | os.PathLike
        The path to the coinc.xml file.

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
        A dictionary corresponding to the LIGO_LW data elements present in the file.
    """
    xmldoc = load_ligolw_xmldoc(path)
    data = {"tables": get_tables_from_xmldoc(xmldoc)}
    if psds := build_psd_series_from_xmldoc(xmldoc):
        data["psds"] = psds
    if snrs := build_snr_series_from_xmldoc(xmldoc):
        data["snrs"] = snrs
    if p_astro := get_p_astro_from_xmldoc(xmldoc):
        data["p_astro"] = p_astro

    return data
