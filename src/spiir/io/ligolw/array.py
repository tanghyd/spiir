import logging
from os import PathLike
from typing import Union, Dict

import pandas as pd
import numpy as np

from gwpy.frequencyseries import FrequencySeries

import lal.series
import ligo.lw.ligolw
import ligo.lw.param
import ligo.lw.table
import ligo.lw.utils

from .ligolw import load_ligolw_xmldoc

logger = logging.getLogger(__name__)


def load_all_ligolw_frequency_arrays(
    path: Union[str, bytes, PathLike],
    ilwdchar_compat: bool = False,
    verbose: bool = False,
) -> pd.Series:
    """Reads a valid LIGO_LW XML Document from a file path and returns a dictionary of 
    pd.Series containing a REAL8FrequencySeries array for each frequency series present.

    The LIGO_LW Document should contain exactly one LIGO_LW REAL8FrequencySeries
    element per interferometer each as an Array element. Each REAL8FrequencySeries is
    paired with an instrument string corresponding to the interferometer name.

    The returned dictionary has keys describing each interferometer name, with each
    value being a frequency array with an index specifying its frequency bins. Each
    array is indexed separately to handle the case where each frequency array
    may not have matching resolutions (i.e. PSDs may not have matching frequency
    components recorded), but if they are the same they can simply be concatenated.

    Parameters
    ----------
    path: str | bytes | PathLike
        A path-like to a file containing a valid LIGO_LW XML Document.
    ilwdchar_compat: bool
        Whether to add ilwdchar conversion compatibility.
    verbose: bool
        Whether to enable verbose output for ligo.lw.utils.load_filename.

    Returns
    -------
    pd.Series
        A pd.Series array containing the value and index of each frequency component.

    Examples
    --------
        >> psds = load_all_ligolw_frequency_arrays("coinc.xml")

    """
    xmldoc = load_ligolw_xmldoc(path, ilwdchar_compat=ilwdchar_compat, verbose=verbose)

    return get_all_ligolw_frequency_arrays_from_xmldoc(xmldoc)


def get_all_ligolw_frequency_arrays_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Element,
) -> Dict[int, pd.Series]:
    """Reads a valid LIGO_LW XML Document from memory and returns a dictionary of 
    pd.Series containing a REAL8FrequencySeries array for each frequency series present.

    The LIGO_LW Document should contain exactly one LIGO_LW REAL8FrequencySeries
    element per interferometer each as an Array element. Each REAL8FrequencySeries is
    paired with an instrument string corresponding to the interferometer name.

    The returned dictionary has keys describing each interferometer name, with each
    value being a frequency array with an index specifying its frequency bins. Each
    array is indexed separately to handle the case where each frequency array
    may not have matching resolutions (i.e. PSDs may not have matching frequency
    components recorded), but if they are the same they can simply be concatenated.

    Parameters
    ----------
    xmldoc: ligo.lw.ligolw.Element
        A LIGO_LW XML Document, or Element, containing the necessary LIGO_LW elements.

    Returns
    -------
    dict[int, pd.Series]
        A dictionary of pd.Series where the key refers to the interferometer name,
        and the values contain the respective real-valued frequency array.
    """
    # get frequency series from LIGO_LW elements
    psds = {}
    for elem in xmldoc.getElements(
        lambda e: (
            (e.tagName == "LIGO_LW")
            and (e.hasAttribute("Name"))
            and (e.Name == "REAL8FrequencySeries")
        )
    ):
        # get ifo and starting frequency param
        for param in elem.getElements(
            lambda e: (e.tagName == ligo.lw.param.Param.tagName)
        ):
            if param.getAttribute("Name") == "f0:param":
                f0 = param.value
            elif param.getAttribute("Name") == "instrument:param":
                ifo = param.value

        # build SNR time series array
        psd = lal.series.parse_REAL8FrequencySeries(elem)
        if f0 != psd.f0:
            logger.warning(f"f0:param {f0} does not match Array Dim Start={psd.f0}!")
        num = len(psd.data.data)
        index = np.linspace(start=f0, stop=psd.deltaF * num, num=num, endpoint=False)

        # build complex snr as a pandas Series object
        psds[ifo] = pd.Series(
            data=psd.data.data, index=pd.Index(index, name="frequency"), name=ifo
        )

    return psds


def load_all_ligolw_snr_arrays(
    path: Union[str, bytes, PathLike],
    add_epoch_time: bool = True,
    ilwdchar_compat: bool = False,
    verbose: bool = False,
) -> Dict[int, pd.Series]:
    """Reads a valid LIGO_LW XML Document from a file path and returns a dictionary
    containing the complex SNR timeseries arrays associated with each interferometer.

    The LIGO_LW Document must contain exactly one ligo.lw.lsctables.SnglInspiralTable
    and at least one LIGO_LW COMPLEX8TimeSeries element containing the timeseries as an
    Array named snr:array. Each row in the SnglInspiralTable should be matched to a
    corresponding COMPLEX8TimeSeries by an event_id:param.

    The returned DataFrame has columns describing each interferometer name, an index
    specifying the timestamp, and values that define the complex SNR time series array.

    Note: The true starting epoch GPS time is added to the index values of each
    pd.Series as each SNR timeseries array may not perfectly align with another.
    To undo this addition and reset all arrays to index starting from 0, simply
    subtract the first element of the pd.Series index from every element for each
    SNR series respectively.

    Parameters
    ----------
    path: str | bytes | PathLike
        A path-like to a file containing a valid LIGO_LW XML Document.
    add_epoch_time: bool
        Whether to add the epoch time to each SNR series array for correct timestamps.
    ilwdchar_compat: bool
        Whether to add ilwdchar conversion compatibility.
    verbose: bool
        Whether to enable verbose output for ligo.lw.utils.load_filename.

    Returns
    -------
    dict[int, pd.Series]
        A dictionary of pd.Series where the key refers to the sngl_inspiral event_id,
        and the values contain the respective SNR timeseries array
        (each with their own timestamped indices).
    """
    xmldoc = load_ligolw_xmldoc(path, ilwdchar_compat=ilwdchar_compat, verbose=verbose)

    return get_all_ligolw_snr_arrays_from_xmldoc(xmldoc, add_epoch_time)


def get_all_ligolw_snr_arrays_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Element,
    add_epoch_time: bool = True,
) -> Dict[int, pd.Series]:
    """Reads a valid LIGO_LW XML Document from a ligo.lw.ligolw.Document object and
    returns a dictionary containing the complex SNR timeseries arrays associated with
    each interferometer ('ifo').

    The LIGO_LW Document must contain exactly one ligo.lw.lsctables.SnglInspiralTable
    and at least one LIGO_LW COMPLEX8TimeSeries element containing the timeseries as an
    Array named snr:array. Each row in the SnglInspiralTable should be matched to a
    corresponding COMPLEX8TimeSeries by an event_id:param.

    The returned DataFrame has columns describing each interferometer name, an
    index specifying the timestamp, and values that define the complex SNR time series.

    Note: The true starting epoch GPS time is added to the index values of each
    pd.Series as each SNR timeseries array may not perfectly align with another.
    To undo this addition and reset all arrays to index starting from 0, simply
    subtract the first element of the pd.Series index from every element for each
    SNR series respectively.

    Parameters
    ----------
    xmldoc: ligo.lw.ligolw.Element
        A LIGO_LW XML Document, or Element, containing the necessary LIGO_LW elements.
    add_epoch_time: bool
        Whether to add the epoch time to each SNR series array for correct timestamps.

    Returns
    -------
    dict[int, pd.Series]
        A dictionary of pd.Series where the key refers to the sngl_inspiral event_id,
        and the values contain the respective SNR timeseries array
        (each with their own timestamped indices).
    """

    # get inspiral rows from sngl_inspiral table
    sngl_inspiral_table = ligo.lw.table.Table.get_table(xmldoc, name="sngl_inspiral")

    # get SNR series from LIGO_LW elements
    data = {}
    for elem in xmldoc.getElements(
        lambda e: (
            (e.tagName == "LIGO_LW")
            and (e.hasAttribute("Name"))
            and (e.Name == "COMPLEX8TimeSeries")
        )
    ):
        # get event_id param
        params = elem.getElements(
            lambda e: (
                (e.tagName == ligo.lw.param.Param.tagName)
                and (e.getAttribute("Name") == "event_id:param")
            )
        )
        assert len(params) == 1, "Expected only one param argument to match conditions."
        event_id = params[0].value

        # match snr series event_id:param with sngl_inspiral event_id
        sngl_inspiral = list(
            filter(lambda row: row.event_id == event_id, sngl_inspiral_table)
        )
        assert (
            len(sngl_inspiral) == 1
        ), "Expected only one sngl_inspiral to match conditions."
        ifo = sngl_inspiral[0].ifo

        # build SNR time series array
        snr = lal.series.parse_COMPLEX8TimeSeries(elem)
        num = len(snr.data.data)
        timesteps = np.linspace(
            start=0.0, stop=snr.deltaT * num, num=num, endpoint=False
        )
        if add_epoch_time:
            timesteps += float(snr.epoch)

        # build complex snr as a pandas Series object
        data[event_id] = pd.Series(
            data=snr.data.data, index=pd.Index(timesteps, name="time"), name=ifo
        )

    return data
