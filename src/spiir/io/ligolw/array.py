import concurrent.futures
import logging
from functools import partial
from os import PathLike
from typing import Dict, Optional, Sequence, Union

import lal
import lal.series
import ligo.lw.array
import ligo.lw.ligolw
import ligo.lw.param
import ligo.lw.table
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..processing import get_unique_index_diff, validate_cpu_count
from .ligolw import get_ligolw_element, load_ligolw_xmldoc

logger = logging.getLogger(__name__)


def get_array_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Document,
    array: str,
) -> np.ndarray:
    """Retrieves a numpy array from a provided LIGO_LW Document element in memory.

    This is a wrapper function around the ligo.lw.array.Array.get_array class method,
    made primarily for convenience.

    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A valid LIGO_LW XML Document element that contains Array element(s).
    array: str
        The array to retrieve from the XML Document element.

    Return
    ------
    np.ndarray
        A numpy array loaded from the LIGO_LW xmldoc array.
    """
    return ligo.lw.array.Array.get_array(xmldoc, array).array


def load_array_from_xml(
    path: Union[str, bytes, PathLike],
    array: str,
    ilwdchar_compat: bool = True,
) -> np.ndarray:
    """Loads one array from one LIGO_LW XML file path and returns a numpy array.

    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A valid LIGO_LW XML Document element that contains Array element(s).
    array: str
        The array to retrieve from the XML Document element.
    ilwdchar_compat: bool, default: `True`
        Whether to add ilwdchar conversion compatibility.

    Return
    ------
    np.ndarray
        A numpy array loaded from the LIGO_LW xmldoc array.
    """
    xmldoc = load_ligolw_xmldoc(path, ilwdchar_compat=ilwdchar_compat)
    return get_array_from_xmldoc(xmldoc, array)


def get_arrays_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Document,
    arrays: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Retrieves numpy arrays from a provided LIGO_LW xmldoc object in memory.

    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A valid LIGO_LW XML Document element that contains Array element(s).
    arrays: :class:`~Sequence[str]`, optional
        If provided, retrieve the specified arrays by their name.
        Otherwise if arrays is None, retrieve all arrays from the xmldoc object.

    Return
    ------
    dict[str, np.ndarray]
        A dictionary where keys are array names are values are the numpy arrays.
    """
    if arrays is None:  # return all arrays in xmldoc
        # FIXME: Arrays with shared names will silently overwrite eachother (snr, psd)
        return {
            array.Name: array.array
            for array in xmldoc.getElements(
                lambda elem: elem.tagName == ligo.lw.array.Array.tagName
            )
        }
    else:
        if isinstance(arrays, str):
            arrays = [arrays]
        return {array: get_array_from_xmldoc(xmldoc, array) for array in arrays}


def load_arrays_from_xml(
    path: Union[str, bytes, PathLike],
    arrays: Optional[Sequence[str]] = None,
    ilwdchar_compat: bool = True,
) -> Dict[str, np.ndarray]:
    """Loads one array from one LIGO_LW XML file path and returns a numpy array.

    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A valid LIGO_LW XML Document element that contains Array element(s).
    arrays: :class:`~Sequence[str]`, optional
        If provided, retrieve the specified arrays by their name.
        Otherwise if arrays is None, retrieve all arrays from the xmldoc object.
    ilwdchar_compat: bool, default: `True`
        Whether to add ilwdchar conversion compatibility.

    Return
    ------
    dict[str, :class:`~np.ndarray`]
        A dictionary where keys are array names are values are the numpy arrays.
    """
    xmldoc = load_ligolw_xmldoc(path, ilwdchar_compat=ilwdchar_compat)
    return get_arrays_from_xmldoc(xmldoc, arrays)


# stacks on first axis
def load_arrays_from_xmls(
    paths: Union[str, bytes, PathLike, Sequence[Union[str, bytes, PathLike]]],
    arrays: Optional[Sequence[str]] = None,
    ilwdchar_compat: bool = True,
    skip_exceptions: bool = False,
    nproc: int = 1,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Loads one or more arrays from one LIGO_LW XML file path and returns a dictionary
    of numpy arrays that have been stacked together.

    Note that each XML file must have the same set of arrays present in each document.

    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A valid LIGO_LW XML Document element that contains Array element(s).
    arrays: :class:`~Sequence[str]`, optional
        If provided, retrieve the specified arrays by their name.
        Otherwise if arrays is None, retrieve all arrays from the xmldoc object.
    ilwdchar_compat: bool, default: `True`
        Whether to add ilwdchar conversion compatibility.
    skip_exceptions: bool
        If True, exceptions raised by a process during mulitprocessing will be ignored.
    nproc: int, default: 1
        Number of CPU processes to use for multiprocessing. Default: 1, recommended: 4.
    verbose: bool, default: `True`
        If True, displays a loading progress bar.

    Return
    ------
    dict[str, :class:`~np.ndarray`]
        A dictionary where keys are array names are values are the numpy arrays.
    """
    if isinstance(paths, (str, bytes, PathLike)):
        return load_arrays_from_xml(paths, arrays, ilwdchar_compat)
    else:
        nproc = validate_cpu_count(nproc)
        _load_arrays_from_xml = partial(
            load_arrays_from_xml,
            arrays=arrays,
            ilwdchar_compat=ilwdchar_compat,
        )

        desc = "Loading arrays from LIGOLW XML files"
        with tqdm(total=len(paths), desc=desc, disable=not verbose) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
                futures = [executor.submit(_load_arrays_from_xml, p) for p in paths]

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

                return {
                    key: np.stack([data[key] for data in results])
                    for key in results[0].keys()
                }


def build_array_element(
    array: np.ndarray,
    name: str,
    dim_names: Optional[Union[str, Sequence[str]]] = None,
) -> ligo.lw.array.Array:
    """Convenience function to build a LIGO_LW Array element."""
    return ligo.lw.array.Array.build(name, array, dim_names)


def append_array_to_xmldoc(
    xmldoc: ligo.lw.ligolw.Document,
    array: ligo.lw.array.Array,
    overwrite: bool = True,
):
    """Writes the given LIGO_LW array into a LIGO_LW XML document.

    This code is based on gwpy.io.table.py.

    Parameters
    ----------
    xmldoc : :class:`~ligo.lw.ligolw.Document`
        the document to write into
    array : `list` of :class:`~ligo.lw.table.Table`
        the set of arrays to write
    overwrite : `bool`, optional, default: `True`
        if `True`, delete an all existing instances matching the provided array name,
        else append the array to the document as is.
    """
    try:
        llw = get_ligolw_element(xmldoc)
    except ValueError:
        llw = ligo.lw.ligolw.LIGO_LW()
        xmldoc.appendChild(llw)

    if overwrite:
        for existing_array in ligo.lw.array.Array.getArraysByName(xmldoc, array.Name):
            llw.removeChild(existing_array)
            existing_array.unlink()

    llw.appendChild(array)
    return xmldoc


def build_psd_series_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Document,
) -> Dict[str, pd.Series]:
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
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A LIGO_LW XML Document element, containing the necessary PSD array elements.

    Returns
    -------
    dict[str, :class:`~pd.Series`]
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


def load_psd_series_from_xml(
    path: Union[str, bytes, PathLike],
    ilwdchar_compat: bool = True,
) -> Dict[str, pd.Series]:
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
    path: str | bytes | :class:`~os.PathLike`
        A path-like to an XML file containing a valid LIGO_LW XML Document.
    ilwdchar_compat: bool, default: `True`
        Whether to add ilwdchar conversion compatibility.

    Returns
    -------
    dict[str, :class:`~pd.Series`]
        A dictionary of pd.Series where the key refers to the interferometer name,
        and the values contain the respective real-valued frequency array.

    Examples
    --------
    The following code snippet will load a coinc.xml file and return a pd.DataFrame.

    >>> psds = load_all_ligolw_frequency_arrays("coinc.xml")
    >>> psds = pd.DataFrame(psds)

    """
    xmldoc = load_ligolw_xmldoc(path, ilwdchar_compat=ilwdchar_compat)
    psds = build_psd_series_from_xmldoc(xmldoc)
    if len(psds) == 0:
        logger.debug(f"No PSD series found in {str(path)}.")
    return psds


def append_psd_series_to_ligolw(
    xmldoc: ligo.lw.ligolw.Document,
    psds: Dict[str, Union[pd.Series, np.ndarray]],
    f0: float = 0.0,
    delta_f: Optional[float] = None,
    epoch_time: Union[lal.LIGOTimeGPS, float] = 0.0,
) -> ligo.lw.ligolw.Document:
    """Assembles and append a LIGO_LW REAL8FreqencySeries array to a LIGO_LW XML.

    The LIGO_LW REAL8FrequencySeries Array is assembled from a dictionary where keys
    are ifo strings and values are the PSD as aa np.ndarray or pd.Series for each ifo.
    The PSD LIGO_LW element will be appended to the xmldoc containing both a
    REAL8FrequencySeries Array object and a Param object that specifies the ifo string.

    Parameters
    ----------
    psds: dict[str, :class:`~pd.Series` | :class:`~np.ndarray`]
        A dictionary of frequency series that refer to the PSD for each ifo.
    f0: float, default: 0.
        The lower bound minimum frequency considered during analysis of the strain.
    delta_f: float, optional
        The increment between each frequency bin of the array (the index) of the PSD.
    epoch_time: float | :class:`~lal.LIGOTimeGPS`, default: 0.
        The GPS epoch time at the time the estimated PSD was calculated.

    Returns
    -------
    :class:`~ligo.lw.ligow.Document`
        The LIGO_LW XML Document element with the LIGO_LW PSD element attached.
    """
    ligo_lw_psds = ligo.lw.ligolw.LIGO_LW(attrs={"Name": "psd"})
    for ifo, psd in psds.items():
        if isinstance(psd, pd.Series):
            psd = psd.dropna()

        if isinstance(psd, pd.Series) and delta_f is None:
            index_diff = float(get_unique_index_diff(psd.index, precision=12))
        else:
            raise ValueError(
                f"Cannot determine delta_f automatically if psd is a {type(psd)}."
            )

        psd_freq_series = lal.CreateREAL8FrequencySeries(
            name="psd",
            epoch=lal.LIGOTimeGPS(epoch_time),
            f0=f0,
            deltaF=delta_f or index_diff,
            sampleUnits=lal.Unit("s strain^2"),
            length=len(psd),
        )
        psd_freq_series.data.data = psd.values if isinstance(psd, pd.Series) else psd
        ligo_lw_psd = lal.series.build_REAL8FrequencySeries(psd_freq_series)
        ligo_lw_psd.appendChild(ligo.lw.param.Param.build("instrument", "lstring", ifo))
        ligo_lw_psds.appendChild(ligo_lw_psd)
    xmldoc.appendChild(ligo_lw_psds)
    return xmldoc


def build_snr_series_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Document,
    add_epoch_time: bool = True,
) -> Dict[str, pd.Series]:
    """Reads a valid LIGO_LW XML Document from a ligo.lw.ligolw.Document object and
    returns a dictionary containing the complex SNR timeseries arrays associated with
    each interferometer ('ifo').

    The LIGO_LW Document must contain exactly one ligo.lw.lsctables.SnglInspiralTable
    and at least one LIGO_LW COMPLEX8TimeSeries element containing the timeseries as an
    Array named snr:array. Each row in the SnglInspiralTable should be matched to a
    corresponding COMPLEX8TimeSeries by an event_id:param, but we discard this event_id
    in the returned Series and replace it with the interferometer name as a string.

    The returned Series has columns describing each interferometer name, an
    index specifying the timestamp, and values that define the complex SNR time series.

    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A LIGO_LW XML Document element containing the necessary SNR series elements.
    add_epoch_time: bool, default: `True`
        If True, adds the GPS epoch time to the SNR series for correct timestamps.

    Returns
    -------
    dict[str, :class:`~pd.Series`]
        A dictionary of pd.Series where the key refers to the sngl_inspiral ifo,
        and the values contain the respective SNR timeseries array
        (each with their own timestamped indices).

    Notes
    -----
    The true starting epoch GPS time is added to the index values of each
    pd.Series as each SNR timeseries array may not perfectly align with another.
    To undo this addition and reset all arrays to index starting from 0, simply
    subtract the first element of the pd.Series index from every element for each
    SNR series respectively.
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
        assert len(sngl_inspiral) == 1, "Expected only one sngl_inspiral in xmldoc."
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
        data[ifo] = pd.Series(
            data=snr.data.data, index=pd.Index(timesteps, name="time"), name=ifo
        )

    return data


def load_snr_series_from_xml(
    path: Union[str, bytes, PathLike],
    add_epoch_time: bool = True,
    ilwdchar_compat: bool = True,
) -> Dict[str, pd.Series]:
    """Reads a valid LIGO_LW XML Document from a file path and returns a dictionary
    containing the complex SNR timeseries arrays associated with each interferometer.

    The LIGO_LW Document must contain exactly one ligo.lw.lsctables.SnglInspiralTable
    and at least one LIGO_LW COMPLEX8TimeSeries element containing the timeseries as an
    Array named snr:array. Each row in the SnglInspiralTable should be matched to a
    corresponding COMPLEX8TimeSeries by an event_id:param, but we discard this event_id
    in the returned Series and replace it with the interferometer name as a string.

    The returned Series has columns describing each interferometer name, an index
    specifying the timestamp, and values that define the complex SNR time series array.

    Parameters
    ----------
    path: str | bytes | :class:`~os.PathLike`
        A path-like to an XML file containing a valid LIGO_LW XML Document.
    add_epoch_time: bool, default: `True`
        Whether to add the epoch time to each SNR series array for correct timestamps.
    ilwdchar_compat: bool, default: `True`
        Whether to add ilwdchar conversion compatibility.

    Returns
    -------
    dict[str, :class:`~pd.Series`]
        A dictionary of pd.Series where the key refers to the sngl_inspiral ifo,
        and the values contain the respective SNR timeseries array
        (each with their own timestamped indices).

    Notes
    -----
    The true starting epoch GPS time is added to the index values of each
    pd.Series as each SNR timeseries array may not perfectly align with another.
    To undo this addition and reset all arrays to index starting from 0, simply
    subtract the first element of the pd.Series index from every element for each
    SNR series respectively.
    """
    xmldoc = load_ligolw_xmldoc(path, ilwdchar_compat=ilwdchar_compat)
    snr_series = build_snr_series_from_xmldoc(xmldoc, add_epoch_time)
    if len(snr_series) == 0:
        logger.debug(f"No SNR series found in {str(path)}.")
    return snr_series


def append_snr_series_to_ligolw(
    xmldoc: ligo.lw.ligolw.Document,
    snr: Union[np.ndarray, pd.Series],
    ifo: str,
    f0: float = 0.0,
    delta_t: Optional[float] = None,
    event_id: Optional[int] = None,
    epoch_time: Optional[Union[lal.LIGOTimeGPS, float]] = None,
) -> ligo.lw.ligolw.Document:
    """Assembles and append a LIGO_LW COMPLEX8TimeSeries array to a LIGO_LW XML.

    The LIGO_LW COMPLEX8TimeSeries Array is assembled from a np.ndarray or pd.Series
    that defines the SNR series and an ifo string that specifies the interfereometer.
    The SNR LIGO_LW element will be appended to the xmldoc containing both a
    COMPLEX8TimeSeries Array object and a Param object that specifies an event_id that
    maps to a corresponding sngl_inspiral table row in an LIGO_LW Table element.

    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A valid LIGO_LW XML Document element with a sngl_inspiral table element.
    snr: dict[str, :class:`~pd.Series` | :class:`~np.ndarray`]
        A dictionary of the complex time series that refers to the SNR for an ifo.
    ifo: str
        The name of the interferometer as a string, e.g. H1, L1, V1, or K1.
    f0: float, default: 0.
        The lower bound minimum frequency considered during analysis of the strain.
    delta_t: float, optional
        The increment between each time bin of the array (the index) of the SNR series.
    event_id: int, optional
        An id that maps to an event_id of a sngl_inspiral table row of the LIGO_LW XML.
    epoch_time: float | :class:`~lal.LIGOTimeGPS`, default: 0.
        The GPS epoch time at the time the estimated SNR was calculated.

    Returns
    -------
    :class:`~ligo.lw.ligow.Document`
        The LIGO_LW XML Document element with the LIGO_LW PSD element attached.
    """
    if event_id is None:
        # get inspiral rows from sngl_inspiral table to map event_id automatically
        sngl_inspiral_table = ligo.lw.table.Table.get_table(xmldoc, "sngl_inspiral")
        sngl_inspiral = [row for row in sngl_inspiral_table if row.ifo == ifo]
        try:
            event_id = sngl_inspiral[0].event_id
        except IndexError as exc:
            raise IndexError(
                f"No event found in sngl_inspiral table to match with the provided "
                f"ifo {ifo} - unable to build a valid snr series element"
            ) from exc

    if isinstance(snr, pd.Series) and delta_t is None:
        index_diff = float(get_unique_index_diff(snr.index, precision=12))
    else:
        raise ValueError(
            f"Cannot determine delta_t automatically if snr is a {type(snr)}."
        )

    if epoch_time is None:
        if isinstance(snr, pd.Series):
            epoch_time = lal.LIGOTimeGPS(snr.index[0])
        else:
            raise ValueError(
                f"Cannot determine starting epoch time if snr is a {type(snr)}."
            )

    lal_snr = lal.CreateCOMPLEX8TimeSeries(
        name=ifo,
        epoch=epoch_time,
        f0=f0,
        deltaT=delta_t or index_diff,
        sampleUnits="s^-1",
        length=len(snr),
    )

    snr_time_series = snr.values if isinstance(snr, pd.Series) else snr
    lal_snr.data.data = snr_time_series
    ligo_lw_snr = lal.series.build_COMPLEX8TimeSeries(lal_snr)
    ligo_lw_snr.appendChild(ligo.lw.param.Param.build("event_id", "int_8s", event_id))
    xmldoc.appendChild(ligo_lw_snr)
    return xmldoc
