import logging
from os import PathLike
from typing import Optional, Union, Dict, Sequence

import ligo.lw.ligolw
import ligo.lw.param
import numpy as np

from .ligolw import load_ligolw_xmldoc

logger = logging.getLogger(__name__)


def get_params_from_xmldoc(
    xmldoc: ligo.lw.ligolw.Document,
    params: Optional[Union[Sequence[str], str]] = None,
) -> dict:
    """Retrieves params values from a LIGO_LW XML Document as a dictionary.
    
    Note that this function automatically ignores LIGO_LW elements that have Name
    attributes, such as the LIGO_LW PSD, SNR, or P_Astro elements.

    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A valid LIGO_LW XML Document element that contains Param element(s).
    params: :class:`~Sequence[str]`, optional
        If provided, retrieve the specified parameters by their name.
        Otherwise if params is None, retrieve all parameters from the xmldoc object.
    
    Returns
    -------
    dict
        The params defined in the XML document.
    """
    if params is not None:
        parameters_to_get = [params] if isinstance(params, str) else params

    parameters = {}
    for elem in xmldoc.getElements(
        lambda e: (
            (e.tagName == "LIGO_LW")
            and ~(e.hasAttribute("Name"))
        )
    ):
        for param in elem.getElements(
            lambda e: e.tagName == ligo.lw.param.Param.tagName
        ):
            if params is None or param.Name in parameters_to_get:
                parameters[param.Name] = param.value

    return parameters

def load_parameters_from_xml(
    path: Union[str, bytes, PathLike],
    params: Optional[Union[Sequence[str], str]] = None,
) -> dict:
    """Reads a valid LIGO_LW XML Document from a file path and returns a dictionary
    containing the LIGO_LW Param elements present in the file.

    Parameters
    ----------
    path: str | bytes | :class:`~os.PathLike`
        A path-like to an XML file containing a valid LIGO_LW XML Document.
    params: :class:`~Sequence[str]`, optional
        If provided, retrieve the specified parameters by their name.
        Otherwise if params is None, retrieve all parameters from the xmldoc object.

    Returns
    -------
    dict
        The params defined in the XML document.
    """
    xmldoc = load_ligolw_xmldoc(path)
    parameters = get_params_from_xmldoc(xmldoc, params)
    if len(parameters) == 0:
        logger.debug(f"No LIGO_LW params found in {str(path)}.")
    return parameters

def get_p_astro_from_xmldoc(xmldoc: ligo.lw.ligolw.Document) -> Dict[str, float]:
    """Retrieves p_astro source probability values from a LIGO_LW XML Document.
    
    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A LIGO_LW Document object that contains a p_astro LIGO_LW element.

    Returns
    -------
    dict[str, float]
        The astrophysical source probabiltiies (p_astro) defined in the XML document.
    """
    p_astro = {}
    for elem in xmldoc.getElements(
        lambda e: (
            (e.tagName == "LIGO_LW")
            and (e.hasAttribute("Name"))
            and (e.Name == "p_astro")
        )
    ):
        for param in elem.getElements(
            lambda e: e.tagName == ligo.lw.param.Param.tagName
        ):
            p_astro[param.Name] = param.value
    return p_astro


def append_p_astro_to_ligolw(
    xmldoc: ligo.lw.ligolw.Document,
    p_astro: Dict[str, float],
    overwrite: bool = True,
) -> ligo.lw.ligolw.Document:
    """Appends p_astro source probability values to a LIGO_LW XML Document.
    
    Parameters
    ----------
    xmldoc: :class:`~ligo.lw.ligolw.Document`
        A LIGO_LW Document object.
    p_astro: dict[str, float]
        Astrophysical source probabilities.
    overwrite: bool, default: `True`
        If true, removes all other LIGO_LW p_astro elements before appending.

    Returns
    -------
    :class:`~ligo.lw.ligolw.Document`
        A LIGO_LW Document object that contains a LIGO_LW p_astro element.
    """
    if not np.allclose(sum(p_astro.values()), 1.0):
        logger.warning(f"p_astro values do not sum to 1.0.")

    llw = ligo.lw.ligolw.LIGO_LW(attrs={"Name": "p_astro"})
    if overwrite:
        for existing_llw_p_astro in xmldoc.getElements(
            lambda e: (
                (e.tagName == "LIGO_LW")
                and (e.hasAttribute("Name"))
                and (e.Name == "p_astro")
            )
        ):
            xmldoc.removeChild(existing_llw_p_astro)
            existing_llw_p_astro.unlink()
    
    for key in sorted(p_astro):
        llw.appendChild(ligo.lw.param.Param.from_pyvalue(key.lower(), p_astro[key]))

    xmldoc.appendChild(llw)
    return xmldoc