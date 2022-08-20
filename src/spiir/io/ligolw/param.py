import logging
from typing import Dict

import ligo.lw.ligolw
import ligo.lw.param
import numpy as np

logger = logging.getLogger(__name__)


def get_p_astro_from_xmldoc(xmldoc: ligo.lw.ligolw.Document) -> Dict[str, float]:
    """Retrieves p_astro source probability values from a LIGO_LW XML Document.
    
    Parameters
    ----------
    xmldoc: ligo.lw.ligolw.Document
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
    xmldoc: ligo.lw.ligolw.Document
        A LIGO_LW Document object.
    p_astro: dict[str, float]
        Astrophysical source probabilities.
    overwrite: bool
        If true, removes all other LIGO_LW p_astro elements before appending.

    Returns
    -------
    ligo.lw.ligolw.Document
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