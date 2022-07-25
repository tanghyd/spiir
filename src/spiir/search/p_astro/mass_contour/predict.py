# Module with utilities for estimating candidate events source probabilities
# Initial code by A. Curiel Barroso, August 2019
# Modified by V. Villa-Ortega, January 2020, March 2021
# Modified by Daniel Tang for SPIIR March 2022

# Code forked from PyCBC for SPIIR development (March 2022)
# Source: https://github.com/gwastro/pycbc/blob/master/pycbc/mchirp_area.py
# Additional edits drawn from prior work done by V. Villa-Ortega:
# https://github.com/veronica-villa/source_probabilities_estimation_pycbclive
"""Functions to compute the area corresponding to different CBC on the m1 & m2
plane when given a central mchirp value and uncertainty.

It also includes a function that calculates the source frame mass when given the
detector frame mass and redshift.
"""

import logging
import math

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from pycbc.conversions import mass2_from_mchirp_mass1 as mcm1_to_m2
from pycbc.cosmology import _redshift
from scipy.integrate import quad

# from spiir.config.logging import logger

logger = logging.getLogger(__name__)


def estimate_redshift_from_distance(
    distance: float,
    distance_std: float,
    truncate_lower_dist: float | None = None,
    lal_cosmology: bool = True,
) -> tuple[float, float]:
    """
    Takes values of distance and its uncertainty and returns a
    dictionary with estimates of the redshift and its uncertainty.

    Constants for lal_cosmology taken from Planck15_lal_cosmology() in
    https://git.ligo.org/lscsoft/pesummary/-/blob/master/pesummary/gw/cosmology.py.

    Parameters
    ----------
    distance: float
        Estimated luminosity distance (D_L).
    distance_std: float
        Uncertainty (standard deviation) associated with D_L.
    lal_cosmology: bool
        If True, it uses the Planck15 cosmology model
        as defined in lalsuite instead of the astropy default.

    Returns
    -------
    tuple[float, float]
        An estimate of redshift and its corresponding uncertainty.
    """
    # define cosmology parameters
    cosmology = FlatLambdaCDM(H0=67.90, Om0=0.3065) if lal_cosmology else None

    # estimate redshift with pycbc
    # this sometimes triggers a warning with multiple valid values, see:
    # https://docs.astropy.org/en/stable/cosmology/index.html \
    #   #finding-the-redshift-at-a-given-value-of-a-cosmological-quantity
    z_estimation = _redshift(distance, cosmology=cosmology)
    z_est_max = _redshift(distance + distance_std, cosmology=cosmology)

    # high distance uncertainty can lead to negative distance estimates
    distance_lower_bound = distance - distance_std
    if truncate_lower_dist:
        distance_lower_bound = max(distance_lower_bound, truncate_lower_dist)
    z_est_min = _redshift(distance_lower_bound, cosmology=cosmology)
    z_std_estimation = 0.5 * (z_est_max - z_est_min)

    return z_estimation, z_std_estimation


def estimate_source_mass(
    mdet: float, mdet_std: float, z: float, z_std: float
) -> tuple[float, float]:
    """
    Takes values of redshift, redshift uncertainty, detector mass and its
    uncertainty and computes the source mass and its uncertainty.

    Parameters
    ----------
    mdet: float
        The mass in the detector frame
    mdet_std: float
        The uncertainty in detector frame source mass
    z: float
        The estimated redshift value
    z_std: float
        The uncertainty in redshift (standard deviation?)

    Returns
    -------
    tuple[float, float]
        The source mass and its uncertainty as a tuple.

    """
    msrc = mdet / (1.0 + z)  # source frame mass
    msrc_std = msrc * ((mdet_std / mdet) ** 2.0 + (z_std / (1.0 + z)) ** 2.0) ** 0.5
    return msrc, msrc_std


def integrate_chirp_mass(mchirp: float, m_min: float, m_max: float) -> float:
    """
    Returns the integral of a component mass as a function of the mass of
    the other component, taking mchirp as an argument.

    Parameters
    ----------
    mchirp: float
        The chirp mass value set as fixed
    x_min: float
        The minimum m2 value
    x_max: float
        The maximum m2 value

    Returns
    -------
    float
        The result of the integration of m2 over m1.

    """
    return quad(lambda m, mchirp: mcm1_to_m2(mchirp, m), m_min, m_max, args=mchirp)[0]


def get_area(
    msrc: float,
    msrc_std: float,
    lim_h1: float | str,  # consider refactor?
    lim_h2: float,
    lim_v1: float,
    lim_v2: float,
) -> float:
    """
    Returns the area of the chirp mass contour in each region of the m1m2 plane
    taking horizontal and vertical limits of the region as arguments. (m1 > m2).

    Parameters
    ----------
    msrc: float
        Chirp mass in source frame
    msrc_std: float
        The uncertainty (standard deviation) of the chirp mass
    lim_h1: float | str
        float or "diagonal" - upper horizontal limit (limit on m2)
    lim_h2: float
        lower horizontal limit of the region (limit on m2)
    lim_v1: float
        right vertical limits of the region (limits on m1)
    lim_v2: float
        left vertical limits of the region (limits on m1)

    Returns
    -------
    float
        The area calculated within provided limits of the mass plane.
    """
    # type check inputs according to implemented logic
    if isinstance(lim_h1, str):
        assert lim_h1 == "diagonal", "if lim_h1 is a str it must be 'diagonal'."
    assert isinstance(lim_h2, float), "get_area not compatible with lim_h2 as str."

    # get bounds of chirp mass contour given uncertainty
    mchirp_max = msrc + msrc_std
    mchirp_min = msrc - msrc_std

    if lim_h1 == "diagonal":
        # the points where the equal mass line and a chirp mass
        # curve intersect is at m1 = m2 = 2**0.2 * mchirp
        max_h1 = mchirp_max * (2.0**0.2)
        min_h1 = mchirp_min * (2.0**0.2)
        fun_sup = lambda x: x
    else:
        max_h1 = mcm1_to_m2(mchirp_max, lim_h1)
        min_h1 = mcm1_to_m2(mchirp_min, lim_h1)
        fun_sup = lambda x: lim_h1

    max_h2 = mcm1_to_m2(mchirp_max, lim_h2)
    min_h2 = mcm1_to_m2(mchirp_min, lim_h2)
    fun_inf = lambda x: lim_h2

    lim_max1 = np.clip(max_h1, lim_v1, lim_v2)
    lim_max2 = np.clip(max_h2, lim_v1, lim_v2)
    lim_min1 = np.clip(min_h1, lim_v1, lim_v2)
    lim_min2 = np.clip(min_h2, lim_v1, lim_v2)

    int_max = integrate_chirp_mass(mchirp_max, lim_max1, lim_max2)
    int_min = integrate_chirp_mass(mchirp_min, lim_min1, lim_min2)
    intline_sup = quad(fun_sup, lim_min1, lim_max1)[0]
    intline_inf = quad(fun_inf, lim_min2, lim_max2)[0]
    area = int_max + intline_sup - int_min - intline_inf

    return area


def calc_areas(
    mchirp: float,
    mchirp_std: float,
    z: float,
    z_std: float,
    m_bounds: tuple[float, float],
    mgap_bounds: tuple[float, float],
    group_mgap: bool = True,
) -> dict[str, float]:
    """
    Computes the area inside the lines of the second component mass as a
    function of the first component mass for the two extreme valuesmsrc, msrc_std

    Parameters
    ----------
    mchirp: float
        The detector frame chirp mass.
    mchirp_std: float
        The uncertainty (standard deviation) in the chirp mass.
    z: float
        The estimated redshift between detector and source frame of the event.
    z_std: float
        The uncertainty (standard deviation) in the estimated redshift.
    m_bounds: tuple[float, float]
        The bounds on all possible component masses (let m1 = m2).
    mgap_bounds: tuple[float, float]
        The bounds on the mass gap category (max NS & min BH mass).
    group_mgap: bool
        If True, aggregates Mass Gap from BH+Gap, Gap+NS, and Gap+Gap.

    Returns
    -------
    dict[str, float]
        The area covered by each source class within a contour on the mass plane.
    """
    # check valid input arguments for mass bounds [lower, upper]
    m_min, m_max = m_bounds
    mgap_min, mgap_max = mgap_bounds
    assert (0 < m_min <= m_max) and (0 < mgap_min <= mgap_max)  # <1 not astrophysical

    # compute source frame chirp mass given detector frame mass and redshift
    mc_src, mc_src_std = estimate_source_mass(mchirp, mchirp_std, z, z_std)

    # compute relative area within chirp mass contour for each class
    abbh = get_area(mc_src, mc_src_std, "diagonal", mgap_max, mgap_max, m_max)
    abhg = get_area(mc_src, mc_src_std, mgap_max, mgap_min, mgap_max, m_max)
    ansbh = get_area(mc_src, mc_src_std, mgap_min, m_min, mgap_max, m_max)
    agg = get_area(mc_src, mc_src_std, "diagonal", mgap_min, mgap_min, mgap_max)
    agns = get_area(mc_src, mc_src_std, mgap_min, m_min, mgap_min, mgap_max)
    abns = get_area(mc_src, mc_src_std, "diagonal", m_min, m_min, mgap_min)

    areas = {"BNS": abns, "NSBH": ansbh, "BBH": abbh}
    if group_mgap:
        areas["MassGap"] = agns + agg + abhg
    else:
        areas.update({"GNS": agns, "GG": agg, "BHG": abhg})

    return areas


def predict_redshift(
    coefficients: dict[str, float],
    snr: float,
    eff_distance: float,
    lal_cosmology: bool = True,
    truncate_lower_dist: float | None = None,
) -> tuple[float, float]:
    logger.debug(f"truncate_lower_dist: {truncate_lower_dist}")
    # compute estimated luminosity distance and redshift and their uncertainties
    dist_est = coefficients["a0"] * eff_distance
    dist_std_est = dist_est * math.exp(coefficients["b0"]) * snr ** coefficients["b1"]
    z, z_std = estimate_redshift_from_distance(
        dist_est, dist_std_est, truncate_lower_dist, lal_cosmology
    )

    return z, z_std


def calc_probabilities(
    coefficients: dict[str, float],
    mchirp: float,
    z: float,
    z_std: float,
    m_bounds: tuple[float, float],
    mgap_bounds: tuple[float, float],
    group_mgap: bool = True,
) -> dict[str, float]:
    # determine chirp mass bounds in detector frame for classification
    get_redshifted_mchirp = lambda m: (m / (2**0.2)) * (1 + z)  # Mc_det = (1+z)*Mc
    mchirp_min, mchirp_max = (get_redshifted_mchirp(m) for m in m_bounds)

    # determine astrophysical source class probabilities given estimated parameters
    if mchirp > mchirp_max:
        # if observed detector frame chirp mass is greater than mchirp bounds => BBH
        probabilities = {"BNS": 0.0, "NSBH": 0.0, "BBH": 1.0}
        if group_mgap:
            probabilities["MassGap"] = 0.0
        else:
            probabilities.update({"GNS": 0.0, "GG": 0.0, "BHG": 0.0})

    elif mchirp < mchirp_min:
        # if observed detector frame chirp mass is less than mchirp bounds => BNS
        probabilities = {"BNS": 1.0, "NSBH": 0.0, "BBH": 0.0}
        if group_mgap:
            probabilities["MassGap"] = 0.0
        else:
            probabilities.update({"GNS": 0.0, "GG": 0.0, "BHG": 0.0})

    else:
        # compute probabilities according to proportional areas in mass contour
        mc_std = mchirp * coefficients["m0"]  # inherent uncertainty in chirp mass
        areas = calc_areas(mchirp, mc_std, z, z_std, m_bounds, mgap_bounds, group_mgap)
        total_area = sum(areas.values())
        probabilities = {key: areas[key] / total_area for key in areas}

    return probabilities


def predict_source_p_astro(
    coefficients: dict[str, float],
    mchirp: float,
    snr: float,
    eff_distance: float,
    m_bounds: tuple[float, float],
    mgap_bounds: tuple[float, float],
    group_mgap: bool = True,
    lal_cosmology: bool = True,
    truncate_lower_dist: float | None = None,
) -> dict[str, float]:
    """
    Computes the different probabilities that a candidate event belongs to
    each CBC source category taking as arguments the chirp mass, the
    coincident SNR and the effective distance, and estimating the
    chirp mass uncertainty, the luminosity distance (and its uncertainty)
    and the redshift (and its uncertainty). Probability is estimated to be
    directly proportional to the area of the corresponding CBC region.

    $$ \tilde{D}_L = a_0 \cdot min(\tilde{D}_{eff, i}) $$
    $$ \tilde{\sigma}_{D_L} = e^{b_0} \cdot \tilde{D}_L \cdot\rho_{c}^{b_1} $$

    Parameters
    ----------
    coefficients: float
        The estimated model coefficients of fitted mass/distance models.
    mchirp: float
        The source frame chirp mass.
    snr: float
        The coincident signal-to-noise ratio (SNR)
    eff_distance: float
        The estimated effective distance, usually taken as the minimum across all coincident detectors.
    m_bounds: tuple[float, float]
        The upper and lower bounds for both component masses (m1 >= m2).
    mgap_bounds: tuple[float, float]
        The boundaries that define the mass gap between BH and NS.
    group_mgap: bool
        If True, aggregates Mass Gap from BH+Gap, Gap+NS, and Gap+Gap.
    lal_cosmology: bool
        If True, it uses the Planck15 cosmology model as defined in lalsuite instead of the astropy default.

    Returns
    -------
    dict[str, float]
        The astrophysical source probabilities for each class.
    """
    # predict redshift according to model coefficients
    z, z_std = predict_redshift(
        coefficients, snr, eff_distance, lal_cosmology, truncate_lower_dist
    )

    # calculate class probabilities given mchirp and redshift uncertainty
    probabilities = calc_probabilities(
        coefficients, mchirp, z, z_std, m_bounds, mgap_bounds, group_mgap
    )

    return probabilities
