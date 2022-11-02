"""Module for source probability estimation and and plotting for the "mchirp_area" 
method as proposed by Villa-Ortega et. al. (2020).

Initial code by A. Curiel Barroso, August 2019
Modified by V. Villa-Ortega, January 2020, March 2021
Modified by Daniel Tang for SPIIR March 2022

Code forked from PyCBC for SPIIR development (March 2022)
Source: https://github.com/gwastro/pycbc/blob/master/pycbc/mchirp_area.py
Additional edits drawn from prior work done by V. Villa-Ortega:
https://github.com/veronica-villa/source_probabilities_estimation_pycbclive

"""
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.polynomial import Polynomial
from pycbc.conversions import mass2_from_mchirp_mass1 as mcm1_to_m2
from pycbc.cosmology import _redshift
from scipy.integrate import quad

logger = logging.getLogger(__name__)


_source_colour_map = {
    "BNS": "#A2C8F5",  # light blue
    "NSBH": "#FFB482",  # light orange
    "BBH": "#FE9F9B",  # light red
    "MG": "#8EE5A1",  # light green
    "MGNS": "#98D6CB",  # turquoise
    "MGMG": "#79BB87",  # green
    "BHMG": "#C6C29E",  # dark khaki
}


def get_source_colour(source: str) -> str:
    """Gets the class-based colour to maintain plotting consistency across classes."""
    return _source_colour_map[source]


def estimate_redshift_from_distance(
    distance: float,
    distance_std: float,
    truncate_lower_dist: Optional[float] = None,
    lal_cosmology: bool = True,
) -> Tuple[float, float]:
    """Takes values of distance and its uncertainty and returns a dictionary with
    estimates of the redshift and its uncertainty.

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
) -> Tuple[float, float]:
    """Takes values of redshift, redshift uncertainty, detector mass and its
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
    """Returns the integral of a component mass as a function of the mass of the other
    component, taking mchirp as an argument.

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
    lim_h1: Union[float, str],  # consider refactor?
    lim_h2: float,
    lim_v1: float,
    lim_v2: float,
) -> float:
    """
    Returns the area of the chirp mass area in each region of the m1m2 plane
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

    # get bounds of chirp mass area given uncertainty
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
    mass_bounds: Tuple[float, float],
    ns_max: float = 3.0,
    mass_gap_max: Optional[float] = None,
    separate_mass_gap: bool = False,
) -> Dict[str, float]:
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
    mass_bounds: tuple[float, float]
        The bounds on all possible component masses (let m1 = m2).
    ns_max: float
        The boundary that separates a classification between BH and NS.
    mass_gap_max: float | None
        If mass_gap_max is set, we assign a Mass Gap (MG) category above ns_max.
    separate_mass_gap: bool
        If True, splits Mass Gap category into BH+Gap, Gap+NS, and Gap+Gap.

    Returns
    -------
    dict[str, float]
        The area covered by each source class within a contour on the mass plane.
    """
    # check valid input arguments for mass bounds [lower, upper]
    m_min, m_max = mass_bounds
    mgap_min, mgap_max = ns_max, mass_gap_max or ns_max
    assert (0 < m_min <= m_max) and (0 < mgap_min <= mgap_max)  # <1 not astrophysical

    # compute source frame chirp mass given detector frame mass and redshift
    mc_src, mc_src_std = estimate_source_mass(mchirp, mchirp_std, z, z_std)

    # compute relative area within chirp mass area for each class
    abbh = get_area(mc_src, mc_src_std, "diagonal", mgap_max, mgap_max, m_max)
    ansbh = get_area(mc_src, mc_src_std, mgap_min, m_min, mgap_max, m_max)
    abns = get_area(mc_src, mc_src_std, "diagonal", m_min, m_min, mgap_min)

    areas = {"BNS": abns, "NSBH": ansbh, "BBH": abbh}

    if mgap_max > mgap_min:
        abhmg = get_area(mc_src, mc_src_std, mgap_max, mgap_min, mgap_max, m_max)
        amgmg = get_area(mc_src, mc_src_std, "diagonal", mgap_min, mgap_min, mgap_max)
        amgns = get_area(mc_src, mc_src_std, mgap_min, m_min, mgap_min, mgap_max)

        if not separate_mass_gap:
            areas["MG"] = amgns + amgmg + abhmg
        else:
            areas.update({"MGNS": amgns, "MGMG": amgmg, "BHMG": abhmg})

    return areas


def predict_redshift(
    a0: float,
    b0: float,
    b1: float,
    snr: float,
    eff_distance: float,
    lal_cosmology: bool = True,
    truncate_lower_dist: Optional[float] = None,
) -> Tuple[float, float]:
    # compute estimated luminosity distance and redshift and their uncertainties
    # logger.debug(f"truncate_lower_dist: {truncate_lower_dist}")
    dist_est = a0 * eff_distance
    dist_std_est = dist_est * math.exp(b0) * np.power(snr, b1)
    z, z_std = estimate_redshift_from_distance(
        dist_est, dist_std_est, truncate_lower_dist, lal_cosmology
    )

    return z, z_std


def calc_probabilities(
    m0: float,
    mchirp: float,
    z: float,
    z_std: float,
    mass_bounds: Tuple[float, float],
    ns_max: float = 3.0,
    mass_gap_max: Optional[float] = None,
    separate_mass_gap: bool = False,
) -> Dict[str, float]:
    """Computes the astrophysical source probability of a given candidate event."""

    # determine chirp mass bounds in detector frame for classification
    get_redshifted_mchirp = lambda m: (m / (2**0.2)) * (1 + z)  # Mc_det = (1+z)*Mc
    mchirp_min, mchirp_max = (get_redshifted_mchirp(m) for m in mass_bounds)
    mass_gap_max = mass_gap_max or ns_max

    # determine astrophysical source class probabilities given estimated parameters
    if mchirp > mchirp_max:
        # if observed detector frame chirp mass is greater than mchirp bounds => BBH
        probabilities = {"BNS": 0.0, "NSBH": 0.0, "BBH": 1.0}

        if mass_gap_max > ns_max:
            if not separate_mass_gap:
                probabilities["MG"] = 0.0
            else:
                probabilities.update({"MGNS": 0.0, "MGMG": 0.0, "BHMG": 0.0})

    elif mchirp < mchirp_min:
        # if observed detector frame chirp mass is less than mchirp bounds => BNS
        probabilities = {"BNS": 1.0, "NSBH": 0.0, "BBH": 0.0}

        if mass_gap_max > ns_max:
            if not separate_mass_gap:
                probabilities["MG"] = 0.0
            else:
                probabilities.update({"MGNS": 0.0, "MGMG": 0.0, "BHMG": 0.0})

    else:
        # compute probabilities according to proportional areas in chirp mass area
        mc_std = mchirp * m0  # inherent uncertainty in chirp mass
        areas = calc_areas(
            mchirp,
            mc_std,
            z,
            z_std,
            mass_bounds,
            ns_max,
            mass_gap_max,
            separate_mass_gap,
        )
        total_area = sum(areas.values())
        probabilities = {key: areas[key] / total_area for key in areas}

    return probabilities


def predict_source_p_astro(
    a0: float,
    b0: float,
    b1: float,
    m0: float,
    mchirp: float,
    snr: float,
    eff_distance: float,
    mass_bounds: Tuple[float, float],
    ns_max: float = 3.0,
    mass_gap_max: Optional[float] = None,
    separate_mass_gap: bool = False,
    lal_cosmology: bool = True,
    truncate_lower_dist: Optional[float] = None,
) -> Dict[str, float]:
    """Computes the astrophysical source probability of a given candidate event.

    Computes the different probabilities that a candidate event belongs to
    each CBC source category taking as arguments the chirp mass, the
    coincident SNR and the effective distance, and estimating the
    chirp mass uncertainty, the luminosity distance (and its uncertainty)
    and the redshift (and its uncertainty). Probability is estimated to be
    directly proportional to the area of the corresponding CBC region.

    Parameters
    ----------
    coefficients: float
        The estimated model coefficients of fitted mass/distance models.
    mchirp: float
        The source frame chirp mass.
    snr: float
        The coincident signal-to-noise ratio (SNR).
    eff_distance: float
        The estimated effective distance, usually taken as the minimum across all
        coincident detectors.
    mass_bounds: tuple[float, float]
        The upper and lower bounds for both component masses (m1 >= m2).
    ns_max: float
        The boundary that separates a classification between BH and NS.
    mass_gap_max: float | None
        If mass_gap_max is set, we assign a Mass Gap (MG) category above ns_max.
    separate_mass_gap: bool
        If True, splits Mass Gap into BH+Gap, Gap+NS, and Gap+Gap.
    lal_cosmology: bool
        If True, it uses the Planck15 cosmology model as defined in lalsuite,
        instead of the astropy default.

    Returns
    -------
    dict[str, float]
        The astrophysical source probabilities for each class.
    """
    z, z_std = predict_redshift(
        a0, b0, b1, snr, eff_distance, lal_cosmology, truncate_lower_dist
    )

    return calc_probabilities(
        m0, mchirp, z, z_std, mass_bounds, ns_max, mass_gap_max, separate_mass_gap
    )


def plot_mchirp_area_figure(
    mchirp: float,
    mchirp_std: float,
    z: float,
    z_std: float,
    mass_bounds: Tuple[float, float],
    ns_max: float = 3.0,
    mass_gap_max: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 6),
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Draws a full matplotlib Figure visualising the probability chirp mass area plane."""

    fig, ax = plt.subplots(figsize=figsize)
    _draw_mchirp_area_axes(
        ax=ax,
        mchirp=mchirp,
        mchirp_std=mchirp_std,
        z=z,
        z_std=z_std,
        mass_bounds=mass_bounds,
        ns_max=ns_max,
        mass_gap_max=mass_gap_max,
        xlims=xlims,
        ylims=ylims,
    )

    return fig


def _draw_mchirp_area_axes(
    ax: Axes,
    mchirp: float,
    mchirp_std: float,
    z: float,
    z_std: float,
    mass_bounds: Tuple[float, float],
    ns_max: float = 3.0,
    mass_gap_max: Optional[float] = None,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
) -> Axes:
    """Draws one matplotlib.axes.Axes visualising the probability chirp mass area plane."""

    # estimate source frame chirp mass and uncertainty boundary
    mc, mc_std = estimate_source_mass(mchirp, mchirp_std, z, z_std)
    mcb = mc + mc_std
    mcs = mc - mc_std

    # determine component masses (when m1 = m2) given chirp mass boundaries
    mib = (2.0**0.2) * mcb
    mis = (2.0**0.2) * mcs

    # get mass boundary limits
    m2_min, m1_max = mass_bounds
    mass_gap_max = mass_gap_max or ns_max

    lim_m1b = min(m1_max, mcm1_to_m2(mcb, m2_min))
    m1b = np.linspace(mib, lim_m1b, num=100)
    m2b = mcm1_to_m2(mcb, m1b)

    lim_m1s = min(m1_max, mcm1_to_m2(mcs, m2_min))
    m1s = np.linspace(mis, lim_m1s, num=100)
    m2s = mcm1_to_m2(mcs, m1s)

    # plot contour
    if mib > m1_max:
        ax.plot((m1_max, m1_max), (mcm1_to_m2(mcs, lim_m1s), m1_max), "b")
    else:
        ax.plot(m1b, m2b, "b")
        ax.plot(
            (m1_max, m1_max), (mcm1_to_m2(mcs, lim_m1s), mcm1_to_m2(mcb, lim_m1b)), "b"
        )
    if mis >= m2_min:
        ax.plot(m1s, m2s, "b")
        ax.plot((lim_m1s, lim_m1b), (m2_min, m2_min), "b")
    else:
        ax.plot((m2_min, lim_m1b), (m2_min, m2_min), "b")

    # plot limits
    ax.plot((m2_min, m1_max), (m2_min, m1_max), "k--")
    ax.plot((ns_max, ns_max), (m2_min, ns_max), "k:")
    ax.plot((mass_gap_max, mass_gap_max), (m2_min, mass_gap_max), "k:")
    ax.plot((ns_max, m1_max), (ns_max, ns_max), "k:")
    ax.plot((mass_gap_max, m1_max), (mass_gap_max, mass_gap_max), "k:")

    # colour plot
    ax.fill_between(
        np.arange(0.0, ns_max - 0.01, 0.01),
        mass_gap_max,
        m1_max,
        color=get_source_colour("NSBH"),
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(mass_gap_max, m1_max, 0.01),
        0.0,
        ns_max,
        color=get_source_colour("NSBH"),
    )
    ax.fill_between(
        np.arange(mass_gap_max, m1_max, 0.01),
        np.arange(mass_gap_max, m1_max, 0.01),
        m1_max,
        color=get_source_colour("BBH"),
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(mass_gap_max, m1_max, 0.01),
        np.arange(mass_gap_max, m1_max, 0.01),
        mass_gap_max,
        color=get_source_colour("BBH"),
    )
    ax.fill_between(
        np.arange(0.0, ns_max, 0.01),
        0.0,
        np.arange(0.0, ns_max, 0.01),
        color=get_source_colour("BNS"),
    )
    ax.fill_between(
        np.arange(0.0, ns_max, 0.01),
        ns_max,
        np.arange(0.0, ns_max, 0.01),
        color=get_source_colour("BNS"),
        alpha=0.5,
    )

    if mass_gap_max > ns_max:
        ax.fill_between(
            np.arange(0.0, ns_max, 0.01),
            ns_max,
            mass_gap_max,
            color=get_source_colour("MG"),
            alpha=0.5,
        )
        ax.fill_between(
            np.arange(ns_max, mass_gap_max, 0.01),
            np.arange(ns_max, mass_gap_max, 0.01),
            m1_max,
            color=get_source_colour("MG"),
            alpha=0.5,
        )
        ax.fill_between(
            np.arange(ns_max, mass_gap_max, 0.01),
            np.arange(ns_max, mass_gap_max, 0.01),
            color=get_source_colour("MG"),
        )
        ax.fill_between(
            np.arange(mass_gap_max, m1_max, 0.01),
            ns_max,
            mass_gap_max,
            color=get_source_colour("MG"),
        )

    # colour contour
    x1 = np.arange(mis, mib + 0.01, 0.01)
    x2 = np.arange(mib, lim_m1b, 0.01)

    if len(x1) > 0:
        ax.fill_between(
            x1,
            x1,
            mcm1_to_m2(mcs, x1),
            facecolor=(1, 1, 1, 0.5),
            edgecolor=(0, 0, 0, 0),
        )

    if len(x2) > 0:
        # errors if mib >= lim_m1b (x2 is empty)
        ax.fill_between(
            x2,
            mcm1_to_m2(mcb, x2),
            mcm1_to_m2(mcs, x2),
            facecolor=(1, 1, 1, 0.5),
            edgecolor=(0, 0, 0, 0),
        )

    # plot_details
    xlims = xlims or mass_bounds
    ylims = ylims or (1.0, 20.0)
    ax.set(xlim=xlims, ylim=ylims, xlabel=r"$m_1$", ylabel=r"$m_2$")

    return ax


def _draw_prob_pie_axes(ax: Axes, probabilities: Dict[str, float]) -> Axes:
    prob_plot = {k: v for k, v in probabilities.items() if v != 0.0}
    labels, sizes = zip(*prob_plot.items())
    colours = [get_source_colour(label) for label in labels]
    ax.pie(
        sizes,
        labels=labels,
        colors=colours,
        autopct="%1.1f%%",
        textprops={"fontsize": 15},
    )
    ax.axis("equal")
    return ax


def plot_prob_pie_figure(
    probabilities: Dict[str, float],
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    _draw_prob_pie_axes(ax, probabilities)
    return fig


class ChirpMassAreaModel:
    def __init__(
        self,
        a0: Optional[float] = None,
        b0: Optional[float] = None,
        b1: Optional[float] = None,
        m0: Optional[float] = None,
        mass_bounds: Tuple[float, float] = (1.0, 45.0),
        ns_max: float = 3.0,
        mass_gap_max: Optional[float] = None,
        separate_mass_gap: bool = False,
        lal_cosmology: bool = True,
    ):
        """Defines class-based Compact Binary Coalescence source classifier based on
        the PyCBC Chirp Mass Area method by Villa-Ortega et. al. (2021).

        Parameters
        ----------
        a0: float | None
            Model parameter used as the coefficient to estimate BAYESTAR distance from
            the minimum effective distance of a given single instrument event trigger.
        b0: float | None
            Model parameter used to estimate distance uncertainty?
        b1: float | None
            Model parameter used to estimate distance uncertainty?
        m0: float | None
            Inherent percentage uncertainty in chirp mass, set to 1% (0.01) by default.
        mass_bounds: tuple[float, float]
            The upper and lower bounds for both component masses (m1 >= m2).
        ns_max: float
            The boundary that separates a classification between BH and NS.
        mass_gap_max: float | None
            If mass_gap_max is set, we assign a Mass Gap (MG) category above ns_max.
        separate_mass_gap: bool
            If True, splits Mass Gap into BH+Gap, Gap+NS, and Gap+Gap.
        lal_cosmology: bool
            If True, it uses the Planck15 cosmology model
            as defined in lalsuite instead of the astropy default.

        Returns
        -------
        dict[str, float]
            A dictionary of probabilities predicted for each CBC source class.
        """

        self.a0 = a0
        self.b0 = b0
        self.b1 = b1
        self.m0 = m0

        self.mass_bounds = mass_bounds  # component mass bounds

        self.ns_max = ns_max
        self.mass_gap_max = mass_gap_max
        assert 0 < ns_max <= (self.mass_gap_max or ns_max)

        self.separate_mass_gap = separate_mass_gap
        self.lal_cosmology = lal_cosmology

    def __repr__(self, precision: int = 4):
        """Overrides string representation of cls when printed."""
        coefficents = ", ".join(
            [
                f"{key}={self.coefficients[key]!r}"
                if self.coefficients[key] is None
                else f"{key}={self.coefficients[key]:.{precision}f}"
                for key in self.coefficients
            ]
        )
        return f"{type(self).__name__}({coefficents})"

    # def __call__(self, mchirp: float, snr: float, eff_dist: float) -> dict[str, float]:
    #     return self.predict(mchirp, snr, eff_dist)

    @property
    def coefficients(self):
        return {"a0": self.a0, "b0": self.b0, "b1": self.b1, "m0": self.m0}

    #     @property
    #     def mass_bounds(self) -> Tuple[float, float]:
    #         return self._mass_bounds

    #     @mass_bounds.setter
    #     def mass_bounds(self, value):
    #         m_min, m_max = self.mass_bounds
    #         if not (0 < m_min <= m_max):
    #             raise ValueError("mass_bounds requires 0 < m_min <= m_max")
    #         try:
    #             self._mass_bounds = (float(m_min), float(m_max))
    #         except TypeError as error:
    #             raise TypeError("mass_bounds must be a tuple of floats") from error

    #     @property
    #     def mass_gap_bounds(self) -> Tuple[float, float]:
    #         return self._mass_gap_bounds

    #     @mass_gap_bounds.setter
    #     def mass_gap_bounds(self, value: Tuple[float, float]):
    #         mgap_min, mgap_max = self.mass_gap_bounds
    #         if not (0 < mgap_min <= mgap_max):
    #             raise ValueError("mass_gap_bounds requires 0 < m_min <= m_max")
    #         try:
    #             self._mass_bounds = (float(mgap_min), float(mgap_max))
    #         except TypeError as error:
    #             raise TypeError("mass_gap_bounds must be a tuple of floats") from error

    #     @property
    #     def separate_mass_gap(self) -> bool:
    #         return self._separate_mass_gap

    #     @separate_mass_gap.setter
    #     def separate_mass_gap(self, value: bool):
    #         if not isinstance(value, bool):
    #             raise TypeError("separate_mass_gap must be a bool")
    #         self._separate_mass_gap = value

    #     @property
    #     def lal_cosmology(self) -> bool:
    #         return self._lal_cosmology

    #     @lal_cosmology.setter
    #     def lal_cosmology(self, value: bool):
    #         if not isinstance(value, bool):
    #             raise TypeError("lal_cosmology must be a bool")
    #         self._lal_cosmology = value

    #     @property
    #     def coefficients(self) -> Dict[str, float]:
    #         return self._coefficients

    #     @coefficients.setter
    #     def coefficients(self, coeffs: Dict[str, float]):
    #         valid_coeffs = ("m0", "a0", "b0", "b1")
    #         for key in coeffs:
    #             if key not in valid_coeffs:
    #                 raise KeyError(f"{key} not in valid coeffs {valid_coeffs}")
    #             if not isinstance(coeffs[key], float):
    #                 raise TypeError(f"{key} type must be float, not {type(coeffs[key])}")
    #         if not (0 < coeffs["m0"] < 1):
    #             raise ValueError(f"m0 coeff should be within 0 and 1; m0 = {coeffs['m0']}")

    #         self._coefficients = coeffs

    # define distance estimation functions
    def _estimate_lum_dist(
        self,
        eff_distance: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Function to estimate luminosity distance from the minimum effective distance.

        TODO: Add mathematics and explanation to docstring.
        """
        assert self.a0 is not None, f"a0 coefficient is not initialised."
        return eff_distance * self.a0

    def _estimate_lum_dist_std(
        self,
        eff_distance: Union[float, np.ndarray],
        snr: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Function to estimate standard deviation of luminosity distance.

        TODO: Add mathematics and explanation to docstring.
        """
        assert self.a0 is not None, f"a0 coefficient is not initialised."
        assert self.b0 is not None, f"b0 coefficient is not initialised."
        assert self.b1 is not None, f"b1 coefficient is not initialised."

        lum_dist_std = (
            np.power(snr, self.b0)
            + np.exp(self.b1)
            + self._estimate_lum_dist(eff_distance)
        )

        return lum_dist_std

    def fit(
        self,
        bayestar_distances: np.ndarray,
        bayestar_stds: np.ndarray,
        eff_distances: np.ndarray,
        snrs: np.ndarray,
        m0: Optional[float] = None,
    ):
        """Fits a Chirp Mass Area model with equal length arrays for BAYESTAR luminosity
        distances against corresponding SNRs and (minimum) effective distances
        recovered by a gravitational wave search pipeline.

        The fitted coefficients are saved to the model instance's attributes as a0, b0,
        b1, and m0; and they can be accessed conveniently via the self.coefficients
        property. If m0 is provided but self.m0 is already initialised, m0 will be
        overwritten with the new value.

        This function uses numpy's Polynomial function to fit the coefficients for
        the chirp mass area model - specifically for estimating luminosity distance
        uncertainty (standard deviation) as a function of the estimated BAYESTAR
        luminosty distance and the recovered trigger Signal to Noise (SNR) ratios.

        See: https://numpy.org/doc/stable/reference/routines.polynomials.html

        Parameters
        ----------
        bayestar_distances: np.ndarray
            An array of BAYESTAR approximated luminosity distances as returned by the
            ligo.skymap BAYESTAR algorithm.
        bayestar_stds: np.ndarray
            An array of BAYESTAR approximated luminosity distance standard deviations
            as returned by the ligo.skymap BAYESTAR algorithm.
        eff_distances: np.ndarray
            An array of trigger effective distances recovered from a search pipeline.
        snrs: np.ndarray
            An array of trigger SNR values recovered from a search pipeline.
        m0: float | None
            A constant that defines the uncertainty in chirp mass.
        """
        # specify chirp mass uncertainty constant
        self.m0 = m0 or self.m0
        if self.m0 is None and m0 is None:
            raise ValueError(f"m0 coefficent not initialised - provide a value for m0.")

        # a0 taken as mean ratio between lum dist and (minimum) effective distances)
        self.a0 = float(np.mean(bayestar_distances / eff_distances))

        # estimate BAYESTAR luminosity distances
        bayes_dist_std = bayestar_stds / self._estimate_lum_dist(eff_distances)

        # fit luminosity distance uncertainty as a function of SNR
        b = Polynomial.fit(np.log(snrs), np.log(bayes_dist_std), 1)
        self.b1, self.b0 = b.convert().coef
        logger.info(f"Fitted coefficients for {self.__repr__}.")
        return self

    def predict(
        self,
        mchirp: float,
        snr: float,
        eff_dist: float,
        truncate_lower_dist: Optional[float] = 0.0003,
    ) -> Dict[str, float]:
        """
        Computes the different probabilities that a candidate event belongs to each
        CBC source class according to search.classify.mchirp_areas.calc_probabilities.

        Parameters
        ----------
        mchirp: float
            The source frame chirp mass.
        snr: float
            The coincident signal-to-noise ratio (SNR)
        eff_distance: float
            The estimated effective distance to the event,
            usually taken as the minimum across all coincident detectors.
        truncate_lower_dist: float | None
            If provided, takes the ceiling of truncate_lower_dist and the estimated
            lower uncertainty bound for distance to prevent negative or unrealistic
            distance estimates.

        Returns
        -------
        dict[str, float]
            The astrophysical source probabilities for each class.
        """
        assert self.m0 is not None, f"m0 coefficient is not initialised."
        assert self.a0 is not None, f"a0 coefficient is not initialised."
        assert self.b0 is not None, f"b0 coefficient is not initialised."
        assert self.b1 is not None, f"b1 coefficient is not initialised."

        # calc_probabilities does not type check mutable self.config nor coefficients
        probs = predict_source_p_astro(
            self.a0,
            self.b0,
            self.b1,
            self.m0,
            mchirp,
            snr,
            eff_dist,
            self.mass_bounds,  # component mass bounds
            self.ns_max,
            self.mass_gap_max,
            self.separate_mass_gap,
            self.lal_cosmology,
            truncate_lower_dist,
        )

        # # remove Mass Gap if bounds are the same
        # # TODO: implement a more elegant solution to remove MassGap
        # if self.mass_gap_bounds[0] == self.mass_gap_bounds[1]:
        #     del probs["MassGap"]

        return probs

    def save_pkl(self, path: Union[str, Path]):
        with Path(path).open(mode="wb") as f:
            pickle.dump(self.__dict__, f)

    def load_pkl(self, path: Union[str, Path]):
        with Path(path).open(mode="rb") as f:
            self.__dict__ = pickle.load(f)

        for key in ["a0", "b0", "b1", "m0"]:
            if getattr(self, key, None) is None:
                logger.info(f"{type(self).__name__} coefficient {key} not initialised.")

    def save_json(self, path: Union[str, Path], indent: int = 4):
        with Path(path).open(mode="w") as f:
            json.dump(self.__dict__, f, indent=indent)

    def load_json(self, path: Union[str, Path]):
        with Path(path).open(mode="r") as f:
            state = json.load(f)
        for key in state:
            setattr(self, key, state[key])

        for key in ["a0", "b0", "b1", "m0"]:
            if getattr(self, key, None) is None:
                logger.info(f"{type(self).__name__} coefficient {key} not initialised.")

    def load(self, path: Union[str, Path]):
        config_path = Path(path)
        if config_path.suffix == ".pkl":
            self.load_pkl(config_path)
        elif config_path.suffix == ".json":
            self.load_json(config_path)
        else:
            raise RuntimeError(f"{path} file type must be .pkl or .json")
