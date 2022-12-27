"""Module for source probability estimation and and plotting for the
'mchirp_area' method as proposed by Villa-Ortega et. al. (2020).

Initial code by A. Curiel Barroso, August 2019
Modified by V. Villa-Ortega, January 2020, March 2021
Modified by D. Tang for SPIIR March 2022

Code forked from PyCBC for SPIIR development (March 2022)
Source: https://github.com/gwastro/pycbc/blob/master/pycbc/mchirp_area.py
Additional edits drawn from prior work done by V. Villa-Ortega:
https://github.com/veronica-villa/source_probabilities_estimation_pycbclive

"""
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.polynomial import Polynomial
from pycbc.conversions import mass2_from_mchirp_mass1 as mcm1_to_m2
from pycbc.mchirp_area import calc_areas, redshift_estimation, src_mass_from_z_det_mass

logger = logging.getLogger(__name__)

SOURCE_COLOR_MAP = {
    "BNS": "#A2C8F5",  # light blue
    "NSBH": "#FFB482",  # light orange
    "BBH": "#FE9F9B",  # light red
    "MG": "#8EE5A1",  # light green
    "MGNS": "#98D6CB",  # turquoise
    "MGMG": "#79BB87",  # green
    "BHMG": "#C6C29E",  # dark khaki
}


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
        """Defines a Compact Binary Coalescence source classifier class based on the
        PyCBC Chirp Mass Area method (mchirp_area.py) by Villa-Ortega et. al. (2021).

        Parameters
        ----------
        a0: float | None
            Model parameter used as the coefficient to estimate BAYESTAR distance from
            the minimum effective distance of a given single instrument event trigger.
        b0: float | None
            Model parameter used to estimate luminosity distance uncertainty.
        b1: float | None
            Model parameter used to estimate luminosity distance uncertainty.
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
            If True, it uses the Planck15 cosmology model as defined in
             lalsuite instead of the astropy default.
        """
        # model coefficients
        self.a0, self.b0, self.b1, self.m0 = a0, b0, b1, m0

        # specify component mass value boundaries
        self.mass_bounds = mass_bounds  # component mass bounds for integration
        self.separate_mass_gap = separate_mass_gap
        self.mass_gap_max = mass_gap_max  # if None, no mass_gap classes used
        self.ns_max = ns_max
        assert 0 < ns_max <= (self.mass_gap_max or ns_max)

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

    @property
    def coefficients(self):
        return {"a0": self.a0, "b0": self.b0, "b1": self.b1, "m0": self.m0}

    def estimate_luminosity_distance(self, eff_distance: float) -> float:
        assert self.a0 is not None, "coefficient 'a0' is not initialised"
        return eff_distance * self.a0

    def estimate_luminosity_distance_uncertainty(
        self,
        eff_distance: float,
        snr: float,
    ) -> float:
        assert self.b0 is not None, "coefficient 'b0' is not initialised"
        assert self.b1 is not None, "coefficient 'b1' is not initialised"

        return (
            np.power(snr, self.b0)
            + np.exp(self.b1)
            + self.estimate_luminosity_distance(eff_distance)
        )

    def estimate_distance(self, eff_distance: float, snr: float) -> Tuple[float, float]:
        distance = self.estimate_luminosity_distance(eff_distance)
        distance_std = self.estimate_luminosity_distance_uncertainty(eff_distance, snr)
        return distance, distance_std

    def estimate_redshift(
        self, distance: float, distance_std: float
    ) -> Tuple[float, float]:
        with warnings.catch_warnings():
            # hide astropy z_at_value warning when input has negative distance
            warnings.simplefilter("ignore", category=AstropyUserWarning)
            z = redshift_estimation(distance, distance_std, self.lal_cosmology)
        return z["central"], z["delta"]

    def calculate_probabilities(
        self,
        mchirp: float,
        z: float,
        z_std: float,
    ) -> Dict[str, float]:
        """Computes the astrophysical source probability of a given candidate event."""
        assert self.m0 is not None, "coefficient 'm0' is not initialised"

        # determine chirp mass bounds in detector frame - mc_det = (1+z)*mc
        get_redshifted_mchirp = lambda m: (m / (2**0.2)) * (1 + z)
        mchirp_min, mchirp_max = (get_redshifted_mchirp(m) for m in self.mass_bounds)

        # determine astrophysical source class probabilities given estimated parameters
        if mchirp > mchirp_max:
            # if observed detector frame chirp mass is greater than mchirp bounds => BBH
            probabilities = {"BNS": 0.0, "NSBH": 0.0, "BBH": 1.0}

            if self.mass_gap_max is not None and self.mass_gap_max > self.ns_max:
                if not self.separate_mass_gap:
                    probabilities["MG"] = 0.0
                else:
                    probabilities.update({"MGNS": 0.0, "MGMG": 0.0, "BHMG": 0.0})

        elif mchirp < mchirp_min:
            # if observed detector frame chirp mass is less than mchirp bounds => BNS
            probabilities = {"BNS": 1.0, "NSBH": 0.0, "BBH": 0.0}

            if self.mass_gap_max is not None and self.mass_gap_max > self.ns_max:
                if not self.separate_mass_gap:
                    probabilities["MG"] = 0.0
                else:
                    probabilities.update({"MGNS": 0.0, "MGMG": 0.0, "BHMG": 0.0})

        else:
            # specify mass gap class maximum if provided, else match neutron star
            mass_gap = True if self.mass_gap_max is not None else False
            mass_gap_max = self.mass_gap_max or self.ns_max

            # compute probabilities according to proportional areas in chirp mass area
            areas = calc_areas(
                trig_mc_det={"central": mchirp, "delta": mchirp * self.m0},
                mass_limits=dict(zip(("min_m2", "max_m1"), self.mass_bounds)),
                mass_bdary={"ns_max": self.ns_max, "gap_max": mass_gap_max},
                z={"central": z, "delta": z_std},
                mass_gap=mass_gap,
                mass_gap_separate=self.separate_mass_gap,
            )

            # rename keys output by calc_areas if component mass gaps are computed
            if mass_gap:
                key_map = {"Mass Gap": "MG", "GG": "MGMG", "GNS": "MGNS"}
                for key in list(areas):
                    if key in key_map:
                        areas[key_map[key]] = areas.pop(key)

            # normalize mass contour area to get probability estimates
            total_area = sum(areas.values())
            probabilities = {key: areas[key] / total_area for key in areas}

        return probabilities

    def fit(
        self,
        snr: np.ndarray,
        eff_distance: np.ndarray,
        bayestar_distance: np.ndarray,
        bayestar_distance_std: np.ndarray,
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
        snr: np.ndarray
            An array of trigger SNR values recovered from a search pipeline.
        eff_distance: np.ndarray
            An array of trigger effective distances recovered from a search pipeline.
        bayestar_distance: np.ndarray
            An array of BAYESTAR approximated luminosity distances as returned by the
            ligo.skymap BAYESTAR algorithm.
        bayestar_distance_std: np.ndarray
            An array of BAYESTAR approximated luminosity distance standard deviations
            as returned by the ligo.skymap BAYESTAR algorithm.
        m0: float | None
            A constant that defines the uncertainty in chirp mass.
        """
        # specify chirp mass uncertainty constant
        self.m0 = m0 or self.m0
        assert self.m0 is not None, "coefficient 'm0' is not initialised"

        # a0 taken as mean ratio between lum dist and (minimum) effective distances)
        self.a0 = float(np.mean(bayestar_distance / eff_distance))

        # estimate BAYESTAR luminosity distances
        luminosity_distance = self.estimate_luminosity_distance(eff_distance)
        norm_bayestar_distance_std = bayestar_distance_std / luminosity_distance

        # fit luminosity distance uncertainty as a function of SNR
        b = Polynomial.fit(np.log(snr), np.log(norm_bayestar_distance_std), 1)
        self.b1, self.b0 = b.convert().coef

        return self

    def predict(
        self,
        mchirp: float,
        snr: float,
        eff_distance: float,
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

        Returns
        -------
        dict[str, float]
            The astrophysical source probabilities for each class.
        """

        distance, distance_std = self.estimate_distance(eff_distance, snr)
        z, z_std = self.estimate_redshift(distance, distance_std)
        return self.calculate_probabilities(mchirp, z, z_std)

    def plot(
        self,
        mchirp: float,
        snr: float,
        eff_distance: float,
        ax: Optional[Axes] = None,
        *args,
        **kwargs,
    ) -> Axes:
        """Draws the estimated proportional chirp mass area plane with the detector
        frame trigger data as input to the model.

        Parameters
        ----------
        mchirp: float
            The source frame chirp mass.
        snr: float
            The coincident signal-to-noise ratio (SNR)
        eff_distance: float
            The estimated effective distance to the event,
            usually taken as the minimum across all coincident detectors.
        ax: Axes, optional
            The matplotlib Axes object to draw on. If None, one is created.

        Returns
        -------
        matplotlib.axes.Axes
            The mass contour on the chirp mass area plane as a matplotlib Axes.
        """
        # TODO: Implement a more efficient solution when we want to predict + plot.

        # estimate source frame chirp mass from trigger data
        assert self.m0 is not None
        distance, distance_std = self.estimate_distance(eff_distance, snr)
        z, z_std = self.estimate_redshift(distance, distance_std)
        m_src, m_src_std = src_mass_from_z_det_mass(z, z_std, mchirp, mchirp * self.m0)

        return draw_mass_contour_plane(
            ax or plt.gca(),  # use the provided matplotlib axes, else create one
            m_src - m_src_std,  # source frame chirp mass lower bound
            m_src + m_src_std,  # source frame chirp mass upper bound
            self.mass_bounds,
            self.ns_max,
            self.mass_gap_max,
            *args,
            **kwargs,
        )

    def save(self, path: Union[str, Path]):
        file_path = Path(path)
        if file_path.suffix == ".pkl":
            self.save_pkl(file_path)
        elif file_path.suffix == ".json":
            self.load_json(file_path)
        else:
            raise RuntimeError(
                f"Save failed - cannot detect file type: {file_path.suffix}. "
                "Valid file types are '.pkl' or '.json'."
            )

    def load(self, path: Union[str, Path]):
        file_path = Path(path)
        if file_path.suffix == ".pkl":
            self.load_pkl(file_path)
        elif file_path.suffix == ".json":
            self.load_json(file_path)
        else:
            raise RuntimeError(
                f"Save failed - cannot detect file type: {file_path.suffix}. "
                "Valid file types are '.pkl' or '.json'."
            )

    def save_pkl(self, path: Union[str, Path]):
        with Path(path).open(mode="wb") as f:
            pickle.dump(self.__dict__, f)

    def save_json(self, path: Union[str, Path], indent: int = 4):
        with Path(path).open(mode="w") as f:
            json.dump(self.__dict__, f, indent=indent)

    def load_pkl(self, path: Union[str, Path]):
        with Path(path).open(mode="rb") as f:
            self.__dict__ = pickle.load(f)

    def load_json(self, path: Union[str, Path]):
        with Path(path).open(mode="r") as f:
            state = json.load(f)
        for key in state:
            setattr(self, key, state[key])


def draw_mass_contour_plane(
    ax: Axes,
    mchirp_lower: float,
    mchirp_upper: float,
    mass_bounds: Tuple[float, float],
    ns_max: float = 3.0,
    mass_gap_max: Optional[float] = None,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    *args,
    **kwargs,
) -> Axes:
    """Draws a matplotlib Axes visualising the proportional chirp mass area plane.

    Source: Veronica Villa-Ortega, March 2021
    - https://github.com/veronica-villa/source_probabilities_estimation_pycbclive
        /blob/f2656b4762a232d4758db88569e0b7ab45756ead/mc_area_plots.py
    """
    # determine component masses (when m1 = m2) given chirp mass boundaries
    mcs, mcb = mchirp_lower, mchirp_upper
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
        color=SOURCE_COLOR_MAP["NSBH"],
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(mass_gap_max, m1_max, 0.01),
        0.0,
        ns_max,
        color=SOURCE_COLOR_MAP["NSBH"],
    )
    ax.fill_between(
        np.arange(mass_gap_max, m1_max, 0.01),
        np.arange(mass_gap_max, m1_max, 0.01),
        m1_max,
        color=SOURCE_COLOR_MAP["BBH"],
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(mass_gap_max, m1_max, 0.01),
        np.arange(mass_gap_max, m1_max, 0.01),
        mass_gap_max,
        color=SOURCE_COLOR_MAP["BBH"],
    )
    ax.fill_between(
        np.arange(0.0, ns_max, 0.01),
        0.0,
        np.arange(0.0, ns_max, 0.01),
        color=SOURCE_COLOR_MAP["BNS"],
    )
    ax.fill_between(
        np.arange(0.0, ns_max, 0.01),
        ns_max,
        np.arange(0.0, ns_max, 0.01),
        color=SOURCE_COLOR_MAP["BNS"],
        alpha=0.5,
    )

    if mass_gap_max > ns_max:
        ax.fill_between(
            np.arange(0.0, ns_max, 0.01),
            ns_max,
            mass_gap_max,
            color=SOURCE_COLOR_MAP["MG"],
            alpha=0.5,
        )
        ax.fill_between(
            np.arange(ns_max, mass_gap_max, 0.01),
            np.arange(ns_max, mass_gap_max, 0.01),
            m1_max,
            color=SOURCE_COLOR_MAP["MG"],
            alpha=0.5,
        )
        ax.fill_between(
            np.arange(ns_max, mass_gap_max, 0.01),
            np.arange(ns_max, mass_gap_max, 0.01),
            color=SOURCE_COLOR_MAP["MG"],
        )
        ax.fill_between(
            np.arange(mass_gap_max, m1_max, 0.01),
            ns_max,
            mass_gap_max,
            color=SOURCE_COLOR_MAP["MG"],
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
