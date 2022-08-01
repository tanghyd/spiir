import json

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .plot import _draw_mass_contour_axes, _draw_prob_pie_axes
from .predict import calc_probabilities, predict_source_p_astro, predict_redshift


# Class based method for estimation
class MassContourEstimator:
    def __init__(
        self,
        coefficients: dict[str, float],
        m_bounds: tuple[float, float] = (1.0, 45.0),
        mgap_bounds: tuple[float, float] = (3.0, 5.0),
        group_mgap: bool = True,
        lal_cosmology: bool = True,
    ):
        """
        Defines class-based Compact Binary Coalescence source classifier based on the
        PyCBC Mass Plane Contour method by Villa-Ortega et. al. (2021).

        Parameters
        ----------
        coefficients: dict[str, float]
            The estimated model coefficients of fitted mass/distance models.
        m_bounds: tuple[float, float]
            The upper and lower bounds for both component masses (m1 >= m2).
        mgap_bounds: tuple[float, float]
            The boundaries that define the mass gap between BH and NS.
        group_mgap: bool
            If True, aggregates Mass Gap from BH+Gap, Gap+NS, and Gap+Gap.
        lal_cosmology: bool
            If True, it uses the Planck15 cosmology model
            as defined in lalsuite instead of the astropy default.

        Returns
        -------
        dict[str, float]
            A dictionary of probabilities predicted for each CBC source class.

        """
        self._coefficients = coefficients  # fitted model coefficients
        self._m_bounds = m_bounds  # component mass bounds
        self._mgap_bounds = mgap_bounds  # mass gap class bounds
        self._group_mgap = group_mgap
        self._lal_cosmology = lal_cosmology

    @property
    def m_bounds(self) -> tuple[float, float]:
        return self._m_bounds

    @m_bounds.setter
    def m_bounds(self, value):
        m_min, m_max = self.m_bounds
        if not (0 < m_min <= m_max):
            raise ValueError("m_bounds requires 0 < m_min <= m_max")
        try:
            self._m_bounds = (float(m_min), float(m_max))
        except TypeError as error:
            raise TypeError("m_bounds must be a tuple of floats") from error

    @property
    def mgap_bounds(self) -> tuple[float, float]:
        return self._mgap_bounds

    @mgap_bounds.setter
    def mgap_bounds(self, value: tuple[float, float]):
        mgap_min, mgap_max = self.mgap_bounds
        if not (0 < mgap_min <= mgap_max):
            raise ValueError("mgap_bounds requires 0 < m_min <= m_max")
        try:
            self._m_bounds = (float(mgap_min), float(mgap_max))
        except TypeError as error:
            raise TypeError("mgap_bounds must be a tuple of floats") from error

    @property
    def group_mgap(self) -> bool:
        return self._group_mgap

    @group_mgap.setter
    def group_mgap(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("group_mgap must be a bool")
        self._group_mgap = value

    @property
    def lal_cosmology(self) -> bool:
        return self._lal_cosmology

    @lal_cosmology.setter
    def lal_cosmology(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("lal_cosmology must be a bool")
        self._lal_cosmology = value

    @property
    def coefficients(self) -> dict[str, float]:
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coeffs: dict[str, float]):
        valid_coeffs = ("m0", "a0", "b0", "b1")
        for key in coeffs:
            if key not in valid_coeffs:
                raise KeyError(f"{key} not in valid coeffs {valid_coeffs}")
            if not isinstance(coeffs[key], float):
                raise TypeError(f"{key} type must be float, not {type(coeffs[key])}")
        if not (0 < coeffs["m0"] < 1):
            raise ValueError(f"m0 coeff should be within 0 and 1; m0 = {coeffs['m0']}")

        self._coefficients = coeffs

    def plot(
        self,
        mchirp: float,
        snr: float,
        eff_dist: float,
        truncate_lower_dist: float = 0.0003,
        suptitle: str | None = None,
        figsize: tuple[float, float] = (16, 6),
        outfile: str | None = None,
    ) -> Figure:
        # closest black hole is 1000LYrs / 0.0003Mpc [https://doi.org/10.1051/0004-6361/202038020]

        # predict redshift and mass uncertainties according to model coefficients
        mchirp_std = mchirp * self.coefficients["m0"]
        z, z_std = predict_redshift(
            self.coefficients, snr, eff_dist, self._lal_cosmology, truncate_lower_dist
        )

        # calculate class probabilities given mchirp and redshift uncertainty
        probabilities = calc_probabilities(
            self.coefficients,
            mchirp,
            z,
            z_std,
            self._m_bounds,
            self._mgap_bounds,
            self._group_mgap,
        )

        # plot paired figure plot
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)
        _draw_mass_contour_axes(
            ax1, mchirp, mchirp_std, z, z_std, self._m_bounds, self._mgap_bounds
        )
        _draw_prob_pie_axes(ax2, probabilities)

        if suptitle:
            fig.suptitle(suptitle)

        if outfile is not None:
            fig.savefig(outfile)

        return fig

    def predict(
        self,
        mchirp: float,
        snr: float,
        eff_dist: float,
        truncate_lower_dist: float | None = 0.0003,
    ) -> dict[str, float]:
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
            If provided, takes the ceiling of truncate_lower_dist and the estimated lower uncertainty
            bound for distance to prevent negative or unrealistic distance estimates.

        Returns
        -------
        dict[str, float]
            The astrophysical source probabilities for each class.
        """

        # calc_probabilities does not type check mutable self.config nor coefficients
        return predict_source_p_astro(
            self.coefficients,
            mchirp,
            snr,
            eff_dist,
            self._m_bounds,  # component mass bounds
            self._mgap_bounds,  # mass gap class bounds
            self._group_mgap,
            self._lal_cosmology,
            truncate_lower_dist,
        )

    def __call__(self, mchirp: float, snr: float, eff_dist: float) -> dict[str, float]:
        return self.predict(mchirp, snr, eff_dist)
