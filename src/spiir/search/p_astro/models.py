"""Module containing the computation of p_astro by source category.

Code sourced from https://git.ligo.org/lscsoft/p-astro/-/tree/master/ligo.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
from ligo.p_astro import MarginalizedPosterior, SourceType
from ligo.p_astro.computation import get_f_over_b

from .mchirp_area import ChirpMassAreaModel

logger = logging.getLogger(__name__)


class TwoComponentModel:
    def __init__(
        self,
        far_star: float = 3e-4,
        snr_star: float = 8.5,
        thresholds: Dict[str, Dict[str, float]] = None,
        far_live_time: Optional[float] = None,
        prior_type: str = "Uniform",
    ):
        # set FAR and SNR thresholds to classify as astro source for bayes factor model
        self.far_star = far_star
        self.snr_star = snr_star
        self.thresholds = thresholds

        # assign prior distribution type to counts
        valid_prior_types = ("Uniform", "Jeffreys")
        if prior_type not in valid_prior_types:
            raise ValueError(f"{prior_type} must be one of {valid_prior_types}.")
        self.prior_type = prior_type

        # mean posterior counts
        self.marginalized_posterior: Optional[MarginalizedPosterior] = None
        self.mean_counts: Optional[Dict[str, float]] = None
        self.far_live_time = far_live_time  # if not None, set noise counts with this

    def __repr__(self, precision: int = 4):
        """Overrides string representation of cls when printed."""
        if self.mean_counts is not None:
            mean_counts = ", ".join(
                [
                    f"{key}={self.mean_counts[key]:.{precision}f}"
                    for key in self.mean_counts
                ]
            )
            return f"{type(self).__name__}({mean_counts})"
        else:
            return f"{type(self).__name__}()"

    def bound_snr(self, far: float, snr: float, ifos: Iterable[str]) -> float:
        """Return an SNR that does not exceed the SNR threshold for FARs below a
        given FAR threshold for each interferometer combination.

        THis function is based on the choose_snr function from ligo.p_astro.computation.

        Parameters
        ----------
        far: float
            The false alarm rate (FAR) of the event
        snr: float
            The signal to noise ratio (SNR) of the event.
        ifos : Iterable[str]
            The set of interferometers that detected the event.

        Returns
        -------
        float
            The maximum between the observed SNR value and the SNR threshold.
        """

        def parse_ifos(key):
            return key if isinstance(key, str) else ",".join(key)

        if isinstance(snr, Iterable):
            far_threshold = [self.thresholds[parse_ifos(key)]["far"] for key in ifos]
            snr_threshold = [self.thresholds[parse_ifos(key)]["snr"] for key in ifos]
        else:
            far_threshold = self.thresholds[parse_ifos(ifos)]["far"]
            snr_threshold = self.thresholds[parse_ifos(ifos)]["snr"]

        far_threshold = np.array(far_threshold)
        snr_threshold = np.array(snr_threshold)
        is_beyond_threshold = (snr > snr_threshold) & (far < far_threshold)
        bounded_snr = np.where(is_beyond_threshold, snr_threshold, snr)
        return bounded_snr if isinstance(snr, Iterable) else bounded_snr.item()

    def fit(
        self, far: np.ndarray, snr: np.ndarray, far_live_time: Optional[float] = None
    ):
        # approximate bayes factor
        bayes_factors = get_f_over_b(far, snr, self.far_star, self.snr_star)
        assert len(bayes_factors.shape) == 1, "bayes_factors should be a 1-dim array."

        # construct two component posterior for signal vs. noise
        astro = SourceType(label="Astro", w_fgmc=np.ones(len(bayes_factors)))
        terr = SourceType(label="Terr", w_fgmc=np.ones(len(bayes_factors)))
        self.marginalized_posterior = MarginalizedPosterior(
            f_divby_b=bayes_factors,
            prior_type=self.prior_type,
            terr_source=terr,
            **{"Astro": astro},
        )

        # update expected mean counts given observed data
        self.mean_counts = {
            key: self.marginalized_posterior.getOneDimMean(category=key)
            for key in ("Astro", "Terr")
        }

        far_live_time = far_live_time or self.far_live_time
        if far_live_time is not None:
            self.mean_counts["Terr"] = self.far_star * far_live_time

        return self

    def predict(
        self,
        far: float,
        snr: float,
        ifos: Optional[Iterable[str]] = None,
    ) -> float:
        assert self.marginalized_posterior is not None, "Model not fit - call .fit()."

        # Ensure SNR does not increase indefinitely beyond limiting FAR
        if self.thresholds is not None and ifos is not None:
            snr = self.bound_snr(far, snr, ifos)

        # compute bayes factor for foreground vs background trigger distribution
        bayes_factors = get_f_over_b(far, snr, self.far_star, self.snr_star)

        # return p_astro calculation for the given trigger
        return self.marginalized_posterior.pastro_update(
            categories=["Astro"],
            bayesfac_dict={"Astro": bayes_factors},
            mean_values_dict=self.mean_counts,
        )

    def save(self, path: Union[str, Path]):
        file_path = Path(path)
        if file_path.suffix == ".pkl":
            self.save_pkl(file_path)
        elif file_path.suffix == ".json":
            raise NotImplementedError("JSON compatibility not yet implemented.")
        else:
            raise RuntimeError(
                f"Save failed - cannot detect file type: {file_path.suffix}. "
                "Valid file types are '.pkl'."
            )

    def load(self, path: Union[str, Path]):
        file_path = Path(path)
        if file_path.suffix == ".pkl":
            self.load_pkl(file_path)
        elif file_path.suffix == ".json":
            raise NotImplementedError("JSON compatibility not yet implemented.")
        else:
            raise RuntimeError(
                f"Save failed - cannot detect file type: {file_path.suffix}. "
                "Valid file types are '.pkl'."
            )

    def save_pkl(self, path: Union[str, Path]):
        with Path(path).open(mode="wb") as f:
            pickle.dump(self.__dict__, f)

    def load_pkl(self, path: Union[str, Path]):
        with Path(path).open(mode="rb") as f:
            self.__dict__ = pickle.load(f)


class CompositeModel:
    def __init__(
        self,
        signal_model: Optional[TwoComponentModel] = None,
        source_model: Optional[ChirpMassAreaModel] = None,
    ):
        self.signal_model = signal_model or TwoComponentModel()
        self.source_model = source_model or ChirpMassAreaModel()

    def load(
        self,
        signal_config: str,
        source_config: str,
    ):
        self.signal_model.load(signal_config)
        self.source_model.load(source_config)

    def predict(
        self,
        far: float,
        snr: float,
        mchirp: float,
        eff_dist: float,
        ifos: Optional[Iterable[str]] = None,
    ) -> Dict[str, float]:
        astro_prob = self.signal_model.predict(far, snr, ifos)
        source_probs = self.source_model.predict(mchirp, snr, eff_dist)
        probs = {key: source_probs[key] * astro_prob for key in source_probs}
        probs.update({"Terrestrial": 1 - astro_prob})
        return probs
