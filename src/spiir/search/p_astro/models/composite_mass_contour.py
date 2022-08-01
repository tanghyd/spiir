import logging
from typing import Optional

import pandas as pd
import numpy as np

from .. import fgmc, mass_contour

logger = logging.getLogger(__name__)


class FGMCMassContourModel:
    def __init__(
        self,
        fgmc_model: fgmc.TwoComponentFGMCToyModel,
        mass_contour_model: Optional[mass_contour.MassContourModel]=None,
    ):
        # to do: intiailise from config file(?)
        self.fgmc_model = fgmc_model
        self.mass_contour_model = mass_contour_model or mass_contour.MassContourModel(
            a0=0.7598851608618243,
            b0=0.6849582413938586,
            b1=0.4668975492108711,
            m0=0.01
        )

    def predict(
        self,
        far: float,
        snr: float,
        mchirp: float,
        eff_dist: float,
    ):
        astro_prob = self.fgmc_model.predict(far, snr)
        source_probs = self.mass_contour_model.predict(mchirp, snr, eff_dist)
        probs = {key: source_probs[key]*astro_prob for key in source_probs}
        probs.update({"Terrestrial": 1 - astro_prob})
        return probs