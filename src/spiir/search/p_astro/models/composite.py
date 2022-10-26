import logging
from typing import Optional

import numpy as np
import pandas as pd

from .. import fgmc, mchirp_area

logger = logging.getLogger(__name__)


class FGMCChirpMassAreaModel:
    def __init__(
        self,
        fgmc_model: fgmc.TwoComponentFGMCToyModel,
        mchirp_area_model: Optional[mchirp_area.ChirpMassAreaModel] = None,
    ):
        # to do: intiailise from config file(?)
        self.fgmc_model = fgmc_model
        self.mchirp_area_model = mchirp_area_model or mchirp_area.ChirpMassAreaModel(
            a0=0.7598851608618243, b0=0.6849582413938586, b1=0.4668975492108711, m0=0.01
        )

    def predict(
        self,
        far: float,
        snr: float,
        mchirp: float,
        eff_dist: float,
    ):
        astro_prob = self.fgmc_model.predict(far, snr)
        source_probs = self.mchirp_area_model.predict(mchirp, snr, eff_dist)
        probs = {key: source_probs[key] * astro_prob for key in source_probs}
        probs.update({"Terrestrial": 1 - astro_prob})
        return probs
