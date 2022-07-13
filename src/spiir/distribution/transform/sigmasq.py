import concurrent.futures
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ...io.ligolw import load_all_ligolw_frequency_arrays
from ...filter import compute_sigmasq
from ...inspiral import waveform

logger = logging.getLogger(__name__)


# TODO: Add documentation explaining what SPIIRSigmaSqTransform does
class SigmaSqTransform:
    transform = "sigmasq"

    def __init__(
        self,
        nprocs: int = 1,
        **kwargs,
    ):
        # multiprocessing settings
        self.nprocs = nprocs

        # load psd data
        self.kwargs = kwargs
        self.psds: Optional[pd.DataFrame] = None
        if "psds" in self.kwargs:
            self.psds = pd.DataFrame(
                load_all_ligolw_frequency_arrays(psd_xml)
                for ifo, psd_xml in self.kwargs["psds"].items()
            )

        elif "psd" in self.kwargs:
            psd_xml_path = Path(self.kwargs["psd"])
            assert psd_xml_path.is_file()
            if "ifos" in self.kwargs:
                self.psds = pd.DataFrame(
                    load_all_ligolw_frequency_arrays(psd_xml_path)
                    for ifo in self.kwargs["ifos"]
                )

            else:
                raise KeyError(f"no 'ifos' specified in kwargs: {self.kwargs}")
        else:
            logger.warning("No 'psds' or 'psd' argument supplied to SigmaSqTransform.")

        if self.psds is None:
            self.ifos = self.kwargs["ifos"]
        else:
            self.ifos = self.psds.columns.tolist()

        self.delta_f = 0.0625

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.apply(data)

    def apply(self, samples: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        logger.warning("SigmaSqTransform.apply method in incomplete testing state!")
        assert isinstance(samples, pd.DataFrame)

        # instantiate array
        freq_bins = int((1024) / (1 / 16)) + 1  # f_high / delta_f inclusive
        sigmasqs = np.empty(samples.shape[0], dtype=np.float64)

        # transform precessing parameters
        precession_parameters = [
            "mass_1",
            "mass_2",
            "theta_jn",
            "theta_1",
            "theta_2",
            "phi_jl",
            "chi_1",
            "chi_2",
            "phi_12",
            "phase",
        ]

        # run concurrent.futures
        with tqdm(
            total=len(samples),
            desc="Simulating CBC inspiral waveforms to compute sigmasq",
            miniters=1,
            disable=not verbose,
        ) as progress_bar:
            records = samples[precession_parameters].to_dict("records")
            for i, sample in enumerate(records):
                output_params = waveform.transform_precessing_spins(
                    **sample, f_ref=20.0
                )
                (
                    incl,
                    spin_1x,
                    spin_1y,
                    spin_1z,
                    spin_2x,
                    spin_2y,
                    spin_2z,
                ) = output_params
                params = dict(
                    mass1=sample["mass_1"],
                    mass2=sample["mass_2"],
                    spin1x=spin_1x,
                    spin1y=spin_1y,
                    spin1z=spin_1z,
                    spin2x=spin_2x,
                    spin2y=spin_2y,
                    spin2z=spin_2z,
                    phase=sample["phase"],
                    duration=16.0,
                    f_low=20,
                    f_high=1024,
                    approximant="IMRPhenomPv2",
                    inclination=incl,
                )

                # simulate waveforms
                try:
                    hp, _ = waveform.simulate_frequency_domain_inspiral(**params)
                except RuntimeError as exc:
                    print(f"params: {params}")
                    raise exc
                # waveforms[i] = hp.data.data.real

                # compute expected SNR at standard fiducial distance (sigmasq) for each ifo
                if self.psds is not None:
                    psd = self.psds.iloc[:freq_bins]["H1"].values
                else:
                    psd = self.psds

                sigmasq = compute_sigmasq(
                    hp.data.data.real,
                    delta_f=self.delta_f,
                    psd=psd,
                    f_low_cutoff=20,
                    f_high_cutoff=1024,
                )

                assert isinstance(sigmasq, np.ndarray)
                sigmasqs[i] = sigmasq.max()
                progress_bar.update(1)

        # aggregate an example "network" sigmasq
        sigmasq_df = pd.DataFrame(sigmasqs, columns=["sigmasq"], index=samples.index)

        # concat extra transform data and return
        return pd.concat([samples, sigmasq_df], axis=1)
