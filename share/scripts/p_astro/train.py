#!/usr/bin/env python

import functools
import logging
import time
from pathlib import Path
from typing import Optional, Union, Sequence

import click
import pandas as pd
from ligo.p_astro.computation import get_f_over_b
from spiir.cli import click_logger_options
from spiir.logging import configure_logger
from spiir.search import p_astro

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)


def load_mchirp_area_data(data_dir: Path) -> pd.DataFrame:
    """Loads the training data used to fit the p_astro Chirp Mass Area model."""
    mchirp_area_train_data = pd.read_csv(data_dir /
                                         "mchirp_area_train_data.csv")
    mchirp_area_test_data = pd.read_csv(data_dir / "mchirp_area_test_data.csv")
    mchirp_area_data = pd.concat(
        {
            "Train": mchirp_area_train_data,
            "Test": mchirp_area_test_data
        },
        names=["split", "index"])

    return mchirp_area_data.reset_index()


def load_fgmc_data(
    data_dir: Path,
    far_star: Optional[float] = None,
    snr_star: Optional[float] = None,
    signal_graceids: Optional[Sequence[str]] = None,
    start_gps_time: Optional[int] = None,
    split_gps_time: Optional[int] = None,
    test_split_size: Optional[float] = None,
) -> pd.DataFrame:
    """Loads the training data used to fit the p_astro FGMC model."""
    triggers = pd.read_csv(data_dir / "fgmc_data.csv")

    if signal_graceids is not None:
        triggers["signal"] = triggers.graceid.isin(signal_graceids)

        n_signal, n_total = len(
            triggers.loc[triggers["signal"]]), len(triggers)
        ratio = f"{n_signal} / {n_total}"
        percent = n_signal / n_total  # percentage
        logger.debug(
            f"Assigned {ratio} ({percent:.2%}) triggers with signal labels.")

    if start_gps_time is not None:
        triggers = triggers.loc[triggers.end_time >= start_gps_time]
        logger.debug(
            f"Selected triggers with end_time later than {start_gps_time} gps time."
        )

    if split_gps_time is not None and test_split_size is not None:
        raise RuntimeError(
            "Either split_gps_time or test_split_size can be set, not both.")

    elif split_gps_time is not None:
        # treat test_split variable as a
        triggers["split"] = [
            "Train" if x > split_gps_time else "Test"
            for x in triggers.end_time
        ]

        train_split = (triggers["split"]
                       == "Train").astype(int).sum() / len(triggers)
        test_split = (triggers["split"]
                      == "Test").astype(int).sum() / len(triggers)

        logger.debug(f"Split data at {split_gps_time} into train "
                     f"({train_split:.2%}) and test ({test_split:.2%}).")

    elif test_split_size is not None:
        from sklearn.model_selection import train_test_split
        assert 0 < test_split_size <= 1, f"test_split_size must be between 0 and 1."

        indices, _ = train_test_split(triggers.index,
                                      test_size=test_split_size,
                                      random_state=0)
        triggers["split"] = [
            "Train" if x in indices else "Test" for x in triggers.index
        ]

        train_split = 1 - test_split_size
        logger.debug(f"Split data into train ({train_split:.2%}) "
                     f"and test ({test_split_size:.2%}).")

    if far_star is not None and snr_star is not None:
        triggers["bayes_factor"] = get_f_over_b(triggers.far,
                                                triggers.snr,
                                                far_star=far_star,
                                                snr_star=snr_star)

        logger.debug("Approxiated bayes factor for FGMC triggers with "
                     f"far_star={far_star} and snr_star={snr_star}.")

    return triggers


def train_fgmc_model(
    triggers: pd.DataFrame,
    far_star: float,
    snr_star: float,
    out_dir: Optional[Path] = None,
    overwrite: bool = False,
):
    # train signal classification model from ligo.p_astro's method (FGMC)
    signal_model = p_astro.models.TwoComponentModel(
        far_star=far_star,
        snr_star=snr_star,
    )
    signal_model.fit(far=triggers["far"].values, snr=triggers["snr"].values)
    logger.debug(f"{signal_model} training complete.")

    if out_dir is not None:
        file_path = out_dir / "fgmc.pkl"
        if file_path.exists() and not overwrite:
            logger.debug(f"File alrady exists at {file_path}.")
        else:
            signal_model.save(file_path)
            logger.debug(f"p_astro signal model state saved to {file_path}.")

    return signal_model


def train_mchirp_area_model(
    triggers: pd.DataFrame,
    out_dir: Optional[Path] = None,
    overwrite: bool = False,
):
    # train source classification model from PyCBC's mchirp_area method
    source_model = p_astro.mchirp_area.ChirpMassAreaModel()
    source_model.fit(
        snr=triggers["cohsnr"],
        eff_distance=triggers["min_dist"],
        bayestar_distance=triggers["bay_dist"],
        bayestar_distance_std=triggers["bay_std"],
        m0=0.01,
    )
    logger.debug(f"{source_model} training complete.")

    if out_dir is not None:
        file_path = out_dir / "mchirp_area.pkl"
        if file_path.exists() and not overwrite:
            logger.debug(f"File alrady exists at {file_path}.")
        else:
            source_model.save(file_path)
            logger.debug(f"p_astro source model state saved to {file_path}.")

    return source_model


@click.command()
@click.argument("data-dir", type=click.Path(file_okay=False))
@click.argument("out-dir", type=click.Path(file_okay=False))
@click.option("--far-star", type=float, default=3e-4, show_default=True)
@click.option("--snr-star", type=float, default=8.5, show_default=True)
@click.option("--overwrite", is_flag=True)
@click_logger_options
def main(
    data_dir: Union[str, Path],
    out_dir: Union[str, Path],
    far_star: float = 3e-4,
    snr_star: float = 8.5,
    overwrite: bool = False,
    log_level: int = logging.WARNING,
    log_file: Optional[Union[str, Path]] = None,
):
    duration = time.perf_counter()
    configure_logger(logger, log_level, log_file)

    # load GraceDB data to train and evaluate p_astro model
    data_dir = Path(data_dir)
    mchirp_area_triggers = load_mchirp_area_data(data_dir)
    fgmc_triggers = load_fgmc_data(
        data_dir,
        far_star=far_star,
        snr_star=snr_star,
        start_gps_time=1238166018,  # 1238166018 is the start of O3a 
        # split_gps_time=1256655618,  # 1256655618 is the start of O3b
        # test_split_size=0.2,
        signal_graceids=mchirp_area_triggers.graceid,
    )

    # run training for both p_astro model components and save model to file
    out_dir = Path(out_dir)
    if "split" in fgmc_triggers.columns:
        logger.debug(f"Using training data split to train FGMC model.")
        fgmc_triggers = fgmc_triggers[fgmc_triggers["split"] == "Train"]
    else:
        logger.debug(f"Using full data set to train FGMC model.")
    train_fgmc_model(fgmc_triggers, far_star, snr_star, out_dir, overwrite)

    if "split" in mchirp_area_triggers.columns:
        logger.debug(
            f"Using training data split to train Chirp Mass Area model.")
        mchirp_area_triggers = mchirp_area_triggers[
            mchirp_area_triggers["split"] == "Train"]
    else:
        logger.debug(f"Using full data set to train Chirp Mass Area model.")
    train_mchirp_area_model(mchirp_area_triggers, out_dir, overwrite)

    duration = time.perf_counter() - duration
    logger.debug(
        f"{Path(__file__).stem} script complete in {duration} seconds.")


if __name__ == "__main__":
    main()
