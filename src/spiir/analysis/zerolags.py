"""Module for common analysis and plots used for SPIIR pipeline zerolags/triggers.

Note that all functions in this module are under development and are subject to change.
"""

import logging
from os import PathLike
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import astropy
import astropy_healpix as ah
import ligo.skymap.io
import ligo.skymap.plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib.figure import Figure
from scipy.stats import halfnorm, poisson
from tqdm import tqdm

from ..processing.array import chunk_iterable

logger = logging.getLogger(__name__)


def match_coincident_zerolags(*args, **kwargs) -> pd.DataFrame:
    import warnings

    warnings.warn(
        "match_coincident_zerolags is deprecated and will be removed in the future. "
        "Please use match_coincident_injections instead."
    )
    return match_coincident_injections(*args, **kwargs)


def match_coincident_injections(
    zerolags: pd.DataFrame,
    injections: pd.DataFrame,
    max_far: float = 1e-4,
    max_cmbchisq: float = 3.0,
    mchirp_percentage: float = 0.1,
    time_interval: float = 0.9,
    chunk_size: Optional[int] = None,
    verbose: bool = False,
    tie_break: Optional[str] = None,
    zerolag_end_times: tuple[str, str] = ("end_time_sngl_H1", "end_time_ns_sngl_H1"),
    injection_end_times: tuple[str, str] = ("h_end_time", "h_end_time_ns"),
) -> pd.DataFrame:
    """Matches zerolags found by the pipeline that are coincident with an injection set.

    Compares a DataFrame or candidate zerolags found by the search pipeline and
    compares  them against a DataFrame of signal injections that have been injected
    into the same strain the pipeline has searched over. If a trigger is considered to
    be statistically significant (i.e within a maximum false alarm rate and combined
    Chi squared value) and within some bounds of the parameter space similar to a
    given injection (i.e. chirp mass and detection gps time), then we can consider the
    trigger to be coincident with the injection. In other words, we can claim that the
    search pipeline has successfully recovered an injection.

    If there are multiple zerolags that match with an injection given the conditions
    provided, then we select the zerolag that has the maximum coherent SNR by default.
    However, the tie_break argument allows the user to specify how a tie break is
    implemented with other column data, or whether any tie break is implemented at all.

    Parameters
    ----------
    zerolags: pd.DataFrame
        A DataFrame of zerolags detected by the pipeline, typically sourced from entries
        in a LIGOLW PostcohInspiralTable object, csv file, or similar.
    injections: pd.DataFrame
        A DataFrame of the known injection set added to the searched strain.
    max_far: float
        The maximum false alarm rate threshold for statistical significance.
    max_cmbchisq: float
        The maximum combined Chi squared threshold for statistical significance.
    mchirp_percentage: float
        The maximum bounds allowed for the chirp mass range between zerolag and
        injection as a percentage. For example 0.1 corresponds to ±10%, and so we say a
        trigger matches if it's chirp is within 90% to 110% of the injection chirp mass.
    time_interval: float
        The maximum bounds allowed for the gps time between zerolag and injection in
        seconds. For example, 0.5 corresponds to ±0.5 seconds from the injection time.
    chunk_size: int | None
        If not None, breaks the array processing into intermediate chunks of size
        chunk_size to constrain maximum memory usage. Otherwise processes all at once.
    verbose: bool
        Whether to enable verbose output for processing progress bars.
    tie_break: str | None
        If tie_break is not None, then for duplicate matching zerolags, this function
        will match on the *max* values as specified by the provided column string.
        Otherwise if tie_break is None, all potentially matching zerolags are returned.
    zerolag_end_times: tuple[str, str]
        A length 2 tuple that specifies the columns to use for the end time in seconds
        (first element) and nanoseconds (second element) from the zerolag DataFrame.
        They are used to match coincident times between zerolags and injections.
    injection_end_times: tuple[str, str]
        A length 2 tuple that specifies the columns to use for the end time in seconds
        (first element) and nanoseconds (second element) from the injection DataFrame.
        They are used to match coincident times between zerolags and injections.

    Returns
    -------
    pd.DataFrame
        A DataFrame of all zerolags with a column specifying the matched injection idx.

    Raises
    ------
    AttributeError
        If required columns are missing from either the `zerolags` or `injections`
        dataframes, then an AttributeError will be raised.
    TypeError
        If tie_break is provided but it is not a str else a TypeError will be raised.
    ValueError
        If chunk_size is not a positive integer then a ValueError will be raised.

    Examples
    --------
    Suppose we would like to find candidate foreground signals in a collection of
    zerolags where we have a known injection set. We can tie break on "cohsnr" with:

        >> match_coincident_zerolags(zerolags, injections, tie_break="cohsnr")

    """
    # validate columns for injection dataframes
    required_injection_cols = ["mchirp"] + list(injection_end_times)
    for col in required_injection_cols:
        if col not in injections.columns:
            raise AttributeError(f"injections do not contain required column {col}.")

    # validate columns for zerolag dataframes
    required_zerolag_cols = ["far", "cmbchisq", "mchirp"]
    required_zerolag_cols += list(zerolag_end_times)

    # handle tie_break column specifications
    if tie_break is not None:
        if not isinstance(tie_break, str):
            raise TypeError(f"tie_break must be a str.")
        required_zerolag_cols += [tie_break]

    for col in required_zerolag_cols:
        if col not in zerolags.columns:
            raise AttributeError(f"zerolags do not contain required column {col}.")

    # determine array chunk_size for memory constrained data processing
    chunk_size = chunk_size or len(zerolags)  # if None, then we do all rows at once
    if not isinstance(chunk_size, int) and chunk_size > 0:
        raise ValueError(f"chunk_size {chunk_size} must be an integer greater than 0.")

    # match candidate zerolags on injection set to search for foreground signals
    found = []
    with tqdm(
        total=len(zerolags),
        desc=f"Matching zerolags against injection set",
        disable=not verbose,
    ) as progress_bar:
        for chunk in chunk_iterable(range(len(zerolags)), size=chunk_size):
            indices = list(chunk)
            candidates = zerolags.iloc[indices]

            # check zerolag will pass statistical tests in pipeline
            is_valid_far = (candidates["far"] != 0) & (candidates["far"] < max_far)
            is_valid_chisq = candidates["cmbchisq"] <= max_cmbchisq
            is_significant = np.stack([is_valid_far, is_valid_chisq]).all(axis=0)
            candidates = candidates.loc[is_significant]

            # check candidate is similar to an injection time and mchirp
            lower_mchirp = (injections["mchirp"] * (1 - mchirp_percentage)).to_numpy()
            upper_mchirp = (injections["mchirp"] * (1 + mchirp_percentage)).to_numpy()
            candidate_mchirp = candidates["mchirp"].to_numpy()
            is_valid_mchirp = (candidate_mchirp[:, None] >= lower_mchirp) & (
                candidate_mchirp[:, None] <= upper_mchirp
            )

            # TODO: Enable matching coincidences for more than one IFO (list of tuples?)
            mid_time = (
                injections[injection_end_times[0]]
                + injections[injection_end_times[1]] * 1e-9
            )
            mid_time = mid_time.to_numpy()
            lower_time, upper_time = mid_time - time_interval, mid_time + time_interval
            candidate_time = (
                candidates[zerolag_end_times[0]]
                + candidates[zerolag_end_times[1]] * 1e-9
            )
            candidate_time = candidate_time.to_numpy()

            is_valid_time = (candidate_time[:, None] >= lower_time) & (
                candidate_time[:, None] <= upper_time
            )

            # 2d array checks zerolags for all bounds of injection time and mchirp
            is_candidate = np.stack([is_valid_mchirp, is_valid_time]).all(axis=0)
            candidate_indices = np.argwhere(is_candidate)
            injection_index_map = np.zeros_like(candidates.index) - 1  # if -1 we remove
            injection_index_map[candidate_indices[:, 0]] = candidate_indices[:, 1]
            candidates["injection_index"] = injection_index_map
            candidates = candidates.loc[candidates["injection_index"] != -1]

            if tie_break is not None:
                # faster select based on maximum value of a single column
                idxs = candidates.groupby("injection_index")[tie_break].idxmax()
                candidates = candidates.loc[idxs]

            found.append(candidates)
            progress_bar.update(len(indices))

    return pd.concat(found)


def plot_zerolag_background(
    data: Dict[str, pd.DataFrame],
    title: str = "Background Zerolag Analysis for Coherent SNR vs Combined ChiSq",
    figsize: Tuple[float, float] = (12, 8),
):
    # TODO: revise and document required input data format(s)
    col: str = ("cmbchisq",)

    # this function produces two comparison plots across two runs
    # i.e. a inj and noninj run for two different runs, comparing cohsnr/cmbchisq
    # TODO: the input columns should be customisable
    # TODO: more thought should be put into generalising this function
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True
    )

    for i, key in enumerate(data):
        ax = axes[i // ncols, i % ncols]
        ax.scatter(
            data[key]["cohsnr"], data[key]["cmbchisq"], s=20, alpha=1.0, marker="."
        )
        ax.grid(False)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Coherent SNR",
            ylabel=r"Combined $\chi^{2}$",
            title=key,
        )

    fig.suptitle(title, fontsize=14)

    return fig


def plot_zerolag_foreground(
    data: Dict[str, pd.DataFrame],
    title: str = "Foreground Determinism Comparison for Combined ChiSq",
) -> Figure:
    # TODO: revise and document required input data format(s)
    fig, ax = plt.subplots(figsize=(12, 8), layout="tight")
    for i in range(2):
        key = list(data.keys())[i + 2]
        data[key]["cmbchisq"].hist(
            bins=100, histtype="step", lw=2 - i, label=key, ax=ax
        )
    ax.set_xlim(right=10)
    ax.set(title=title, xlabel=r"Combined $\chi^{2}$")
    ax.legend(loc="upper right")

    return fig


def plot_expected_far(
    data: Dict[str, pd.DataFrame],
    title: str = "Cumulative Counts of Zerolag Triggers vs. iFAR",
    figsize: Tuple[float, float] = (16, 8),
) -> Figure:
    # TODO: revise and document required input data format(s)
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, layout="tight")

    # calculate event counts at each ifar tick label
    for i, key in enumerate(data):
        # plot observed data
        data[key]["ifar"] = [None if y == 0 else 1 / y for y in data[key]["far"]]
        ticks = np.geomspace(
            min(data[key]["ifar"].dropna()), max(data[key]["ifar"].dropna())
        )
        counts = [
            int(data[key]["ifar"][data[key]["ifar"] > tick].count()) for tick in ticks
        ]
        ax = axes[i // ncols]
        ax.plot(ticks, counts, label=key, linewidth=2)

        # set axis details for every odd (last) iteration
        if i % ncols == 1:
            # estimate theoretical iFAR candidate frequencies over n_seconds duration
            # n_seconds = 7*24*(60**2)  # we don't have duration metadata
            # TODO: Should n_seconds be a user input arg if we don't have this metadata?
            n_seconds = data[key]["end_time"].max() - data[key]["end_time"].min()
            n_seconds = int(np.ceil(n_seconds))
            r = np.geomspace(1, n_seconds)
            ax.plot(r, n_seconds / r, label="Theoretical", c="k")

            # plot expected variance in iFAR
            for nsigma in reversed(range(1, 4)):
                ifar_grid = np.geomspace(1, 2e6)
                prob = halfnorm.cdf(nsigma)
                lo, hi = np.asarray(poisson(n_seconds / ifar_grid).interval(prob))
                ax.fill_between(
                    ifar_grid,
                    lo,
                    hi,
                    color=str(1 - 0.5 / nsigma),
                    label=r"{}$\sigma$".format(nsigma),
                )

            # format axes
            ax.grid(True, which="major", color="k", linestyle="-", alpha=0.4)
            ax.grid(True, which="minor", color="k", linestyle="--", alpha=0.2)
            ax.legend()
            ax.set(
                title=" ".join(key.split(" ")[:2]),
                xlabel="iFAR",
                xscale="log",
                xlim=(1, n_seconds),
                ylabel="Cumulative Counts",
                yscale="log",
            )

    fig.suptitle(title, fontsize=14)

    return fig
