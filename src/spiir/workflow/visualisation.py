"""Module for common plotting functions used to visualise SPIIR pipeline results.

Note that all functions in this module are currently placeholders and will be subject 
to significant changes in the near future.
"""

import logging
from os import PathLike
from typing import Union, Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import halfnorm, poisson


logger = logging.getLogger(__name__)


def save_background_analysis_plots(
    data: Dict[str, pd.DataFrame],
    path: Union[str, PathLike],
):
    # this function produces two comparison plots across two runs
    # i.e. a inj and noninj run for two different runs, comparing cohsnr/cmbchisq
    # TODO: the input columns should be customisable
    # TODO: more thought should be put into generalising this function 
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 8), sharex=True, sharey=True
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

    fig.suptitle("Background Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)


def save_foreground_analysis_plots(
    data: Dict[str, pd.DataFrame],
    path: Union[str, PathLike],
):

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(2):
        key = list(data.keys())[i + 2]
        data[key]["cmbchisq"].hist(
            bins=100, histtype="step", lw=2 - i, label=key, ax=ax
        )  # should probably replace column here with an input argument
    ax.set_xlim(right=10)
    ax.set_title("Foreground Determinism Comparison for Combined ChiSq")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)


def save_zerolag_trigger_plots(
    data: Dict[str, pd.DataFrame],
    path: Union[str, PathLike],
):
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))

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

    fig.suptitle("Cumulative Counts of Zerolag Triggers vs. iFAR", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)

