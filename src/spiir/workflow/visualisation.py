"""Module for common plotting functions used to visualise SPIIR pipeline results.

Note that all functions in this module are currently placeholders and will be subject 
to significant changes in the near future.
"""

import logging
from os import PathLike
from typing import Optional, Union, Tuple, Dict, Sequence, Any

import astropy
import astropy_healpix as ah
import ligo.skymap.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib.figure import Figure
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


def plot_skymap_from_fits(
    path: Union[str, bytes, PathLike],
    contours: Optional[Union[float, Sequence[float]]]=None,
    inset_args: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]]=None,
    ground_truth: Optional[SkyCoord]=None,
    annotate: bool=True,
    figsize: Tuple[float, float]=(16, 7),
    title: Optional[str]=None,
) -> Figure:
    """Plots a detailed probability skymap from a FITS file.
    
    TODO:
        - Implement automatic inset axis sizing and ordering based on ra / dec.
        - Ensure logic works for greater than two inset axes.
        - Test various figsizes to ensure figure is consistently formatted.

    Parameters
    ----------
    path: str | bytes | os.PathLike
        A path to a valid FITS file, typically generated from ligo.skymap.io.fits.
    contours: float | Sequence[float] | None
        A sequence of probability contour levels (credible intervals) used to draw on 
        the skymap. For example, (0.5, 0.9) would correspond to the 50% and 90% levels.
    inset_kwargs: dict | Sequence[dict] | None
        A set of keyword arguments passed used to define one or more inset skymap axes. 
        If a sequence of argument dicts are provided, one inset axis is made for each 
        element of the list. At a minimum, each must have a 'center' key with a valid 
        astropy.coordinates.SkyCoord to center the inset axis on the skymap canvas.
    ground_truth: astropy.coordinates.Skycoord | None
        An astropy SkyCoord input for a "ground truth" marker for the true location.
    annotate: bool
        Whether to annotate the plot with information about additional metadata such as 
        the LIGOLW event id and the probability per square degree.
    figsize: tuple[float, float] | None
        An optional tuple specifying the figure size. Default is (16, 7).
    title: str | None
        An optional string specifying the figure title.


    Returns
    -------
    matplotlib.figure.Figure
        The probability skymap as a matplotlib Figure object.
    
    """
    skymap, metadata = ligo.skymap.io.fits.read_sky_map(path, nest=None)
    nside = ah.npix_to_nside(len(skymap))
    deg2perpix = ah.nside_to_pixel_area(nside).to_value(astropy.units.deg**2)
    probperdeg2 = skymap / deg2perpix
            
    # get sky position of max point of probability once, if required later
    healpix = ah.HEALPix(nside=nside, order='nested')
    max_longitude, max_latitude = healpix.healpix_to_lonlat(probperdeg2.argmax())
    
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.axes(projection='astro mollweide')
    
    if title:
        if not isinstance(title, str): raise TypeError("title argument must be a str.")
        ax.set_title(title, fontsize=16, pad=15)

    ax.imshow_hpx(
        (probperdeg2, "ICRS"),
        nested=metadata["nest"],
        vmin=0.,
        vmax=probperdeg2.max(),
        cmap="cylon",
    )

    if ground_truth:
        marker = "X"
        ax.plot_coord(
            ground_truth,
            marker,
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=8,
            linewidth=1,
        )

    if contours:
        contours = [contours] if isinstance(contours, float) else contours
        credible_levels = 100 * find_greedy_credible_levels(skymap)
        contour = ax.contour_hpx(
            (credible_levels, 'ICRS'),
            nested=metadata['nest'],
            colors='k',
            linewidths=0.5,
            levels=contours,
        )
        # fmt = r'%g\%%' if mpl.rcParams['text.usetex'] else '%g%%'
        contour.clabel(fmt=r'%g\%%', fontsize=6, inline=True)

    if annotate:
        text = []
        try:
            objid = metadata['objid']
        except KeyError:
            pass
        else:
            text.append('event ID: {}'.format(objid))
        if contours:
            pp = np.round(contours).astype(int)
            ii = np.searchsorted(np.sort(credible_levels), contours) * deg2perpix
            ii = np.round(ii).astype(int)
            for i, p in zip(ii, pp):
                text.append(r'{:d}% area: {:,d} deg$^2$'.format(p, i))
        ax.text(.05, 0.95, '\n'.join(text), transform=ax.transAxes, ha='right')

    if inset_args is not None:
        if isinstance(inset_args, dict):
            inset_args = [inset_args]

        for i, inset_arg in enumerate(inset_args):
            if not isinstance(inset_arg, dict):
                raise TypeError(f"inset_args must be a dict or sequence of dicts.")
            
            if "center" not in inset_arg:
                raise KeyError(f"inset_arg must contain a valid 'center' SkyCoord.")
            
            # define inset axis size
            inset_width = inset_arg.get("width", 0.3)
            inset_height = inset_arg.get("height", 0.3)
            inset_left = inset_arg.get("left", 0.8)
            inset_bottom = inset_arg.get("bottom", None)
            if inset_bottom is None:
                if len(inset_args) == 1:
                    inset_bottom = 0.5 - inset_height/2
                else:
                    # FIXME: Handle more than 2 inset axes
                    inset_bottom = 0.85 - ((i%2)+1)*inset_height - (i%2)*0.1

            # center point as astropy.coordinates.SkyCord
            # FIXME: implement more robust input argument logic and exception handling
            inset_center = inset_arg.get("center", "max")
            if isinstance(inset_center, str) and inset_center.lower() == "max":
                inset_center = SkyCoord(max_longitude, max_latitude)
            if not isinstance(inset_center, SkyCoord):
                try:
                    inset_center = SkyCoord(inset_center)
                except Exception as exc:
                    print(exc)
                    try:
                        inset_center = SkyCoord(**inset_center)
                    except Exception as exc:
                        raise ValueError(f"inset_arg['center'] is an invalid SkyCoord.")
                
            ax_inset = plt.axes(
                [inset_left, inset_bottom, inset_width, inset_height],
                projection='astro zoom',
                center=inset_center,
                radius=(inset_arg.get("radius", None) or 10)*astropy.units.deg,
            )

            for coords in ax_inset.coords:
                coords.set_ticklabel(exclude_overlapping=True)
                coords.set_ticklabel(size=8)

            ax.mark_inset_axes(ax_inset)
            ax.connect_inset_axes(ax_inset, 'upper left')
            ax.connect_inset_axes(ax_inset, 'lower left')
            ax_inset.scalebar((0.1, 0.1), 5 * astropy.units.deg).label()
            ax_inset.compass(0.9, 0.1, 0.2)
            ax_inset.set_xlabel(inset_arg.get("xlabel", " "))  # empty string required
            ax_inset.set_ylabel(inset_arg.get("ylabel", " "))  # None breaks x/y labels
            ax_inset.set_title(inset_arg.get("title", None))
                        
            ax_inset.imshow_hpx(
                (probperdeg2, "ICRS"),
                nested=metadata["nest"],
                vmin=0.,
                vmax=probperdeg2.max(),
                cmap="cylon",
            )

            if ground_truth:
                ax_inset.plot_coord(
                    ground_truth,
                    marker,
                    markerfacecolor='white',
                    markeredgecolor='black',
                    markersize=8,
                    linewidth=1,
                )
                
            if contours:
                contour_inset = ax_inset.contour_hpx(
                    (credible_levels, 'ICRS'),
                    nested=metadata['nest'],
                    colors='k',
                    linewidths=0.5,
                    levels=contours,
                )
                # fmt = r'%g\%%' if mpl.rcParams['text.usetex'] else '%g%%'
                contour_inset.clabel(fmt=r'%g\%%', fontsize=8, inline=True)

    return fig