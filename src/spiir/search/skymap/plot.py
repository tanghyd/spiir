"""Module for skymap plotting functions based on ligo.skymap for use by SPIIR."""

import logging
from os import PathLike
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import astropy
import astropy_healpix as ah
import ligo.skymap.io
import ligo.skymap.plot
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

logger = logging.getLogger(__name__)


def plot_skymap(
    probs: np.ndarray,
    nested: bool = True,
    event_id: Optional[str] = None,
    contours: Optional[Union[float, Sequence[float]]] = None,
    inset_kwargs: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
    ground_truth: Optional[Union[Tuple[float, float], SkyCoord]] = None,
    annotate: bool = True,
    colorbar: bool = False,
    figsize: Tuple[float, float] = (14, 7),
    title: Optional[str] = None,
) -> Figure:
    """Plots a detailed probability skymap.

    TODO:
        - Implement automatic inset axis sizing and ordering based on ra / dec.
        - Ensure logic works for greater than two inset axes.
        - Test various figsizes to ensure figure is consistently formatted.

    Parameters
    ----------
    probs: np.ndarray
        The probability per pixel over the full skymap.
    nested: bool, default = True
        The order of HEALPix pixels - either 'nested' (True) or 'ring' (False).
    event_id: str, optional
        An optional event_id string to uniquely label the skymap.
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
    annotate: bool, default = True
        Whether to annotate the plot with information about additional metadata such as
        the LIGOLW event id and the probability per square degree.
    colobar: bool, default = False
        Whether to add a colorbar corresponding to the probability per square degree.
    figsize: tuple[float, float] | None
        An optional tuple specifying the figure size. Default is (16, 7).
    title: str | None
        An optional string specifying the figure title.


    Returns
    -------
    matplotlib.figure.Figure
        The probability skymap as a matplotlib Figure object.

    """
    # get array of log probabilities per pixel on the sky
    nside = ah.npix_to_nside(len(probs))
    deg2perpix = ah.nside_to_pixel_area(nside).to_value(astropy.units.deg**2)
    probperdeg2 = probs / deg2perpix

    # get sky position of max point of probability once, if required later
    healpix = ah.HEALPix(nside=nside, order="nested" if nested else "ring")
    max_longitude, max_latitude = healpix.healpix_to_lonlat(probperdeg2.argmax())

    # create skymap figure
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.axes(projection="astro mollweide")
    ax.grid()

    if title:
        if not isinstance(title, str):
            raise TypeError("title argument must be a str.")
        ax.set_title(title, fontsize=16, pad=15)

    vmin, vmax = probperdeg2.min(), probperdeg2.max()
    img = ax.imshow_hpx(
        (probperdeg2, "ICRS"),
        nested=nested,
        vmin=vmin,
        vmax=vmax,
        cmap="cylon",
    )

    if ground_truth:
        if not isinstance(ground_truth, SkyCoord):
            logging.debug("ground_truth not defined as a SkyCoord, assuming unit='rad'")
            ground_truth = SkyCoord(*ground_truth, unit="rad")
        marker = "X"
        ax.plot_coord(
            ground_truth,
            marker,
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=8,
            linewidth=1,
        )

    if contours:
        contours = [contours] if isinstance(contours, float) else contours
        credible_levels = 100 * find_greedy_credible_levels(probs)
        contour = ax.contour_hpx(
            (credible_levels, "ICRS"),
            nested=nested,
            colors="k",
            linewidths=0.5,
            levels=contours,
        )
        # fmt = r'%g\%%' if mpl.rcParams['text.usetex'] else '%g%%'
        contour.clabel(fmt=r"%g\%%", fontsize=6, inline=True)

    if colorbar:
        # cb_ticks = np.linspace(vmin, vmax, 2, endpoint=True)
        cb = fig.colorbar(
            img,
            orientation='horizontal',
            location="bottom",
            fraction=0.045,
            pad=0.05,
            format=PercentFormatter(1),
        )
        cb.set_label(r'Probability per deg$^2$', fontsize=11)

    if annotate:
        text = []
        if event_id is not None:
            text.append(f"event ID: {str(event_id)}")
        if contours:
            pp = np.round(contours).astype(int)
            ii = np.searchsorted(np.sort(credible_levels), contours) * deg2perpix
            ii = np.round(ii).astype(int)
            for i, p in zip(ii, pp):
                text.append(r"{:d}% area: {:,d} deg$^2$".format(p, i))
        ax.text(0.05, 0.95, "\n".join(text), transform=ax.transAxes, ha="right")

    if inset_kwargs is not None:
        if isinstance(inset_kwargs, dict):
            inset_kwargs = [inset_kwargs]

        for i, inset_kwarg in enumerate(inset_kwargs):
            if not isinstance(inset_kwarg, dict):
                raise TypeError("inset_kwargs must be a dict or sequence of dicts.")

            if "center" not in inset_kwarg:
                raise KeyError("inset_kwarg must contain a valid 'center' SkyCoord.")

            # define inset axis size
            inset_width = inset_kwarg.get("width", 0.3)
            inset_height = inset_kwarg.get("height", 0.3)
            inset_left = inset_kwarg.get("left", 0.8)
            inset_bottom = inset_kwarg.get("bottom", None)
            if inset_bottom is None:
                if len(inset_kwargs) == 1:
                    inset_bottom = 0.5 - inset_height / 2
                else:
                    # FIXME: Handle more than 2 inset axes
                    inset_bottom = 0.85 - ((i % 2) + 1) * inset_height - (i % 2) * 0.1

            # center point as astropy.coordinates.SkyCord
            # FIXME: implement more robust input argument logic and exception handling
            inset_center = inset_kwarg.get("center", "max")
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
                        error_msg = "inset_kwarg['center'] is an invalid SkyCoord."
                        raise ValueError(error_msg) from exc

            ax_inset = plt.axes(
                [inset_left, inset_bottom, inset_width, inset_height],
                projection="astro zoom",
                center=inset_center,
                radius=(inset_kwarg.get("radius", None) or 10) * astropy.units.deg,
            )

            for coords in ax_inset.coords:
                coords.set_ticklabel(exclude_overlapping=True)
                coords.set_ticklabel(size=8)

            ax.mark_inset_axes(ax_inset)
            ax.connect_inset_axes(ax_inset, "upper left")
            ax.connect_inset_axes(ax_inset, "lower left")
            ax_inset.scalebar((0.1, 0.1), 5 * astropy.units.deg).label()
            ax_inset.compass(0.9, 0.1, 0.2)
            ax_inset.set_xlabel(inset_kwarg.get("xlabel", " "))  # empty string required
            ax_inset.set_ylabel(inset_kwarg.get("ylabel", " "))  # None breaks x/ylabels
            ax_inset.set_title(inset_kwarg.get("title", None))

            ax_inset.imshow_hpx(
                (probperdeg2, "ICRS"),
                nested=nested,
                vmin=0.0,
                vmax=probperdeg2.max(),
                cmap="cylon",
            )

            if ground_truth:
                ax_inset.plot_coord(
                    ground_truth,
                    marker,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=8,
                    linewidth=1,
                )

            if contours:
                contour_inset = ax_inset.contour_hpx(
                    (credible_levels, "ICRS"),
                    nested=nested,
                    colors="k",
                    linewidths=0.5,
                    levels=contours,
                )
                # fmt = r'%g\%%' if mpl.rcParams['text.usetex'] else '%g%%'
                contour_inset.clabel(fmt=r"%g\%%", fontsize=8, inline=True)

    return fig


def plot_skymap_from_fits(
    path: Union[str, bytes, PathLike],
    contours: Optional[Union[float, Sequence[float]]] = None,
    inset_kwargs: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
    ground_truth: Optional[SkyCoord] = None,
    annotate: bool = True,
    figsize: Tuple[float, float] = (16, 7),
    title: Optional[str] = None,
) -> Figure:
    """Plots a detailed probability skymap from a FITS file.

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
    probs, metadata = ligo.skymap.io.fits.read_sky_map(path, nest=None)

    return plot_skymap(
        probs,
        nested=metadata["nested"],
        event_id=metadata.get("objid", None),
        contours=contours,
        inset_kwargs=inset_kwargs,
        ground_truth=ground_truth,
        annotate=annotate,
        figsize=figsize,
        title=title,
    )
