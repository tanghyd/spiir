# Functions to make plots related to the source probability calculation.
# By V. Villa-Ortega, March 2021
# Minor modifications for SPIIR By Daniel Tang, March 2022

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pycbc.conversions import mass2_from_mchirp_mass1 as mcm1_to_m2

from .predict import estimate_source_mass

_source_colour_map = {
    "BNS": "#A2C8F5",  # light blue
    "NSBH": "#FFB482",  # light orange
    "BBH": "#FE9F9B",  # light red
    "MassGap": "#8EE5A1",  # light green
    "GNS": "#98D6CB",  # turquoise
    "GG": "#79BB87",  # green
    "BHG": "#C6C29E",  # dark khaki
}


def get_source_colour(source: str) -> str:
    return _source_colour_map[source]


def plot_mass_contour_figure(
    mchirp: float,
    mchirp_std: float,
    z: float,
    z_std: float,
    mass_limits: tuple[float, float],
    mass_bdary: tuple[float, float],
    figsize: tuple[float, float] = (8, 6),
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
) -> Figure:
    """Draws a full matplotlib Figure visualising the probability mass contour plane."""

    fig, ax = plt.subplots(figsize=figsize)
    _draw_mass_contour_axes(
        ax=ax,
        mchirp=mchirp,
        mchirp_std=mchirp_std,
        z=z,
        z_std=z_std,
        mass_limits=mass_limits,
        mass_bdary=mass_bdary,
        xlims=xlims,
        ylims=ylims,
    )

    return fig


def _draw_mass_contour_axes(
    ax: Axes,
    mchirp: float,
    mchirp_std: float,
    z: float,
    z_std: float,
    mass_limits: tuple[float, float],
    mass_bdary: tuple[float, float],
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
) -> Axes:
    """Draws one matplotlib.axes.Axes visualising the probability mass contour plane."""

    # estimate source frame chirp mass and uncertainty boundary
    mc, mc_std = estimate_source_mass(mchirp, mchirp_std, z, z_std)
    mcb = mc + mc_std
    mcs = mc - mc_std

    # determine component masses (when m1 = m2) given chirp mass boundaries
    mib = (2.0**0.2) * mcb
    mis = (2.0**0.2) * mcs

    # get mass boundary limits
    m2_min, m1_max = mass_limits
    ns_max, gap_max = mass_bdary  # range to define mass gap class

    lim_m1b = min(m1_max, mcm1_to_m2(mcb, m2_min))
    m1b = np.linspace(mib, lim_m1b, num=100)
    m2b = mcm1_to_m2(mcb, m1b)

    lim_m1s = min(m1_max, mcm1_to_m2(mcs, m2_min))
    m1s = np.linspace(mis, lim_m1s, num=100)
    m2s = mcm1_to_m2(mcs, m1s)

    # plot contour
    if mib > m1_max:
        ax.plot((m1_max, m1_max), (mcm1_to_m2(mcs, lim_m1s), m1_max), "b")
    else:
        ax.plot(m1b, m2b, "b")
        ax.plot(
            (m1_max, m1_max), (mcm1_to_m2(mcs, lim_m1s), mcm1_to_m2(mcb, lim_m1b)), "b"
        )
    if mis >= m2_min:
        ax.plot(m1s, m2s, "b")
        ax.plot((lim_m1s, lim_m1b), (m2_min, m2_min), "b")
    else:
        ax.plot((m2_min, lim_m1b), (m2_min, m2_min), "b")

    # plot limits
    ax.plot((m2_min, m1_max), (m2_min, m1_max), "k--")
    ax.plot((ns_max, ns_max), (m2_min, ns_max), "k:")
    ax.plot((gap_max, gap_max), (m2_min, gap_max), "k:")
    ax.plot((ns_max, m1_max), (ns_max, ns_max), "k:")
    ax.plot((gap_max, m1_max), (gap_max, gap_max), "k:")

    # colour plot
    ax.fill_between(
        np.arange(0.0, ns_max - 0.01, 0.01),
        gap_max,
        m1_max,
        color=get_source_colour("NSBH"),
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(gap_max, m1_max, 0.01), 0.0, ns_max, color=get_source_colour("NSBH")
    )
    ax.fill_between(
        np.arange(ns_max, gap_max, 0.01),
        np.arange(ns_max, gap_max, 0.01),
        m1_max,
        color=get_source_colour("MassGap"),
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(0.0, ns_max, 0.01),
        ns_max,
        gap_max,
        color=get_source_colour("MassGap"),
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(gap_max, m1_max, 0.01),
        np.arange(gap_max, m1_max, 0.01),
        m1_max,
        color=get_source_colour("BBH"),
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(gap_max, m1_max, 0.01),
        np.arange(gap_max, m1_max, 0.01),
        gap_max,
        color=get_source_colour("BBH"),
    )
    ax.fill_between(
        np.arange(0.0, ns_max, 0.01),
        0.0,
        np.arange(0.0, ns_max, 0.01),
        color=get_source_colour("BNS"),
    )
    ax.fill_between(
        np.arange(0.0, ns_max, 0.01),
        ns_max,
        np.arange(0.0, ns_max, 0.01),
        color=get_source_colour("BNS"),
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(ns_max, gap_max, 0.01),
        np.arange(ns_max, gap_max, 0.01),
        color=get_source_colour("MassGap"),
    )
    ax.fill_between(
        np.arange(gap_max, m1_max, 0.01),
        ns_max,
        gap_max,
        color=get_source_colour("MassGap"),
    )

    # colour contour
    x1 = np.arange(mis, mib + 0.01, 0.01)
    x2 = np.arange(mib, lim_m1b, 0.01)

    if len(x1) > 0:
        ax.fill_between(
            x1,
            x1,
            mcm1_to_m2(mcs, x1),
            facecolor=(1, 1, 1, 0.5),
            edgecolor=(0, 0, 0, 0),
        )

    if len(x2) > 0:
        # errors if mib >= lim_m1b (x2 is empty)
        ax.fill_between(
            x2,
            mcm1_to_m2(mcb, x2),
            mcm1_to_m2(mcs, x2),
            facecolor=(1, 1, 1, 0.5),
            edgecolor=(0, 0, 0, 0),
        )

    # plot_details
    xlims = xlims or mass_limits
    ylims = ylims or (1.0, 20.0)
    ax.set(xlim=xlims, ylim=ylims, xlabel=r"$m_1$", ylabel=r"$m_2$")

    return ax


def plot_prob_pie_figure(
    probabilities: dict[str, float],
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    _draw_prob_pie_axes(ax, probabilities)
    return fig


def _draw_prob_pie_axes(ax: Axes, probabilities: dict[str, float]) -> Axes:
    prob_plot = {k: v for k, v in probabilities.items() if v != 0.0}
    labels, sizes = zip(*prob_plot.items())
    colours = [get_source_colour(label) for label in labels]
    ax.pie(
        sizes,
        labels=labels,
        colors=colours,
        autopct="%1.1f%%",
        textprops={"fontsize": 15},
    )
    ax.axis("equal")
    return ax
