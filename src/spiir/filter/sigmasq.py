import logging
from typing import Optional, Union, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def get_cutoff_indices(
    f_low: Optional[float], f_high: Optional[float], df: float, n: int
) -> Tuple[int, int]:
    """
    Gets the indices of a frequency series at which to stop an overlap
    calculation.

    Parameters
    ----------
    f_low: float
        The frequency (in Hz) of the lower index.
    f_high: float
        The frequency (in Hz) of the upper index.
    df: float
        The frequency step (in Hz) of the frequency series.
    n: int
        The number of points in the **time** series(?). Can be odd or even.

    Returns
    -------
    (k_min, k_max): tuple(int, int)
    """
    if f_low:
        k_min = int(f_low / df)
        if k_min < 0:
            raise ValueError(
                f"Start frequency cannot be negative. \
                    Supplied value and kmin {f_low} and {k_min}"
            )
    else:
        k_min = 1

    if f_high:
        k_max = int(f_high / df)
        if k_max > int((n + 1) / 2.0):
            k_max = int((n + 1) / 2.0)
    else:
        # int() truncates towards 0, so this is equivalent to the floor of the float
        k_max = int((n + 1) / 2.0)

    if k_max <= k_min:
        raise ValueError(
            f"Kmax cannot be less than or equal to kmin. \
            Provided values of freqencies (min,max) were {f_low} and {f_high} \
            corresponding to (kmin, kmax) of ({k_min, k_max}"
        )

    return k_min, k_max


def compute_sigmasq(
    frequency_series: np.ndarray,
    delta_f: float,
    psd: Optional[np.ndarray] = None,
    f_low_cutoff: Optional[float] = None,
    f_high_cutoff: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Return the loudness of the waveform. This is defined (see Duncan
    Brown's thesis) as the unnormalized matched-filter of the input waveform,
    htilde, with itself. This quantity is usually referred to as (sigma)^2
    and is then used to normalize matched-filters with the data.

    Parameters
    ----------
    htilde : np.ndarray
        The input vector containing a waveform.
    psd : np.ndarray | None
        The psd used to weight the accumulated power.
    low_frequency_cutoff : float | None
        The frequency to begin considering waveform power.
    high_frequency_cutoff : float | None
        The frequency to stop considering waveform power.

    Returns
    -------
    sigmasq: float
    """
    n = (len(frequency_series) - 1) * 2
    norm = 4.0 * delta_f
    k_min, k_max = get_cutoff_indices(f_low_cutoff, f_high_cutoff, delta_f, n)
    frequency_series = frequency_series[k_min:k_max]

    assert np.isreal(frequency_series).all()

    # compute inner product with complex conjugation conditional on dtype
    sigmasq = np.vdot(frequency_series, frequency_series)

    if psd is not None:
        # weight frequency components by power spectral density
        psd_len = len(psd[k_min:k_max])
        series_len = len(frequency_series)
        if psd_len != series_len:
            raise ValueError(
                f"PSD length {psd_len} does not match frequency series {series_len}."
            )

        # if delta_f of psd != delta_f of frequency series
        # raise ValueError("Waveform does not have same delta_f as psd")

        # weight frequency components by PSD
        sigmasq /= psd[k_min:k_max]

    return sigmasq.real * norm
