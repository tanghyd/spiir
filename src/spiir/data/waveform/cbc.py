import logging
from typing import Tuple, Union

import lal
import lalsimulation as lalsim
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# TODO: Replace functions with new interface from Waveforms group


def simulate_inspiral(
    domain: str, coordinates: str, df: bool = False, **kwargs
) -> Union[np.ndarray, pd.DataFrame]:
    if domain == "frequency":
        return simulate_frequency_domain_inspiral(df=df, **kwargs)
    elif domain == "time":
        return simulate_time_domain_inspiral(df=df, **kwargs)
    else:
        raise KeyError(
            f"Invalid domain {domain} provided - must be 'time' or 'frequency'."
        )


def simulate_frequency_domain_inspiral(
    mass1: float,
    mass2: float,
    spin1x: float,
    spin1y: float,
    spin1z: float,
    spin2x: float,
    spin2y: float,
    spin2z: float,
    phase: float,
    # duration: float,
    delta_f: float,
    approximant: str,
    distance: float = 1.0e6 * lal.PC_SI,
    inclination: float = 0.0,
    f_ref: float = 20.0,
    f_min: float = 20.0,
    f_max: float = 2048.0,
    df: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Generates a single frequency-domain template which:
        (1) is band-limited between f_low and f_high,
        (2) has an IFFT which is duration seconds long, and
        (3) has an IFFT which is sampled at sample_rate Hz.

    """
    sim_kwargs = dict(
        m1=lal.MSUN_SI * mass1,
        m2=lal.MSUN_SI * mass2,
        S1x=spin1x,
        S1y=spin1y,
        S1z=spin1z,
        S2x=spin2x,
        S2y=spin2y,
        S2z=spin2z,
        distance=distance,
        inclination=inclination,
        phiRef=phase,
        longAscNodes=0.0,
        eccentricity=0.0,
        meanPerAno=0.0,
        deltaF=delta_f,
        # deltaF=1.0 / duration,
        f_min=f_min,
        f_max=f_max,
        f_ref=f_ref,
        LALparams=None,
    )

    sim_kwargs["approximant"] = lalsim.GetApproximantFromString(approximant)

    hp, hc = lalsim.SimInspiralFD(**sim_kwargs)

    if df:
        freqs = pd.Index(np.arange(len(hp.data.data)) * hp.deltaF, name="frequency")
        hp = pd.Series(hp.data.data, index=freqs, name="plus")
        hc = pd.Series(hc.data.data, index=freqs, name="cross")
        return pd.DataFrame([hp, hc]).T

    return np.stack([hp.data.data, hc.data.data])


def simulate_time_domain_inspiral(
    mass1: float,
    mass2: float,
    spin1x: float,
    spin1y: float,
    spin1z: float,
    spin2x: float,
    spin2y: float,
    spin2z: float,
    phase: float,
    # sample_rate: float,
    delta_t: float,
    approximant: str,
    distance: float = 1.0e6 * lal.PC_SI,
    inclination: float = 0.0,
    f_ref: float = 20.0,
    f_min: float = 20.0,
    df: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    sim_kwargs = dict(
        m1=lal.MSUN_SI * mass1,
        m2=lal.MSUN_SI * mass2,
        S1x=spin1x,
        S1y=spin1y,
        S1z=spin1z,
        S2x=spin2x,
        S2y=spin2y,
        S2z=spin2z,
        inclination=inclination,
        distance=distance,
        phiRef=phase,
        longAscNodes=0.0,
        eccentricity=0.0,
        meanPerAno=0.0,
        deltaT=delta_t,
        # deltaT=1.0 / sample_rate,
        f_ref=f_ref,
        f_min=f_min,
        LALparams=None,
    )

    # load valid approximants
    sim_kwargs["approximant"] = lalsim.GetApproximantFromString(approximant)

    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__c.html
    hp, hc = lalsim.SimInspiralTD(**sim_kwargs)

    if df:
        shift = hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds / 1e9
        times = pd.Index(np.arange(len(hp.data.data)) * hp.deltaT + shift, name="time")
        hp = pd.Series(hp.data.data, index=times, name="plus")
        hc = pd.Series(hc.data.data, index=times, name="cross")
        return pd.DataFrame([hp, hc]).T

    return np.stack([hp.data.data, hc.data.data])


def convert_inference_to_simulation_spins(theta_jn, theta_1, theta_2, a_1, a_2):
    raise NotImplementedError
    # See bilby: https://github.com/lscsoft/bilby/blob/master/bilby/gw/conversion.py#L53
    # if (a_1 == 0.0 or theta_1 in [0, np.pi]) and (a_2 == 0.0 or theta_2 in [0, np.pi]):
    #     spin_1x, spin_1y, spin_1z = 0.0, 0.0, float(a_1 * np.cos(theta_1))
    #     spin_2x, spin_2y, spin_2z = 0.0, 0.0, float(a_2 * np.cos(theta_2))
    #     iota = theta_jn
    # else:
    #     pass
    # transform_source_to_radiation_frame()


# transform_from_source_frame_to_radiation?
# @np.vectorize
def transform_precessing_spins(
    mass_1: float,
    mass_2: float,
    theta_jn: float,
    theta_1: float,
    theta_2: float,
    phi_jl: float,
    chi_1: float,
    chi_2: float,
    phi_12: float,
    phase: float,
    f_ref: float = 20.0,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Converts inspiral parameters defined in the source frame to the radiation frame.

    Inverse function to transform_radiation_to_source_frame, and wrapper function
    around lalsimulation.XLALSimInspiralTransformPrecessingNewInitialConditions.

    Usually used to convert from waveform parameters that are defined for parameter
    inference (LALInference) to inspiral simulation (LALSimulation).

    See: XLALSimInspiralTransformPrecessingNewInitialConditions() from
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html.

    Function to specify the desired orientation of a precessing binary in terms of
    several angles and then compute the vector components with respect to orbital
    angular momentum as needed to specify binary configuration for ChooseTDWaveform.


    Parameters
    ----------
    mass_1: float
        The mass of the 1st object (in Mpc).
    mass_2: float
        The mass of the 2nd object (in Mpc), typically follows convention m_1 >= m_2).
    theta_jn: float
        zenith angle between J and N (rad)
    phi_jl: float
        Angle. (?)
    theta_1: float
        Inclination of the 1st object measured from Newtonian orbital angular momentum.
    theta_2: float
        Inclination of the 2nd object measured from Newtonian orbital angular momentum.
    phi_12: float
        Difference between the azimuthal angles of each object's spin: theta_1, theta_2.
        Angle between two object's orbital angular momentum (?).
    chi_1: float
        Dimensionless spin of the first object.
    chi_2: float
        The dimensionless spin magnitude object.
    phase: float
        Reference orbital phase phi.
    f_ref: float
        Reference gravitational wave frequency (Hz).

    Returns
    -------
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z: tuple[float, ...]
        The inclination angle of the plane of coalecence and
        the spins in the (x,y,z) radiation frame co-ordinates.
    """

    # convert masses from Mpc to SI units
    mass_1_SI = mass_1 * lal.MSUN_SI
    mass_2_SI = mass_2 * lal.MSUN_SI

    if f_ref == 0:
        raise ValueError(
            "f_ref = 0 is invalid. LALSimulation suggests using min frequency (f_min)."
        )
    incl, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
        lalsim.SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, theta_1, theta_2, phi_12,
            chi_1, chi_2, mass_1_SI, mass_2_SI, f_ref, phase
        )
    )  # fmt: skip

    return incl, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z


# @np.vectorize
def inverse_transform_precessing_spins(
    mass_1: float,
    mass_2: float,
    spin_1x: float,
    spin_1y: float,
    spin_1z: float,
    spin_2x: float,
    spin_2y: float,
    spin_2z: float,
    incl: float,
    phase: float,
    f_ref: float = 20.0,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Converts inspiral parameters defined in the radiation frame to the source frame.
    Wrapper function around lalsimulation.XLALSimInspiralTransformPrecessingWvf2PE and
    inverse function to transform_source_to_radiation_frame.

    Usually used to convert from waveform parameters that are defined for
    inspiral simulation (LALSimulation) to parameter inference (LALInference).

    See: XLALSimInspiralTransformPrecessingWvf2PE() from
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html.

    Function to specify the desired orientation of a precessing binary in terms of
    several angles and then compute the vector components with respect to orbital
    angular momentum as needed to specify binary configuration for ChooseTDWaveform.


    Parameters
    ----------
    mass_1: float
        The mass of the 1st object (in Mpc).
    mass_2: float
        The mass of the 2nd object (in Mpc), typically follows convention m_1 >= m_2).
    theta_jn: float
        Zenith angle between J and N (rad)
    phi_jl: float
        Azimuthal angle of L_N on its cone about J (rad).
    theta_1: float
        Inclination of the 1st object measured from Newtonian orbital angular momentum.
    theta_2: float
        Inclination of the 2nd object measured from Newtonian orbital angular momentum.
    phi_12: float
        Difference between the azimuthal angles of each object's spin: theta_1, theta_2.
    chi_1: float
        Dimensionless spin of the first object.
    chi_2: float
        The dimensionless spin magnitude object.
    phase: float
        Reference orbital phase phi.
    f_ref: float
        Reference gravitational wave frequency (Hz).

    Returns
    -------
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z: tuple[float, ...]
        The inclination angle of the plane of coalecence and
        the spins in the (x,y,z) radiation frame co-ordinates.
    """
    if f_ref == 0:
        raise ValueError(
            "f_ref = 0 is invalid. LALSimulation suggests using min frequency (f_min)."
        )

    spin_1 = (spin_1x, spin_1y, spin_1z)
    spin_2 = (spin_2x, spin_2y, spin_2z)
    theta_jn, phi_jl, theta_1, theta_2, phi_12, chi_1, chi_2 = (
        lalsim.XLALSimInspiralTransformPrecessingWvf2PE(
            incl, *spin_1, *spin_2, mass_1, mass_2, f_ref, phase
        )
    )  # fmt: skip

    return theta_jn, phi_jl, theta_1, theta_2, phi_12, chi_1, chi_2
