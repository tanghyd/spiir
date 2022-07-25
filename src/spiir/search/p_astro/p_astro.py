import numpy as np

from .mass_contour import predict_source_p_astro

def estimate_bayes_factor(
    far: float,
    cohsnr: float,
    far_threshold: float=1e-4,
    cohsnr_threshold: float=6,
) -> float:
    """
    Compute bayesfactor for using an approximate ("Toy") model.

    Parameters
    ----------
    far: float
        false alarm rate of the event
    cohsnr: float
        Coherent SNR of the event
    far_threshold : float
        threshold false alarm rate
    cohsnr_threshold : float
        threshold Coherent SNR

    Returns
    -------
    bayesfactor : float
        bayesfactor of event
    """
    # Compute astrophysical bayesfactor for GraceDB event
    foreground = 3 * cohsnr_threshold**3 / (cohsnr**4)
    background = far / far_threshold
    return foreground / background


# def evaluate_p_astro_from_bayesfac(
#     astro_bayesfac: float,
#     mean_values_dict: dict[str, float],
#     coefficients: dict[str, float],
#     mchirp: float,
#     snr: float,
#     eff_distance: float,
#     m_bounds: tuple[float, float],
#     mgap_bounds: tuple[float, float],
#     group_mgap: bool = True,
#     lal_cosmology: bool = True,
#     truncate_lower_dist: float | None = 0.003,
#     activation_counts: float | None = None
# ):
#     """
#     Evaluates `p_astro` for a new event using Bayes factor, masses, and number
#     of astrophysical categories. Invoked with every new GraceDB entry.

#     Parameters
#     ----------
#     astro_bayesfac : float
#         astrophysical Bayes factor
#     mean_values_dict: dictionary
#         mean values of Poisson counts
#     coefficients: float
#         The estimated model coefficients of fitted mass/distance models.
#     mchirp: float
#         The source frame chirp mass.
#     snr: float
#         The coincident signal-to-noise ratio (SNR)
#     eff_distance: float
#         The estimated effective distance, usually taken as the minimum across all coincident detectors.
#     m_bounds: tuple[float, float]
#         The upper and lower bounds for both component masses (m1 >= m2).
#     mgap_bounds: tuple[float, float]
#         The boundaries that define the mass gap between BH and NS.
#     group_mgap: bool
#         If True, aggregates Mass Gap from BH+Gap, Gap+NS, and Gap+Gap.
#     lal_cosmology: bool
#         If True, it uses the Planck15 cosmology model as defined in lalsuite instead of the astropy default.

#     Returns
#     -------
#     p_astro : dictionary
#         p_astro for all source categories
#     """

#     source_probabilities = predict_source_p_astro(
#         coefficients,
#         mchirp,
#         snr,
#         eff_distance,
#         m_bounds,
#         mgap_bounds,
#         group_mgap,
#         lal_cosmology,
#         truncate_lower_dist=truncate_lower_dist,
#     )

#     a_hat_bns = probs["BNS"]
#     a_hat_bbh = probs["BBH"]
#     a_hat_nsbh = probs["NSBH"]

#     # Compute category-wise Bayes factors
#     # from astrophysical Bayes factor
#     rescaled_fb = len(probs) * astro_bayesfac
#     bns_bayesfac = a_hat_bns * rescaled_fb
#     nsbh_bayesfac = a_hat_nsbh * rescaled_fb
#     bbh_bayesfac = a_hat_bbh * rescaled_fb

#     # Construct category-wise Bayes factor dictionary
#     event_bayesfac_dict = {
#         "counts_BNS": bns_bayesfac,
#         "counts_NSBH": nsbh_bayesfac,
#         "counts_BBH": bbh_bayesfac
#     }

#     # Compute the p-astro values for each source category
#     # using the mean values
#     p_astro_values = {}
#     for category in mean_values_dict:
#         p_astro_values[category.split("_")[1]] = p_astro_update(
#             category=category,
#             event_bayesfac_dict=event_bayesfac_dict,
#             mean_values_dict=mean_values_dict
#         )

#     return p_astro_values


# def compute_pastro(
#     p_astro_labels: list[str] = ["BNS", "NSBH", "BBH", "Terrestrial"]
# ) -> dict[str, float]:
#     p_astro_values = np.abs(np.random.randn(4))
#     p_astro_values /= p_astro_values.sum()  # normalise
#     p_astro = {}
#     for label, value in zip(p_astro_labels, p_astro_values.tolist()):
#         p_astro[label] = value
#     return p_astro

def p_astro_update(category, event_bayesfac_dict, mean_values_dict):
    """
    Compute `p_astro` for a new event using mean values of Poisson expected
    counts constructed from all the previous events. Invoked with every new
    GraceDB entry.

    Parameters
    ----------
    category : string
        source category
    event_bayesfac_dict : dictionary
        event Bayes factors
    mean_values_dict : dictionary
        mean values of Poisson counts

    Returns
    -------
    p_astro : float
        p_astro by source category
    """
    if category == "counts_Terrestrial":
        numerator = mean_values_dict["counts_Terrestrial"]
    else:
        numerator = event_bayesfac_dict[category] * mean_values_dict[category]

    denominator = (
        mean_values_dict["counts_Terrestrial"]
        + np.sum([
            mean_values_dict[key] * event_bayesfac_dict[key] 
            for key in event_bayesfac_dict
        ])
    )

    return numerator / denominator


def compute_p_astro(
    # astro_bayesfac: float,
#     mean_values_dict: dict[str, float],
    coefficients: dict[str, float],
    mchirp: float,
    cohsnr: float,
    far: float,
    eff_distance: float,
    m_bounds: tuple[float, float],
    mgap_bounds: tuple[float, float],
    far_threshold: float=1e-4,
    cohsnr_threshold: float=8.5,
) -> dict[str, float]:

    astro_probs = predict_source_p_astro(
        coefficients,
        mchirp,
        cohsnr,
        eff_distance,
        m_bounds,
        mgap_bounds,
        group_mgap=True,
        lal_cosmology=True,
        truncate_lower_dist=0.003,
    )

    # Compute category-wise Bayes factors from astrophysical Bayes factor
    astro_bayesfac = estimate_bayes_factor(far, cohsnr, far_threshold, cohsnr_threshold)

    return {
        "Terrestrial": (64/86400)*14394240,
        **{key: val*len(astro_probs)*astro_bayesfac for key, val in astro_probs.items()}
    }