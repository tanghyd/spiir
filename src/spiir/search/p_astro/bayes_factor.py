from typing import Dict, Iterable, Optional, Union

import numpy as np


def compute_autocorr_norm(autocorr: np.ndarray) -> np.ndarray:
    # should we only calculate these values with the real component?
    return np.sum([2 * (1 - np.abs(x.real) ** 2) for x in autocorr], axis=1)


def compute_sum_of_squares(x: np.ndarray) -> np.ndarray:
    return np.sum(x.real**2, axis=1)


def compute_log_odds(
    snr: float,
    chisq: float,
    dof: float,  # autocorr_norm
    a: float,  # autocorr_sum_sq
    beta: float,  # signal => 0.5, noise => -1.0
) -> float:
    # TODO: Verify this function has correctly implemented the mathematics
    x = chisq * dof  # unreduced is multiplied by dof
    a = a - 1

    # calculate mean and variance of unreduced chisq (g)
    chi_mean = dof + np.power(beta * snr, 2) * a
    chi_std = np.sqrt(2 * dof + 4 * np.power(beta * snr, 2) * a)

    bound = dof + np.power(0.06 * snr, 2) * a
    if isinstance(x, np.ndarray):
        x = np.where(x < bound, bound, x)
    elif x < bound:
        x = bound

    return -0.5 * np.power((chisq - chi_mean) / chi_std, 2) - np.log(
        chi_std * np.sqrt(2 * np.pi)
    )


class BayesFactorModel:
    """New ranking statistic model using log bayes factor by Victor Oloworaran et al."""

    LOG_BF_THRESHOLD = 200.0

    def __init__(
        self,
        autocorr_norm: Union[Dict[str, np.ndarray], np.ndarray],
        autocorr_sum_sq: Union[Dict[str, np.ndarray], np.ndarray],
        ifos: Optional[Iterable] = None,
        beta_signal: float = 0.5,
        beta_noise: float = -1.0,
    ):
        if ifos is None:
            if isinstance(autocorr_norm, dict) and isinstance(autocorr_sum_sq, dict):
                assert sorted(autocorr_norm) == sorted(autocorr_sum_sq)
                self.ifos = tuple(sorted(autocorr_norm.keys()))
                self.autocorr_norm = np.array([autocorr_norm[i] for i in self.ifos])
                self.autocorr_sum_sq = np.array([autocorr_sum_sq[i] for i in self.ifos])
            else:
                raise ValueError("ifos not provided and autocorr data not a dict.")
        else:
            self.ifos = tuple(ifos)
            self.autocorr_norm = autocorr_norm  # degrees of freedom
            self.autocorr_sum_sq = autocorr_sum_sq  # sum of sq(autocorr)

        # beta hyperparameters for signal and noise models
        self.beta_signal = beta_signal
        self.beta_noise = beta_noise

        # check input arguments
        if self.autocorr_norm.shape != self.autocorr_sum_sq.shape:
            raise ValueError("Autocorrelation array shapes do not match.")
        if self.autocorr_norm.shape[0] != len(self.ifos):
            raise ValueError("Autocorrelation array counts do not match ifos provided.")
        for ifo in self.ifos:
            if ifo not in (valid_ifos := {"H1", "L1", "V1", "K1"}):
                raise ValueError(f"ifo must be one of {valid_ifos} not {ifo}.")

    def __repr__(self):
        """Overrides string representation of cls when printed."""
        shape = next(iter(self.autocorr_norm.values())).shape
        return f"{type(self).__name__}(ifos={self.ifos}, templates={shape})"

    def predict(
        self,
        snr: np.ndarray,
        chisq: np.ndarray,
        bankid: np.ndarray,
        tmplt_idx: np.ndarray,
        ifo: Optional[str] = None,
        log: bool = False,
    ) -> np.ndarray:
        """Computes the log odds ratio for a given single detector trigger."""
        if isinstance(tmplt_idx, np.ndarray):
            tmplt_idx = tmplt_idx.astype(int)
        if isinstance(bankid, np.ndarray):
            bankid = bankid.astype(int)

        if ifo is not None:
            idx = self.ifos.index(ifo)
            norm = self.autocorr_norm[idx, bankid, tmplt_idx]
            sum_sq = self.autocorr_sum_sq[idx, bankid, tmplt_idx]
        else:
            # TODO: test logic for network snr case (consider weighting ifo by SNR?)
            norm = self.autocorr_norm[:, bankid, tmplt_idx].mean()
            sum_sq = self.autocorr_sum_sq[:, bankid, tmplt_idx].mean()

        signal = compute_log_odds(snr, chisq, norm, sum_sq, self.beta_signal)
        noise = compute_log_odds(snr, chisq, norm, sum_sq, self.beta_noise)
        log_bf = signal - noise + np.power(snr, 2) / 2
        if log:
            return log_bf
        else:
            # truncate extreme log bayes factor to avoid inf/-inf values
            log_bf[np.where(log_bf > self.LOG_BF_THRESHOLD)] = self.LOG_BF_THRESHOLD
            log_bf[np.where(log_bf < -self.LOG_BF_THRESHOLD)] = -self.LOG_BF_THRESHOLD
            return np.exp(log_bf)
