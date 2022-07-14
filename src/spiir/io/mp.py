import multiprocessing as mp


def validate_cpu_count(nproc: int) -> int:
    """Validates whether the provided nproc argument is within a valid range of
    multiprocessing CPU cores based on the user's machine specifications.

    Parameters
    ----------
    nproc: int
        The number of processes to be used for CPU multiprocessing.

    Returns
    -------
    int
        A valid CPU count (nproc) integer for use in multiprocessing programs.
    """
    if nproc == -1:
        nproc = mp.cpu_count()
    if not isinstance(nproc, int):
        raise TypeError(f"nproc must be of type int, not type {type(nproc)}.")
    if not 1 <= nproc <= mp.cpu_count():
        raise ValueError(f"nproc ({nproc}) must be 1 <= nproc <= {mp.cpu_count()}.")
    return nproc
