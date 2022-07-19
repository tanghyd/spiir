import concurrent.futures
import logging
import multiprocessing as mp
from collections import Callable
from tqdm import tqdm
from typing import Optional

import numpy as np

from spiir.io.array import chunk_iterable

logger = logging.getLogger(__name__)


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


def apply_parallel_array_func(
    func: Callable,
    array: np.ndarray,
    output_dtype: Optional[type]=None,
    output_shape: tuple=(),
    nproc: int=4,
    chunk_size: Optional[int]=None,
    desc: Optional[str]=None,
    verbose: bool=True,
) -> np.ndarray:
    """Simple multiprocessing for arbitrary functions or class methods on large arrays.
    
    This function will break up an input `array` into chunks of size `chunk_size`, and 
    pass each chunk to the pool of `nproc` processes that each runs the `func` callable.
    The input array is iterated over its **first dimension only**, and will return an 
    output array of equal length with an output dtype and shape specified by 
    `output_dtype` and `output_shape`.

    Parameters
    ----------
    func: Callable
        An instantiated class object or function that processes numpy arrays when 
        called. It must either have a method named accordingly given the `method` 
        input argument, else it must have a `__call__` method.
    array: np.ndarray
        An n-dimensional array of samples that match the dimensions of data fitted to 
        the model provided - i.e. if a function input requires 2-dimensional data, 
        then we would require array to be of shape (n, 2), where n is the array length.
    output_dtype: type | None
        The dtype of the output array. If None, copies the dtype of the input `array`.
    output_shape: tuple
        The shape of each output array element. If output_sample_shape=(), then the 
        output array would be a 1-dimensional array of length `len(array)`. If, for 
        example output_sample_shape=(2, 128), then we would have an output array 
        of shape (len(array), 2, 128).
    nproc: int = 4
        The number of CPU processes to use for multiprocessing.
    chunk_size: int | None
        The intermediate size to break the input array before being passed to processes.
        If None, chunks will be evenly split over the number of processes with a 
        ceiling if the array length is not evenly divisible by the number of processes.
    desc: str | None
        The text description for the progress bar.
    verbose: bool
        Whether or not to display the progress bar.

    Returns
    -------
    np.ndarray:
        The model output array computed for each sample in the input array.

    Notes
    -----
    If any parameters need to be passed to the `func` function beforehand, we recommend 
    using functools.partial to partially instantiate a version of their function with 
    their appropriate arguments and keyword arguments, as `apply_parallel_array_func` 
    does not provide the functionality to do this inside its execution logic.

    For more complex cases (e.g. having different function parameters for each sample, 
    or where multiple multiprocessing steps are required per sample), we recommend that 
    the user writes their own multiprocessing implementation using the code in this 
    function as a skeleton.

    Examples
    --------
    For example, consider a Kernel Density Estimator model. The computational cost for 
    scoring samples from a fitted KDE greatly increases with respect to the number of 
    samples. Additionally, KDEs fitted with a large number of samples also take much 
    longer to estimate scores than KDEs fitted with a smaller number of samples. 
    For these reasons, we may wish to improve the speed of computing the score_samples 
    method of KDE models via multiprocessing with this function. 
    
    For a thorough summary of the computational efficiency and trade-offs for KDE see: 
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/.
    
    >>> import numpy as np    
    >>> from sklearn.neighbors import KernelDensity
    ...
    >>> # assume we have some 2-dimensional data we want to fit a KDE to
    >>> seed = 100
    >>> rng = np.random.default_rng(seed)
    >>> data = np.stack([
    ...    rng.normal(loc=0, scale=1, size=100000),
    ...    rng.normal(loc=5, scale=3, size=100000)
    ... ], axis=1)
    ...
    >>> # randomly create train and test split
    >>> train_idx = rng.choice(range(data.shape[0]), 50000, replace=False)
    >>> train = data[train_idx]
    >>> test = data[~train_idx]
    ...
    >>> # fit kde to training data
    >>> kde = KernelDensity(bandwidth=0.3, rtol=1e-4)
    >>> kde.fit(train)  # cannot parallelise .fit method
    ...
    >>> # estimate likelihood scores for test samples with 4 workers in parallel
    >>> log_density = apply_parallel_array_func(kde.score_samples, test, nproc=4)
    """
    # validate chunk size
    nproc = validate_cpu_count(nproc)
    total = len(array)
    chunk_size = chunk_size or int(np.ceil(total / nproc))
    if not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size {chunk_size} must be an integer.")
    if not (1 <= chunk_size <= total):
        raise ValueError(f"chunk_size {chunk_size} must be 1 <= chunk_size <= {total}")

    output = np.empty((total, *output_shape), dtype=output_dtype or array.dtype)
    with tqdm(total=total, desc=desc, disable=not verbose) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = {}
            # submit futures to worker pool while keeping track of indices
            for chunk in chunk_iterable(range(total), size=chunk_size):
                indices = list(chunk)  # converts from an iterable to list for indexing
                future = executor.submit(func, array[indices])
                futures[future] = indices  # stores array indices with future object
                
            for future in concurrent.futures.as_completed(futures):
                # assign result to correct output array indices (dict value)
                output[futures[future]] = future.result()
                pbar.update(chunk_size)
            pbar.refresh()
    return output