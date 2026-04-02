"""Exact tail counts for additive fitness landscapes via Touzet's algorithm.

Uses iterative granularity refinement with dynamic programming on
discretized scores (Touzet & Varré 2007), adapted from the TFMPvalue
C++ implementation.
"""

import numpy as np
from _touzet_cpp import score_to_pvalue as _score_to_pvalue
from _touzet_cpp import deficit_spectrum as _deficit_spectrum


def tail_count(theta, F, tol=0.0, initial_granularity=0.1,
               max_granularity=1e-10, refinement_factor=10.0,
               max_time=10.0, max_dict_size=1_000_000):
    """Compute N_geq(F): number of sequences with fitness >= F.

    Parameters
    ----------
    theta : array_like, shape (L, C)
        Fitness matrix. theta[l, c] is the fitness contribution of
        character c at position l.
    F : float
        Fitness threshold.
    tol : float
        Fractional error tolerance. 0 means refine until the result is
        certifiably exact. A positive value (e.g. 0.01) stops iteration
        once (upper - lower) / upper <= tol.
    initial_granularity : float
        Starting granularity for the iterative refinement.
    max_granularity : float
        Finest granularity to attempt before stopping.
    refinement_factor : float
        Factor by which the granularity is divided on each iteration.
    max_time : float
        Wall-clock time limit in seconds. 0 means no limit.
    max_dict_size : int
        Maximum number of entries in the DP dictionary at any single
        position. 0 means no limit.

    Returns
    -------
    int or tuple of int
        If tol == 0, returns the exact count. If tol > 0, returns
        (lower_bound, upper_bound).

    Raises
    ------
    ValueError
        If the computation exceeds max_time or max_dict_size.
    """
    theta = np.asarray(theta, dtype=np.float64)
    lo, hi = _score_to_pvalue(theta, float(F), float(tol),
                              initial_granularity, max_granularity,
                              refinement_factor, max_time,
                              int(max_dict_size))
    lo_int = int(round(lo))
    hi_int = int(round(hi))
    if tol == 0.0 or lo_int == hi_int:
        return hi_int
    return (lo_int, hi_int)


def deficit_spectrum(theta, eps_max, tol=0.0, initial_granularity=0.1,
                     max_granularity=1e-10, refinement_factor=10.0,
                     max_time=10.0, max_dict_size=1_000_000):
    """Compute the deficit spectrum: all achievable deficit values and counts.

    The deficit of a sequence x is d(x) = f_max - f(x) >= 0.

    Parameters
    ----------
    theta : array_like, shape (L, C)
        Fitness matrix.
    eps_max : float
        Maximum deficit to include.
    tol : float
        Fractional error tolerance. 0 means refine until certified exact.
    initial_granularity : float
        Starting granularity for the iterative refinement.
    max_granularity : float
        Finest granularity to attempt before stopping.
    refinement_factor : float
        Factor by which the granularity is divided on each iteration.
    max_time : float
        Wall-clock time limit in seconds. 0 means no limit.
    max_dict_size : int
        Maximum number of entries in the DP dictionary at any single
        position. 0 means no limit.

    Returns
    -------
    deficits : ndarray
        Sorted array of achievable deficit values d <= eps_max.
    counts : ndarray
        Number of sequences at each deficit value.

    Raises
    ------
    ValueError
        If the computation exceeds max_time or max_dict_size.
    """
    theta = np.asarray(theta, dtype=np.float64)
    deficits, counts = _deficit_spectrum(theta, float(eps_max), float(tol),
                                         initial_granularity, max_granularity,
                                         refinement_factor, max_time,
                                         int(max_dict_size))
    deficits = np.array(deficits)
    counts = np.array(counts)
    order = np.argsort(deficits)
    return deficits[order], np.round(counts[order]).astype(int)
