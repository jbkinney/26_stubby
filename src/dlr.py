"""Daniels-Lugannani-Rice (DLR) density and mass for additive fitness landscapes.

Provides density(theta, F, expansion_order) and mass(theta, F, expansion_order)
at two validated levels of saddle-point accuracy:

    expansion_order=0  Daniels density / numerical integral of density
    expansion_order=1  Leading Lugannani-Rice (rho_LR, N>^(0))

expansion_order=2 is DISABLED pending resolution of two issues:
  1. The density formula _density_2 had a factor-of-3 error in the Daniels
     correction coefficients (lambda_4/24 and 5*lambda_3^2/72 should be
     lambda_4/8 and 5*lambda_3^2/24).  This has been fixed in the code below
     but not yet validated end-to-end.
  2. The mass formula _mass_2 uses the far-tail expansion of A_1-B_1, which
     diverges near the mean (beta*->0).  The Lugannani-Rice series is supposed
     to be uniformly valid, but the document's derivation of q''(w) drops
     O(1/w^5) terms that are essential for the cancellation at w=0.  A
     uniformly valid implementation requires the exact q''(w) recurrence,
     which has not yet been validated numerically.
"""

import numpy as np
from scipy.special import erfc
from scipy.integrate import quad
from scipy.optimize import brentq
from cgf import (
    solve_beta, log_partition, psi_1, psi_2, psi_3, psi_4,
)


def density(theta, F, expansion_order=0):
    """Density of states rho(F) at the specified expansion order.

    Parameters
    ----------
    theta : array_like, shape (L, C)
        Per-locus fitness effects.
    F : float or array_like
        Fitness value(s).
    expansion_order : int
        0 = Daniels saddle-point density (L0).
        1 = Lugannani-Rice density rho_LR (L1b).
        2 = DISABLED (see module docstring).

    Returns
    -------
    float or ndarray
    """
    if expansion_order == 2:
        raise NotImplementedError(
            "expansion_order=2 is disabled: the density coefficient and mass "
            "formula have known issues (see dlr.py module docstring)."
        )
    theta = np.asarray(theta, dtype=float)
    scalar = np.ndim(F) == 0
    F_arr = np.atleast_1d(np.asarray(F, dtype=float))
    func = {0: _density_0, 1: _density_1}[expansion_order]
    result = np.array([func(theta, f) for f in F_arr])
    return float(result[0]) if scalar else result


def mass(theta, F, expansion_order=0):
    """Tail count N>(F) at the specified expansion order.

    Parameters
    ----------
    theta : array_like, shape (L, C)
        Per-locus fitness effects.
    F : float or array_like
        Fitness value(s).
    expansion_order : int
        0 = Numerical integral of the Daniels density.
        1 = Leading Lugannani-Rice formula N>^(0) (L1a).
        2 = DISABLED (see module docstring).

    Returns
    -------
    float or ndarray
    """
    if expansion_order == 2:
        raise NotImplementedError(
            "expansion_order=2 is disabled: the density coefficient and mass "
            "formula have known issues (see dlr.py module docstring)."
        )
    theta = np.asarray(theta, dtype=float)
    scalar = np.ndim(F) == 0
    F_arr = np.atleast_1d(np.asarray(F, dtype=float))
    func = {0: _mass_0, 1: _mass_1}[expansion_order]
    result = np.array([func(theta, f) for f in F_arr])
    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Density helpers
# ---------------------------------------------------------------------------

def _density_0(theta, F):
    """Daniels saddle-point density rho(F) (L0)."""
    f_max = theta.max(axis=1).sum()
    f_min = theta.min(axis=1).sum()
    if F >= f_max or F <= f_min:
        return 0.0
    beta = solve_beta(theta, F)
    var = psi_2(theta, beta)
    if var < 1e-30:
        return 0.0
    psi_b = log_partition(theta, beta)
    log_rho = psi_b - beta * F - 0.5 * np.log(2.0 * np.pi * var)
    return np.exp(log_rho)


def _density_1(theta, F):
    """Lugannani-Rice density rho_LR(F) (L1b)."""
    f_max = theta.max(axis=1).sum()
    f_min = theta.min(axis=1).sum()
    if F >= f_max or F <= f_min:
        return 0.0

    beta = solve_beta(theta, F)
    if abs(beta) < 1e-14:
        return _density_0(theta, F)

    psi_pp = psi_2(theta, beta)
    psi_ppp = psi_3(theta, beta)
    psi_b = log_partition(theta, beta)

    log_rho = psi_b - beta * F - 0.5 * np.log(2.0 * np.pi * psi_pp)
    rho = np.exp(log_rho)

    L, C = theta.shape
    log_CL = L * np.log(C)
    Delta = max(beta * F - psi_b + log_CL, 0.0)

    correction = (1.0
                  + 1.0 / (beta ** 2 * psi_pp)
                  + psi_ppp / (2.0 * beta * psi_pp ** 2)
                  - abs(beta) * np.sqrt(psi_pp) / (2.0 * Delta) ** 1.5)
    return rho * correction


def _density_2(theta, F):
    """Corrected Daniels density rho_D1(F) (L2b).  DISABLED — see module docstring.

    rho_D1 = rho * [1 + lambda_4/8 - 5*lambda_3^2/24]

    The original code had lambda_4/24 and 5*lambda_3^2/72, which is off by a
    factor of 3 from the standard Daniels (1954) result.  Verified numerically
    against the exact exponential-sum density.
    """
    rho = _density_0(theta, F)
    if rho == 0.0:
        return 0.0

    beta = solve_beta(theta, F)
    sigma2 = psi_2(theta, beta)
    sigma = np.sqrt(sigma2)
    lam3 = psi_3(theta, beta) / sigma ** 3
    lam4 = psi_4(theta, beta) / sigma ** 4

    return rho * (1.0 + lam4 / 8.0 - 5.0 * lam3 ** 2 / 24.0)


# ---------------------------------------------------------------------------
# Mass helpers
# ---------------------------------------------------------------------------

def _mass_0(theta, F):
    """Numerical integral of the Daniels density from F to f_max."""
    L, C = theta.shape
    f_max = theta.max(axis=1).sum()
    f_min = theta.min(axis=1).sum()
    if F > f_max:
        return 0.0
    if F < f_min:
        return float(C ** L)
    upper = f_max - 1e-10 * (f_max - f_min)
    result, _ = quad(lambda f: _density_0(theta, f), F, upper,
                     limit=200, epsrel=1e-10)
    return result


def _mass_1(theta, F):
    """Leading Lugannani-Rice tail count N>^(0)(F) (L1a)."""
    L, C = theta.shape
    log_CL = L * np.log(C)
    f_max = theta.max(axis=1).sum()
    f_min = theta.min(axis=1).sum()
    if F > f_max:
        return 0.0
    if F < f_min:
        return np.exp(log_CL)

    beta = solve_beta(theta, F)
    if abs(beta) < 1e-14:
        return np.exp(log_CL) / 2.0

    psi_b = log_partition(theta, beta)
    psi_pp = psi_2(theta, beta)
    sigma = np.sqrt(psi_pp)

    Delta = max(beta * F - psi_b + log_CL, 0.0)
    w = np.sign(beta) * np.sqrt(2.0 * Delta)

    log_rho = psi_b - beta * F - 0.5 * np.log(2.0 * np.pi * psi_pp)
    rho = np.exp(log_rho)

    erfc_term = np.exp(log_CL) / 2.0 * erfc(w / np.sqrt(2.0))
    correction = rho * (1.0 / beta - sigma / w)
    return erfc_term + correction


def _mass_2(theta, F):
    """Corrected Lugannani-Rice tail count N>^(1)(F) (L2a).  DISABLED — see module docstring.

    N>^(1) = N>^(0) + rho/beta* * [1/u_hat^2 + lam3/(3*u_hat)
                                     + (5*lam3^2 - 3*lam4)/144]

    BUG: This is the far-tail expansion of A_1-B_1, valid only when |u_hat|>>1.
    It diverges near the mean (beta*->0) because the O(1/w^5) terms that cancel
    the divergence in the exact q''(w) have been dropped.  A uniformly valid
    implementation should compute q''(w) via the exact recurrence
    q(w)=(phi(w)-1)/w, q'(w)=(phi'(w)-q(w))/w, q''(w)=(phi''(w)-2*q'(w))/w,
    then use A_1-B_1 = -C^L*exp(-Delta)/(2*sqrt(2*pi)) * q''(w).
    """
    N0 = _mass_1(theta, F)

    f_max = theta.max(axis=1).sum()
    f_min = theta.min(axis=1).sum()
    if F > f_max or F < f_min:
        return N0

    beta = solve_beta(theta, F)
    if abs(beta) < 1e-14:
        return N0

    rho = _density_0(theta, F)
    sigma2 = psi_2(theta, beta)
    sigma = np.sqrt(sigma2)
    u_hat = beta * sigma
    lam3 = psi_3(theta, beta) / sigma ** 3
    lam4 = psi_4(theta, beta) / sigma ** 4

    bracket = (1.0 / u_hat ** 2
               + lam3 / (3.0 * u_hat)
               + (5.0 * lam3 ** 2 - 3.0 * lam4) / 144.0)
    return N0 + (rho / beta) * bracket


# ---------------------------------------------------------------------------
# Density class
# ---------------------------------------------------------------------------

def _sigma_at(theta, F):
    """Saddle-point standard deviation at fitness F."""
    return np.sqrt(max(psi_2(theta, solve_beta(theta, F)), 1e-30))


def _adaptive_grid(theta, F_low, F_high, relative_bin_width):
    """Adaptive grid of F values spanning [F_low, F_high].

    Uses the same root-finding algorithm as _adaptive_bins in sampling.py:
    each grid point F_k satisfies F_k = b_k + 0.5 * rbw * sigma(F_k).
    """
    rbw = relative_bin_width
    f_min = theta.min(axis=1).sum()
    f_max = theta.max(axis=1).sum()
    b0 = (F_low + F_high) / 2

    def _solve_right(b_k):
        sigma_b = _sigma_at(theta, min(b_k, f_max - 1e-10))
        lo = b_k + 1e-14
        hi = min(b_k + rbw * sigma_b * 2, f_max - 1e-10)
        if hi <= lo:
            return None
        def g(F):
            return F - b_k - 0.5 * rbw * _sigma_at(theta, F)
        if g(lo) * g(hi) > 0:
            return None
        return brentq(g, lo, hi, xtol=1e-10)

    def _solve_left(b_k):
        sigma_b = _sigma_at(theta, max(b_k, f_min + 1e-10))
        lo = max(b_k - rbw * sigma_b * 2, f_min + 1e-10)
        hi = b_k - 1e-14
        if lo >= hi:
            return None
        def g(F):
            return F - b_k + 0.5 * rbw * _sigma_at(theta, F)
        if g(lo) * g(hi) > 0:
            return None
        return brentq(g, lo, hi, xtol=1e-10)

    right_centers = []
    b_k = b0
    for _ in range(10_000):
        F_k = _solve_right(b_k)
        if F_k is None or F_k >= F_high:
            break
        s_k = _sigma_at(theta, F_k)
        b_next = b_k + rbw * s_k
        if b_next >= F_high:
            right_centers.append(0.5 * (b_k + F_high))
            break
        right_centers.append(F_k)
        b_k = b_next

    left_centers = []
    b_k = b0
    for _ in range(10_000):
        F_k = _solve_left(b_k)
        if F_k is None or F_k <= F_low:
            break
        s_k = _sigma_at(theta, F_k)
        b_next = b_k - rbw * s_k
        if b_next <= F_low:
            left_centers.append(0.5 * (F_low + b_k))
            break
        left_centers.append(F_k)
        b_k = b_next

    left_centers.reverse()
    return np.array(left_centers + right_centers)


