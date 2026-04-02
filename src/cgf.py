"""Cumulant generating function (CGF) primitives for additive fitness landscapes.

The CGF is Psi(beta) = sum_l log(sum_c exp(beta * theta_lc)), where theta_lc
is the fitness contribution of character c at locus l.  All saddle-point
methods (Daniels, Lugannani-Rice) build on these primitives.
"""

import numpy as np


def tilted_probs(theta, beta):
    """Tilted probabilities p_lc(beta) = exp(beta*theta_lc) / Z_l.

    Parameters
    ----------
    theta : ndarray, shape (L, C)
    beta : float

    Returns
    -------
    ndarray, shape (L, C)
    """
    lw = beta * theta
    lw -= lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def log_partition(theta, beta):
    """Log partition function Psi(beta) = sum_l log(sum_c exp(beta*theta_lc))."""
    lw = beta * theta
    mx = lw.max(axis=1)
    return np.sum(mx + np.log(np.exp(lw - mx[:, None]).sum(axis=1)))


def psi_1(theta, beta):
    """First derivative Psi'(beta) = expected fitness under the tilted measure."""
    p = tilted_probs(theta, beta)
    return np.sum(p * theta)


def psi_2(theta, beta):
    """Second derivative Psi''(beta) = fitness variance under the tilted measure."""
    p = tilted_probs(theta, beta)
    mu_l = (p * theta).sum(axis=1)
    return np.sum(p * (theta - mu_l[:, None]) ** 2)


def psi_3(theta, beta):
    """Third derivative Psi'''(beta) = third central moment under the tilted measure."""
    p = tilted_probs(theta, beta)
    mu_l = (p * theta).sum(axis=1)
    return np.sum(p * (theta - mu_l[:, None]) ** 3)


def psi_4(theta, beta):
    """Fourth derivative Psi^(4)(beta) = fourth cumulant under the tilted measure."""
    p = tilted_probs(theta, beta)
    mu_l = (p * theta).sum(axis=1)
    d = theta - mu_l[:, None]
    m4 = (p * d ** 4).sum(axis=1)
    m2 = (p * d ** 2).sum(axis=1)
    return np.sum(m4 - 3.0 * m2 ** 2)


def solve_beta(theta, F, tol=1e-12, max_iter=200):
    """Find the saddle point beta* such that Psi'(beta*) = F.

    Uses Newton's method: beta <- beta + (F - Psi'(beta)) / Psi''(beta).
    """
    beta = 0.0
    for _ in range(max_iter):
        residual = F - psi_1(theta, beta)
        if abs(residual) < tol:
            return beta
        var = psi_2(theta, beta)
        if var < 1e-30:
            break
        beta += residual / var
    return beta


def entropy_deficit(theta, beta, F):
    """Entropy deficit Delta = beta*F - Psi(beta) + Psi(0).

    Equals beta*F - K(beta) where K is the normalized CGF.
    """
    return beta * F - log_partition(theta, beta) + log_partition(theta, 0.0)
