"""Landscape: density of states for additive fitness landscapes."""

import numpy as np
from scipy.stats import norm

from cgf import psi_1, psi_2
from dlr import density as _dlr_density, mass as _dlr_mass

_METHODS = ('gaussian', 'saddlepoint', 'lugannanirice')
_METHOD_TO_ORDER = {'saddlepoint': 0, 'lugannanirice': 1}


def _validate_method(method):
    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {_METHODS!r}, got {method!r}")


class Landscape:
    """Additive fitness landscape with density-of-states computation.

    Parameters
    ----------
    theta : ndarray, shape (L, C)
        Per-locus fitness contributions.
    alphabet : list of str, optional
        Character labels (default: ['A','C','G','T'][:C]).
    """

    def __init__(self, theta, alphabet=None):
        self.theta = np.asarray(theta, dtype=float)
        L, C = self.theta.shape

        if alphabet is None:
            default = ['A', 'C', 'G', 'T']
            self.alphabet = default[:C] if C <= 4 else [str(i) for i in range(C)]
        else:
            self.alphabet = list(alphabet)

        self.F_min = float(self.theta.min(axis=1).sum())
        self.F_max = float(self.theta.max(axis=1).sum())

    def density(self, F, method='saddlepoint'):
        """Density of states rho(F).

        Parameters
        ----------
        F : float or array_like
        method : {'gaussian', 'saddlepoint', 'lugannanirice'}
        """
        _validate_method(method)
        if method == 'gaussian':
            return self._density_gaussian(F)
        return _dlr_density(self.theta, F, _METHOD_TO_ORDER[method])

    def mass(self, F, method='lugannanirice'):
        """Tail count N_>(F).

        Parameters
        ----------
        F : float or array_like
        method : {'gaussian', 'saddlepoint', 'lugannanirice'}
        """
        _validate_method(method)
        if method == 'gaussian':
            return self._mass_gaussian(F)
        return _dlr_mass(self.theta, F, _METHOD_TO_ORDER[method])

    def _density_gaussian(self, F):
        L, C = self.theta.shape
        CL = float(C ** L)
        mu = psi_1(self.theta, 0.0)
        sigma2 = psi_2(self.theta, 0.0)
        sigma = np.sqrt(sigma2)
        scalar = np.ndim(F) == 0
        F_arr = np.atleast_1d(np.asarray(F, dtype=float))
        result = CL * norm.pdf(F_arr, loc=mu, scale=sigma)
        return float(result[0]) if scalar else result

    def _mass_gaussian(self, F):
        L, C = self.theta.shape
        CL = float(C ** L)
        mu = psi_1(self.theta, 0.0)
        sigma2 = psi_2(self.theta, 0.0)
        sigma = np.sqrt(sigma2)
        scalar = np.ndim(F) == 0
        F_arr = np.atleast_1d(np.asarray(F, dtype=float))
        result = CL * norm.sf(F_arr, loc=mu, scale=sigma)
        return float(result[0]) if scalar else result
