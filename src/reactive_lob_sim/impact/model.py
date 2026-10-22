"""Propagator impact state with square-root size dependence.

The power-law decay g(tau) = (1 + tau/tau0)^(-beta) is approximated by a non-negative sum of exponentials
fitted on a log-spaced lag grid (non-negative least squares), so the impact state updates in O(K) per trade
instead of summing over the whole trade history (Bochud and Challet 2007 use the same device).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from reactive_lob_sim.params import ImpactParams

# Lag grid of the exponential-sum fit, in units of tau0, and its resolution.
_GRID_DECADES = (-2.0, 4.0)
_GRID_POINTS = 200


def power_law_decay(tau: np.ndarray, tau0: float, beta: float) -> np.ndarray:
    return np.asarray((1.0 + np.asarray(tau) / tau0) ** (-beta), dtype=np.float64)


def exponential_sum(tau0: float, beta: float, n_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """Weights ``w`` and rates ``r`` with sum_k w_k exp(-r_k tau) close to (1 + tau/tau0)^(-beta)."""
    if beta == 0.0:
        return np.ones(1), np.zeros(1)
    grid = tau0 * np.concatenate([[0.0], np.logspace(*_GRID_DECADES, _GRID_POINTS)])
    rates = 1.0 / (tau0 * np.logspace(_GRID_DECADES[0], _GRID_DECADES[1], n_terms))
    design = np.exp(-np.outer(grid, rates))
    target = power_law_decay(grid, tau0, beta)
    weights, _ = nnls(design, target)
    return np.asarray(weights), np.asarray(rates)


def peak_impact(params: ImpactParams, qty: float, price: float) -> float:
    """Immediate impact Y * sigma_w * P * (q / V_w)^delta of one trade of size ``qty`` (before kappa)."""
    return float(
        params.Y * params.sigma_window * price * (max(qty, 0.0) / params.volume_window) ** params.delta
    )


class ImpactState:
    """Running impact I(t) of past trades, scaled by the coupling ``kappa``."""

    def __init__(self, params: ImpactParams) -> None:
        self.params = params
        self._w, self._r = exponential_sum(params.tau0_s, params.beta, params.n_exponentials)
        self._state = np.zeros_like(self._w)
        self._t = 0.0

    def _decay_to(self, t: float) -> None:
        dt = t - self._t
        if dt > 0:
            self._state *= np.exp(-self._r * dt)
            self._t = t

    def add_trade(self, t: float, sign: int, qty: float, price: float) -> None:
        self._decay_to(t)
        if self.params.kappa == 0.0:
            return
        self._state += sign * peak_impact(self.params, qty, price)

    def value(self, t: float) -> float:
        self._decay_to(t)
        return float(self.params.kappa * np.dot(self._w, self._state))
