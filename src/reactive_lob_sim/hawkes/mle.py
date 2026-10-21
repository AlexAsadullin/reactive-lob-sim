"""Maximum-likelihood calibration and simulation of multivariate Hawkes order flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from scipy.optimize import minimize

from reactive_lob_sim.hawkes import kernels
from reactive_lob_sim.params import HawkesParams
from reactive_lob_sim.settings import HawkesFitSettings

log = structlog.get_logger(__name__)

# Objective value returned where the intensity is non-positive; finite because L-BFGS-B rejects inf.
_INFEASIBLE = 1e300


@dataclass(frozen=True)
class HawkesFit:
    params: HawkesParams
    loglik: float
    n_events: int
    horizon: float
    converged: bool
    rescaled: bool

    @property
    def aic(self) -> float:
        d = self.params.dim
        return 2 * (d + d * d + 1) - 2 * self.loglik


def simulate_hawkes(
    params: HawkesParams,
    horizon: float,
    seed: int,
    breaks: np.ndarray | None = None,
    multipliers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate event times and types on ``[0, horizon)``; see :func:`kernels.simulate`."""
    if breaks is None or multipliers is None:
        breaks = np.zeros(1)
        multipliers = np.ones(1)
    if breaks.shape != multipliers.shape or breaks[0] != 0.0:
        raise ValueError("breaks must start at 0 and match multipliers")
    if params.spectral_radius() >= 1.0:
        raise ValueError("non-stationary Hawkes parameters (spectral radius >= 1)")
    times, types = kernels.simulate(
        params.mu_array(),
        params.alpha_array(),
        float(params.beta),
        float(horizon),
        breaks.astype(np.float64),
        multipliers.astype(np.float64),
        int(seed),
    )
    return np.asarray(times), np.asarray(types)


def log_likelihood(params: HawkesParams, times: np.ndarray, types: np.ndarray, horizon: float) -> float:
    ll, *_ = kernels.loglik_grad(
        params.mu_array(), params.alpha_array(), float(params.beta), times, types.astype(np.int64), horizon
    )
    return float(ll)


def fit_hawkes(
    times: np.ndarray, types: np.ndarray, dim: int, horizon: float, cfg: HawkesFitSettings
) -> HawkesFit:
    """Fit (mu, alpha, beta) by L-BFGS-B on log-parameters with the exact gradient.

    The start point splits each empirical rate into a baseline share ``1 - init_branching`` and an excited
    share spread evenly over the ``dim`` parents. If the optimum is non-stationary, alpha is rescaled to the
    spectral radius ``max_spectral_radius`` (reported in :attr:`HawkesFit.rescaled`).
    """
    times = np.ascontiguousarray(times, dtype=np.float64)
    types = np.ascontiguousarray(types, dtype=np.int64)
    if times.size < dim * 2:
        raise ValueError("too few events to fit a Hawkes process")
    if np.any(np.diff(times) < 0):
        raise ValueError("event times must be sorted")
    if types.min() < 0 or types.max() >= dim:
        raise ValueError("event types outside [0, dim)")
    n = times.size
    rates = np.maximum(np.bincount(types, minlength=dim) / horizon, cfg.param_floor)
    mu0 = rates * (1.0 - cfg.init_branching)
    alpha0 = np.full((dim, dim), cfg.init_branching / dim)
    x0 = np.concatenate([np.log(mu0), np.log(alpha0).ravel(), [np.log(cfg.init_beta)]])
    floor = np.log(cfg.param_floor)
    bounds = [(floor, None)] * (dim + dim * dim) + [(np.log(cfg.beta_min), np.log(cfg.beta_max))]

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        mu = np.exp(x[:dim])
        alpha = np.exp(x[dim : dim + dim * dim]).reshape(dim, dim)
        beta = float(np.exp(x[-1]))
        ll, g_mu, g_alpha, g_beta = kernels.loglik_grad(mu, alpha, beta, times, types, horizon)
        if not np.isfinite(ll):
            return _INFEASIBLE, np.zeros_like(x)
        grad = np.concatenate([g_mu * mu, (g_alpha * alpha).ravel(), [g_beta * beta]])
        return -ll / n, -grad / n

    res = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": cfg.max_iter, "ftol": cfg.tolerance},
    )
    mu = np.exp(res.x[:dim])
    alpha = np.exp(res.x[dim : dim + dim * dim]).reshape(dim, dim)
    beta = float(np.exp(res.x[-1]))
    radius = float(np.max(np.abs(np.linalg.eigvals(alpha))))
    rescaled = radius >= cfg.max_spectral_radius
    if rescaled:
        alpha = alpha * (cfg.max_spectral_radius / radius)
        log.warning("hawkes_rescaled", spectral_radius=radius, target=cfg.max_spectral_radius)
    params = HawkesParams(mu=mu.tolist(), alpha=alpha.tolist(), beta=beta)
    ll = log_likelihood(params, times, types, horizon)
    log.info("hawkes_fit", n_events=n, loglik=ll, beta=beta, spectral_radius=params.spectral_radius())
    return HawkesFit(params, ll, n, horizon, bool(res.success), rescaled)
