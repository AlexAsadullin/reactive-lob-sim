"""Typed parameter models of the simulators.

These models are shared by the settings file (defaults), the calibrators (fitted values) and the calibration
files on disk, so a calibrated simulator is fully described by one ``ABMParams`` document.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Order-event types of the multivariate flow model. The order is part of the calibration file format.
EVENT_TYPES: tuple[str, ...] = ("mkt_buy", "mkt_sell", "lim_bid", "lim_ask", "cxl_bid", "cxl_ask")
N_EVENT_TYPES = len(EVENT_TYPES)
MKT_BUY, MKT_SELL, LIM_BID, LIM_ASK, CXL_BID, CXL_ASK = range(N_EVENT_TYPES)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HawkesParams(_Model):
    """Multivariate Hawkes process with exponential kernels.

    lambda_i(t) = mu_i * m(t) + sum_j alpha_ij * sum_{t_k^j < t} beta * exp(-beta (t - t_k^j)),

    so ``alpha`` is the branching matrix (expected number of type-i children of one type-j event) and the
    process is stationary iff the spectral radius of ``alpha`` is below one. ``m(t)`` is an optional
    regime multiplier on the baseline.
    """

    mu: list[float]
    alpha: list[list[float]]
    beta: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        d = len(self.mu)
        if d == 0 or len(self.alpha) != d or any(len(row) != d for row in self.alpha):
            raise ValueError("alpha must be a square matrix matching mu")
        if any(m < 0 for m in self.mu) or any(a < 0 for row in self.alpha for a in row):
            raise ValueError("Hawkes parameters must be non-negative")
        return self

    @property
    def dim(self) -> int:
        return len(self.mu)

    def mu_array(self) -> np.ndarray:
        return np.asarray(self.mu, dtype=np.float64)

    def alpha_array(self) -> np.ndarray:
        return np.asarray(self.alpha, dtype=np.float64)

    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.alpha_array()))))

    def stationary_rates(self) -> np.ndarray:
        """Mean event rates (I - alpha)^-1 mu of the stationary process."""
        a = self.alpha_array()
        return np.asarray(np.linalg.solve(np.eye(self.dim) - a, self.mu_array()), dtype=np.float64)


class SizeDist(_Model):
    """Log-normal order size in base units, rounded to the lot size and floored at one lot."""

    log_mean: float
    log_sd: float = Field(ge=0)


class LiquidityParams(_Model):
    """Zero-intelligence liquidity provision.

    A limit order is placed ``min_offset_ticks + k`` ticks away from the reference price, with
    ``k ~ Geometric(offset_geom_p) - 1``. The reference price is ``mid + pull_to_value * (V - mid)``, where
    ``V`` is the perceived value (fundamental plus impact state). A cancel event cancels
    ``Poisson(n_orders / n_ref_orders)`` uniformly chosen resting orders on its side, which makes depth
    mean-reverting (cancellation proportional to queue size, as in Smith-Farmer and Cont-Stoikov-Talreja).
    """

    offset_geom_p: float = Field(gt=0, le=1)
    min_offset_ticks: int = Field(ge=0)
    pull_to_value: float = Field(ge=0, le=1)
    n_ref_orders: float = Field(gt=0)
    min_orders_per_side: int = Field(ge=1)
    refill_orders: int = Field(ge=1)
    size: SizeDist


class TraderMix(_Model):
    """How market-order events are attributed to trader types, and how each type picks a direction."""

    noise: float = Field(ge=0)
    fundamental: float = Field(ge=0)
    momentum: float = Field(ge=0)
    fundamental_threshold_ticks: float = Field(ge=0)
    momentum_lookback_s: float = Field(gt=0)
    funding_flow_sensitivity: float = Field(ge=0, le=0.5)
    funding_scale: float = Field(gt=0)
    size: SizeDist


class Regime(_Model):
    vol_multiplier: float = Field(gt=0)
    rate_multiplier: float = Field(gt=0)
    mean_duration_s: float = Field(gt=0)


class LiquidationParams(_Model):
    """Leveraged-position book that turns price moves into forced market orders.

    The long share of open interest is ``0.5 + crowding_sensitivity * tanh(funding / funding_scale)``
    (positive funding means longs pay, i.e. longs are crowded). ``oi_fraction_at_risk`` of open interest is
    spread over positions opened at recent prices with the given leverages; a long opened at ``p`` with
    leverage ``L`` is liquidated (market sell) when the mid falls below ``p * (1 - 1/L + maintenance)``.
    """

    enabled: bool
    leverages: list[float]
    maintenance_margin: float = Field(ge=0, lt=1)
    crowding_sensitivity: float = Field(ge=0, le=0.5)
    funding_scale: float = Field(gt=0)
    oi_fraction_at_risk: float = Field(ge=0, le=1)
    refresh_interval_s: float = Field(gt=0)
    max_positions: int = Field(ge=1)


class MacroParams(_Model):
    """Funding and open interest dynamics of a perpetual swap.

    Funding at each interval is ``base + premium_coeff * (mark - index) / index + noise``, clipped to
    ``[-clip, clip]``, where the index is the fundamental value. Open interest follows a discretised
    Ornstein-Uhlenbeck process in log space and is reduced by liquidations.
    """

    funding_interval_s: float = Field(gt=0)
    funding_base: float
    funding_premium_coeff: float
    funding_noise_sd: float = Field(ge=0)
    funding_clip: float = Field(gt=0)
    oi_initial: float = Field(gt=0)
    oi_reversion_per_s: float = Field(ge=0)
    oi_vol_per_sqrt_s: float = Field(ge=0)


class ImpactParams(_Model):
    """Propagator model with square-root size dependence (Bouchaud et al. 2004; Gatheral 2010).

    A trade of signed size ``s * q`` at time ``t_i`` shifts the perceived value by

        I(t) = kappa * sum_i s_i * Y * sigma_w * P * (q_i / V_w) ** delta * g(t - t_i),
        g(tau) = (1 + tau / tau0_s) ** (-beta),

    where ``sigma_w`` is the relative price volatility and ``V_w`` the traded volume over ``window_s``,
    and ``P`` the current price. ``Y`` and ``delta`` come from the square-root-law fit, ``tau0_s`` and
    ``beta`` from the propagator fit, and ``kappa`` is the coupling that sets how strongly simulated order
    flow reacts to the trades (calibrated so the simulator reproduces the fitted impact).
    """

    Y: float = Field(ge=0)
    delta: float = Field(gt=0, le=1.5)
    sigma_window: float = Field(gt=0)
    volume_window: float = Field(gt=0)
    window_s: float = Field(gt=0)
    tau0_s: float = Field(gt=0)
    beta: float = Field(ge=0)
    kappa: float = Field(ge=0)
    source: Literal["all", "experimental"]
    n_exponentials: int = Field(ge=1)


class InitialBook(_Model):
    levels_per_side: int = Field(ge=1)
    orders_per_level: int = Field(ge=1)
    spread_ticks: int = Field(ge=1)


class ABMParams(_Model):
    """Complete description of an agent-based market."""

    initial_price: float = Field(gt=0)
    hawkes: HawkesParams
    liquidity: LiquidityParams
    traders: TraderMix
    fundamental_vol_per_sqrt_s: float = Field(ge=0)
    regimes: list[Regime] = Field(min_length=1)
    liquidation: LiquidationParams
    macro: MacroParams
    impact: ImpactParams
    initial_book: InitialBook

    @model_validator(mode="after")
    def _check_dims(self) -> Self:
        if self.hawkes.dim != N_EVENT_TYPES:
            raise ValueError(
                f"ABM Hawkes flow must have {N_EVENT_TYPES} dimensions ({', '.join(EVENT_TYPES)})"
            )
        return self
