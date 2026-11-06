"""Calibration of the agent-based market's order-flow parameters from tape-derived events.

Fitted from data: the Hawkes flow (maximum likelihood), the quote-offset distribution (maximum likelihood of
a geometric law), limit and market order sizes (log-normal moments), the cancellation scale ``n_ref_orders``
(from the depth balance, see :func:`fit_n_ref`), the index volatility, and the funding/OI dynamics. The
trader mix and the liquidation book are not identifiable from an anonymous L2 tape and keep their prior
values from the settings file; the impact coupling is calibrated separately (see ``coupling``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reactive_lob_sim.calibration.events import OrderEvents
from reactive_lob_sim.contracts.pit_client import ConditioningSeries
from reactive_lob_sim.hawkes.mle import HawkesFit, fit_hawkes
from reactive_lob_sim.params import (
    CXL_ASK,
    CXL_BID,
    LIM_ASK,
    LIM_BID,
    MKT_BUY,
    MKT_SELL,
    N_EVENT_TYPES,
    ABMParams,
    HawkesParams,
    SizeDist,
)
from reactive_lob_sim.settings import FlowFitSettings, HawkesFitSettings
from reactive_lob_sim.tape.l2 import BookSeries
from reactive_lob_sim.timeutil import NS_PER_S

# The depth balance needs net inflow L - M > 0; when consumption nearly equals arrivals the estimate is
# unstable, so the net inflow is floored at this share of the arrival rate.
_MIN_NET_INFLOW_SHARE = 0.1
# Open interest is logged; this floor (relative to the prior's start value) guards against zeros.
_MIN_OI_FRACTION = 1e-6


@dataclass(frozen=True)
class FlowFit:
    params: ABMParams
    hawkes: HawkesFit
    n_events: int


def fit_size(sizes: np.ndarray) -> SizeDist:
    logs = np.log(sizes[sizes > 0])
    return SizeDist(log_mean=float(logs.mean()), log_sd=float(logs.std()))


def fit_offset(offsets: np.ndarray) -> tuple[int, float]:
    """Minimum offset and the geometric success probability p = 1 / (1 + mean(k - k_min)) (MLE)."""
    k = np.maximum(np.rint(offsets), 0.0)
    k_min = int(k.min()) if k.size else 0
    return k_min, float(1.0 / (1.0 + np.mean(k - k_min))) if k.size else 1.0


def fit_n_ref(events: OrderEvents, book: BookSeries, depth_levels: int, lim_size: float) -> float:
    """Cancellation scale from the stationary depth balance.

    With limit arrivals L, cancel events C and market-order consumption M (all in orders per second on one
    side) and cancellations of Poisson(N / n_ref) orders per cancel event, the stationary number of orders
    is N = n_ref (L - M) / C, so n_ref = N C / (L - M). N is the mean visible depth divided by the mean limit
    size.
    """
    h = events.horizon
    lim = events.of(LIM_BID, LIM_ASK).sum() / (2 * h)
    cxl = events.of(CXL_BID, CXL_ASK).sum() / (2 * h)
    mkt_mask = events.of(MKT_BUY, MKT_SELL)
    consumption = events.size[mkt_mask].sum() / lim_size / (2 * h)
    depth = float(np.mean((book.bid_depth + book.ask_depth) / 2.0))
    n_orders = depth / lim_size
    net = max(lim - consumption, lim * _MIN_NET_INFLOW_SHARE)
    return float(max(n_orders * cxl / net, 1.0))


def fit_index_vol(book: BookSeries, tick_size: float, interval_s: float) -> float:
    grid = np.arange(int(book.ts[0]), int(book.ts[-1]), round(interval_s * NS_PER_S), dtype=np.int64)
    mids = book.sample(grid).mid * tick_size
    r = np.diff(np.log(mids))
    return float(np.std(r) / np.sqrt(interval_s))


def fit_macro(prior: ABMParams, cond: ConditioningSeries) -> dict[str, float]:
    """Funding and open-interest parameters (crypto module only; empty for MOEX)."""
    out: dict[str, float] = {}
    if prior.crypto is None:
        return out
    if cond.funding.size >= 2:
        out["funding_base"] = float(np.mean(cond.funding))
        out["funding_noise_sd"] = float(np.std(cond.funding))
        gaps = np.diff(cond.funding_ts) / NS_PER_S
        if np.all(gaps > 0):
            out["funding_interval_s"] = float(np.median(gaps))
    if cond.oi.size >= 2:
        out["oi_initial"] = float(cond.oi[-1])
        dt = np.diff(cond.oi_ts) / NS_PER_S
        lr = np.diff(np.log(np.maximum(cond.oi, prior.crypto.macro.oi_initial * _MIN_OI_FRACTION)))
        ok = dt > 0
        if ok.any():
            out["oi_vol_per_sqrt_s"] = float(np.std(lr[ok] / np.sqrt(dt[ok])))
    return out


def fit_flow(
    prior: ABMParams,
    events: OrderEvents,
    book: BookSeries,
    cond: ConditioningSeries,
    tick_size: float,
    hawkes_cfg: HawkesFitSettings,
    flow_cfg: FlowFitSettings,
) -> FlowFit:
    if len(events) < flow_cfg.min_events:
        raise ValueError(f"only {len(events)} events; at least {flow_cfg.min_events} are needed to calibrate")
    hfit = fit_hawkes(events.t, events.kind, N_EVENT_TYPES, events.horizon, hawkes_cfg)
    lim_mask = events.of(LIM_BID, LIM_ASK)
    mkt_mask = events.of(MKT_BUY, MKT_SELL)
    k_min, p = fit_offset(events.offset[lim_mask])
    lim_size = fit_size(events.size[lim_mask])
    mkt_size = fit_size(events.size[mkt_mask])
    mean_lim = float(np.mean(events.size[lim_mask]))
    n_ref = fit_n_ref(events, book, flow_cfg.depth_levels, mean_lim)
    liquidity = prior.liquidity.model_copy(
        update={"offset_geom_p": p, "min_offset_ticks": k_min, "n_ref_orders": n_ref, "size": lim_size}
    )
    traders = prior.traders.model_copy(update={"size": mkt_size})
    crypto = prior.crypto
    if crypto is not None:
        crypto = crypto.model_copy(update={"macro": crypto.macro.model_copy(update=fit_macro(prior, cond))})
    first_mid = float(events.mid[0]) * tick_size
    params = prior.model_copy(
        update={
            "initial_price": first_mid,
            "hawkes": hfit.params,
            "liquidity": liquidity,
            "traders": traders,
            "fundamental_vol_per_sqrt_s": fit_index_vol(book, tick_size, flow_cfg.vol_interval_s),
            "regimes": [prior.regimes[0].model_copy(update={"vol_multiplier": 1.0, "rate_multiplier": 1.0})],
            "crypto": crypto,
        }
    )
    return FlowFit(ABMParams.model_validate(params.model_dump()), hfit, len(events))


def zero_intelligence(params: ABMParams) -> ABMParams:
    """Zero-intelligence baseline with the same mean event rates: Poisson flow (no excitation), noise traders
    only, no impact coupling and no liquidation book (in the spirit of Smith, Farmer et al. 2003)."""
    rates = params.hawkes.stationary_rates()
    zero = [[0.0] * N_EVENT_TYPES for _ in range(N_EVENT_TYPES)]
    hawkes = HawkesParams(mu=rates.tolist(), alpha=zero, beta=params.hawkes.beta)
    traders = params.traders.model_copy(update={"noise": 1.0, "fundamental": 0.0, "momentum": 0.0})
    liquidity = params.liquidity.model_copy(update={"pull_to_value": 0.0})
    return params.model_copy(
        update={
            "hawkes": hawkes,
            "traders": traders,
            "liquidity": liquidity,
            "impact": params.impact.model_copy(update={"kappa": 0.0}),
            "crypto": None
            if params.crypto is None
            else params.crypto.model_copy(
                update={"liquidation": params.crypto.liquidation.model_copy(update={"enabled": False})}
            ),
            "fundamental_vol_per_sqrt_s": 0.0,
        }
    )
