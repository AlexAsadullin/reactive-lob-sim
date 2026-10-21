"""Calibration of the impact coupling kappa by a probe-metaorder experiment.

The tape-fitted square-root law gives the total price impact of a metaorder, which in a simulator is the
sum of mechanical impact (liquidity taken from the book) and reactive impact (flow responding to trades,
scaled by kappa). For each kappa on a grid we run the simulator with a probe that buys, and separately sells,
Q = probe_volume_fraction * V_w over one window of ``window_s`` in market-order children, on the same seeds
(the index path and event clock are identical between the buy and sell runs). The measured impact is
E[(r_buy - r_sell) / 2], which cancels the common drift. The calibrated kappa is where the measured curve
crosses the target Y sigma_w (Q / V_w)^delta, by linear interpolation on the grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import structlog

from reactive_lob_sim.engine.l3market import L3Market
from reactive_lob_sim.engine.types import OrderType, Side
from reactive_lob_sim.params import ImpactParams
from reactive_lob_sim.settings import ImpactFitSettings

log = structlog.get_logger(__name__)

MarketFactory = Callable[[float, int, float], L3Market]
"""(kappa, seed, horizon) -> a fresh simulated market."""


@dataclass(frozen=True)
class CouplingFit:
    kappa: float
    grid: list[float]
    measured: list[float]
    measured_se: list[float]
    target: float
    probe_qty: float


def probe_impact(
    factory: MarketFactory, kappa: float, seed: int, qty: float, cfg: ImpactFitSettings, side: Side
) -> float:
    """Relative mid change over the probe window for one run."""
    market = factory(kappa, seed, cfg.probe_duration_s)
    start = cfg.probe_duration_s - cfg.window_s
    market.advance(start)
    mid0 = market.mid_price()
    if mid0 is None:
        raise RuntimeError("no mid at probe start")
    n_child = max(int(cfg.window_s / cfg.probe_child_interval_s), 1)
    child = qty / n_child
    for k in range(n_child):
        market.advance(start + k * cfg.probe_child_interval_s)
        market.submit(side, OrderType.MARKET, child)
    market.advance(cfg.probe_duration_s)
    mid1 = market.mid_price() or mid0
    return (mid1 - mid0) / mid0


def calibrate_coupling(
    factory: MarketFactory, fitted: ImpactParams, cfg: ImpactFitSettings, seed: int
) -> CouplingFit:
    qty = cfg.probe_volume_fraction * fitted.volume_window
    target = fitted.Y * fitted.sigma_window * cfg.probe_volume_fraction**fitted.delta
    measured, errors = [], []
    for kappa in cfg.kappa_grid:
        samples = []
        for s in range(cfg.probe_seeds):
            up = probe_impact(factory, kappa, seed + s, qty, cfg, Side.BUY)
            down = probe_impact(factory, kappa, seed + s, qty, cfg, Side.SELL)
            samples.append((up - down) / 2.0)
        arr = np.asarray(samples)
        measured.append(float(arr.mean()))
        errors.append(float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0)
    kappa = _crossing(cfg.kappa_grid, measured, target)
    log.info("coupling_fit", kappa=kappa, target=target, measured=measured)
    return CouplingFit(kappa, list(cfg.kappa_grid), measured, errors, target, qty)


def _crossing(grid: list[float], measured: list[float], target: float) -> float:
    g = np.asarray(grid)
    m = np.asarray(measured)
    for i in range(len(g) - 1):
        lo, hi = m[i] - target, m[i + 1] - target
        if lo == 0:
            return float(g[i])
        if lo * hi < 0:
            return float(g[i] + (g[i + 1] - g[i]) * (-lo) / (hi - lo))
    return float(g[int(np.argmin(np.abs(m - target)))])
