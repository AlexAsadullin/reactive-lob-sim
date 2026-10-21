"""Impact calibration from tape: square-root law and propagator decay.

1. Square-root law. The sample is cut into windows of ``window_s``. For each window we take the net signed
   volume Q, the traded volume, and the relative mid change r. With sigma_w = std(r) and V_w the mean traded
   volume per window, windows are binned by |Q| / V_w and the bin means of sign(Q) r / sigma_w are fitted
   by least squares in log-log space: E[sign(Q) r] / sigma_w = Y (|Q| / V_w)^delta (Toth et al. 2011;
   Bouchaud, Bonart, Donier and Gould 2018, ch. 12).
2. Propagator. On bins of ``lag_bin_s``, relative mid changes are regressed on lagged order flow
   x_t = sum sign * (q / V_w)^delta (ordinary least squares on lags 0..L). The cumulative response
   G(l) = sum_{m<=l} k_m is normalised at its peak and fitted with (1 + tau/tau0)^(-beta) after the peak
   (Bouchaud et al. 2004; Gatheral 2010).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from scipy.optimize import curve_fit

from reactive_lob_sim.impact.model import power_law_decay
from reactive_lob_sim.params import ImpactParams
from reactive_lob_sim.settings import ImpactFitSettings
from reactive_lob_sim.tape.l2 import BookSeries, TradeSeries
from reactive_lob_sim.timeutil import NS_PER_S

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ImpactFit:
    params: ImpactParams
    bin_x: np.ndarray
    bin_y: np.ndarray
    kernel_lags_s: np.ndarray
    kernel: np.ndarray
    sqrt_r2: float
    n_windows: int


def _mid_at(book: BookSeries, grid_ns: np.ndarray) -> np.ndarray:
    return book.sample(grid_ns).mid


def fit_square_root_law(
    book: BookSeries, trades: TradeSeries, cfg: ImpactFitSettings
) -> tuple[float, float, float, float, np.ndarray, np.ndarray, float, int]:
    """Return ``Y, delta, sigma_w, V_w, bin_x, bin_y, r2, n_windows``."""
    start = max(int(book.ts[0]), int(trades.ts[0]) if trades.ts.size else int(book.ts[0]))
    end = int(book.ts[-1])
    step = round(cfg.window_s * NS_PER_S)
    edges = np.arange(start, end, step, dtype=np.int64)
    if edges.size < cfg.n_bins + 2:
        raise ValueError("sample too short for the square-root-law fit")
    mids = _mid_at(book, edges)
    if mids.size != edges.size:
        raise ValueError("book series does not cover the sample")
    r = np.diff(mids) / mids[:-1]
    idx = np.searchsorted(edges, trades.ts, side="right") - 1
    ok = (idx >= 0) & (idx < r.size)
    q = np.bincount(idx[ok], weights=(trades.sign * trades.size)[ok], minlength=r.size)[: r.size]
    v = np.bincount(idx[ok], weights=trades.size[ok], minlength=r.size)[: r.size]
    sigma_w = float(np.std(r))
    volume_w = float(np.mean(v))
    if sigma_w <= 0 or volume_w <= 0:
        raise ValueError("no price variation or no volume in the sample")
    active = q != 0
    x = np.abs(q[active]) / volume_w
    y = np.sign(q[active]) * r[active] / sigma_w
    qs = np.quantile(x, np.linspace(0, 1, cfg.n_bins + 1))
    which = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, cfg.n_bins - 1)
    bin_x = np.array([x[which == b].mean() for b in range(cfg.n_bins) if np.any(which == b)])
    bin_y = np.array([y[which == b].mean() for b in range(cfg.n_bins) if np.any(which == b)])
    pos = (bin_x > 0) & (bin_y > 0)
    if pos.sum() < 2:
        raise ValueError("impact is not positive in enough bins to fit the square-root law")
    lx, ly = np.log(bin_x[pos]), np.log(bin_y[pos])
    delta, log_y = np.polyfit(lx, ly, 1)
    delta = float(np.clip(delta, cfg.delta_min, cfg.delta_max))
    log_y = float(np.mean(ly - delta * lx))
    pred = log_y + delta * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(np.exp(log_y)), delta, sigma_w, volume_w, bin_x, bin_y, r2, int(r.size)


def fit_propagator(
    book: BookSeries,
    trades: TradeSeries,
    delta: float,
    volume_w: float,
    cfg: ImpactFitSettings,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return ``tau0_s, beta, lags_s, normalised cumulative kernel``."""
    step = round(cfg.lag_bin_s * NS_PER_S)
    edges = np.arange(int(book.ts[0]), int(book.ts[-1]), step, dtype=np.int64)
    lags = cfg.max_lag_bins
    if edges.size < 10 * lags:
        raise ValueError("sample too short for the propagator fit")
    mids = _mid_at(book, edges)
    r = np.diff(mids) / mids[:-1]
    idx = np.searchsorted(edges, trades.ts, side="right") - 1
    ok = (idx >= 0) & (idx < r.size)
    flow = trades.sign * (trades.size / volume_w) ** delta
    x = np.bincount(idx[ok], weights=flow[ok], minlength=r.size)[: r.size]
    design = np.column_stack([np.roll(x, lag) for lag in range(lags + 1)])[lags:]
    coef, *_ = np.linalg.lstsq(design, r[lags:], rcond=None)
    cumulative = np.cumsum(coef)
    lags_s = np.arange(lags + 1) * cfg.lag_bin_s
    peak = int(np.argmax(cumulative))
    if cumulative[peak] <= 0:
        raise ValueError("order flow has no positive price response")
    kernel = cumulative / cumulative[peak]
    tail_t = lags_s[peak:] - lags_s[peak]
    tail_g = np.clip(kernel[peak:], 0.0, None)
    if tail_t.size < 3:
        return cfg.lag_bin_s, 0.0, lags_s, kernel
    (tau0, beta), _ = curve_fit(
        lambda t, a, b: power_law_decay(t, a, b),
        tail_t,
        tail_g,
        p0=(cfg.lag_bin_s, 0.5),
        bounds=((cfg.lag_bin_s / 10.0, 0.0), (lags_s[-1] * 10.0, cfg.beta_max)),
    )
    return float(tau0), float(beta), lags_s, kernel


def fit_impact(book: BookSeries, trades: TradeSeries, cfg: ImpactFitSettings) -> ImpactFit:
    """Fit the full propagator/square-root model; the coupling ``kappa`` is set to 1 (raw tape impact)."""
    y, delta, sigma_w, volume_w, bx, by, r2, n = fit_square_root_law(book, trades, cfg)
    tau0, beta, lags_s, kernel = fit_propagator(book, trades, delta, volume_w, cfg)
    params = ImpactParams(
        Y=y,
        delta=delta,
        sigma_window=sigma_w,
        volume_window=volume_w,
        window_s=cfg.window_s,
        tau0_s=tau0,
        beta=beta,
        kappa=1.0,
        source="all",
        n_exponentials=cfg.n_exponentials,
    )
    log.info("impact_fit", Y=y, delta=delta, tau0_s=tau0, beta=beta, r2=r2, n_windows=n)
    return ImpactFit(params, bx, by, lags_s, kernel, r2, n)
