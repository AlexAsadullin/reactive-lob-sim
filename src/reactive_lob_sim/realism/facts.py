"""Stylised facts of a limit-order-book tape.

Facts (Cont 2001; Bouchaud et al. 2018; Cont, Kukanov and Stoikov 2014):
excess kurtosis and the Hill tail index of mid returns (fat tails);
autocorrelation of returns (close to zero) and of absolute returns (positive: volatility clustering);
spread and top-of-book depth distributions;
the order-flow-imbalance (OFI) regression dmid = a + b OFI over fixed intervals and its R^2;
the market-impact response function R(tau) = E[eps_t (m_{t+tau} - m_t)] over trades, in ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from reactive_lob_sim.settings import RealismSettings
from reactive_lob_sim.tape.l2 import BookSeries, book_series, trade_series
from reactive_lob_sim.tape.tape import Tape
from reactive_lob_sim.timeutil import NS_PER_S


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    denom = float(np.dot(x, x))
    if denom == 0 or x.size <= lags:
        return np.zeros(lags)
    return np.asarray([np.dot(x[:-k], x[k:]) / denom for k in range(1, lags + 1)])


def hill_index(x: np.ndarray, tail_fraction: float) -> float:
    """Hill (1975) estimator of the tail index of |x| over the top ``tail_fraction`` order statistics."""
    a = np.sort(np.abs(x[x != 0]))[::-1]
    k = max(int(a.size * tail_fraction), 2)
    if a.size <= k:
        return float("nan")
    logs = np.log(a[:k]) - np.log(a[k])
    mean = float(np.mean(logs))
    return 1.0 / mean if mean > 0 else float("nan")


def ofi_series(book: BookSeries) -> np.ndarray:
    """Event-level OFI e_n of Cont, Kukanov and Stoikov (2014), eq. (2)."""
    pb, qb, pa, qa = book.bid, book.bid_size, book.ask, book.ask_size
    e = np.zeros(pb.size)
    e[1:] = (
        (pb[1:] >= pb[:-1]) * qb[1:]
        - (pb[1:] <= pb[:-1]) * qb[:-1]
        - (pa[1:] <= pa[:-1]) * qa[1:]
        + (pa[1:] >= pa[:-1]) * qa[:-1]
    )
    return e


@dataclass
class TapeSamples:
    """Raw samples that the distributional distances compare."""

    returns: dict[float, np.ndarray]
    spread: np.ndarray
    depth: np.ndarray
    trade_gaps: np.ndarray
    trade_sizes: np.ndarray
    response: np.ndarray


@dataclass
class StylisedFacts:
    kurtosis: dict[float, float]
    hill: dict[float, float]
    acf_returns: dict[float, list[float]]
    acf_abs_returns: dict[float, list[float]]
    spread_mean: float
    depth_mean: float
    ofi_beta: float
    ofi_r2: float
    response_lags_s: list[float]
    response: list[float]
    n_trades: int
    samples: TapeSamples = field(repr=False)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for dt in self.kurtosis:
            out[f"kurtosis_{dt:g}s"] = self.kurtosis[dt]
            out[f"hill_{dt:g}s"] = self.hill[dt]
            out[f"acf1_{dt:g}s"] = self.acf_returns[dt][0]
            out[f"abs_acf_mean_{dt:g}s"] = float(np.mean(self.acf_abs_returns[dt]))
        out.update(
            spread_mean_ticks=self.spread_mean,
            depth_mean=self.depth_mean,
            ofi_beta=self.ofi_beta,
            ofi_r2=self.ofi_r2,
            n_trades=float(self.n_trades),
        )
        return out


def stylised_facts(tape: Tape, tick_size: float, cfg: RealismSettings) -> StylisedFacts:
    book = book_series(tape, tick_size, cfg.depth_levels)
    trades = trade_series(tape)
    start, end = int(book.ts[0]), int(book.ts[-1])
    kurt, hill, acf_r, acf_a, rets = {}, {}, {}, {}, {}
    for dt in cfg.sample_intervals_s:
        grid = np.arange(start, end, round(dt * NS_PER_S), dtype=np.int64)
        mid = book.sample(grid).mid
        r = np.diff(np.log(mid)) if mid.size > 1 else np.zeros(0)
        rets[dt] = r
        kurt[dt] = float(stats.kurtosis(r)) if r.size > 3 and r.std() > 0 else float("nan")
        hill[dt] = hill_index(r, cfg.hill_tail_fraction) if r.size else float("nan")
        acf_r[dt] = acf(r, cfg.acf_lags).tolist()
        acf_a[dt] = acf(np.abs(r), cfg.acf_lags).tolist()
    one_s = book.sample(np.arange(start, end, NS_PER_S, dtype=np.int64))
    spread = one_s.spread
    depth = (one_s.bid_depth + one_s.ask_depth) / 2.0

    step = round(cfg.ofi_interval_s * NS_PER_S)
    e = ofi_series(book)
    bins = (book.ts - start) // step
    n_bins = int(bins.max()) + 1
    ofi = np.bincount(bins, weights=e, minlength=n_bins)
    edges = start + np.arange(n_bins + 1, dtype=np.int64) * step
    mids = book.sample(edges).mid
    if mids.size == n_bins + 1 and n_bins > 2 and ofi.std() > 0:
        dmid = np.diff(mids)
        slope, _, r_value, _, _ = stats.linregress(ofi, dmid)
        ofi_beta, ofi_r2 = float(slope), float(r_value**2)
    else:
        ofi_beta, ofi_r2 = float("nan"), float("nan")

    response = []
    if trades.ts.size:
        valid = trades.ts >= book.ts[0]
        ts_v, sign = trades.ts[valid], trades.sign[valid]
        m0 = book.sample(ts_v).mid
        for lag in cfg.response_lags_s:
            m1 = book.sample(ts_v + round(lag * NS_PER_S)).mid
            response.append(float(np.mean(sign * (m1 - m0))) if m1.size else float("nan"))
        gaps = np.diff(trades.ts) / NS_PER_S
    else:
        response = [float("nan")] * len(cfg.response_lags_s)
        gaps = np.zeros(0)
    samples = TapeSamples(rets, spread, depth, gaps[gaps > 0], trades.size, np.asarray(response))
    return StylisedFacts(
        kurt,
        hill,
        acf_r,
        acf_a,
        float(spread.mean()),
        float(depth.mean()),
        ofi_beta,
        ofi_r2,
        list(cfg.response_lags_s),
        response,
        int(trades.ts.size),
        samples,
    )
