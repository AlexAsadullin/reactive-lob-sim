"""LOB-Bench-style distributional distances between a real and a simulated tape.

For each scoring function (1 s and 10 s returns, absolute 1 s returns, spread, depth, trade inter-arrival
times, trade sizes) we compare the two samples with the Wasserstein-1 distance and the two-sample
Kolmogorov-Smirnov test, as in LOB-Bench (Nagy et al. 2025, section 4). The Wasserstein distance is also
reported divided by the standard deviation of the real sample, so that scores of different units can be
averaged. The market-impact response functions are compared by the root-mean-square difference across lags,
divided by the root-mean-square of the real response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from reactive_lob_sim.realism.facts import StylisedFacts


@dataclass(frozen=True)
class Distance:
    wasserstein: float
    wasserstein_norm: float
    ks_stat: float
    ks_pvalue: float
    n_real: int
    n_sim: int


def distance(real: np.ndarray, sim: np.ndarray) -> Distance:
    real = real[np.isfinite(real)]
    sim = sim[np.isfinite(sim)]
    if real.size < 2 or sim.size < 2:
        nan = float("nan")
        return Distance(nan, nan, nan, nan, int(real.size), int(sim.size))
    w = float(stats.wasserstein_distance(real, sim))
    scale = float(np.std(real))
    ks = stats.ks_2samp(real, sim)
    return Distance(
        w,
        w / scale if scale > 0 else float("nan"),
        float(ks.statistic),
        float(ks.pvalue),
        int(real.size),
        int(sim.size),
    )


@dataclass(frozen=True)
class RealismScore:
    distances: dict[str, Distance]
    response_error: float
    fact_gaps: dict[str, float]

    @property
    def mean_wasserstein_norm(self) -> float:
        vals = [d.wasserstein_norm for d in self.distances.values() if np.isfinite(d.wasserstein_norm)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def mean_ks(self) -> float:
        vals = [d.ks_stat for d in self.distances.values() if np.isfinite(d.ks_stat)]
        return float(np.mean(vals)) if vals else float("nan")


def score_realism(real: StylisedFacts, sim: StylisedFacts) -> RealismScore:
    rs, ss = real.samples, sim.samples
    d: dict[str, Distance] = {}
    for dt in rs.returns:
        if dt in ss.returns:
            d[f"returns_{dt:g}s"] = distance(rs.returns[dt], ss.returns[dt])
    first = min(rs.returns)
    if first in ss.returns:
        d[f"abs_returns_{first:g}s"] = distance(np.abs(rs.returns[first]), np.abs(ss.returns[first]))
    d["spread"] = distance(rs.spread, ss.spread)
    d["depth"] = distance(rs.depth, ss.depth)
    d["trade_interarrival"] = distance(rs.trade_gaps, ss.trade_gaps)
    d["trade_size"] = distance(rs.trade_sizes, ss.trade_sizes)
    r_real, r_sim = np.asarray(real.response), np.asarray(sim.response)
    ok = np.isfinite(r_real) & np.isfinite(r_sim)
    denom = float(np.sqrt(np.mean(r_real[ok] ** 2))) if ok.any() else 0.0
    resp_err = float(np.sqrt(np.mean((r_real[ok] - r_sim[ok]) ** 2)) / denom) if denom > 0 else float("nan")
    rsum, ssum = real.summary(), sim.summary()
    gaps = {k: ssum[k] - rsum[k] for k in rsum if k in ssum and k != "n_trades"}
    return RealismScore(d, resp_err, gaps)
