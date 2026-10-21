"""Sim-to-real gap score.

Each simulator predicts, for every (strategy, episode) unit, the outcome metrics that the live market later
produced; with several seeds the prediction is the seed mean. Per metric we compute

rank agreement rho: Spearman rank correlation between predicted and live values across units;
calibrated error NMAE: mean |pred - live| divided by the mean |live - mean(live)|, i.e. the error relative
to a predictor that always says "the average live outcome" (NMAE < 1 beats that constant);
coverage: the share of units whose live value lies inside the seed range of the prediction (only with two
or more seeds).

The gap score of a simulator is G = mean over metrics of [w_rank (1 - rho) / 2 + w_error NMAE]; lower is
better and a perfect simulator has G = 0. Confidence intervals come from a nonparametric bootstrap over
units; the same resampled units are used for every simulator (paired bootstrap), which also gives the
probability that each simulator ranks first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

Unit = tuple[str, str]
"""(strategy_id, episode_id)."""


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class MetricScore:
    rho: Interval
    nmae: Interval
    coverage: float
    n_units: int


@dataclass(frozen=True)
class SimulatorScore:
    simulator: str
    gap: Interval
    metrics: dict[str, MetricScore]
    p_best: float
    rank: int


def spearman(pred: np.ndarray, live: np.ndarray) -> float:
    if pred.size < 3 or np.ptp(pred) == 0 or np.ptp(live) == 0:
        return 0.0
    return float(stats.spearmanr(pred, live).statistic)


def nmae(pred: np.ndarray, live: np.ndarray) -> float:
    denom = float(np.mean(np.abs(live - live.mean())))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - live)) / denom)


def _gap(
    pred: dict[str, np.ndarray], live: dict[str, np.ndarray], idx: np.ndarray, w_rank: float, w_err: float
) -> float:
    parts = []
    for m in live:
        p, y = pred[m][idx], live[m][idx]
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        e = nmae(p[ok], y[ok])
        if not np.isfinite(e):
            continue
        parts.append(w_rank * (1.0 - spearman(p[ok], y[ok])) / 2.0 + w_err * e)
    return float(np.mean(parts)) if parts else float("nan")


def gap_scores(
    predictions: Mapping[str, Mapping[Unit, Sequence[Mapping[str, float]]]],
    live: Mapping[Unit, Mapping[str, float]],
    metrics: Sequence[str],
    bootstrap_samples: int,
    ci_level: float,
    w_rank: float,
    w_error: float,
    seed: int,
) -> list[SimulatorScore]:
    """Score and rank simulators. ``predictions[sim][unit]`` is a list of per-seed metric dicts."""
    units = sorted(live)
    if len(units) < 3:
        raise ValueError("the gap score needs at least three (strategy, episode) units")
    y = {m: np.asarray([live[u][m] for u in units], dtype=np.float64) for m in metrics}
    pred_mean: dict[str, dict[str, np.ndarray]] = {}
    pred_lo: dict[str, dict[str, np.ndarray]] = {}
    pred_hi: dict[str, dict[str, np.ndarray]] = {}
    n_seeds: dict[str, int] = {}
    for sim, by_unit in predictions.items():
        missing = [u for u in units if u not in by_unit or not by_unit[u]]
        if missing:
            raise ValueError(f"simulator {sim} has no prediction for {missing[:3]}")
        n_seeds[sim] = min(len(by_unit[u]) for u in units)
        pred_mean[sim], pred_lo[sim], pred_hi[sim] = {}, {}, {}
        for m in metrics:
            vals = [np.asarray([r[m] for r in by_unit[u]], dtype=np.float64) for u in units]
            pred_mean[sim][m] = np.asarray([np.nanmean(v) if np.isfinite(v).any() else np.nan for v in vals])
            pred_lo[sim][m] = np.asarray([np.nanmin(v) if np.isfinite(v).any() else np.nan for v in vals])
            pred_hi[sim][m] = np.asarray([np.nanmax(v) if np.isfinite(v).any() else np.nan for v in vals])

    rng = np.random.default_rng(seed)
    n = len(units)
    full = np.arange(n)
    boot_idx = [rng.integers(0, n, n) for _ in range(bootstrap_samples)]
    alpha = (1.0 - ci_level) / 2.0
    sims = list(predictions)
    boot_gap = np.asarray([[_gap(pred_mean[s], y, idx, w_rank, w_error) for s in sims] for idx in boot_idx])

    def interval(point: float, samples: np.ndarray) -> Interval:
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            return Interval(point, float("nan"), float("nan"))
        return Interval(point, float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha)))

    filled = np.where(np.isfinite(boot_gap), boot_gap, np.inf)
    best = np.argmin(filled, axis=1)
    out = []
    for j, sim in enumerate(sims):
        mscores = {}
        for m in metrics:
            p, t = pred_mean[sim][m], y[m]
            ok = np.isfinite(p) & np.isfinite(t)
            rho_b, err_b = [], []
            for idx in boot_idx:
                pb, tb = p[idx], t[idx]
                okb = np.isfinite(pb) & np.isfinite(tb)
                if okb.sum() >= 3:
                    rho_b.append(spearman(pb[okb], tb[okb]))
                    err_b.append(nmae(pb[okb], tb[okb]))
            rho = spearman(p[ok], t[ok]) if ok.sum() >= 3 else float("nan")
            err = nmae(p[ok], t[ok]) if ok.sum() >= 3 else float("nan")
            if n_seeds[sim] >= 2:
                lo, hi = pred_lo[sim][m], pred_hi[sim][m]
                okc = ok & np.isfinite(lo) & np.isfinite(hi)
                cov = float(np.mean((t[okc] >= lo[okc]) & (t[okc] <= hi[okc]))) if okc.any() else float("nan")
            else:
                cov = float("nan")
            mscores[m] = MetricScore(
                interval(rho, np.asarray(rho_b)), interval(err, np.asarray(err_b)), cov, int(ok.sum())
            )
        point = _gap(pred_mean[sim], y, full, w_rank, w_error)
        out.append(
            SimulatorScore(sim, interval(point, boot_gap[:, j]), mscores, float(np.mean(best == j)), 0)
        )
    order = sorted(
        range(len(out)), key=lambda k: np.inf if not np.isfinite(out[k].gap.value) else out[k].gap.value
    )
    ranked = []
    for r, k in enumerate(order, start=1):
        s = out[k]
        ranked.append(SimulatorScore(s.simulator, s.gap, s.metrics, s.p_best, r))
    return ranked
