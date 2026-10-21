"""Tokenisation of order events and their conditioning features.

Every event becomes four categorical fields: event type (6 classes), quote offset in ticks clipped to
``[0, offset_clip_ticks]``, size bin (training quantiles) and inter-arrival time bin (training quantiles of
log time gaps). The conditioning vector holds the standardised funding rate, the standardised log change of
open interest since the previous fix, and a one-hot volatility regime (training quantiles of the trailing
realised volatility of the mid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reactive_lob_sim.calibration.events import OrderEvents
from reactive_lob_sim.contracts.pit_client import ConditioningSeries
from reactive_lob_sim.settings import GenerativeSettings
from reactive_lob_sim.timeutil import NS_PER_S


@dataclass(frozen=True)
class Tokenizer:
    offset_clip: int
    size_edges: np.ndarray
    size_values: np.ndarray
    dt_edges: np.ndarray
    vol_edges: np.ndarray
    funding_mean: float
    funding_std: float
    oi_std: float
    vol_window_s: float
    dt_floor_s: float

    @property
    def n_offsets(self) -> int:
        return self.offset_clip + 1

    @property
    def n_sizes(self) -> int:
        return int(self.size_values.shape[0])

    @property
    def n_dts(self) -> int:
        return int(self.dt_edges.shape[0]) - 1

    @property
    def n_vol(self) -> int:
        return int(self.vol_edges.shape[0]) + 1

    @property
    def cond_dim(self) -> int:
        return 2 + self.n_vol

    def encode_offsets(self, offsets: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(offsets), 0, self.offset_clip).astype(np.int64)

    def encode_sizes(self, sizes: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.size_edges, sizes, side="right"), 0, self.n_sizes - 1).astype(
            np.int64
        )

    def encode_dts(self, dts: np.ndarray) -> np.ndarray:
        logs = np.log(np.maximum(dts, self.dt_floor_s))
        return np.clip(np.searchsorted(self.dt_edges, logs, side="right") - 1, 0, self.n_dts - 1).astype(
            np.int64
        )

    def decode_dt(self, bin_index: int, u: float) -> float:
        lo, hi = self.dt_edges[bin_index], self.dt_edges[bin_index + 1]
        return float(np.exp(lo + u * (hi - lo)))

    def vol_regime(self, vol: float) -> int:
        return int(np.searchsorted(self.vol_edges, vol, side="right"))

    def cond_vector(self, funding: float, oi_change: float, vol: float) -> np.ndarray:
        out = np.zeros(self.cond_dim, dtype=np.float32)
        out[0] = (funding - self.funding_mean) / self.funding_std
        out[1] = oi_change / self.oi_std
        out[2 + self.vol_regime(vol)] = 1.0
        return out


def trailing_vol(t: np.ndarray, mid: np.ndarray, window_s: float) -> np.ndarray:
    """Realised volatility of log mid changes over the trailing window, evaluated at each event."""
    r2 = np.concatenate([[0.0], np.diff(np.log(mid)) ** 2])
    cum = np.cumsum(r2)
    start = np.searchsorted(t, t - window_s, side="left")
    return np.sqrt(np.maximum(cum - cum[start], 0.0))


def _step_values(ts: np.ndarray, values: np.ndarray, at: np.ndarray, default: float) -> np.ndarray:
    if ts.size == 0:
        return np.full(at.shape, default)
    idx = np.searchsorted(ts, at, side="right") - 1
    out = values[np.clip(idx, 0, None)]
    return np.where(idx >= 0, out, default)


def condition_features(
    events: OrderEvents, cond: ConditioningSeries, tick_size: float, window_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Funding, OI log change and trailing volatility at every event."""
    at = events.start_ns + np.round(events.t * NS_PER_S).astype(np.int64)
    funding = _step_values(
        cond.funding_ts, cond.funding, at, float(cond.funding.mean()) if cond.funding.size else 0.0
    )
    if cond.oi.size >= 2:
        log_oi = np.log(cond.oi)
        changes = np.concatenate([[0.0], np.diff(log_oi)])
        oi_change = _step_values(cond.oi_ts, changes, at, 0.0)
    else:
        oi_change = np.zeros(at.shape)
    vol = trailing_vol(events.t, events.mid * tick_size, window_s)
    return funding, oi_change, vol


def fit_tokenizer(
    events: OrderEvents, cond: ConditioningSeries, tick_size: float, cfg: GenerativeSettings
) -> Tokenizer:
    sizes = events.size
    size_q = np.quantile(sizes, np.linspace(0, 1, cfg.n_size_bins + 1))
    size_edges = np.unique(size_q[1:-1])
    which = np.searchsorted(size_edges, sizes, side="right")
    size_values = np.asarray([np.median(sizes[which == b]) for b in range(size_edges.size + 1)])
    dts = np.diff(np.concatenate([[0.0], events.t]))
    logs = np.log(np.maximum(dts, cfg.dt_floor_s))
    dt_edges = np.unique(np.quantile(logs, np.linspace(0, 1, cfg.n_dt_bins + 1)))
    if dt_edges.size < 2:
        dt_edges = np.asarray([logs.min(), logs.min() + 1.0])
    funding, oi_change, vol = condition_features(events, cond, tick_size, cfg.vol_window_s)
    vol_edges = np.unique(np.quantile(vol, np.linspace(0, 1, cfg.n_vol_regimes + 1)[1:-1]))
    f_std = float(funding.std()) or 1.0
    oi_std = float(oi_change.std()) or 1.0
    return Tokenizer(
        cfg.offset_clip_ticks,
        size_edges,
        size_values,
        dt_edges,
        vol_edges,
        float(funding.mean()),
        f_std,
        oi_std,
        cfg.vol_window_s,
        cfg.dt_floor_s,
    )


@dataclass(frozen=True)
class TokenStream:
    kind: np.ndarray
    offset: np.ndarray
    size: np.ndarray
    dt: np.ndarray
    cond: np.ndarray


def tokenize(tok: Tokenizer, events: OrderEvents, cond: ConditioningSeries, tick_size: float) -> TokenStream:
    dts = np.diff(np.concatenate([[0.0], events.t]))
    funding, oi_change, vol = condition_features(events, cond, tick_size, tok.vol_window_s)
    cvec = np.stack([tok.cond_vector(f, o, v) for f, o, v in zip(funding, oi_change, vol, strict=True)])
    return TokenStream(
        events.kind.astype(np.int64),
        tok.encode_offsets(events.offset),
        tok.encode_sizes(events.size),
        tok.encode_dts(dts),
        cvec.astype(np.float32),
    )
