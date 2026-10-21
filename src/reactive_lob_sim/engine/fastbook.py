"""Numba batch matching kernel for exogenous order streams.

The object book in :mod:`reactive_lob_sim.engine.book` serves the interactive simulators, where every event
depends on the current state and on Python agents. Many workloads, however, push a pre-generated stream of
limit, market and cancel events through the book (stress replays, benchmark sweeps, throughput tests). For
those, this kernel stores the book in flat arrays: a fixed tick grid with per-level FIFO queues kept as doubly
linked lists over an order pool. The matching rules are the same price-time priority rules as the object
book, and :func:`run_events_reference` runs the identical stream through :class:`OrderBook`, so the two
implementations can be checked against each other and timed against each other.

Event encoding (one row per event): ``kind`` 0 = limit (GTC), 1 = market, 2 = cancel; ``side`` 0 = buy,
1 = sell; ``tick`` the limit price on the grid; ``qty`` the size; ``ref`` the event index of the limit order
to
cancel. The order id of a limit order is its event index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from reactive_lob_sim.engine.book import OrderBook
from reactive_lob_sim.engine.types import OrderRequest, OrderType, Side

EV_LIMIT, EV_MARKET, EV_CANCEL = 0, 1, 2


@dataclass(frozen=True)
class EventBatch:
    kind: np.ndarray
    side: np.ndarray
    tick: np.ndarray
    qty: np.ndarray
    ref: np.ndarray

    def __len__(self) -> int:
        return int(self.kind.shape[0])


@dataclass(frozen=True)
class BatchResult:
    """Per-event filled quantity and the final aggregated levels (bid and ask sizes on the grid)."""

    filled: np.ndarray
    bid_qty: np.ndarray
    ask_qty: np.ndarray


@njit(cache=True)
def _run(kind, side, tick, qty, ref, n_ticks, eps):  # type: ignore[no-untyped-def]  # pragma: no cover
    n = kind.shape[0]
    o_qty = np.zeros(n, dtype=np.float64)
    o_side = np.zeros(n, dtype=np.int64)
    o_tick = np.zeros(n, dtype=np.int64)
    o_next = np.full(n, -1, dtype=np.int64)
    o_prev = np.full(n, -1, dtype=np.int64)
    o_alive = np.zeros(n, dtype=np.bool_)
    head = np.full((2, n_ticks), -1, dtype=np.int64)
    tail = np.full((2, n_ticks), -1, dtype=np.int64)
    lvl = np.zeros((2, n_ticks), dtype=np.float64)
    filled = np.zeros(n, dtype=np.float64)
    best_bid = -1
    best_ask = n_ticks

    for e in range(n):
        k = kind[e]
        if k == 2:
            o = ref[e]
            if o < 0 or not o_alive[o]:
                continue
            s = o_side[o]
            t = o_tick[o]
            p, q = o_prev[o], o_next[o]
            if p >= 0:
                o_next[p] = q
            else:
                head[s, t] = q
            if q >= 0:
                o_prev[q] = p
            else:
                tail[s, t] = p
            lvl[s, t] -= o_qty[o]
            o_alive[o] = False
            if head[s, t] < 0:
                lvl[s, t] = 0.0
                if s == 0 and t == best_bid:
                    while best_bid >= 0 and head[0, best_bid] < 0:
                        best_bid -= 1
                elif s == 1 and t == best_ask:
                    while best_ask < n_ticks and head[1, best_ask] < 0:
                        best_ask += 1
            continue

        s = side[e]
        remaining = qty[e]
        limit = tick[e]
        is_market = k == 1
        opp = 1 - s
        while remaining > eps:
            if s == 0:
                if best_ask >= n_ticks or (not is_market and best_ask > limit):
                    break
                t = best_ask
            else:
                if best_bid < 0 or (not is_market and best_bid < limit):
                    break
                t = best_bid
            o = head[opp, t]
            while o >= 0 and remaining > eps:
                take = min(remaining, o_qty[o])
                remaining -= take
                o_qty[o] -= take
                lvl[opp, t] -= take
                filled[e] += take
                if o_qty[o] <= eps:
                    o_alive[o] = False
                    nxt = o_next[o]
                    head[opp, t] = nxt
                    if nxt >= 0:
                        o_prev[nxt] = -1
                    else:
                        tail[opp, t] = -1
                    o = nxt
            if head[opp, t] < 0:
                lvl[opp, t] = 0.0
                if opp == 1:
                    while best_ask < n_ticks and head[1, best_ask] < 0:
                        best_ask += 1
                else:
                    while best_bid >= 0 and head[0, best_bid] < 0:
                        best_bid -= 1
        if is_market or remaining <= eps:
            continue
        o_qty[e] = remaining
        o_side[e] = s
        o_tick[e] = limit
        o_alive[e] = True
        last = tail[s, limit]
        o_prev[e] = last
        if last >= 0:
            o_next[last] = e
        else:
            head[s, limit] = e
        tail[s, limit] = e
        lvl[s, limit] += remaining
        if s == 0 and limit > best_bid:
            best_bid = limit
        elif s == 1 and limit < best_ask:
            best_ask = limit
    return filled, lvl[0].copy(), lvl[1].copy()


def run_events(batch: EventBatch, n_ticks: int, qty_tolerance: float) -> BatchResult:
    """Run a stream through the Numba kernel. Ticks must lie in ``[0, n_ticks)``."""
    if len(batch) and (batch.tick.min() < 0 or batch.tick.max() >= n_ticks):
        raise ValueError("event ticks outside the price grid")
    filled, bid, ask = _run(
        batch.kind.astype(np.int64),
        batch.side.astype(np.int64),
        batch.tick.astype(np.int64),
        batch.qty.astype(np.float64),
        batch.ref.astype(np.int64),
        n_ticks,
        qty_tolerance,
    )
    return BatchResult(filled, bid, ask)


def run_events_reference(batch: EventBatch, n_ticks: int, qty_tolerance: float) -> BatchResult:
    """Run the same stream through the pure-Python object book."""
    book = OrderBook(qty_tolerance)
    ids: dict[int, int] = {}
    filled = np.zeros(len(batch))
    kinds, sides = batch.kind.tolist(), batch.side.tolist()
    ticks, qtys, refs = batch.tick.tolist(), batch.qty.tolist(), batch.ref.tolist()
    for e in range(len(batch)):
        if kinds[e] == EV_CANCEL:
            oid = ids.get(refs[e])
            if oid is not None:
                book.cancel(oid)
            continue
        side = Side.BUY if sides[e] == 0 else Side.SELL
        if kinds[e] == EV_MARKET:
            if book.best(side.opposite) is None:
                continue
            res = book.submit(OrderRequest(0, side, OrderType.MARKET, qtys[e]))
        else:
            res = book.submit(OrderRequest(0, side, OrderType.LIMIT, qtys[e], ticks[e]))
            ids[e] = res.order_id
        filled[e] = res.filled_qty
    bid = np.zeros(n_ticks)
    ask = np.zeros(n_ticks)
    for t, q in book.levels(Side.BUY):
        bid[t] = q
    for t, q in book.levels(Side.SELL):
        ask[t] = q
    return BatchResult(filled, bid, ask)


def zero_intelligence_stream(
    n_events: int,
    n_ticks: int,
    p_limit: float,
    p_market: float,
    offset_geom_p: float,
    size_log_mean: float,
    size_log_sd: float,
    lot_size: float,
    seed: int,
) -> EventBatch:
    """Synthetic zero-intelligence stream around the grid centre (used for benchmarks and equivalence tests).

    Limit prices are ``centre -/+ (Geometric(p) - 1)`` ticks from the centre on the bid/ask side; cancels
    pick a uniformly random earlier limit event (which may already be gone, as in a real feed).
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n_events)
    kind = np.where(u < p_limit, EV_LIMIT, np.where(u < p_limit + p_market, EV_MARKET, EV_CANCEL))
    side = rng.integers(0, 2, n_events)
    centre = n_ticks // 2
    offset = rng.geometric(offset_geom_p, n_events)
    # Offsets start at zero on both sides, so the two sides can meet at the centre and limits can cross.
    tick = np.clip(np.where(side == 0, centre - offset + 1, centre + offset - 1), 0, n_ticks - 1)
    qty = np.maximum(np.round(rng.lognormal(size_log_mean, size_log_sd, n_events) / lot_size), 1.0) * lot_size
    limit_idx = np.flatnonzero(kind == EV_LIMIT)
    ref = np.full(n_events, -1, dtype=np.int64)
    cancels = np.flatnonzero(kind == EV_CANCEL)
    n_earlier = np.searchsorted(limit_idx, cancels)
    has_earlier = n_earlier > 0
    pick = (rng.random(cancels.size) * np.maximum(n_earlier, 1)).astype(np.int64)
    ref[cancels[has_earlier]] = limit_idx[pick[has_earlier]]
    return EventBatch(kind.astype(np.int64), side.astype(np.int64), tick.astype(np.int64), qty, ref)
