"""Order-event extraction from an L2 tape.

An L2 feed does not label orders, so events are inferred:
consecutive trades with the same timestamp and aggressor form one market order;
a level size increase is a limit order of the increase;
a level size decrease is first attributed to recorded trades at that level, and the rest is a cancellation.

Offsets are measured in ticks from the mid before the event, positive away from the mid, and follow the
quoting rule of the agent-based market: a bid at tick b has offset floor(mid - 0.5) - b.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from reactive_lob_sim.params import CXL_ASK, CXL_BID, LIM_ASK, LIM_BID, MKT_BUY, MKT_SELL
from reactive_lob_sim.tape.l2 import KIND_DELTA, KIND_SNAPSHOT, KIND_TRADE, L2Book, merge_events
from reactive_lob_sim.tape.tape import Tape
from reactive_lob_sim.timeutil import NS_PER_S


@dataclass(frozen=True)
class OrderEvents:
    t: np.ndarray
    kind: np.ndarray
    offset: np.ndarray
    size: np.ndarray
    mid: np.ndarray
    start_ns: int
    horizon: float

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def of(self, *kinds: int) -> np.ndarray:
        return np.isin(self.kind, kinds)


def extract_events(tape: Tape, tick_size: float, qty_tolerance: float) -> OrderEvents:
    ev = merge_events(tape, tick_size)
    if len(ev) == 0:
        raise ValueError("empty tape")
    start_ns = int(ev.ts[0])
    book = L2Book()
    pending: dict[tuple[bool, int], float] = {}
    rows: list[tuple[float, int, float, float, float]] = []
    ts_l, kind_l, bid_l = ev.ts.tolist(), ev.kind.tolist(), ev.is_bid.tolist()
    tick_l, size_l, group_l = ev.tick.tolist(), ev.size.tolist(), ev.group.tolist()
    snap_group = -1
    n = len(ev)
    i = 0
    while i < n:
        kind = kind_l[i]
        ts = ts_l[i]
        t = (ts - start_ns) / NS_PER_S
        mid = book.mid_tick()
        if kind == KIND_SNAPSHOT:
            if group_l[i] != snap_group:
                book.clear()
                snap_group = group_l[i]
            book.set(bid_l[i], tick_l[i], size_l[i])
            i += 1
            continue
        if kind == KIND_TRADE:
            buy = bid_l[i]
            total = 0.0
            while i < n and kind_l[i] == KIND_TRADE and ts_l[i] == ts and bid_l[i] == buy:
                key = (not buy, tick_l[i])
                pending[key] = pending.get(key, 0.0) + size_l[i]
                total += size_l[i]
                i += 1
            if mid is not None:
                rows.append((t, MKT_BUY if buy else MKT_SELL, 0.0, total, mid))
            continue
        if kind == KIND_DELTA:
            is_bid, tick, size = bid_l[i], tick_l[i], size_l[i]
            old = book.set(is_bid, tick, size)
            change = size - old
            if mid is not None and abs(change) > qty_tolerance:
                offset = float(math.floor(mid - 0.5) - tick) if is_bid else float(tick - math.ceil(mid + 0.5))
                if change > 0:
                    rows.append((t, LIM_BID if is_bid else LIM_ASK, offset, change, mid))
                else:
                    key = (is_bid, tick)
                    explained = min(-change, pending.get(key, 0.0))
                    if explained:
                        pending[key] -= explained
                    cancelled = -change - explained
                    if cancelled > qty_tolerance:
                        rows.append((t, CXL_BID if is_bid else CXL_ASK, offset, cancelled, mid))
        i += 1
    if not rows:
        raise ValueError("no order events could be inferred from the tape")
    arr = _merge_ties(np.asarray(rows, dtype=np.float64))
    horizon = (int(ev.ts[-1]) - start_ns) / NS_PER_S
    return OrderEvents(
        arr[:, 0],
        arr[:, 1].astype(np.int64),
        arr[:, 2],
        arr[:, 3],
        arr[:, 4],
        start_ns,
        max(horizon, float(arr[-1, 0])),
    )


def _merge_ties(arr: np.ndarray) -> np.ndarray:
    """Merge events of the same type at the same timestamp into one event (sizes add up).

    One engine operation can produce several same-type book changes at one timestamp (for example a refill
    of several quotes). Treated as separate events, these zero-gap ties would drive the fitted Hawkes decay
    to its upper bound.
    """
    keep = np.ones(arr.shape[0], dtype=bool)
    same = (np.diff(arr[:, 0]) == 0) & (np.diff(arr[:, 1]) == 0)
    first = 0
    for i in range(1, arr.shape[0]):
        if same[i - 1]:
            arr[first, 3] += arr[i, 3]
            keep[i] = False
        else:
            first = i
    return arr[keep]
