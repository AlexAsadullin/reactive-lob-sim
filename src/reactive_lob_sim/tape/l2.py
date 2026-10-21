"""L2 book reconstruction from the tape and derived top-of-book series."""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from reactive_lob_sim.tape.schema import BOOK_DELTA, BOOK_SNAPSHOT, FUNDING, OPEN_INTEREST, TRADES
from reactive_lob_sim.tape.tape import Tape

# Merge order of channels sharing a timestamp: a snapshot resets the book first, a trade precedes the book
# deltas it caused, and slow-moving channels come last.
KIND_SNAPSHOT, KIND_TRADE, KIND_DELTA, KIND_FUNDING, KIND_OI = range(5)


class L2Book:
    """Aggregated price levels keyed by integer tick."""

    def __init__(self) -> None:
        self._size: dict[bool, dict[int, float]] = {True: {}, False: {}}
        self._prices: dict[bool, list[int]] = {True: [], False: []}

    def clear(self) -> None:
        for is_bid in (True, False):
            self._size[is_bid].clear()
            self._prices[is_bid].clear()

    def set(self, is_bid: bool, tick: int, size: float) -> float:
        """Set a level's size (0 deletes it) and return the previous size."""
        levels = self._size[is_bid]
        old = levels.get(tick, 0.0)
        if size > 0:
            if tick not in levels:
                insort(self._prices[is_bid], tick)
            levels[tick] = size
        elif tick in levels:
            del levels[tick]
            prices = self._prices[is_bid]
            del prices[bisect_left(prices, tick)]
        return old

    def size(self, is_bid: bool, tick: int) -> float:
        return self._size[is_bid].get(tick, 0.0)

    def best(self, is_bid: bool) -> int | None:
        prices = self._prices[is_bid]
        if not prices:
            return None
        return prices[-1] if is_bid else prices[0]

    def levels(self, is_bid: bool, n: int | None = None) -> list[tuple[int, float]]:
        prices = self._prices[is_bid]
        ordered = prices[::-1] if is_bid else prices
        chosen = ordered if n is None else ordered[:n]
        return [(t, self._size[is_bid][t]) for t in chosen]

    def depth(self, is_bid: bool, n: int) -> float:
        return sum(q for _, q in self.levels(is_bid, n))

    def mid_tick(self) -> float | None:
        bid, ask = self.best(True), self.best(False)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0


@dataclass(frozen=True)
class MergedEvents:
    """All book-relevant tape rows in event order, as parallel arrays."""

    ts: np.ndarray
    kind: np.ndarray
    is_bid: np.ndarray
    tick: np.ndarray
    size: np.ndarray
    value: np.ndarray
    group: np.ndarray

    def __len__(self) -> int:
        return int(self.ts.shape[0])


def _col(table: pa.Table, name: str, dtype: type, fill: float = 0.0) -> np.ndarray:
    if table.num_rows == 0:
        return np.empty(0, dtype=dtype)
    arr = table.column(name).to_numpy(zero_copy_only=False)
    out = np.asarray(arr, dtype=np.float64 if dtype is float else dtype)
    if out.dtype.kind == "f":
        out = np.where(np.isnan(out), fill, out)
    return out


def merge_events(tape: Tape, tick_size: float) -> MergedEvents:
    """Merge snapshot, trade, delta, funding and open-interest rows into one ordered stream.

    ``is_bid`` is the book side for snapshots and deltas and ``aggressor == buy`` for trades. ``value`` is the
    funding rate or the open interest. ``group`` is an integer per snapshot id (else -1).
    """
    parts: list[tuple[np.ndarray, ...]] = []

    def add(channel: str, kind: int, side_col: str | None, bid_label: str, value_col: str | None) -> None:
        table = tape[channel]
        n = table.num_rows
        if n == 0:
            return
        ts = tape.ts(channel)
        seq = _col(table, "seq", float, -1.0)
        if side_col is None:
            is_bid = np.zeros(n, dtype=bool)
        else:
            is_bid = np.asarray(
                table.column(side_col).to_numpy(zero_copy_only=False) == bid_label, dtype=bool
            )
        if "price" in table.column_names:
            tick = np.rint(_col(table, "price", float) / tick_size).astype(np.int64)
            size = _col(table, "size", float)
        else:
            tick = np.zeros(n, dtype=np.int64)
            size = np.zeros(n, dtype=np.float64)
        value = _col(table, value_col, float) if value_col else np.zeros(n)
        if channel == BOOK_SNAPSHOT:
            ids = table.column("snapshot_id").to_numpy(zero_copy_only=False)
            _, group = np.unique(ids, return_inverse=True)
        else:
            group = np.full(n, -1, dtype=np.int64)
        parts.append((ts, np.full(n, kind), seq, is_bid, tick, size, value, group.astype(np.int64)))

    add(BOOK_SNAPSHOT, KIND_SNAPSHOT, "side", "bid", None)
    add(TRADES, KIND_TRADE, "side", "buy", None)
    add(BOOK_DELTA, KIND_DELTA, "side", "bid", None)
    add(FUNDING, KIND_FUNDING, None, "", "rate")
    add(OPEN_INTEREST, KIND_OI, None, "", "open_interest")
    if not parts:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return MergedEvents(empty_i, empty_i, np.empty(0, dtype=bool), empty_i, empty_f, empty_f, empty_i)
    cols = [np.concatenate([p[i] for p in parts]) for i in range(8)]
    ts, kind, seq, is_bid, tick, size, value, group = cols
    order = np.lexsort((seq, kind, ts))
    return MergedEvents(
        ts[order].astype(np.int64),
        kind[order].astype(np.int64),
        is_bid[order].astype(bool),
        tick[order].astype(np.int64),
        size[order].astype(np.float64),
        value[order].astype(np.float64),
        group[order].astype(np.int64),
    )


@dataclass(frozen=True)
class BookSeries:
    """Top-of-book state after every book update (ticks and sizes)."""

    ts: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    bid_depth: np.ndarray
    ask_depth: np.ndarray

    @property
    def mid(self) -> np.ndarray:
        return np.asarray((self.bid + self.ask) / 2.0, dtype=np.float64)

    @property
    def spread(self) -> np.ndarray:
        return np.asarray(self.ask - self.bid, dtype=np.float64)

    def sample(self, grid_ns: np.ndarray) -> BookSeries:
        """Last observation at or before each grid time (grid times before the first update are dropped)."""
        idx = np.searchsorted(self.ts, grid_ns, side="right") - 1
        keep = idx >= 0
        idx = idx[keep]
        return BookSeries(
            grid_ns[keep],
            self.bid[idx],
            self.ask[idx],
            self.bid_size[idx],
            self.ask_size[idx],
            self.bid_depth[idx],
            self.ask_depth[idx],
        )


def book_series(tape: Tape, tick_size: float, depth_levels: int) -> BookSeries:
    """Replay snapshots and deltas through an :class:`L2Book` and record the two-sided top of book."""
    events = merge_events(tape, tick_size)
    book = L2Book()
    rows: list[tuple[int, int, int, float, float, float, float]] = []
    current_group = -1
    n = len(events)
    ts_l, kind_l, bid_l = events.ts.tolist(), events.kind.tolist(), events.is_bid.tolist()
    tick_l, size_l, group_l = events.tick.tolist(), events.size.tolist(), events.group.tolist()
    for i in range(n):
        kind = kind_l[i]
        if kind == KIND_SNAPSHOT:
            if group_l[i] != current_group:
                book.clear()
                current_group = group_l[i]
            book.set(bid_l[i], tick_l[i], size_l[i])
        elif kind == KIND_DELTA:
            book.set(bid_l[i], tick_l[i], size_l[i])
        else:
            continue
        last_of_ts = i + 1 >= n or ts_l[i + 1] != ts_l[i] or kind_l[i + 1] not in (KIND_SNAPSHOT, KIND_DELTA)
        if not last_of_ts:
            continue
        bid, ask = book.best(True), book.best(False)
        if bid is None or ask is None or bid >= ask:
            continue
        rows.append(
            (
                ts_l[i],
                bid,
                ask,
                book.size(True, bid),
                book.size(False, ask),
                book.depth(True, depth_levels),
                book.depth(False, depth_levels),
            )
        )
    if not rows:
        raise ValueError("tape has no two-sided book")
    arr = np.asarray(rows, dtype=np.float64)
    return BookSeries(
        arr[:, 0].astype(np.int64),
        arr[:, 1],
        arr[:, 2],
        arr[:, 3],
        arr[:, 4],
        arr[:, 5],
        arr[:, 6],
    )


@dataclass(frozen=True)
class TradeSeries:
    ts: np.ndarray
    price: np.ndarray
    size: np.ndarray
    sign: np.ndarray


def trade_series(tape: Tape) -> TradeSeries:
    table = tape[TRADES]
    ts = tape.ts(TRADES)
    order = np.argsort(ts, kind="stable")
    price = _col(table, "price", float)
    size = _col(table, "size", float)
    if table.num_rows:
        sign = np.where(table.column("side").to_numpy(zero_copy_only=False) == "buy", 1.0, -1.0)
    else:
        sign = np.empty(0)
    return TradeSeries(ts[order], price[order], size[order], sign[order])


def channel_values(tape: Tape, channel: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered ``(ts, value)`` of a scalar channel such as funding rate or open interest."""
    if channel not in (FUNDING, OPEN_INTEREST):
        raise ValueError(f"not a scalar channel: {channel}")
    ts = tape.ts(channel)
    values = _col(tape[channel], column, float)
    order = np.argsort(ts, kind="stable")
    return ts[order], values[order]
