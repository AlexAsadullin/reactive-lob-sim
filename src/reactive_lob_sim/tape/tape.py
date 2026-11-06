"""In-memory tape (one table per I1 channel) and a recorder that simulators write to."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from reactive_lob_sim.engine.types import Side
from reactive_lob_sim.tape.schema import (
    BOOK_DELTA,
    BOOK_SNAPSHOT,
    CHANNELS,
    FUNDING,
    LIQUIDATIONS,
    OPEN_INTEREST,
    SCHEMAS,
    TRADES,
    TRADING_STATUS,
    conform,
    empty_table,
)
from reactive_lob_sim.timeutil import seconds_to_ns

_PRICE_DECIMALS = 12


def _ts_column(table: pa.Table) -> np.ndarray:
    """Event time: ``ts_exchange`` where present, else ``ts_local``."""
    if table.num_rows == 0:
        return np.empty(0, dtype=np.int64)
    ts = pc.coalesce(table.column("ts_exchange"), table.column("ts_local"))
    return np.asarray(ts.to_numpy(zero_copy_only=False), dtype=np.int64)


@dataclass(frozen=True)
class Tape:
    """One symbol's tape. Tables follow the contract 6.2 schemas exactly."""

    venue: str
    symbol: str
    tables: dict[str, pa.Table]

    @classmethod
    def from_tables(cls, venue: str, symbol: str, tables: dict[str, pa.Table]) -> Tape:
        full = {ch: conform(ch, tables[ch]) if ch in tables else empty_table(ch) for ch in CHANNELS}
        return cls(venue, symbol, full)

    def __getitem__(self, channel: str) -> pa.Table:
        return self.tables[channel]

    def ts(self, channel: str) -> np.ndarray:
        return _ts_column(self.tables[channel])

    def time_range(self) -> tuple[int, int]:
        stamps = [self.ts(ch) for ch in CHANNELS if self.tables[ch].num_rows]
        if not stamps:
            raise ValueError("empty tape")
        joined = np.concatenate(stamps)
        return int(joined.min()), int(joined.max())

    def slice(self, start_ns: int, end_ns: int) -> Tape:
        """Rows with ``start_ns <= ts < end_ns``. Snapshots are kept only if inside the window."""
        out: dict[str, pa.Table] = {}
        for ch, table in self.tables.items():
            ts = self.ts(ch)
            mask = (ts >= start_ns) & (ts < end_ns)
            out[ch] = table.filter(pa.array(mask)) if table.num_rows else table
        return Tape(self.venue, self.symbol, out)


@dataclass
class _Columns:
    names: tuple[str, ...]
    data: dict[str, list[object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = {n: [] for n in self.names}

    def append(self, row: Iterable[object]) -> None:
        for name, value in zip(self.names, row, strict=True):
            self.data[name].append(value)


class TapeRecorder:
    """Collects simulator output as I1 tape rows.

    Simulated time ``t_s`` (seconds since the run start) maps to ``epoch_ns + t_s * 1e9``; both
    ``ts_exchange`` and ``ts_local`` carry that stamp (a simulator has no network latency). ``seq`` is one
    counter across channels, so the original event order is recoverable.
    """

    def __init__(
        self,
        venue: str,
        symbol: str,
        epoch_ns: int,
        tick_size: float,
        capture_region: str,
        connection_id: str,
    ) -> None:
        self.venue = venue
        self.symbol = symbol
        self._epoch_ns = epoch_ns
        self._tick = tick_size
        self._region = capture_region
        self._conn = connection_id
        self._seq = 0
        self._trade_id = 0
        self._snapshot_id = 0
        self._last_delta_seq: dict[tuple[str, int], int] = {}
        self.trading_session: str | None = None
        self._cols = {ch: _Columns(tuple(SCHEMAS[ch].names)) for ch in CHANNELS}

    def _common(self, t_s: float) -> tuple[object, ...]:
        self._seq += 1
        ts = self._epoch_ns + seconds_to_ns(t_s)
        return (self.venue, self.symbol, ts, ts, self._region, self._conn, self._seq)

    def ts_ns(self, t_s: float) -> int:
        return self._epoch_ns + seconds_to_ns(t_s)

    def price(self, tick: int) -> float:
        # Rounding strips float residue such as 100.01000000000001 from tick * tick_size.
        return round(tick * self._tick, _PRICE_DECIMALS)

    def trade(self, t_s: float, price_tick: int, size: float, aggressor: Side) -> None:
        self._trade_id += 1
        common = self._common(t_s)
        self._cols[TRADES].append(
            (
                *common,
                f"{self._conn}-{self._trade_id}",
                self.price(price_tick),
                size,
                aggressor.value,
                self.trading_session,
            )
        )

    def trading_status(self, t_s: float, status: str, limit_allowed: bool, market_allowed: bool) -> None:
        common = self._common(t_s)
        self._cols[TRADING_STATUS].append((*common, status, limit_allowed, market_allowed))

    def book_delta(self, t_s: float, side: Side, price_tick: int, size: float) -> None:
        common = self._common(t_s)
        key = (side.book_side, price_tick)
        prev = self._last_delta_seq.get(key)
        self._last_delta_seq[key] = self._seq
        self._cols[BOOK_DELTA].append((*common, side.book_side, self.price(price_tick), max(size, 0.0), prev))

    def snapshot(self, t_s: float, bids: list[tuple[int, float]], asks: list[tuple[int, float]]) -> None:
        self._snapshot_id += 1
        snap = f"{self._conn}-snap-{self._snapshot_id}"
        for side, levels in ((Side.BUY, bids), (Side.SELL, asks)):
            for level, (tick, qty) in enumerate(levels):
                common = self._common(t_s)
                self._cols[BOOK_SNAPSHOT].append(
                    (*common, side.book_side, self.price(tick), qty, level, snap)
                )

    def funding(self, t_s: float, rate: float, mark: float, index: float, next_funding_s: float) -> None:
        common = self._common(t_s)
        self._cols[FUNDING].append((*common, rate, mark, index, self.ts_ns(next_funding_s)))

    def liquidation(self, t_s: float, side: Side, price: float, size: float) -> None:
        common = self._common(t_s)
        self._cols[LIQUIDATIONS].append((*common, side.value, price, size, None))

    def open_interest(self, t_s: float, oi: float, price: float) -> None:
        common = self._common(t_s)
        self._cols[OPEN_INTEREST].append((*common, oi, oi * price))

    def to_tape(self) -> Tape:
        tables = {ch: pa.Table.from_pydict(cols.data, schema=SCHEMAS[ch]) for ch, cols in self._cols.items()}
        return Tape(self.venue, self.symbol, tables)


__all__ = [
    "BOOK_DELTA",
    "BOOK_SNAPSHOT",
    "FUNDING",
    "LIQUIDATIONS",
    "OPEN_INTEREST",
    "TRADES",
    "TRADING_STATUS",
    "Tape",
    "TapeRecorder",
]
