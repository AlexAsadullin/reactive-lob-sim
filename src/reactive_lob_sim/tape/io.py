"""Parquet I/O in the I1 hive layout (contract 6.2).

``<root>/venue=<venue>/channel=<channel>/symbol=<symbol>/date=<YYYY-MM-DD>/part-<n>.parquet``
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from reactive_lob_sim.tape.schema import CHANNELS, conform, empty_table
from reactive_lob_sim.tape.tape import Tape
from reactive_lob_sim.timeutil import NS_PER_S, from_ns

_NS_PER_DAY = 86_400 * NS_PER_S


def partition_dir(root: Path, venue: str, channel: str, symbol: str, day: date) -> Path:
    return root / f"venue={venue}" / f"channel={channel}" / f"symbol={symbol}" / f"date={day.isoformat()}"


def _next_part(directory: Path) -> Path:
    existing = sorted(directory.glob("part-*.parquet"))
    numbers = [int(p.stem.split("-", 1)[1]) for p in existing if p.stem.split("-", 1)[1].isdigit()]
    return directory / f"part-{max(numbers, default=-1) + 1}.parquet"


def write_tape(tape: Tape, root: Path) -> list[Path]:
    """Write every non-empty channel, split by UTC date, as new ``part-<n>`` files."""
    written: list[Path] = []
    for channel in CHANNELS:
        table = tape[channel]
        if table.num_rows == 0:
            continue
        ts = tape.ts(channel)
        days = ts // _NS_PER_DAY
        for day_index in np.unique(days):
            part = table.filter(pa.array(days == day_index))
            day = from_ns(int(day_index) * _NS_PER_DAY).date()
            directory = partition_dir(root, tape.venue, channel, tape.symbol, day)
            directory.mkdir(parents=True, exist_ok=True)
            path = _next_part(directory)
            pq.write_table(part, path)
            written.append(path)
    return written


def _days(start_ns: int, end_ns: int) -> list[date]:
    first, last = from_ns(start_ns).date(), from_ns(max(end_ns - 1, start_ns)).date()
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def read_tape(
    root: Path,
    venue: str,
    symbol: str,
    start_ns: int | None = None,
    end_ns: int | None = None,
    channels: tuple[str, ...] = CHANNELS,
) -> Tape:
    """Read a symbol's tape. Without a window, every date partition is read."""
    tables: dict[str, pa.Table] = {}
    for channel in channels:
        base = root / f"venue={venue}" / f"channel={channel}" / f"symbol={symbol}"
        if start_ns is not None and end_ns is not None:
            dirs = [base / f"date={d.isoformat()}" for d in _days(start_ns, end_ns)]
        else:
            dirs = sorted(base.glob("date=*"))
        parts = [
            conform(channel, pq.read_table(p))
            for d in dirs
            if d.is_dir()
            for p in sorted(d.glob("*.parquet"))
        ]
        tables[channel] = pa.concat_tables(parts) if parts else empty_table(channel)
    tape = Tape.from_tables(venue, symbol, tables)
    if start_ns is not None and end_ns is not None:
        tape = tape.slice(start_ns, end_ns)
    return tape
