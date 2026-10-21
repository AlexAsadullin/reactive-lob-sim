"""Fills export of the Shadow Exchange (contract 6.4): schema, writer and reader.

The writer is used for the synthetic "live" market of ``make reproduce`` and for exporting simulator fills
in the same format, so the gap score reads live and simulated fills through one code path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reactive_lob_sim.timeutil import NS_PER_S, from_ns
from reactive_lob_sim.venue import AgentFill

FILL_SCHEMA = pa.schema(
    [
        ("fill_id", pa.string()),
        ("order_id", pa.string()),
        ("client_order_id", pa.string()),
        ("account_id", pa.string()),
        ("strategy_id", pa.string()),
        ("venue", pa.string()),
        ("symbol", pa.string()),
        ("side", pa.string()),
        ("order_type", pa.string()),
        ("ts_submit", pa.int64()),
        ("ts_fill", pa.int64()),
        ("price", pa.float64()),
        ("qty", pa.float64()),
        ("fee", pa.float64()),
        ("liquidity", pa.string()),
        ("arrival_mid", pa.float64()),
        ("distance_to_touch_bps", pa.float64()),
        ("queue_ahead_at_entry", pa.float64()),
        ("latency_ns", pa.int64()),
        ("scenario_id", pa.string()),
    ]
)


@dataclass(frozen=True)
class FillContext:
    """Identifiers that the export attaches to every fill of one strategy run."""

    account_id: str
    strategy_id: str
    venue: str
    symbol: str
    epoch_ns: int
    scenario_id: str | None = None


def fills_table(fills: Iterable[AgentFill], ctx: FillContext) -> pa.Table:
    rows: dict[str, list[object]] = {name: [] for name in FILL_SCHEMA.names}
    for k, f in enumerate(fills):
        values: dict[str, object] = {
            "fill_id": f"{ctx.account_id}-{ctx.strategy_id}-{k}",
            "order_id": str(f.order_id),
            "client_order_id": f"{ctx.strategy_id}-{f.order_id}",
            "account_id": ctx.account_id,
            "strategy_id": ctx.strategy_id,
            "venue": ctx.venue,
            "symbol": ctx.symbol,
            "side": f.side.value,
            "order_type": f.order_type.value,
            "ts_submit": ctx.epoch_ns + round(f.ts_submit_s * NS_PER_S),
            "ts_fill": ctx.epoch_ns + round(f.ts_fill_s * NS_PER_S),
            "price": f.price,
            "qty": f.qty,
            "fee": float(f.fee),
            "liquidity": f.liquidity.value,
            "arrival_mid": f.arrival_mid,
            "distance_to_touch_bps": f.distance_to_touch_bps,
            "queue_ahead_at_entry": f.queue_ahead_at_entry,
            "latency_ns": 0,
            "scenario_id": ctx.scenario_id,
        }
        for name in FILL_SCHEMA.names:
            rows[name].append(values[name])
    return pa.Table.from_pydict(rows, schema=FILL_SCHEMA)


def write_fills(table: pa.Table, root: Path, part_name: str) -> list[Path]:
    """Write ``table`` under ``root/date=<YYYY-MM-DD>/part-<name>.parquet`` (one file per fill date)."""
    if table.num_rows == 0:
        return []
    ts = table.column("ts_fill").to_pylist()
    days: dict[date, list[int]] = {}
    for i, value in enumerate(ts):
        days.setdefault(from_ns(int(value)).date(), []).append(i)
    written = []
    for day, idx in sorted(days.items()):
        directory = root / f"date={day.isoformat()}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"part-{part_name}.parquet"
        pq.write_table(table.take(pa.array(idx)), path)
        written.append(path)
    return written


def read_fills(root: Path, start: date | None = None, end: date | None = None) -> pa.Table:
    """Read all fills partitions (optionally restricted to ``start <= date <= end``)."""
    parts = []
    for directory in sorted(root.glob("date=*")):
        day = date.fromisoformat(directory.name.split("=", 1)[1])
        if (start and day < start) or (end and day > end):
            continue
        for path in sorted(directory.glob("*.parquet")):
            table = pq.read_table(path)
            missing = [n for n in FILL_SCHEMA.names if n not in table.column_names]
            if missing:
                raise ValueError(f"{path} lacks contract 6.4 columns: {missing}")
            parts.append(table.select(FILL_SCHEMA.names).cast(FILL_SCHEMA))
    return pa.concat_tables(parts) if parts else FILL_SCHEMA.empty_table()
