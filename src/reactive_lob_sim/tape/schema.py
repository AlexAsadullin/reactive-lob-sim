"""Arrow schemas of the I1 normalised tape (contract 6.2 revision 2, MOEX).

The crypto channels (funding, liquidations, open interest) stay readable, as the contract requires, but the
MOEX path never writes them.
"""

from __future__ import annotations

import pyarrow as pa

TRADES = "trades"
BOOK_DELTA = "book_delta"
BOOK_SNAPSHOT = "book_snapshot"
FUNDING = "funding"
LIQUIDATIONS = "liquidations"
OPEN_INTEREST = "open_interest"
TRADING_STATUS = "trading_status"
CHANNELS: tuple[str, ...] = (
    TRADES,
    BOOK_DELTA,
    BOOK_SNAPSHOT,
    TRADING_STATUS,
    FUNDING,
    LIQUIDATIONS,
    OPEN_INTEREST,
)

_COMMON: list[tuple[str, pa.DataType]] = [
    ("venue", pa.string()),
    ("symbol", pa.string()),
    ("ts_exchange", pa.int64()),
    ("ts_local", pa.int64()),
    ("capture_region", pa.string()),
    ("connection_id", pa.string()),
    ("seq", pa.int64()),
]

_SPECIFIC: dict[str, list[tuple[str, pa.DataType]]] = {
    TRADES: [
        ("trade_id", pa.string()),
        ("price", pa.float64()),
        ("size", pa.float64()),
        ("side", pa.string()),
        ("trading_session", pa.string()),
    ],
    TRADING_STATUS: [
        ("status", pa.string()),
        ("limit_order_allowed", pa.bool_()),
        ("market_order_allowed", pa.bool_()),
    ],
    BOOK_DELTA: [
        ("side", pa.string()),
        ("price", pa.float64()),
        ("size", pa.float64()),
        ("prev_seq", pa.int64()),
    ],
    BOOK_SNAPSHOT: [
        ("side", pa.string()),
        ("price", pa.float64()),
        ("size", pa.float64()),
        ("level", pa.int64()),
        ("snapshot_id", pa.string()),
    ],
    FUNDING: [
        ("rate", pa.float64()),
        ("mark_price", pa.float64()),
        ("index_price", pa.float64()),
        ("next_funding_ts", pa.int64()),
    ],
    LIQUIDATIONS: [
        ("side", pa.string()),
        ("price", pa.float64()),
        ("size", pa.float64()),
        ("trader", pa.string()),
    ],
    OPEN_INTEREST: [("open_interest", pa.float64()), ("open_interest_value", pa.float64())],
}

SCHEMAS: dict[str, pa.Schema] = {ch: pa.schema(_COMMON + _SPECIFIC[ch]) for ch in CHANNELS}


def empty_table(channel: str) -> pa.Table:
    return SCHEMAS[channel].empty_table()


def conform(channel: str, table: pa.Table) -> pa.Table:
    """Select and cast the contract columns of ``channel``; missing nullable columns are filled with nulls."""
    schema = SCHEMAS[channel]
    columns = []
    for field in schema:
        if field.name in table.column_names:
            columns.append(table.column(field.name).cast(field.type))
        else:
            columns.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(columns, schema=schema)
