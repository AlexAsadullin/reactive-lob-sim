"""Strategy outcomes: PnL, fill rate and slippage, from a run record or from contract 6.4 fills.

PnL is cash plus inventory at the final mid, net of fees, in quote currency. Fill rate is filled quantity
over submitted quantity. Slippage is the quantity-weighted mean of side * (price - arrival_mid) /
arrival_mid in basis points (positive is a cost; maker fills usually have negative slippage).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pyarrow as pa

from reactive_lob_sim.engine.accounting import Account, to_decimal
from reactive_lob_sim.engine.types import Side
from reactive_lob_sim.strategies.base import StrategyRun

_BPS = 10_000.0
METRICS = ("pnl", "fill_rate", "slippage_bps")


@dataclass(frozen=True)
class Outcome:
    pnl: float
    fill_rate: float
    slippage_bps: float
    filled_qty: float
    n_fills: int

    def metric(self, name: str) -> float:
        return float(getattr(self, name))


def _outcome(
    sides: list[Side],
    prices: list[float],
    qtys: list[float],
    fees: list[Decimal],
    arrivals: list[float],
    submitted: float | None,
    final_mid: float,
) -> Outcome:
    acct = Account()
    for side, price, qty, fee in zip(sides, prices, qtys, fees, strict=True):
        acct.apply_fill(side, price, qty, fee)
    q = np.asarray(qtys)
    filled = float(q.sum())
    if filled > 0:
        sign = np.asarray([s.sign for s in sides], dtype=np.float64)
        p = np.asarray(prices)
        a = np.asarray(arrivals)
        slip = float(np.sum(q * sign * (p - a) / a) / filled * _BPS)
    else:
        slip = float("nan")
    rate = filled / submitted if submitted else float("nan")
    return Outcome(float(acct.equity(final_mid)), rate, slip, filled, len(qtys))


def outcome_from_run(run: StrategyRun) -> Outcome:
    f = run.fills
    return _outcome(
        [x.side for x in f],
        [x.price for x in f],
        [x.qty for x in f],
        [x.fee for x in f],
        [x.arrival_mid for x in f],
        run.submitted_qty,
        run.final_mid,
    )


def outcomes_from_fills(
    fills: pa.Table,
    final_mid: dict[tuple[str, str], float],
    submitted: dict[tuple[str, str], float] | None = None,
) -> dict[tuple[str, str], Outcome]:
    """Outcomes per ``(strategy_id, account_id)`` from an I6 fills export.

    Contract 6.4 fills carry no submitted quantity or closing mark, so the caller supplies the final mark per
    run, and optionally the submitted quantity (without it the fill rate is NaN and is left out of scores).
    """
    rows = fills.to_pylist()
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in rows:
        groups.setdefault((str(r["strategy_id"]), str(r["account_id"])), []).append(r)
    out: dict[tuple[str, str], Outcome] = {}
    for key in final_mid:
        g = sorted(groups.get(key, []), key=lambda r: int(str(r["ts_fill"])))
        out[key] = _outcome(
            [Side(str(r["side"])) for r in g],
            [float(str(r["price"])) for r in g],
            [float(str(r["qty"])) for r in g],
            [to_decimal(float(str(r["fee"]))) for r in g],
            [float(str(r["arrival_mid"])) for r in g],
            None if submitted is None else submitted.get(key),
            final_mid[key],
        )
    return out
