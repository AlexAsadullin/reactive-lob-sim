"""Strategy protocol, the strategy runner and the run record."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from reactive_lob_sim.engine.types import OrderStatus, OrderType, Side, TimeInForce
from reactive_lob_sim.venue import AgentFill, MarketBackend, MarketView, OrderAck


class Gateway:
    """Order gateway handed to strategies: forwards to the market and counts submitted quantity."""

    def __init__(self, market: MarketBackend) -> None:
        self._market = market
        self.submitted_qty = 0.0
        self.n_orders = 0
        self.n_rejects = 0

    @property
    def tick_size(self) -> float:
        return self._market.tick_size

    @property
    def lot_size(self) -> float:
        return self._market.lot_size

    def submit(
        self,
        side: Side,
        order_type: OrderType,
        qty: float,
        price: float | None = None,
        tif: TimeInForce = TimeInForce.GTC,
    ) -> OrderAck:
        ack = self._market.submit(side, order_type, qty, price, tif)
        self.n_orders += 1
        if ack.status is OrderStatus.REJECTED:
            self.n_rejects += 1
        else:
            self.submitted_qty += qty
        return ack

    def cancel(self, order_id: int) -> bool:
        return self._market.cancel(order_id)

    def open_orders(self) -> list[int]:
        return self._market.open_orders()

    def cancel_all(self) -> None:
        for oid in self._market.open_orders():
            self._market.cancel(oid)


class Strategy(Protocol):
    strategy_id: str

    def on_start(self, view: MarketView, horizon: float) -> None: ...

    def on_step(self, view: MarketView, gw: Gateway) -> None: ...

    def on_fill(self, fill: AgentFill) -> None: ...


@dataclass
class StrategyRun:
    strategy_id: str
    fills: list[AgentFill]
    submitted_qty: float
    n_orders: int
    n_rejects: int
    start_mid: float
    final_mid: float
    duration_s: float
    meta: dict[str, str | int | float] = field(default_factory=dict)


def run_strategy(
    market: MarketBackend,
    strategy: Strategy,
    duration_s: float,
    decision_interval_s: float,
    start_s: float = 0.0,
) -> StrategyRun:
    """Drive ``strategy`` over ``[start_s, start_s + duration_s]`` with one decision per interval.

    At each decision time the market is advanced, new fills are delivered, and the strategy acts. At the end
    all open orders are cancelled; inventory is marked at the final mid (no forced liquidation).
    """
    gw = Gateway(market)
    market.advance(start_s)
    view = market.view()
    if view.mid is None:
        raise RuntimeError("market has no two-sided book at the strategy start")
    start_mid = view.mid
    strategy.on_start(view, duration_s)
    fills: list[AgentFill] = []
    steps = math.floor(duration_s / decision_interval_s + 1e-9)
    for k in range(1, steps + 1):
        market.advance(start_s + k * decision_interval_s)
        for fill in market.drain_fills():
            fills.append(fill)
            strategy.on_fill(fill)
        strategy.on_step(market.view(), gw)
        for fill in market.drain_fills():
            fills.append(fill)
            strategy.on_fill(fill)
    gw.cancel_all()
    market.advance(start_s + duration_s)
    fills.extend(market.drain_fills())
    final = market.view().mid
    return StrategyRun(
        strategy.strategy_id,
        fills,
        gw.submitted_qty,
        gw.n_orders,
        gw.n_rejects,
        start_mid,
        final if final is not None else start_mid,
        duration_s,
    )
