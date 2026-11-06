"""The interface between strategies and simulated markets.

Every simulator (replay, agent-based, generative and the ground-truth "live" market) implements
:class:`MarketBackend`, so the same strategy object runs unchanged in each of them, which is what the
sim-to-real gap score needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from reactive_lob_sim.engine.types import Liquidity, OrderStatus, OrderType, RejectReason, Side, TimeInForce
from reactive_lob_sim.tape.tape import Tape


@dataclass(frozen=True, slots=True)
class MarketView:
    t: float
    best_bid: float | None
    best_ask: float | None
    bid_size: float
    ask_size: float
    last_trade: float | None
    funding_rate: float
    open_interest: float
    tick_size: float
    lot_size: float = 1.0
    phase: str = "continuous"

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


@dataclass(frozen=True, slots=True)
class OrderAck:
    order_id: int
    status: OrderStatus
    reject_reason: RejectReason | None
    filled_qty: float


@dataclass(frozen=True, slots=True)
class AgentFill:
    """A fill of the experimental agent; the fields mirror the I6 fills export (contract 6.4)."""

    order_id: int
    side: Side
    order_type: OrderType
    ts_submit_s: float
    ts_fill_s: float
    price: float
    qty: float
    fee: Decimal
    liquidity: Liquidity
    arrival_mid: float
    distance_to_touch_bps: float
    queue_ahead_at_entry: float


class MarketBackend(Protocol):
    @property
    def tick_size(self) -> float: ...

    @property
    def lot_size(self) -> float: ...

    @property
    def t(self) -> float: ...

    def advance(self, t: float) -> None:
        """Process market events up to and including simulated time ``t`` (seconds)."""

    def view(self) -> MarketView: ...

    def submit(
        self,
        side: Side,
        order_type: OrderType,
        qty: float,
        price: float | None = None,
        tif: TimeInForce = TimeInForce.GTC,
    ) -> OrderAck: ...

    def cancel(self, order_id: int) -> bool: ...

    def open_orders(self) -> list[int]: ...

    def drain_fills(self) -> list[AgentFill]: ...

    def tape(self) -> Tape | None:
        """The recorded tape, if the backend records one."""
