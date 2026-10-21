"""Value types shared by the matching engine and its callers.

Prices inside the engine are integer ticks, so price comparisons and level keys are exact. Quantities are
float64 in base units; the book treats anything at or below ``qty_tolerance`` as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Side(StrEnum):
    """Order side. ``BUY`` orders rest on the bid side, ``SELL`` orders on the ask side."""

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def book_side(self) -> str:
        """Book-side label of the I1 tape schema (contract 6.2)."""
        return "bid" if self is Side.BUY else "ask"


class OrderType(StrEnum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "POST_ONLY"


class OrderStatus(StrEnum):
    """Order statuses of the shared order API (contract 6.4)."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


class RejectReason(StrEnum):
    INVALID_QTY = "invalid_qty"
    INVALID_PRICE = "invalid_price"
    POST_ONLY_WOULD_CROSS = "post_only_would_cross"
    FOK_INSUFFICIENT_LIQUIDITY = "fok_insufficient_liquidity"
    NO_LIQUIDITY = "no_liquidity"
    UNKNOWN_ORDER = "unknown_order"


class Liquidity(StrEnum):
    MAKER = "maker"
    TAKER = "taker"


class Owner(IntEnum):
    """Who owns an order inside a simulated market. ``EXPERIMENT`` is the strategy under test."""

    EXPERIMENT = 1
    LIQUIDITY = 2
    NOISE = 3
    FUNDAMENTAL = 4
    MOMENTUM = 5
    LIQUIDATION = 6
    GENERATIVE = 7


@dataclass(frozen=True, slots=True)
class OrderRequest:
    owner: int
    side: Side
    order_type: OrderType
    qty: float
    price_tick: int | None = None
    tif: TimeInForce = TimeInForce.GTC


@dataclass(slots=True)
class RestingOrder:
    order_id: int
    owner: int
    side: Side
    price_tick: int
    qty: float
    arrival: int


@dataclass(frozen=True, slots=True)
class Fill:
    maker_order_id: int
    maker_owner: int
    taker_order_id: int
    taker_owner: int
    taker_side: Side
    price_tick: int
    qty: float


@dataclass(frozen=True, slots=True)
class SubmitResult:
    order_id: int
    status: OrderStatus
    fills: tuple[Fill, ...]
    resting_qty: float
    reject_reason: RejectReason | None
    queue_ahead: float
    touched: tuple[tuple[Side, int], ...]

    @property
    def filled_qty(self) -> float:
        return sum(f.qty for f in self.fills)


@dataclass(frozen=True, slots=True)
class CancelResult:
    order_id: int
    cancelled: bool
    qty: float
    touched: tuple[tuple[Side, int], ...]
