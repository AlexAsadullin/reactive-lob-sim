"""Fees and the experimental agent's account.

Analytics run in float64, but the agent's cash, position and fees are kept in ``decimal.Decimal`` so that fee
accrual over many small fills is exact and reproducible. Floats enter through ``repr`` (shortest round-trip
form), which avoids binary-expansion noise such as ``Decimal(0.1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from reactive_lob_sim.engine.types import Liquidity, Side

_BPS = Decimal(10_000)


def to_decimal(x: float) -> Decimal:
    return Decimal(repr(float(x)))


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Maker and taker fees in basis points of notional. A negative maker fee is a rebate."""

    maker_bps: Decimal
    taker_bps: Decimal

    @classmethod
    def from_bps(cls, maker_bps: float, taker_bps: float) -> FeeSchedule:
        return cls(to_decimal(maker_bps), to_decimal(taker_bps))

    def fee(self, price: float, qty: float, liquidity: Liquidity) -> Decimal:
        bps = self.maker_bps if liquidity is Liquidity.MAKER else self.taker_bps
        return to_decimal(price) * to_decimal(qty) * bps / _BPS


@dataclass(slots=True)
class Account:
    """Cash and inventory of a single trading agent in one instrument."""

    cash: Decimal = field(default_factory=lambda: Decimal(0))
    position: Decimal = field(default_factory=lambda: Decimal(0))
    fees: Decimal = field(default_factory=lambda: Decimal(0))
    traded_qty: Decimal = field(default_factory=lambda: Decimal(0))

    def apply_fill(self, side: Side, price: float, qty: float, fee: Decimal) -> None:
        q = to_decimal(qty)
        notional = to_decimal(price) * q
        if side is Side.BUY:
            self.position += q
            self.cash -= notional
        else:
            self.position -= q
            self.cash += notional
        self.cash -= fee
        self.fees += fee
        self.traded_qty += q

    def equity(self, mark: float) -> Decimal:
        """Mark-to-market value: cash plus inventory at ``mark`` (fees are already deducted from cash)."""
        return self.cash + self.position * to_decimal(mark)
