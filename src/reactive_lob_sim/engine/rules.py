"""Exchange trading rules of one instrument: whole lots, the price tick and the daily price band.

Quantities are in shares and must be a positive whole number of lots; prices are in roubles and must lie on
the tick grid. The band is ``[band_low_tick, band_high_tick]`` in integer ticks (the low edge rounded up and
the high edge rounded down, so every allowed price is inside the band). A limit order outside the band is
rejected, and a market order never executes outside it (it carries the band edge as its protection price).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reactive_lob_sim.engine.types import RejectReason

# Relative tolerance of the lot and tick grid checks (float64 residue such as 0.30000000000000004).
GRID_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class TradingRules:
    tick: float
    lot: float
    band_low_tick: int | None = None
    band_high_tick: int | None = None

    @classmethod
    def with_band(cls, tick: float, lot: float, low: float, high: float) -> TradingRules:
        if not 0 < low < high:
            raise ValueError("price band must satisfy 0 < low < high")
        low_tick = math.ceil(low / tick - GRID_TOLERANCE)
        return cls(tick, lot, low_tick, math.floor(high / tick + GRID_TOLERANCE))

    def lots(self, qty: float) -> float:
        return qty / self.lot

    def is_whole_lots(self, qty: float) -> bool:
        n = qty / self.lot
        return n >= 1 - GRID_TOLERANCE and abs(n - round(n)) <= GRID_TOLERANCE

    def round_qty(self, qty: float) -> float:
        """Nearest whole number of lots, at least one lot (used by background flow)."""
        return max(round(qty / self.lot), 1) * self.lot

    def floor_qty(self, qty: float) -> float:
        """Whole lots not exceeding ``qty`` (zero when ``qty`` is below one lot)."""
        return math.floor(qty / self.lot + GRID_TOLERANCE) * self.lot

    def on_tick(self, price: float) -> bool:
        k = price / self.tick
        return abs(k - round(k)) <= GRID_TOLERANCE

    def to_tick(self, price: float) -> int:
        return round(price / self.tick)

    def in_band(self, tick: int) -> bool:
        if self.band_low_tick is not None and tick < self.band_low_tick:
            return False
        return self.band_high_tick is None or tick <= self.band_high_tick

    def clamp(self, tick: int) -> int:
        if self.band_low_tick is not None:
            tick = max(tick, self.band_low_tick)
        if self.band_high_tick is not None:
            tick = min(tick, self.band_high_tick)
        return tick

    def protection_tick(self, is_buy: bool) -> int | None:
        """The worst price a market order may reach: the band edge on its side."""
        return self.band_high_tick if is_buy else self.band_low_tick

    def check(self, qty: float, price: float | None) -> RejectReason | None:
        """Why an order of ``qty`` shares at ``price`` breaks the rules, or ``None``."""
        if qty <= 0 or not self.is_whole_lots(qty):
            return RejectReason.INVALID_QTY
        if price is None:
            return None
        if price <= 0 or not self.on_tick(price):
            return RejectReason.INVALID_PRICE
        if not self.in_band(self.to_tick(price)):
            return RejectReason.PRICE_OUT_OF_BAND
        return None
